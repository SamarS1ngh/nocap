package com.nocap.hybrid

import com.nocap.embedding.EmbeddingEngine
import com.nocap.onlinehead.StructuredFeatures
import com.nocap.onlinehead.TwoLayerNet
import com.nocap.vectorstore.VectorStore
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max

/**
 * Orchestrator. Blends kNN + online-head, falls back to an LLM when local signals
 * disagree or the store hasn't warmed up.
 *
 * Wire LLM at the app layer via the [llmFallback] lambda — the predictor itself
 * has no network dependency.
 *
 * NOT thread-safe. Caller should serialize calls (e.g., one CoroutineScope on Dispatchers.IO).
 */
class HybridPredictor(
    private val engine: EmbeddingEngine,
    private val store: VectorStore,
    private val head: TwoLayerNet,
    private val features: StructuredFeatures,
    private val config: Config = Config(),
    private val llmFallback: (suspend (text: String, ctx: PredictContext) -> Float?)? = null,
) {

    data class Config(
        val alpha: Float = 0.6f,
        /**
         * Number of nearest-neighbour past notifications consulted per prediction.
         * Bigger k = smoother / more stable. Smaller k = more reactive to rare patterns.
         */
        val kNeighbours: Int = 10,
        val knnSimilarityThreshold: Float = 0.7f,
        val knnMinNeighbours: Int = 3,
        val disagreementThreshold: Float = 0.4f,
        val coldStartUnder: Int = 30,        // store size below this → LLM only
        // Store size below this → ignore the online head's vote (kNN only).
        // The store grows ONLY on explicit user feedback, so 500 was effectively
        // unreachable — the head trained on every feedback event but never got to
        // vote, leaving every prediction kNN-only and mushy. 50 lets the head start
        // contributing once it has a real (if small) training signal.
        val suppressHeadUnder: Int = 50,
        /** Store row count that triggers age-based pruning on next learn(). */
        val pruneOver: Int = 50_000,
        /** Drop rows older than this when pruning fires. Default 90 days. */
        val pruneCutoffAgeMs: Long = 90L * 24 * 60 * 60 * 1000,
        /**
         * Half-life (ms) for exponential recency weighting of kNN votes.
         * A neighbour's vote is multiplied by max(knnMinWeight, exp(-age/halfLife)).
         * Default 180 days: recent votes dominate but ancient ones still contribute.
         */
        val knnRecencyHalfLifeMs: Long = 180L * 24 * 60 * 60 * 1000,
        /** Floor for kNN vote weight — old neighbours can never go fully silent. */
        val knnMinWeight: Float = 0.2f,
    )

    enum class Source { LLM_COLD_START, KNN_ONLY, HEAD_ONLY, BLEND, LLM_DISAGREEMENT, FALLBACK_DEFAULT }

    data class PredictContext(
        val packageName: String,
        val title: String,
        val body: String,
        val postedAtMs: Long,
    )

    data class Prediction(
        val importance: Float,
        val source: Source,
        val pKnn: Float?,
        val pHead: Float?,
        val neighbours: List<VectorStore.Hit>,
    )

    suspend fun predict(input: StructuredFeatures.Input, text: String): Prediction {
        val ctx = PredictContext(input.packageName, input.title, input.body, input.postedAtMs)

        val storeSize = store.size()

        // Phase 0: cold start — too little data to trust local signals.
        if (storeSize < config.coldStartUnder) {
            val p = llmFallback?.invoke(text, ctx)
            val src = if (p == null) Source.FALLBACK_DEFAULT else Source.LLM_COLD_START
            return Prediction(p ?: 0.5f, src, null, null, emptyList())
        }

        val embedding = engine.encode(text)
        val structured = features.extract(input)
        val combined = concat(embedding, structured)

        // Prefer same-app neighbours so the kNN vote reflects how THIS app's
        // notifications have been treated, not whatever happens to embed nearby
        // across every app. Fall back to the global search when the app hasn't
        // accumulated enough of its own labeled rows to vote on its own.
        val sameApp = store.nearest(embedding, config.kNeighbours, input.packageName)
        val neighbours =
            if (sameApp.size >= config.knnMinNeighbours) sameApp
            else store.nearest(embedding, config.kNeighbours)
        val nowMs = System.currentTimeMillis()
        val pKnn = computeKnn(neighbours, nowMs)
        val pHead = if (storeSize >= config.suppressHeadUnder) head.predict(combined) else null

        return decide(ctx, text, neighbours, pKnn, pHead)
    }

    /**
     * Apply a labeled example to both store and head. LR is decayed by the
     * head's own schedule.
     *
     * @param notificationKey StatusBarNotification.key — used as the vector-store
     *        dedup key so re-classifying the same notification doesn't leave
     *        contradictory duplicate vectors. Null means "no dedup; always append"
     *        (only legitimate for tests or backfills).
     */
    suspend fun learn(
        input: StructuredFeatures.Input,
        text: String,
        label: Float,
        notificationKey: String?,
        baseLr: Float = TwoLayerNet.DEFAULT_LR,
    ) {
        val embedding = engine.encode(text)
        val structured = features.extract(input)
        val combined = concat(embedding, structured)
        if (notificationKey != null) {
            store.upsert(
                notificationKey = notificationKey,
                vector = embedding,
                label = label,
                packageName = input.packageName,
                postedAt = input.postedAtMs,
            )
        } else {
            store.append(
                vector = embedding,
                label = label,
                packageName = input.packageName,
                postedAt = input.postedAtMs,
            )
        }
        head.update(combined, label, head.decayedLr(baseLr))
        store.pruneIfOversized(config.pruneOver, config.pruneCutoffAgeMs)
    }

    private suspend fun decide(
        ctx: PredictContext,
        text: String,
        neighbours: List<VectorStore.Hit>,
        pKnn: Float?,
        pHead: Float?,
    ): Prediction = when {
        pKnn != null && pHead != null -> {
            if (abs(pKnn - pHead) > config.disagreementThreshold) {
                // Disagreement is recoverable — if LLM unavailable, use the midpoint of local signals.
                val p = llmFallback?.invoke(text, ctx) ?: ((pKnn + pHead) / 2f)
                Prediction(p, Source.LLM_DISAGREEMENT, pKnn, pHead, neighbours)
            } else {
                val a = config.alpha
                val p = a * pKnn + (1 - a) * pHead
                Prediction(p, Source.BLEND, pKnn, pHead, neighbours)
            }
        }
        pKnn != null -> Prediction(pKnn, Source.KNN_ONLY, pKnn, null, neighbours)
        pHead != null -> Prediction(pHead, Source.HEAD_ONLY, null, pHead, neighbours)
        else -> {
            val p = llmFallback?.invoke(text, ctx) ?: 0.5f
            Prediction(p, Source.FALLBACK_DEFAULT, null, null, neighbours)
        }
    }

    /**
     * Recency-weighted mean of neighbour labels.
     * weight(i) = max(knnMinWeight, exp(-age_i / halfLife))
     */
    private fun computeKnn(neighbours: List<VectorStore.Hit>, nowMs: Long): Float? {
        if (neighbours.size < config.knnMinNeighbours) return null
        val topSim = neighbours.first().similarity
        if (topSim < config.knnSimilarityThreshold) return null

        var sumW = 0f
        var sumWL = 0f
        for (h in neighbours) {
            val ageMs = (nowMs - h.postedAt).coerceAtLeast(0L)
            val expWeight = exp(-ageMs.toDouble() / config.knnRecencyHalfLifeMs).toFloat()
            val w = max(config.knnMinWeight, expWeight)
            sumW += w
            sumWL += w * h.label
        }
        return if (sumW <= 0f) null else sumWL / sumW
    }

    private fun concat(embedding: FloatArray, structured: FloatArray): FloatArray {
        val out = FloatArray(embedding.size + structured.size)
        System.arraycopy(embedding, 0, out, 0, embedding.size)
        System.arraycopy(structured, 0, out, embedding.size, structured.size)
        return out
    }
}
