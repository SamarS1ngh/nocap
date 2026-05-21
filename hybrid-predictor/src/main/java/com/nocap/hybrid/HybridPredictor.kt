package com.nocap.hybrid

import com.nocap.embedding.EmbeddingEngine
import com.nocap.onlinehead.StructuredFeatures
import com.nocap.onlinehead.TwoLayerNet
import com.nocap.vectorstore.VectorStore
import kotlin.math.abs

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
        val kNeighbours: Int = 5,
        val knnSimilarityThreshold: Float = 0.7f,
        val knnMinNeighbours: Int = 3,
        val disagreementThreshold: Float = 0.4f,
        val coldStartUnder: Int = 30,        // store size below this → LLM only
        val suppressHeadUnder: Int = 500,    // store size below this → ignore head vote
        /** Store row count that triggers age-based pruning on next learn(). */
        val pruneOver: Int = 50_000,
        /** Drop rows older than this when pruning fires. Default 90 days. */
        val pruneCutoffAgeMs: Long = 90L * 24 * 60 * 60 * 1000,
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

        val neighbours = store.nearest(embedding, config.kNeighbours)
        val pKnn = computeKnn(neighbours)
        val pHead = if (storeSize >= config.suppressHeadUnder) head.predict(combined) else null

        return decide(ctx, text, neighbours, pKnn, pHead)
    }

    /** Apply a labeled example to both store and head. LR is decayed by the head's own schedule. */
    suspend fun learn(
        input: StructuredFeatures.Input,
        text: String,
        label: Float,
        baseLr: Float = TwoLayerNet.DEFAULT_LR,
    ) {
        val embedding = engine.encode(text)
        val structured = features.extract(input)
        val combined = concat(embedding, structured)
        store.append(
            vector = embedding,
            label = label,
            packageName = input.packageName,
            postedAt = input.postedAtMs,
        )
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

    private fun computeKnn(neighbours: List<VectorStore.Hit>): Float? {
        if (neighbours.size < config.knnMinNeighbours) return null
        val topSim = neighbours.first().similarity
        if (topSim < config.knnSimilarityThreshold) return null
        var sum = 0f
        for (h in neighbours) sum += h.label
        return sum / neighbours.size
    }

    private fun concat(embedding: FloatArray, structured: FloatArray): FloatArray {
        val out = FloatArray(embedding.size + structured.size)
        System.arraycopy(embedding, 0, out, 0, embedding.size)
        System.arraycopy(structured, 0, out, embedding.size, structured.size)
        return out
    }
}
