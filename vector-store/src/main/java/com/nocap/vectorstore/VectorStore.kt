package com.nocap.vectorstore

import android.content.Context

/**
 * Facade over Room + cosine kNN. Used by hybrid-predictor as the "memory book."
 *
 * Brute-force scan: O(N) per query. Fine up to ~100K labeled rows on a modern phone.
 * Beyond that, swap in HNSW (hnswlib via JNI) without changing this API.
 */
class VectorStore private constructor(
    private val dao: VectorDao,
) {

    data class Hit(
        val id: Long,
        val similarity: Float,
        val label: Float,
        val packageName: String,
        val postedAt: Long,
        val metaJson: String?,
    )

    /**
     * Insert a new vector with no dedup. Prefer [upsert] when you have a
     * notification key; this exists for backfills/tests and the legacy path.
     */
    suspend fun append(
        vector: FloatArray,
        label: Float?,
        packageName: String,
        postedAt: Long,
        notificationKey: String? = null,
        metaJson: String? = null,
    ): Long = dao.insert(
        VectorRow(
            vectorBytes = FloatArrayCodec.encode(vector),
            label = label,
            packageName = packageName,
            postedAt = postedAt,
            notificationKey = notificationKey,
            metaJson = metaJson,
        )
    )

    /**
     * Idempotent write keyed by [notificationKey]. If a row already exists for
     * the key, ONLY its label is updated (the embedding for that notification
     * is fixed — its text didn't change). If no row exists, a new one is
     * inserted with the given vector.
     *
     * Use this from training paths so re-classifying a notification doesn't
     * leave contradictory duplicate vectors in the store.
     */
    suspend fun upsert(
        notificationKey: String,
        vector: FloatArray,
        label: Float,
        packageName: String,
        postedAt: Long,
        metaJson: String? = null,
    ): Long {
        val existing = dao.findIdByKey(notificationKey)
        if (existing != null) {
            dao.updateLabelByKey(notificationKey, label)
            return existing
        }
        return dao.insert(
            VectorRow(
                vectorBytes = FloatArrayCodec.encode(vector),
                label = label,
                packageName = packageName,
                postedAt = postedAt,
                notificationKey = notificationKey,
                metaJson = metaJson,
            )
        )
    }

    suspend fun setLabel(id: Long, label: Float) = dao.setLabel(id, label)

    /**
     * Top-k labeled rows by cosine similarity. Returns descending sim.
     * Assumes stored vectors are L2-normalized (EmbeddingEngine guarantees this).
     *
     * When [packageName] is non-null, only rows from that package are considered.
     * Same-app neighbours keep the kNN vote from bleeding across apps — e.g. a job
     * alert shouldn't borrow "want" votes from chat notifications that merely embed
     * close to it. Callers fall back to the global (packageName = null) search when
     * an app has too few of its own labeled rows to vote.
     */
    suspend fun nearest(query: FloatArray, k: Int, packageName: String? = null): List<Hit> {
        val rows = dao.allLabeled()
        if (rows.isEmpty()) return emptyList()

        val scored = ArrayList<Hit>(rows.size)
        for (row in rows) {
            if (packageName != null && row.packageName != packageName) continue
            val vec = FloatArrayCodec.decode(row.vectorBytes)
            if (vec.size != query.size) continue  // skip dim mismatch (model upgrade scenario)
            val sim = Cosine.similarityNormalized(query, vec)
            scored += Hit(
                id = row.id,
                similarity = sim,
                label = row.label!!,
                packageName = row.packageName,
                postedAt = row.postedAt,
                metaJson = row.metaJson,
            )
        }
        scored.sortByDescending { it.similarity }
        return if (scored.size <= k) scored else scored.subList(0, k)
    }

    suspend fun size(): Int = dao.labeledCount()
    suspend fun total(): Int = dao.totalCount()

    suspend fun prune(strategy: PruneStrategy) {
        when (strategy) {
            is PruneStrategy.OlderThan -> dao.deleteOlderThan(strategy.cutoffMs)
            PruneStrategy.All -> dao.deleteAll()
        }
    }

    /**
     * If total row count exceeds [maxRows], delete rows whose postedAt is older
     * than now - [maxAgeMs]. No-op below the threshold.
     */
    suspend fun pruneIfOversized(maxRows: Int, maxAgeMs: Long) {
        if (dao.totalCount() <= maxRows) return
        val cutoff = System.currentTimeMillis() - maxAgeMs
        dao.deleteOlderThan(cutoff)
    }

    sealed interface PruneStrategy {
        data class OlderThan(val cutoffMs: Long) : PruneStrategy
        data object All : PruneStrategy
    }

    companion object {
        fun create(context: Context): VectorStore =
            VectorStore(VectorDatabase.get(context).vectors())
    }
}
