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
     * Insert a new vector. Label may be null (we don't have feedback yet);
     * such rows are stored but ignored by [nearest] until [setLabel] sets one.
     */
    suspend fun append(
        vector: FloatArray,
        label: Float?,
        packageName: String,
        postedAt: Long,
        metaJson: String? = null,
    ): Long = dao.insert(
        VectorRow(
            vectorBytes = FloatArrayCodec.encode(vector),
            label = label,
            packageName = packageName,
            postedAt = postedAt,
            metaJson = metaJson,
        )
    )

    suspend fun setLabel(id: Long, label: Float) = dao.setLabel(id, label)

    /**
     * Top-k labeled rows by cosine similarity. Returns descending sim.
     * Assumes stored vectors are L2-normalized (EmbeddingEngine guarantees this).
     */
    suspend fun nearest(query: FloatArray, k: Int): List<Hit> {
        val rows = dao.allLabeled()
        if (rows.isEmpty()) return emptyList()

        val scored = ArrayList<Hit>(rows.size)
        for (row in rows) {
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
