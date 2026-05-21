package com.nocap.vectorstore

import kotlin.math.sqrt

/**
 * Cosine similarity in [-1, 1]. Higher = more similar.
 * If either vector is zero-length, returns 0.
 */
object Cosine {
    fun similarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "vector dims differ: ${a.size} vs ${b.size}" }
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            val av = a[i]
            val bv = b[i]
            dot += av * bv
            normA += av * av
            normB += bv * bv
        }
        if (normA < 1e-12f || normB < 1e-12f) return 0f
        return dot / (sqrt(normA) * sqrt(normB))
    }

    /**
     * Optimized variant when both vectors are already L2-normalized.
     * Skips the norm computation.
     */
    fun similarityNormalized(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "vector dims differ: ${a.size} vs ${b.size}" }
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot
    }
}
