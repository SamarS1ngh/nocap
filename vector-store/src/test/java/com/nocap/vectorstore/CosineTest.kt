package com.nocap.vectorstore

import org.junit.Assert.assertEquals
import org.junit.Test

class CosineTest {

    @Test
    fun identityIsOne() {
        val v = floatArrayOf(1f, 2f, 3f)
        assertEquals(1f, Cosine.similarity(v, v), 1e-6f)
    }

    @Test
    fun oppositeIsMinusOne() {
        val a = floatArrayOf(1f, 0f)
        val b = floatArrayOf(-1f, 0f)
        assertEquals(-1f, Cosine.similarity(a, b), 1e-6f)
    }

    @Test
    fun orthogonalIsZero() {
        val a = floatArrayOf(1f, 0f)
        val b = floatArrayOf(0f, 1f)
        assertEquals(0f, Cosine.similarity(a, b), 1e-6f)
    }

    @Test
    fun zeroVectorReturnsZero() {
        val a = floatArrayOf(0f, 0f)
        val b = floatArrayOf(1f, 1f)
        assertEquals(0f, Cosine.similarity(a, b), 1e-6f)
    }

    @Test
    fun normalizedVariantMatchesSlowPath() {
        // L2-normalize two arbitrary vectors, then check both paths agree.
        fun norm(v: FloatArray): FloatArray {
            val n = kotlin.math.sqrt(v.sumOf { (it * it).toDouble() }).toFloat()
            return FloatArray(v.size) { v[it] / n }
        }
        val a = norm(floatArrayOf(0.3f, 0.5f, 0.1f, -0.2f))
        val b = norm(floatArrayOf(0.1f, 0.4f, 0.6f, 0.2f))
        assertEquals(
            Cosine.similarity(a, b),
            Cosine.similarityNormalized(a, b),
            1e-5f,
        )
    }
}
