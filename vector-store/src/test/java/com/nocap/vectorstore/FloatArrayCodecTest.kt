package com.nocap.vectorstore

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class FloatArrayCodecTest {

    @Test
    fun roundTripPreservesValues() {
        val input = floatArrayOf(0f, 1f, -1f, 3.14159f, Float.MIN_VALUE, Float.MAX_VALUE)
        val bytes = FloatArrayCodec.encode(input)
        val decoded = FloatArrayCodec.decode(bytes)
        assertArrayEquals(input, decoded, 0f)
    }

    @Test
    fun encodedSizeIsFourBytesPerFloat() {
        val input = FloatArray(384) { it.toFloat() }
        val bytes = FloatArrayCodec.encode(input)
        assertEquals(384 * 4, bytes.size)
    }

    @Test
    fun emptyArrayRoundTrips() {
        val out = FloatArrayCodec.decode(FloatArrayCodec.encode(FloatArray(0)))
        assertEquals(0, out.size)
    }
}
