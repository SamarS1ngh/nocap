package com.nocap.vectorstore

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Pack/unpack FloatArray to a compact ByteArray for Room storage.
 * Little-endian, raw IEEE-754 floats. No header — caller knows dimension.
 */
object FloatArrayCodec {
    fun encode(vec: FloatArray): ByteArray {
        val buf = ByteBuffer.allocate(vec.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (v in vec) buf.putFloat(v)
        return buf.array()
    }

    fun decode(bytes: ByteArray): FloatArray {
        require(bytes.size % 4 == 0) { "byte length must be multiple of 4, got ${bytes.size}" }
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(bytes.size / 4)
        for (i in out.indices) out[i] = buf.float
        return out
    }
}
