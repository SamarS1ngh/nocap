package com.nocap.embedding

import android.content.Context
import android.content.res.AssetFileDescriptor
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/**
 * Turns text → 384-dim semantic vector using sentence-transformers/all-MiniLM-L6-v2 converted to TFLite.
 *
 * Bundled assets required (drop into embedding-engine/src/main/assets):
 *   minilm.tflite   — model file, ~90MB. See assets/README.md for source.
 *   vocab.txt       — BERT vocab (one piece per line).
 *
 * Expected TFLite graph signature:
 *   inputs:  input_ids (1, 128, int32), attention_mask (1, 128, int32), token_type_ids (1, 128, int32)
 *   output:  token_embeddings (1, 128, 384, float32)
 *
 * Pipeline: tokenize → run model → mean-pool over valid tokens → L2-normalize → 384 floats.
 *
 * Thread safety: TFLite Interpreter is not thread-safe. Calls are serialized via @Synchronized.
 * For higher throughput, instantiate per worker thread.
 */
class EmbeddingEngine private constructor(
    private val interpreter: Interpreter,
    private val tokenizer: WordPieceTokenizer,
) {

    /**
     * Encode a single string to a 384-dim L2-normalized vector.
     * Empty/blank input returns a zero vector (no-op marker).
     */
    @Synchronized
    fun encode(text: String): FloatArray {
        if (text.isBlank()) return FloatArray(EMBED_DIM)

        val enc = tokenizer.encode(text, maxSeqLen = SEQ_LEN)

        val inputIds = intBuffer(enc.inputIds)
        val attentionMask = intBuffer(enc.attentionMask)
        val tokenTypeIds = intBuffer(enc.tokenTypeIds)

        val output = Array(1) { Array(SEQ_LEN) { FloatArray(EMBED_DIM) } }

        interpreter.runForMultipleInputsOutputs(
            arrayOf<Any>(inputIds, attentionMask, tokenTypeIds),
            mapOf(0 to output),
        )

        return l2Normalize(meanPool(output[0], enc.attentionMask))
    }

    private fun meanPool(tokenEmbeds: Array<FloatArray>, mask: IntArray): FloatArray {
        val out = FloatArray(EMBED_DIM)
        var count = 0
        for (t in tokenEmbeds.indices) {
            if (mask[t] == 0) continue
            count++
            val row = tokenEmbeds[t]
            for (d in 0 until EMBED_DIM) out[d] += row[d]
        }
        if (count == 0) return out
        val inv = 1f / count
        for (d in 0 until EMBED_DIM) out[d] *= inv
        return out
    }

    private fun l2Normalize(vec: FloatArray): FloatArray {
        var sumSq = 0f
        for (v in vec) sumSq += v * v
        val norm = sqrt(sumSq)
        if (norm < 1e-12f) return vec
        val inv = 1f / norm
        for (i in vec.indices) vec[i] *= inv
        return vec
    }

    private fun intBuffer(arr: IntArray): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(arr.size * 4).order(ByteOrder.nativeOrder())
        for (v in arr) buf.putInt(v)
        buf.rewind()
        return buf
    }

    fun close() = interpreter.close()

    companion object {
        const val EMBED_DIM = 384
        const val SEQ_LEN = 128

        private const val MODEL_ASSET = "minilm.tflite"
        private const val VOCAB_ASSET = "vocab.txt"

        @Volatile
        private var instance: EmbeddingEngine? = null

        fun get(context: Context): EmbeddingEngine =
            instance ?: synchronized(this) {
                instance ?: build(context.applicationContext).also { instance = it }
            }

        private fun build(context: Context): EmbeddingEngine {
            val model = loadModel(context, MODEL_ASSET)
            val options = Interpreter.Options().apply {
                setNumThreads(2)
                setUseXNNPACK(true)
            }
            val interpreter = Interpreter(model, options)
            val tokenizer = WordPieceTokenizer.fromAssets(context, VOCAB_ASSET)
            return EmbeddingEngine(interpreter, tokenizer)
        }

        private fun loadModel(context: Context, asset: String): MappedByteBuffer {
            val afd: AssetFileDescriptor = context.assets.openFd(asset)
            FileInputStream(afd.fileDescriptor).use { fis ->
                val channel = fis.channel
                return channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            }
        }
    }
}
