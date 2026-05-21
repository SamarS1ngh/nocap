package com.nocap.onlinehead

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * 435 → 100 → 1 feed-forward net with ReLU hidden + sigmoid output.
 * Trained one example at a time via SGD with manual backprop.
 *
 * Weight count: 435*100 + 100 + 100 + 1 = 43,701.
 *
 * Init: He (Kaiming) for hidden weights (good for ReLU), zero biases.
 *   stddev = sqrt(2 / fan_in)
 *
 * Loss: binary cross-entropy. Gradient through sigmoid simplifies to (P - y).
 */
class TwoLayerNet(
    val inputDim: Int = DEFAULT_INPUT_DIM,
    val hiddenDim: Int = DEFAULT_HIDDEN_DIM,
    seed: Long = 1234L,
) {

    // W1[hiddenDim][inputDim], b1[hiddenDim], W2[hiddenDim], b2 scalar.
    val w1: Array<FloatArray> = Array(hiddenDim) { FloatArray(inputDim) }
    val b1: FloatArray = FloatArray(hiddenDim)
    val w2: FloatArray = FloatArray(hiddenDim)
    var b2: Float = 0f
        private set

    /** Number of times update() has been called. Used for LR decay schedule. */
    var updateCount: Long = 0L
        private set

    /**
     * Ring buffer of recent BCE losses, one per update() call. Read via [lossHistory].
     * In-memory only; not persisted across process restarts.
     */
    private val lossRing = FloatArray(LOSS_RING_SIZE)
    private var lossWriteIndex = 0
    private var lossFilled = 0

    init {
        val rng = Random(seed)
        val stddev1 = sqrt(2.0 / inputDim).toFloat()
        val stddev2 = sqrt(2.0 / hiddenDim).toFloat()
        for (i in 0 until hiddenDim) {
            val row = w1[i]
            for (j in 0 until inputDim) row[j] = (rng.nextGaussian() * stddev1).toFloat()
            w2[i] = (rng.nextGaussian() * stddev2).toFloat()
        }
    }

    data class Forward(
        val z1: FloatArray,
        val h: FloatArray,
        val z2: Float,
        val p: Float,
    )

    /** Forward pass. Caches intermediate activations for use during update(). */
    fun forward(x: FloatArray): Forward {
        require(x.size == inputDim) { "expected $inputDim inputs, got ${x.size}" }

        val z1 = FloatArray(hiddenDim)
        val h = FloatArray(hiddenDim)
        for (i in 0 until hiddenDim) {
            var sum = b1[i]
            val row = w1[i]
            for (j in 0 until inputDim) sum += row[j] * x[j]
            z1[i] = sum
            h[i] = if (sum > 0f) sum else 0f
        }

        var z2 = b2
        for (i in 0 until hiddenDim) z2 += w2[i] * h[i]
        val p = sigmoid(z2)

        return Forward(z1, h, z2, p)
    }

    /**
     * One SGD step on a single (x, y) example.
     * @param y true label in [0, 1]
     * @param lr learning rate
     * @return loss BEFORE this update (handy for monitoring)
     */
    /**
     * Decayed LR: halves every [DECAY_EVERY_N_UPDATES] updates, floored at [MIN_LR].
     * Pass the base LR; receive a possibly-smaller effective LR.
     */
    fun decayedLr(baseLr: Float = DEFAULT_LR): Float {
        val halvings = updateCount / DECAY_EVERY_N_UPDATES
        val factor = if (halvings <= 0) 1f else 1f / (1L shl halvings.coerceAtMost(20).toInt())
        return (baseLr * factor).coerceAtLeast(MIN_LR)
    }

    fun update(x: FloatArray, y: Float, lr: Float = DEFAULT_LR): Float {
        val fwd = forward(x)
        val p = fwd.p
        val h = fwd.h
        val z1 = fwd.z1
        updateCount++

        // dL/dz2 = P - y   (collapse of sigmoid + BCE)
        val dz2 = p - y

        // Output layer grads + grad flowing back into hidden.
        // dL/dw2_i = dz2 * h_i ;  dL/db2 = dz2
        // dL/dh_i = dz2 * w2_i ;  dL/dz1_i = dL/dh_i if z1_i>0 else 0
        val dz1 = FloatArray(hiddenDim)
        for (i in 0 until hiddenDim) {
            // Hidden→output update (read w2 first, then mutate w2/b2 later — order doesn't matter
            // here since we cache the old w2 in dz1 calculation).
            val w2i = w2[i]
            val dh = dz2 * w2i
            dz1[i] = if (z1[i] > 0f) dh else 0f
            w2[i] -= lr * dz2 * h[i]
        }
        b2 -= lr * dz2

        // Input→hidden updates
        for (i in 0 until hiddenDim) {
            val g = dz1[i]
            if (g == 0f) continue  // dead ReLU, skip work
            val row = w1[i]
            for (j in 0 until inputDim) row[j] -= lr * g * x[j]
            b1[i] -= lr * g
        }

        // Loss before update (for caller logging)
        val eps = 1e-9f
        val loss = -(y * kotlin.math.ln(p.coerceAtLeast(eps)) +
            (1 - y) * kotlin.math.ln((1 - p).coerceAtLeast(eps)))
        lossRing[lossWriteIndex] = loss
        lossWriteIndex = (lossWriteIndex + 1) % LOSS_RING_SIZE
        if (lossFilled < LOSS_RING_SIZE) lossFilled++
        return loss
    }

    /**
     * Returns recorded losses in chronological order, oldest first.
     * Length = min(updateCount, LOSS_RING_SIZE).
     */
    fun lossHistory(): FloatArray {
        if (lossFilled == 0) return FloatArray(0)
        val out = FloatArray(lossFilled)
        if (lossFilled < LOSS_RING_SIZE) {
            // ring not yet wrapped — write index is also the count
            System.arraycopy(lossRing, 0, out, 0, lossFilled)
        } else {
            // ring wrapped — oldest entry sits at write index
            val tail = LOSS_RING_SIZE - lossWriteIndex
            System.arraycopy(lossRing, lossWriteIndex, out, 0, tail)
            System.arraycopy(lossRing, 0, out, tail, lossWriteIndex)
        }
        return out
    }

    /** Mean of the most recent [window] losses, or null if not enough data. */
    fun recentLoss(window: Int = 100): Float? {
        if (lossFilled < window) return null
        val history = lossHistory()
        var sum = 0f
        for (i in (history.size - window) until history.size) sum += history[i]
        return sum / window
    }

    fun predict(x: FloatArray): Float = forward(x).p

    fun save(file: File) {
        file.parentFile?.mkdirs()
        DataOutputStream(file.outputStream().buffered()).use { out ->
            out.writeInt(MAGIC)
            out.writeInt(VERSION)
            out.writeInt(inputDim)
            out.writeInt(hiddenDim)
            for (i in 0 until hiddenDim) {
                val row = w1[i]
                for (j in 0 until inputDim) out.writeFloat(row[j])
            }
            for (i in 0 until hiddenDim) out.writeFloat(b1[i])
            for (i in 0 until hiddenDim) out.writeFloat(w2[i])
            out.writeFloat(b2)
        }
    }

    fun loadInto(file: File) {
        DataInputStream(file.inputStream().buffered()).use { input ->
            val magic = input.readInt()
            require(magic == MAGIC) { "bad magic: $magic" }
            val version = input.readInt()
            require(version == VERSION) { "unsupported version: $version" }
            val savedInput = input.readInt()
            val savedHidden = input.readInt()
            require(savedInput == inputDim && savedHidden == hiddenDim) {
                "shape mismatch: file=$savedInput x $savedHidden, net=$inputDim x $hiddenDim"
            }
            for (i in 0 until hiddenDim) {
                val row = w1[i]
                for (j in 0 until inputDim) row[j] = input.readFloat()
            }
            for (i in 0 until hiddenDim) b1[i] = input.readFloat()
            for (i in 0 until hiddenDim) w2[i] = input.readFloat()
            b2 = input.readFloat()
        }
    }

    private fun sigmoid(x: Float): Float = (1.0 / (1.0 + exp(-x.toDouble()))).toFloat()

    private fun Random.nextGaussian(): Double {
        // Box-Muller. Discards the second sample. Fine for one-off init.
        var u1: Double
        do { u1 = nextDouble() } while (u1 <= 1e-12)
        val u2 = nextDouble()
        return sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * Math.PI * u2)
    }

    companion object {
        const val DEFAULT_INPUT_DIM = 435
        const val DEFAULT_HIDDEN_DIM = 100
        const val DEFAULT_LR = 0.01f
        const val MIN_LR = 1e-5f
        const val DECAY_EVERY_N_UPDATES = 5_000L
        const val LOSS_RING_SIZE = 1_000

        private const val MAGIC = 0x4F484541  // "OHEA"
        private const val VERSION = 1
    }
}
