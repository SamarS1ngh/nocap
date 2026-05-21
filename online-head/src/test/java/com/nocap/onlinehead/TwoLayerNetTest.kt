package com.nocap.onlinehead

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class TwoLayerNetTest {

    @get:Rule
    val tmp = TemporaryFolder()

    @Test
    fun forwardOutputIsInUnitInterval() {
        val net = TwoLayerNet(inputDim = 10, hiddenDim = 4, seed = 42L)
        val x = FloatArray(10) { it.toFloat() }
        val p = net.predict(x)
        assertTrue("p=$p", p in 0f..1f)
    }

    @Test
    fun updateMovesPredictionTowardLabel() {
        val net = TwoLayerNet(inputDim = 6, hiddenDim = 4, seed = 7L)
        val x = floatArrayOf(0.3f, -0.1f, 0.8f, 0.0f, 0.2f, -0.5f)
        val y = 1.0f
        val pBefore = net.predict(x)
        repeat(50) { net.update(x, y, lr = 0.1f) }
        val pAfter = net.predict(x)
        assertTrue("expected pAfter > pBefore (before=$pBefore, after=$pAfter)", pAfter > pBefore)
    }

    @Test
    fun convergesOnSimpleAndPattern() {
        // Two-layer net should be able to learn an XOR-like pattern that LR cannot.
        // Use 2-input → 4-hidden → 1-output; target is XOR.
        val net = TwoLayerNet(inputDim = 2, hiddenDim = 8, seed = 123L)
        val examples = listOf(
            floatArrayOf(0f, 0f) to 0f,
            floatArrayOf(0f, 1f) to 1f,
            floatArrayOf(1f, 0f) to 1f,
            floatArrayOf(1f, 1f) to 0f,
        )
        // Train many epochs
        repeat(20_000) {
            val (x, y) = examples[it % examples.size]
            net.update(x, y, lr = 0.1f)
        }
        // After training, predictions should match labels within 0.2.
        for ((x, y) in examples) {
            val p = net.predict(x)
            assertEquals("XOR input=${x.toList()} expected=$y got=$p", y, p, 0.2f)
        }
    }

    @Test
    fun saveAndLoadRoundTrip() {
        val net = TwoLayerNet(inputDim = 8, hiddenDim = 5, seed = 99L)
        val x = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f)
        // Train a bit so weights diverge from init.
        repeat(20) { net.update(x, 1f, lr = 0.05f) }
        val before = net.predict(x)

        val file = File(tmp.root, "net.bin")
        net.save(file)

        val restored = TwoLayerNet(inputDim = 8, hiddenDim = 5, seed = 0L)
        // Sanity: fresh net's prediction should differ from trained.
        assertNotEquals(before, restored.predict(x))

        restored.loadInto(file)
        assertEquals(before, restored.predict(x), 1e-6f)
    }

    @Test
    fun adaptiveBurstFiresOnConsistentHighLoss() {
        // Net is barely initialized. Feed it 6 maximally-wrong examples in a row —
        // loss should sit far above the 0.7 initial baseline EMA's drift threshold.
        val net = TwoLayerNet(inputDim = 4, hiddenDim = 4, seed = 0L)
        val x = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
        repeat(8) { net.update(x, 1f, lr = 0.001f) }
        // After ≥5 high-loss updates a burst should have fired
        assertTrue("expected adaptiveBurstCount > 0", net.adaptiveBurstCount > 0)
    }

    @Test
    fun shapeMismatchOnLoadThrows() {
        val net = TwoLayerNet(inputDim = 4, hiddenDim = 3, seed = 0L)
        val file = File(tmp.root, "net.bin")
        net.save(file)
        val wrongShape = TwoLayerNet(inputDim = 5, hiddenDim = 3, seed = 0L)
        try {
            wrongShape.loadInto(file)
            throw AssertionError("expected IllegalArgumentException")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("shape mismatch"))
        }
    }
}
