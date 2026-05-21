package com.nocap.app

import android.content.Context
import android.util.Log
import com.nocap.embedding.EmbeddingEngine
import com.nocap.hybrid.HybridPredictor
import com.nocap.onlinehead.StructuredFeatures
import com.nocap.onlinehead.TwoLayerNet
import com.nocap.vectorstore.VectorStore
import kotlinx.coroutines.runBlocking
import java.io.File

/**
 * Holds the lazy-built [HybridPredictor] for the app. The predictor stays null
 * if assets (minilm.tflite / vocab.txt) aren't bundled yet — listener falls
 * back to the legacy Gemini-only flow in that case.
 *
 * Also owns disk persistence of the [TwoLayerNet] (~50K floats → ~175 KB binary).
 */
class HybridFactory(private val app: NocapApp) {

    private val headFile: File = File(app.filesDir, "online_head.bin")
    private var headRef: TwoLayerNet? = null

    val predictor: HybridPredictor? by lazy { tryBuild() }

    private fun tryBuild(): HybridPredictor? = runCatching {
        val engine = EmbeddingEngine.get(app)
        val store = VectorStore.create(app)

        val head = TwoLayerNet().also { net ->
            if (headFile.exists()) {
                runCatching { net.loadInto(headFile) }
                    .onFailure { Log.w(TAG, "load head failed: ${it.message}") }
            }
            headRef = net
        }

        val topPkgs = runBlocking { app.database.notifications().topPackages(30) }
        val features = StructuredFeatures(topPkgs)

        val alpha = runBlocking { app.prefs.getAlpha() }
        val config = HybridPredictor.Config(alpha = alpha)

        HybridPredictor(
            engine = engine,
            store = store,
            head = head,
            features = features,
            config = config,
            llmFallback = ::geminiFallback,
        )
    }.onFailure { Log.w(TAG, "predictor unavailable: ${it.message}") }.getOrNull()

    private suspend fun geminiFallback(text: String, ctx: HybridPredictor.PredictContext): Float? {
        val openAi = app.prefs.getOpenAiKey()
        val gemini = app.prefs.getGeminiKey()
        if (openAi.isNullOrBlank() && gemini.isNullOrBlank()) return null
        val result = app.gemini.classify(
            openAiKey = openAi,
            geminiKey = gemini,
            packageName = ctx.packageName,
            title = ctx.title,
            body = ctx.body,
        )
        return result.getOrNull()?.let { (it.importance.coerceIn(0, 10) / 10f) }
    }

    /** Persist the trained head weights so they survive process death. */
    fun persistHead() {
        val net = headRef ?: return
        runCatching { net.save(headFile) }
            .onFailure { Log.w(TAG, "persist head failed: ${it.message}") }
    }

    /** Snapshot of head's loss history for the diagnostics chart. */
    fun lossHistory(): FloatArray = headRef?.lossHistory() ?: FloatArray(0)

    /** Mean of last [window] losses, null if head has trained fewer steps. */
    fun recentLoss(window: Int = 100): Float? = headRef?.recentLoss(window)

    /** Total update() calls on the head. */
    fun headUpdateCount(): Long = headRef?.updateCount ?: 0L

    companion object {
        private const val TAG = "HybridFactory"
    }
}
