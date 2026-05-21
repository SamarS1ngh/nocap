package com.nocap.app.listener

import android.content.ComponentName
import android.content.Context
import android.os.PowerManager
import android.provider.Settings
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import com.nocap.app.NocapApp
import com.nocap.app.data.CapturedNotification
import com.nocap.app.diag.PredictionPayload
import com.nocap.app.gemini.FeedbackExample
import com.nocap.hybrid.HybridPredictor
import com.nocap.onlinehead.StructuredFeatures
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

class NocapNotificationListenerService : NotificationListenerService() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onListenerConnected() {
        super.onListenerConnected()
        // Warm the hybrid predictor as soon as the system binds us — without this,
        // the FIRST notification that arrives while the app is fully backgrounded
        // pays a 90MB TFLite-mmap cost on a low-priority worker thread. That init
        // sometimes silently fails under memory pressure, leaving the predictor
        // permanently null and every background notification stuck on FAILED.
        // Doing it here ensures the predictor is ready before any notification.
        val app = applicationContext as NocapApp
        scope.launch {
            val available = try {
                app.hybrid.predictor != null
            } catch (t: Throwable) {
                Log.w(TAG, "predictor warm-up threw", t)
                false
            }
            Log.i(TAG, "listener connected, hybrid predictor available=$available")
        }
    }

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        val app = applicationContext as NocapApp
        val extras = sbn.notification.extras
        val title = extras.getCharSequence("android.title")?.toString().orEmpty()
        val text = extras.getCharSequence("android.text")?.toString().orEmpty()
        val sub = extras.getCharSequence("android.subText")?.toString().orEmpty()
        val bigText = extras.getCharSequence("android.bigText")?.toString().orEmpty()
        val body = listOf(bigText, text, sub).firstOrNull { it.isNotBlank() }.orEmpty()

        val captured = CapturedNotification(
            packageName = sbn.packageName,
            postedAt = sbn.postTime,
            title = title,
            text = body,
            notificationKey = sbn.key,
        )

        scope.launch {
            val dao = app.database.notifications()
            // If a live row already exists for this key, this is an UPDATE (the
            // source app reposted — likely after the user replied / liked / etc.).
            // Bump updateCount instead of inserting a fresh row + re-classifying.
            val live = try {
                dao.findLiveByKey(sbn.key)
            } catch (t: Throwable) {
                Log.w(TAG, "findLiveByKey failed", t)
                null
            }
            if (live != null) {
                try {
                    dao.recordUpdate(live.id, title, body, sbn.postTime)
                } catch (t: Throwable) {
                    Log.w(TAG, "recordUpdate failed for id=${live.id}", t)
                }
                return@launch
            }

            val id = try {
                dao.insert(captured)
            } catch (t: Throwable) {
                Log.e(TAG, "insert failed", t)
                return@launch
            }
            classify(app, id, captured)
        }
    }

    override fun onNotificationRemoved(
        sbn: StatusBarNotification,
        rankingMap: RankingMap?,
        reason: Int,
    ) {
        // Skip cancellations we initiated ourselves — they'd label our own predictions.
        if (reason == REASON_LISTENER_CANCEL) return
        if (reason !in setOf(REASON_CLICK, REASON_CANCEL, REASON_CANCEL_ALL, REASON_APP_CANCEL)) return

        // Snapshot interactive state + actions BEFORE going async — the SBN
        // reference and the screen state are freshest right here.
        val hadActions = (sbn.notification.actions?.size ?: 0) > 0
        val interactiveNow = (getSystemService(POWER_SERVICE) as? PowerManager)?.isInteractive ?: false

        val app = applicationContext as NocapApp
        scope.launch {
            val row = try {
                app.database.notifications().findByKey(sbn.key)
            } catch (t: Throwable) {
                Log.w(TAG, "findByKey failed", t)
                null
            } ?: return@launch

            val labelAndSource: Pair<Float, String> = when (reason) {
                // Tap → user opened. Always WANT, regardless of when.
                REASON_CLICK -> 1.0f to "click"

                // Per-notification swipe. Trust the user — if they specifically
                // dismissed this row, it's a SKIP signal. Timing irrelevant.
                // EXCEPTION: if the notification was updated at least once while
                // it was live, the user likely engaged via an inline action
                // (reply / like / mark-read) and then swiped to clean up. We
                // can't tell that case from "5 spam messages then swipe," so
                // we abstain — no label is safer than a wrong label.
                REASON_CANCEL -> {
                    if (row.updateCount > 0) {
                        markRemoved(app, row.id)
                        return@launch
                    }
                    0.0f to "swipe"
                }

                // Clear-all button hits every notification in the shade at once.
                // User explicitly opted to wipe them all → label every one as SKIP.
                // Source is distinct ("clear_all") so the UI / diagnostics can show
                // it differently from per-notification swipes.
                REASON_CANCEL_ALL -> 0.0f to "clear_all"

                // App-driven cancel — usually fires after the user used an inline
                // action on an app that auto-dismisses (Gmail "Mark read", some
                // calendar apps). We infer:
                //   - the notification had ≥1 action (something to interact with),
                //   - the phone was interactive at remove time (a human was present).
                // Background sync cancellations on a locked phone fall through.
                REASON_APP_CANCEL ->
                    if (hadActions && interactiveNow) 1.0f to "app_action"
                    else {
                        markRemoved(app, row.id)
                        return@launch
                    }

                else -> return@launch
            }

            val (label, source) = labelAndSource
            learn(app, row, label, source)
            markRemoved(app, row.id)
        }
    }

    private suspend fun markRemoved(app: NocapApp, rowId: Long) {
        try {
            app.database.notifications().markRemoved(rowId, System.currentTimeMillis())
        } catch (t: Throwable) {
            Log.w(TAG, "markRemoved failed for id=$rowId", t)
        }
    }

    private suspend fun classify(app: NocapApp, id: Long, n: CapturedNotification) {
        if (n.title.isBlank() && n.text.isBlank()) return

        val hybridEnabled = app.prefs.getHybridEnabled()
        val predictor = if (hybridEnabled) app.hybrid.predictor else null

        val handled = when {
            predictor != null -> classifyWithHybrid(app, id, n, predictor)
            else -> classifyWithGemini(app, id, n)
        }

        if (!handled) {
            val reason = when {
                predictor == null && !hasAnyKey(app) -> "No on-device model and no API key configured"
                predictor == null -> "On-device model not available; LLM call failed"
                else -> "LLM fallback required but failed (no key or call error)"
            }
            try {
                app.database.notifications().markFailed(
                    id = id,
                    reason = reason,
                    classifiedAt = System.currentTimeMillis(),
                )
            } catch (t: Throwable) {
                Log.e(TAG, "markFailed write failed for id=$id", t)
            }
        }
    }

    private suspend fun hasAnyKey(app: NocapApp): Boolean =
        !app.prefs.getOpenAiKey().isNullOrBlank() || !app.prefs.getGeminiKey().isNullOrBlank()

    /**
     * Returns true if the classification was successfully written. If false, the
     * caller should write a failure marker so the row doesn't sit in "classifying...".
     */
    private suspend fun classifyWithHybrid(
        app: NocapApp,
        id: Long,
        n: CapturedNotification,
        predictor: HybridPredictor,
    ): Boolean {
        val input = StructuredFeatures.Input(
            packageName = n.packageName,
            title = n.title,
            body = n.text,
            postedAtMs = n.postedAt,
        )
        val text = (n.title + "\n" + n.text).trim()

        val prediction = try {
            predictor.predict(input, text)
        } catch (t: Throwable) {
            Log.e(TAG, "hybrid predict failed for id=$id", t)
            return false
        }

        // Predictor reports FALLBACK_DEFAULT when an LLM call was required but no key
        // was configured. That counts as a failure — surface it to the user.
        if (prediction.source == HybridPredictor.Source.FALLBACK_DEFAULT) {
            return false
        }

        val importanceInt = (prediction.importance * 10f).roundToInt().coerceIn(0, 10)
        val shouldHide = app.prefs.getFilteringEnabled() && importanceInt < app.prefs.getThreshold()
        if (shouldHide) {
            try {
                cancelNotification(n.notificationKey)
            } catch (t: Throwable) {
                Log.w(TAG, "cancel failed for id=$id key=${n.notificationKey}: ${t.message}")
            }
        }

        val reason = buildString {
            append(prediction.source.name.lowercase())
            prediction.pKnn?.let { append(" knn=").append("%.2f".format(it)) }
            prediction.pHead?.let { append(" head=").append("%.2f".format(it)) }
        }

        val payloadJson = PredictionPayload.encode(
            PredictionPayload.from(prediction, app.prefs.getAlpha())
        )

        return try {
            app.database.notifications().applyHybridPrediction(
                id = id,
                importance = importanceInt,
                category = "hybrid",
                reason = reason,
                summary = "",
                classifiedAt = System.currentTimeMillis(),
                hidden = shouldHide,
                predictionJson = payloadJson,
            )
            true
        } catch (t: Throwable) {
            Log.e(TAG, "update failed for id=$id", t)
            false
        }
    }

    private suspend fun classifyWithGemini(app: NocapApp, id: Long, n: CapturedNotification): Boolean {
        val openAi = app.prefs.getOpenAiKey()
        val gemini = app.prefs.getGeminiKey()
        if (openAi.isNullOrBlank() && gemini.isNullOrBlank()) {
            Log.d(TAG, "no api keys configured; skipping classification for id=$id")
            return false
        }
        val examples = buildFeedbackExamples(app, n.packageName)
        val result = app.gemini.classify(openAi, gemini, n.packageName, n.title, n.text, examples)
        return result.fold(
            onSuccess = { c ->
                val shouldHide = app.prefs.getFilteringEnabled() && c.importance < app.prefs.getThreshold()
                if (shouldHide) {
                    try {
                        cancelNotification(n.notificationKey)
                    } catch (t: Throwable) {
                        Log.w(TAG, "cancel failed for id=$id key=${n.notificationKey}: ${t.message}")
                    }
                }
                try {
                    app.database.notifications().applyClassification(
                        id = id,
                        category = c.category,
                        importance = c.importance,
                        summary = c.summary,
                        reason = c.reason,
                        classifiedAt = System.currentTimeMillis(),
                        hidden = shouldHide,
                    )
                    true
                } catch (t: Throwable) {
                    Log.e(TAG, "update failed for id=$id", t)
                    false
                }
            },
            onFailure = { err ->
                Log.w(TAG, "classify failed for id=$id: ${err.message}")
                false
            },
        )
    }

    private suspend fun learn(
        app: NocapApp,
        n: CapturedNotification,
        label: Float,
        source: String,
    ) {
        val feedbackInt = if (label >= 0.5f) 1 else -1
        val dao = app.database.notifications()
        try {
            dao.setFeedback(n.id, feedbackInt, source)
        } catch (t: Throwable) {
            Log.w(TAG, "setFeedback failed: ${t.message}", t)
        }

        // If the row never got a successful classification (or stuck on FAILED),
        // promote the shade action itself to a classification. The user just told
        // us what they thought of it — that's a stronger signal than "unknown."
        if (n.category == null || n.category == "failed" || n.importance == null) {
            val importanceFromLabel = if (label >= 0.5f) 9 else 1
            try {
                dao.applyManualImportance(
                    id = n.id,
                    importance = importanceFromLabel,
                    classifiedAt = System.currentTimeMillis(),
                )
            } catch (t: Throwable) {
                Log.w(TAG, "applyManualImportance failed: ${t.message}", t)
            }
        }

        val predictor = app.hybrid.predictor ?: return
        val input = StructuredFeatures.Input(
            packageName = n.packageName,
            title = n.title,
            body = n.text,
            postedAtMs = n.postedAt,
        )
        val text = (n.title + "\n" + n.text).trim()
        if (text.isBlank()) return
        try {
            predictor.learn(input, text, label, notificationKey = n.notificationKey)
            app.hybrid.persistHead()
        } catch (t: Throwable) {
            Log.w(TAG, "learn failed: ${t.message}", t)
        }
    }

    private suspend fun buildFeedbackExamples(app: NocapApp, currentPackage: String): List<FeedbackExample> {
        val dao = app.database.notifications()
        val sameApp = dao.recentFeedbackForPackage(currentPackage, limit = 4)
        val general = dao.recentFeedback(limit = 8)
        val merged = (sameApp + general).distinctBy { it.id }.take(10)
        return merged.map { row ->
            FeedbackExample(
                packageName = row.packageName,
                title = row.title,
                body = row.text,
                wanted = row.feedback == 1,
            )
        }
    }

    companion object {
        private const val TAG = "NocapListener"
        // Mirror of platform constants — keep here so the module doesn't drag in
        // the full reason enum surface.
        private const val REASON_CLICK = 1
        private const val REASON_CANCEL = 2
        private const val REASON_CANCEL_ALL = 3
        private const val REASON_APP_CANCEL = 5
        private const val REASON_LISTENER_CANCEL = 10

        fun isEnabled(context: Context): Boolean {
            val enabled = Settings.Secure.getString(
                context.contentResolver,
                "enabled_notification_listeners"
            ) ?: return false
            val component = ComponentName(context, NocapNotificationListenerService::class.java)
            val flat = component.flattenToString()
            return enabled.split(":").any { it == flat }
        }
    }
}
