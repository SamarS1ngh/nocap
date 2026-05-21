package com.nocap.app.ui

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nocap.app.NocapApp
import com.nocap.onlinehead.StructuredFeatures
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class NotificationListViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application as NocapApp

    val notifications = combine(
        app.database.notifications().recent(),
        app.prefs.showHiddenFlow,
    ) { items, showHidden ->
        if (showHidden) items else items.filterNot { it.hidden }
    }

    enum class Choice {
        IMPORTANT,  // importance=9, feedback=+1, label=1.0  (trains model)
        NEUTRAL,    // importance=5, feedback= 0, no training (skipped — ambiguous signal)
        JUNK,       // importance=1, feedback=-1, label=0.0  (trains model)
    }

    /**
     * Unified manual classification: in one tap we
     *   1. set importance (so filtering threshold uses it),
     *   2. set feedback + source (audit trail),
     *   3. for IMPORTANT/JUNK only — also feed the example into the hybrid
     *      predictor's vector store + head so the model learns.
     *
     * Neutral skips step 3 because 0.5 is an ambiguous label that adds noise.
     */
    fun classifyAndLearn(id: Long, choice: Choice) {
        viewModelScope.launch {
            val (importance, feedback, label) = when (choice) {
                Choice.IMPORTANT -> Triple(9, 1, 1.0f)
                Choice.NEUTRAL -> Triple(5, 0, null)
                Choice.JUNK -> Triple(1, -1, 0.0f)
            }
            val now = System.currentTimeMillis()

            val dao = app.database.notifications()
            try {
                dao.applyManualImportance(id = id, importance = importance, classifiedAt = now)
                dao.setFeedback(id = id, value = feedback, source = "manual")
            } catch (t: Throwable) {
                Log.e(TAG, "classifyAndLearn write failed for id=$id", t)
                return@launch
            }

            if (label != null) {
                feedHybrid(id, label)
            }
        }
    }

    private suspend fun feedHybrid(id: Long, label: Float) {
        val predictor = app.hybrid.predictor ?: return
        val row = withContext(Dispatchers.IO) {
            runCatching { app.database.notifications().findById(id) }.getOrNull()
        } ?: return

        val text = (row.title + "\n" + row.text).trim()
        if (text.isBlank()) return

        val input = StructuredFeatures.Input(
            packageName = row.packageName,
            title = row.title,
            body = row.text,
            postedAtMs = row.postedAt,
        )
        try {
            predictor.learn(input, text, label, notificationKey = row.notificationKey)
            app.hybrid.persistHead()
        } catch (t: Throwable) {
            Log.w(TAG, "predictor.learn failed: ${t.message}")
        }
    }

    companion object {
        private const val TAG = "NotifVM"
    }
}
