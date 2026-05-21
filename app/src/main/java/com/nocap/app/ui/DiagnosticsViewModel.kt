package com.nocap.app.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nocap.app.NocapApp
import com.nocap.app.data.CapturedNotification
import com.nocap.app.data.NotificationDao
import com.nocap.vectorstore.VectorStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class DiagnosticsViewModel(application: Application) : AndroidViewModel(application) {

    private val app = application as NocapApp

    data class Health(
        val phase: String,
        val storeSize: Int,
        val totalCaptured: Int,
        val alpha: Float,
        val hybridAvailable: Boolean,
        val headUpdateCount: Long,
        val recentLoss: Float?,
        val llmCalls: Int,
        val disagreementRate: Float,
    )

    data class UiState(
        val health: Health? = null,
        val packageStats: List<NotificationDao.PackageStats> = emptyList(),
        val lossHistory: FloatArray = FloatArray(0),
        val disagreements: List<CapturedNotification> = emptyList(),
        val refreshing: Boolean = false,
    )

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    init {
        refresh()
    }

    fun refresh() = viewModelScope.launch {
        _state.update { it.copy(refreshing = true) }
        withContext(Dispatchers.IO) {
            val dao = app.database.notifications()
            val storeSize = runCatching { VectorStore.create(app).size() }.getOrDefault(0)
            val totalCaptured = dao.totalCount()
            val packageStats = dao.packageStats()
            val disagreements = dao.recentDisagreements()
            val llmCalls = dao.llmCallCount()
            val disagreementRate = if (totalCaptured == 0) 0f else llmCalls.toFloat() / totalCaptured
            val lossHistory = app.hybrid.lossHistory()
            val updateCount = app.hybrid.headUpdateCount()
            val recentLoss = app.hybrid.recentLoss()
            val alpha = app.prefs.getAlpha()
            val hybridAvailable = app.hybrid.predictor != null
            val phase = derivePhase(storeSize)

            _state.update {
                it.copy(
                    health = Health(
                        phase = phase,
                        storeSize = storeSize,
                        totalCaptured = totalCaptured,
                        alpha = alpha,
                        hybridAvailable = hybridAvailable,
                        headUpdateCount = updateCount,
                        recentLoss = recentLoss,
                        llmCalls = llmCalls,
                        disagreementRate = disagreementRate,
                    ),
                    packageStats = packageStats,
                    lossHistory = lossHistory,
                    disagreements = disagreements,
                    refreshing = false,
                )
            }
        }
    }

    private fun derivePhase(storeSize: Int): String = when {
        storeSize < 30 -> "COLD_START (LLM only)"
        storeSize < 500 -> "KNN_PRIMARY (head suppressed)"
        storeSize < 2000 -> "BLEND (warming up)"
        storeSize < 5000 -> "BLEND (mature)"
        else -> "HEAD_DOMINANT"
    }
}
