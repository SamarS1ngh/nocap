package com.nocap.app.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nocap.app.NocapApp
import com.nocap.vectorstore.VectorStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class SettingsViewModel(application: Application) : AndroidViewModel(application) {

    private val app = application as NocapApp

    data class UiState(
        val savedOpenAi: String? = null,
        val savedGemini: String? = null,
        val filteringEnabled: Boolean = false,
        val threshold: Int = 5,
        val showHidden: Boolean = true,
        val hybridEnabled: Boolean = true,
        val alpha: Float = 0.6f,
        val storeSize: Int = 0,
        val message: String? = null,
        val testing: Boolean = false,
    )

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            app.prefs.openAiKeyFlow.collect { k -> _state.update { it.copy(savedOpenAi = k) } }
        }
        viewModelScope.launch {
            app.prefs.geminiKeyFlow.collect { k -> _state.update { it.copy(savedGemini = k) } }
        }
        viewModelScope.launch {
            app.prefs.filteringEnabledFlow.collect { v -> _state.update { it.copy(filteringEnabled = v) } }
        }
        viewModelScope.launch {
            app.prefs.thresholdFlow.collect { v -> _state.update { it.copy(threshold = v) } }
        }
        viewModelScope.launch {
            app.prefs.showHiddenFlow.collect { v -> _state.update { it.copy(showHidden = v) } }
        }
        viewModelScope.launch {
            app.prefs.hybridEnabledFlow.collect { v -> _state.update { it.copy(hybridEnabled = v) } }
        }
        viewModelScope.launch {
            app.prefs.alphaFlow.collect { v -> _state.update { it.copy(alpha = v) } }
        }
        viewModelScope.launch {
            refreshStoreSize()
        }
    }

    fun saveOpenAi(key: String) = launchSave { app.prefs.setOpenAiKey(key.trim()); "OpenAI key ${if (key.isBlank()) "cleared" else "saved"}." }
    fun saveGemini(key: String) = launchSave { app.prefs.setGeminiKey(key.trim()); "Gemini key ${if (key.isBlank()) "cleared" else "saved"}." }

    fun setFilteringEnabled(v: Boolean) = viewModelScope.launch { app.prefs.setFilteringEnabled(v) }
    fun setThreshold(v: Int) = viewModelScope.launch { app.prefs.setThreshold(v) }
    fun setShowHidden(v: Boolean) = viewModelScope.launch { app.prefs.setShowHidden(v) }
    fun setHybridEnabled(v: Boolean) = viewModelScope.launch { app.prefs.setHybridEnabled(v) }
    fun setAlpha(v: Float) = viewModelScope.launch { app.prefs.setAlpha(v) }

    fun resetLearning() = viewModelScope.launch {
        withContext(Dispatchers.IO) {
            VectorStore.create(getApplication()).prune(VectorStore.PruneStrategy.All)
            File(getApplication<Application>().filesDir, "online_head.bin").delete()
        }
        refreshStoreSize()
        _state.update { it.copy(message = "Learning reset. Vector store cleared, head weights deleted.") }
    }

    private suspend fun refreshStoreSize() {
        val size = withContext(Dispatchers.IO) {
            runCatching { VectorStore.create(getApplication()).size() }.getOrDefault(0)
        }
        _state.update { it.copy(storeSize = size) }
    }

    fun testOpenAi(key: String) = launchTest("OpenAI") { app.gemini.pingOpenAi(key.trim()) }
    fun testGemini(key: String) = launchTest("Gemini") { app.gemini.pingGemini(key.trim()) }

    private fun launchSave(block: suspend () -> String) {
        viewModelScope.launch {
            val msg = block()
            _state.update { it.copy(message = msg) }
        }
    }

    private fun launchTest(name: String, block: suspend () -> Result<String>) {
        viewModelScope.launch {
            _state.update { it.copy(testing = true, message = "Testing $name...") }
            val result = block()
            _state.update {
                it.copy(
                    testing = false,
                    message = result.fold(
                        onSuccess = { "$name key works." },
                        onFailure = { err -> "$name failed: ${err.message?.take(120) ?: "unknown error"}" }
                    )
                )
            }
        }
    }
}
