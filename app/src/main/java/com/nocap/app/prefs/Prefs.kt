package com.nocap.app.prefs

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "nocap_prefs")

class Prefs(private val context: Context) {

    val openAiKeyFlow: Flow<String?> = context.dataStore.data.map { it[OPENAI_KEY] }
    val geminiKeyFlow: Flow<String?> = context.dataStore.data.map { it[GEMINI_KEY] }

    val filteringEnabledFlow: Flow<Boolean> = context.dataStore.data.map { it[FILTERING_ENABLED] ?: false }
    val thresholdFlow: Flow<Int> = context.dataStore.data.map { it[THRESHOLD] ?: 5 }
    val showHiddenFlow: Flow<Boolean> = context.dataStore.data.map { it[SHOW_HIDDEN] ?: true }
    val alphaFlow: Flow<Float> = context.dataStore.data.map { it[ALPHA] ?: 0.6f }
    val hybridEnabledFlow: Flow<Boolean> = context.dataStore.data.map { it[HYBRID_ENABLED] ?: true }

    suspend fun setOpenAiKey(value: String) = context.dataStore.edit { it[OPENAI_KEY] = value }
    suspend fun setGeminiKey(value: String) = context.dataStore.edit { it[GEMINI_KEY] = value }
    suspend fun setFilteringEnabled(value: Boolean) = context.dataStore.edit { it[FILTERING_ENABLED] = value }
    suspend fun setThreshold(value: Int) = context.dataStore.edit { it[THRESHOLD] = value.coerceIn(0, 10) }
    suspend fun setShowHidden(value: Boolean) = context.dataStore.edit { it[SHOW_HIDDEN] = value }
    suspend fun setAlpha(value: Float) = context.dataStore.edit { it[ALPHA] = value.coerceIn(0f, 1f) }
    suspend fun setHybridEnabled(value: Boolean) = context.dataStore.edit { it[HYBRID_ENABLED] = value }

    suspend fun getOpenAiKey(): String? = openAiKeyFlow.first()
    suspend fun getGeminiKey(): String? = geminiKeyFlow.first()
    suspend fun getFilteringEnabled(): Boolean = filteringEnabledFlow.first()
    suspend fun getThreshold(): Int = thresholdFlow.first()
    suspend fun getAlpha(): Float = alphaFlow.first()
    suspend fun getHybridEnabled(): Boolean = hybridEnabledFlow.first()

    companion object {
        private val OPENAI_KEY = stringPreferencesKey("openai_api_key")
        private val GEMINI_KEY = stringPreferencesKey("gemini_api_key")
        private val FILTERING_ENABLED = booleanPreferencesKey("filtering_enabled")
        private val THRESHOLD = intPreferencesKey("importance_threshold")
        private val SHOW_HIDDEN = booleanPreferencesKey("show_hidden_in_list")
        private val ALPHA = floatPreferencesKey("hybrid_alpha")
        private val HYBRID_ENABLED = booleanPreferencesKey("hybrid_enabled")
    }
}
