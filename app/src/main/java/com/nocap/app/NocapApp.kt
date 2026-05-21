package com.nocap.app

import android.app.Application
import com.nocap.app.data.AppDatabase
import com.nocap.app.gemini.GeminiClient
import com.nocap.app.prefs.Prefs

class NocapApp : Application() {
    val database: AppDatabase by lazy { AppDatabase.get(this) }
    val prefs: Prefs by lazy { Prefs(this) }
    val gemini: GeminiClient by lazy { GeminiClient() }
    val hybrid: HybridFactory by lazy { HybridFactory(this) }
    val appLabels: AppLabelResolver by lazy { AppLabelResolver(this) }
}
