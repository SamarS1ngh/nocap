package com.nocap.app.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.nocap.app.NocapApp
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch

class NotificationListViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application as NocapApp

    val notifications = combine(
        app.database.notifications().recent(),
        app.prefs.showHiddenFlow,
    ) { items, showHidden ->
        if (showHidden) items else items.filterNot { it.hidden }
    }

    fun setFeedback(id: Long, value: Int) {
        viewModelScope.launch {
            app.database.notifications().setFeedback(id, value)
        }
    }

    fun manualClassify(id: Long, importance: Int) {
        viewModelScope.launch {
            app.database.notifications().applyManualImportance(
                id = id,
                importance = importance.coerceIn(0, 10),
                classifiedAt = System.currentTimeMillis(),
            )
        }
    }
}
