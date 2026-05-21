package com.nocap.app

import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.nocap.app.listener.NocapNotificationListenerService
import com.nocap.app.ui.DiagnosticsScreen
import com.nocap.app.ui.NotificationListScreen
import com.nocap.app.ui.SettingsScreen
import com.nocap.app.ui.theme.NocapTheme

private enum class Screen { List, Settings, Diagnostics }

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            NocapTheme {
                var screen by remember { mutableStateOf(Screen.List) }
                when (screen) {
                    Screen.List -> NocapHome(
                        onOpenSettings = { screen = Screen.Settings },
                        onOpenDiagnostics = { screen = Screen.Diagnostics },
                    )
                    Screen.Settings -> SettingsScreen(onBack = { screen = Screen.List })
                    Screen.Diagnostics -> DiagnosticsScreen(onBack = { screen = Screen.List })
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun NocapHome(
    onOpenSettings: () -> Unit,
    onOpenDiagnostics: () -> Unit,
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("nocap") },
                actions = {
                    TextButton(onClick = onOpenDiagnostics) { Text("Diag") }
                    TextButton(onClick = onOpenSettings) { Text("Settings") }
                }
            )
        }
    ) { padding ->
        Column(Modifier.padding(padding)) {
            ListenerPermissionBanner()
            NotificationListScreen()
        }
    }
}

@Composable
private fun ListenerPermissionBanner() {
    val ctx = LocalContext.current
    var granted by remember { mutableStateOf(NocapNotificationListenerService.isEnabled(ctx)) }

    if (!granted) {
        Card(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
            Column(Modifier.padding(16.dp)) {
                Text(
                    "Notification access needed",
                    style = MaterialTheme.typography.titleMedium
                )
                Spacer(Modifier.height(8.dp))
                Text("nocap needs permission to read your notifications.")
                Spacer(Modifier.height(12.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = {
                        ctx.startActivity(Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS))
                    }) { Text("Grant") }
                    OutlinedButton(onClick = {
                        granted = NocapNotificationListenerService.isEnabled(ctx)
                    }) { Text("Refresh") }
                }
            }
        }
    }
}
