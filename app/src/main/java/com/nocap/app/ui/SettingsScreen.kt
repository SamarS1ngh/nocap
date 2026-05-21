package com.nocap.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onBack: () -> Unit,
    vm: SettingsViewModel = viewModel(),
) {
    val state by vm.state.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = { TextButton(onClick = onBack) { Text("Back") } }
            )
        }
    ) { padding ->
        Column(
            Modifier
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            // ===== API keys =====
            Text("API keys", style = MaterialTheme.typography.titleLarge)
            Spacer(Modifier.height(4.dp))
            Text(
                "OpenAI is primary; Gemini is the fallback. Set at least one.",
                style = MaterialTheme.typography.bodySmall
            )
            Spacer(Modifier.height(16.dp))

            KeyField(
                label = "OpenAI API key (primary)",
                placeholder = "sk-...",
                savedKey = state.savedOpenAi,
                testing = state.testing,
                onSave = vm::saveOpenAi,
                onTest = vm::testOpenAi,
            )

            Spacer(Modifier.height(16.dp))
            KeyField(
                label = "Gemini API key (backup)",
                placeholder = "AIza...",
                savedKey = state.savedGemini,
                testing = state.testing,
                onSave = vm::saveGemini,
                onTest = vm::testGemini,
            )

            Spacer(Modifier.height(24.dp))
            HorizontalDivider()
            Spacer(Modifier.height(24.dp))

            // ===== Filtering =====
            Text("Filtering", style = MaterialTheme.typography.titleLarge)
            Spacer(Modifier.height(4.dp))
            Text(
                "When enabled, notifications classified below the threshold are " +
                        "removed from your system shade right after they arrive. " +
                        "You'll see them briefly (~1-2s) before they disappear.",
                style = MaterialTheme.typography.bodySmall
            )
            Spacer(Modifier.height(16.dp))

            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(Modifier.weight(1f)) {
                    Text("Filtering enabled", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        if (state.filteringEnabled) "Active — low-importance notifications will be hidden"
                        else "Off — every notification stays in your shade",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(
                    checked = state.filteringEnabled,
                    onCheckedChange = vm::setFilteringEnabled
                )
            }

            Spacer(Modifier.height(20.dp))
            Text("Hide below importance: ${state.threshold}/10", style = MaterialTheme.typography.bodyLarge)
            Spacer(Modifier.height(4.dp))
            Text(
                when {
                    state.threshold <= 2 -> "Very lenient — only kills bottom-tier promo/spam"
                    state.threshold <= 4 -> "Lenient — kills promo and routine local news"
                    state.threshold <= 6 -> "Balanced — also kills Reddit, social, outgoing acks"
                    state.threshold <= 8 -> "Strict — only personal DMs and big news survive"
                    else -> "Brutal — almost everything gets hidden"
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(Modifier.height(4.dp))
            Slider(
                value = state.threshold.toFloat(),
                onValueChange = { vm.setThreshold(it.toInt()) },
                valueRange = 0f..10f,
                steps = 9,
            )

            Spacer(Modifier.height(16.dp))
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(Modifier.weight(1f)) {
                    Text("Show filtered in app list", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        "Keeps an audit trail of what got hidden",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(
                    checked = state.showHidden,
                    onCheckedChange = vm::setShowHidden
                )
            }

            Spacer(Modifier.height(24.dp))
            HorizontalDivider()
            Spacer(Modifier.height(24.dp))

            // ===== Hybrid on-device classifier =====
            Text("On-device classifier", style = MaterialTheme.typography.titleLarge)
            Spacer(Modifier.height(4.dp))
            Text(
                "Local kNN + small neural head. Falls back to the LLM only when the " +
                    "two disagree. Disable to use the LLM for every notification.",
                style = MaterialTheme.typography.bodySmall
            )
            Spacer(Modifier.height(16.dp))

            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(Modifier.weight(1f)) {
                    Text("Hybrid mode", style = MaterialTheme.typography.bodyLarge)
                    Text(
                        "Memory: ${state.storeSize} labeled notifications",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Switch(
                    checked = state.hybridEnabled,
                    onCheckedChange = vm::setHybridEnabled
                )
            }

            Spacer(Modifier.height(20.dp))
            Text(
                "kNN ↔ head blend (α = ${"%.2f".format(state.alpha)})",
                style = MaterialTheme.typography.bodyLarge
            )
            Spacer(Modifier.height(4.dp))
            Text(
                when {
                    state.alpha < 0.25f -> "Mostly trust the learned head"
                    state.alpha < 0.55f -> "Lean head, but kNN still matters"
                    state.alpha < 0.75f -> "Balanced (default)"
                    else -> "Mostly trust the kNN memory"
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Slider(
                value = state.alpha,
                onValueChange = vm::setAlpha,
                valueRange = 0f..1f,
            )

            Spacer(Modifier.height(16.dp))
            OutlinedButton(onClick = { vm.resetLearning() }) {
                Text("Reset learning (wipe memory + head)")
            }

            Spacer(Modifier.height(20.dp))
            state.message?.let {
                Text(it, style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}

@Composable
private fun KeyField(
    label: String,
    placeholder: String,
    savedKey: String?,
    testing: Boolean,
    onSave: (String) -> Unit,
    onTest: (String) -> Unit,
) {
    var input by remember(savedKey) { mutableStateOf(savedKey ?: "") }
    var visible by remember { mutableStateOf(false) }

    Text(label, style = MaterialTheme.typography.titleMedium)
    Spacer(Modifier.height(8.dp))
    OutlinedTextField(
        value = input,
        onValueChange = { input = it },
        label = { Text(placeholder) },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
        visualTransformation =
            if (visible) VisualTransformation.None else PasswordVisualTransformation(),
        trailingIcon = {
            TextButton(onClick = { visible = !visible }) {
                Text(if (visible) "Hide" else "Show")
            }
        }
    )
    Spacer(Modifier.height(8.dp))
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(onClick = { onSave(input) }) { Text("Save") }
        OutlinedButton(
            onClick = { onTest(input) },
            enabled = input.isNotBlank() && !testing,
        ) { Text(if (testing) "..." else "Test") }
    }
}
