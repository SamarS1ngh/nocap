package com.nocap.app.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nocap.app.data.CapturedNotification
import com.nocap.app.data.NotificationDao
import java.text.DateFormat
import java.util.Date

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DiagnosticsScreen(
    onBack: () -> Unit,
    vm: DiagnosticsViewModel = viewModel(),
) {
    val state by vm.state.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Diagnostics") },
                navigationIcon = { TextButton(onClick = onBack) { Text("Back") } },
                actions = {
                    TextButton(onClick = { vm.refresh() }, enabled = !state.refreshing) {
                        Text(if (state.refreshing) "..." else "Refresh")
                    }
                },
            )
        }
    ) { padding ->
        Column(
            Modifier
                .padding(padding)
                .padding(horizontal = 16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            Spacer(Modifier.height(12.dp))
            HealthCard(state.health)
            Spacer(Modifier.height(20.dp))
            LossChartCard(state.lossHistory, state.health?.recentLoss)
            Spacer(Modifier.height(20.dp))
            PackageStatsCard(state.packageStats)
            Spacer(Modifier.height(20.dp))
            DisagreementLogCard(state.disagreements)
            Spacer(Modifier.height(32.dp))
        }
    }
}

@Composable
private fun HealthCard(health: DiagnosticsViewModel.Health?) {
    Section(title = "Predictor health") {
        if (health == null) {
            Text("Loading…", style = MaterialTheme.typography.bodySmall)
            return@Section
        }
        InfoRow("Phase", health.phase)
        InfoRow("Hybrid loaded", if (health.hybridAvailable) "yes" else "no (using LLM only)")
        InfoRow("Vector store size", health.storeSize.toString())
        InfoRow("Total captured", health.totalCaptured.toString())
        InfoRow("α (kNN vs head)", "%.2f".format(health.alpha))
        InfoRow("Head update count", health.headUpdateCount.toString())
        InfoRow(
            "Recent loss (last 100)",
            health.recentLoss?.let { "%.3f".format(it) } ?: "—",
        )
        InfoRow("LLM calls (lifetime)", health.llmCalls.toString())
        InfoRow(
            "Disagreement rate",
            "%.1f%%".format(health.disagreementRate * 100),
        )
    }
}

@Composable
private fun LossChartCard(history: FloatArray, recent: Float?) {
    Section(title = "Head loss over time") {
        if (history.isEmpty()) {
            Text(
                "No training data yet. Head learns once you click or fast-swipe notifications.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            return@Section
        }
        Text(
            "Last ${history.size} updates" +
                (recent?.let { ". Mean of last 100 = ${"%.3f".format(it)}" } ?: ""),
            style = MaterialTheme.typography.bodySmall,
        )
        Spacer(Modifier.height(8.dp))
        LossChart(
            data = history,
            modifier = Modifier
                .fillMaxWidth()
                .height(140.dp),
        )
        Spacer(Modifier.height(4.dp))
        Text(
            "Lower = better. Should trend down with use.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun LossChart(data: FloatArray, modifier: Modifier = Modifier) {
    val lineColor = MaterialTheme.colorScheme.primary
    val axisColor = MaterialTheme.colorScheme.outlineVariant

    Canvas(modifier) {
        if (data.isEmpty()) return@Canvas
        val maxY = (data.max().coerceAtLeast(1e-3f))
        val minY = 0f
        val rangeY = (maxY - minY).coerceAtLeast(1e-6f)
        val stepX = if (data.size > 1) size.width / (data.size - 1) else size.width

        // X axis
        drawLine(
            color = axisColor,
            start = Offset(0f, size.height),
            end = Offset(size.width, size.height),
            strokeWidth = 1f,
        )

        // Loss line
        val path = Path()
        data.forEachIndexed { i, loss ->
            val x = i * stepX
            val yNorm = (loss - minY) / rangeY
            val y = size.height - yNorm * size.height
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, color = lineColor, style = Stroke(width = 2f))
    }
}

@Composable
private fun PackageStatsCard(stats: List<NotificationDao.PackageStats>) {
    Section(title = "Per-app stats") {
        if (stats.isEmpty()) {
            Text("No captured notifications.", style = MaterialTheme.typography.bodySmall)
            return@Section
        }
        stats.take(20).forEach { s ->
            PackageStatRow(s)
        }
        if (stats.size > 20) {
            Spacer(Modifier.height(6.dp))
            Text(
                "…and ${stats.size - 20} more",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun PackageStatRow(s: NotificationDao.PackageStats) {
    val hiddenFrac = if (s.total == 0) 0f else s.hiddenCount.toFloat() / s.total
    val avg = s.avgImportance ?: 0.0
    Column(Modifier.padding(vertical = 6.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                s.packageName,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                modifier = Modifier.weight(1f),
            )
            Text(
                "${s.total}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Spacer(Modifier.height(2.dp))
        LinearProgressIndicator(
            progress = { hiddenFrac },
            modifier = Modifier.fillMaxWidth().height(4.dp),
        )
        Spacer(Modifier.height(2.dp))
        Row {
            Text(
                "avg imp %.1f/10 · hidden %d (%.0f%%) · ✓%d · ✗%d".format(
                    avg, s.hiddenCount, hiddenFrac * 100, s.wantedCount, s.skippedCount,
                ),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun DisagreementLogCard(items: List<CapturedNotification>) {
    Section(title = "Recent kNN/head disagreements") {
        if (items.isEmpty()) {
            Text(
                "None yet. Disagreements fire when kNN and head differ by > 0.4.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            return@Section
        }
        items.take(20).forEach { n ->
            DisagreementRow(n)
        }
    }
}

@Composable
private fun DisagreementRow(n: CapturedNotification) {
    Column(Modifier.padding(vertical = 6.dp)) {
        Text(
            n.title.ifBlank { "(no title)" },
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium,
            maxLines = 1,
        )
        Text(
            n.packageName,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            n.reason ?: "—",
            style = MaterialTheme.typography.labelSmall,
        )
        if (n.classifiedAt != null) {
            Text(
                DateFormat.getDateTimeInstance().format(Date(n.classifiedAt)),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        HorizontalDivider(Modifier.padding(top = 6.dp))
    }
}

@Composable
private fun Section(title: String, content: @Composable () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(),
    ) {
        Column(Modifier.padding(14.dp)) {
            Text(
                title,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(Modifier.height(8.dp))
            content()
        }
    }
}

@Composable
private fun InfoRow(key: String, value: String) {
    Row(Modifier.padding(vertical = 3.dp)) {
        Text(
            key,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f),
        )
        Text(value, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Medium)
    }
}
