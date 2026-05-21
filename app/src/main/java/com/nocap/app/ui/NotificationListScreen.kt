package com.nocap.app.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRowDefaults
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nocap.app.NocapApp
import com.nocap.app.data.CapturedNotification
import com.nocap.app.diag.PredictionPayload
import java.text.DateFormat
import java.util.Date

private const val ALL_TAB_KEY = "__all__"

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NotificationListScreen(vm: NotificationListViewModel = viewModel()) {
    val items by vm.notifications.collectAsStateWithLifecycle(initialValue = emptyList())

    if (items.isEmpty()) {
        Box(Modifier.fillMaxSize().padding(24.dp), contentAlignment = Alignment.Center) {
            Text(
                "No notifications captured yet.\nGrant permission, then wait for one to arrive.",
                style = MaterialTheme.typography.bodyLarge
            )
        }
        return
    }

    // Tab list: "All" first, then one tab per package sorted by most-recent activity.
    val groups = remember(items) {
        items
            .groupBy { it.packageName }
            .toList()
            .sortedByDescending { (_, rows) -> rows.maxOf { it.postedAt } }
    }
    val tabs = remember(groups) {
        buildList {
            add(ALL_TAB_KEY to items.size)
            groups.forEach { (pkg, rows) -> add(pkg to rows.size) }
        }
    }

    var selectedKey by remember { mutableStateOf(ALL_TAB_KEY) }
    // Guard against a tab disappearing (app uninstalled while list is open).
    val selectedIndex = tabs.indexOfFirst { it.first == selectedKey }.coerceAtLeast(0)
    if (tabs.getOrNull(selectedIndex)?.first != selectedKey) {
        selectedKey = tabs[selectedIndex].first
    }

    val filteredItems = remember(items, selectedKey) {
        if (selectedKey == ALL_TAB_KEY) items else items.filter { it.packageName == selectedKey }
    }

    val cardExpanded = remember { mutableStateMapOf<Long, Boolean>() }

    Column(Modifier.fillMaxSize()) {
        ScrollableTabRow(
            selectedTabIndex = selectedIndex,
            edgePadding = 8.dp,
            indicator = { positions ->
                TabRowDefaults.SecondaryIndicator(
                    Modifier.tabIndicatorOffset(positions[selectedIndex]),
                )
            },
        ) {
            tabs.forEachIndexed { index, (key, count) ->
                Tab(
                    selected = index == selectedIndex,
                    onClick = { selectedKey = key },
                    text = { TabLabel(key, count) },
                )
            }
        }

        LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(vertical = 8.dp)) {
            items(filteredItems, key = { it.id }) { n ->
                NotificationCard(
                    n = n,
                    expanded = cardExpanded[n.id] ?: false,
                    onToggleExpand = { cardExpanded[n.id] = !(cardExpanded[n.id] ?: false) },
                    onClassify = { choice -> vm.classifyAndLearn(n.id, choice) },
                )
            }
        }
    }
}

@Composable
private fun TabLabel(key: String, count: Int) {
    if (key == ALL_TAB_KEY) {
        Text("All ($count)")
        return
    }
    val ctx = LocalContext.current
    val info = remember(key) {
        (ctx.applicationContext as NocapApp).appLabels.resolve(key)
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        if (info.icon != null) {
            Image(
                bitmap = info.icon.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier
                    .size(18.dp)
                    .clip(RoundedCornerShape(4.dp)),
            )
            Spacer(Modifier.width(6.dp))
        }
        Text("${info.label} ($count)")
    }
}

@Composable
private fun NotificationCard(
    n: CapturedNotification,
    expanded: Boolean,
    onToggleExpand: () -> Unit,
    onClassify: (NotificationListViewModel.Choice) -> Unit,
) {
    Card(
        Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp, horizontal = 8.dp)
            .alpha(if (n.hidden) 0.55f else 1f)
            .clickable(onClick = onToggleExpand)
    ) {
        Column(Modifier.padding(12.dp)) {
            CardTopRow(n = n, onToggleExpand = onToggleExpand)
            Spacer(Modifier.height(4.dp))
            Text(
                n.title.ifBlank { "(no title)" },
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                maxLines = if (expanded) Int.MAX_VALUE else 1,
            )
            if (n.text.isNotBlank()) {
                Spacer(Modifier.height(4.dp))
                Text(
                    n.text,
                    style = MaterialTheme.typography.bodyMedium,
                    maxLines = if (expanded) Int.MAX_VALUE else 2,
                )
            }

            AnimatedVisibility(visible = expanded) {
                Column {
                    if (n.category == "failed") {
                        Spacer(Modifier.height(8.dp))
                        FailedClassifyPanel(
                            reason = n.reason,
                            onMarkImportant = { onClassify(NotificationListViewModel.Choice.IMPORTANT) },
                            onMarkNeutral = { onClassify(NotificationListViewModel.Choice.NEUTRAL) },
                            onMarkJunk = { onClassify(NotificationListViewModel.Choice.JUNK) },
                        )
                    } else if (n.category != null && n.importance != null) {
                        Spacer(Modifier.height(8.dp))
                        AssistChip(
                            onClick = {},
                            label = {
                                Text(
                                    "${n.category.uppercase()} · ${n.importance}/10",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color.Black,
                                )
                            },
                            colors = AssistChipDefaults.assistChipColors(
                                containerColor = importanceColor(n.importance),
                                labelColor = Color.Black,
                            )
                        )
                        if (!n.summary.isNullOrBlank()) {
                            Spacer(Modifier.height(6.dp))
                            Text(
                                n.summary,
                                style = MaterialTheme.typography.bodyMedium,
                                fontStyle = FontStyle.Italic
                            )
                        }
                        if (!n.reason.isNullOrBlank()) {
                            Spacer(Modifier.height(4.dp))
                            Text(
                                n.reason,
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }

                        val payload = remember(n.predictionJson) {
                            n.predictionJson?.let { PredictionPayload.decode(it) }
                        }
                        if (payload != null) {
                            Spacer(Modifier.height(10.dp))
                            PredictorBreakdown(payload)
                        }

                        Spacer(Modifier.height(12.dp))
                        ClassifyStrip(currentImportance = n.importance, onClassify = onClassify)
                    } else {
                        Spacer(Modifier.height(6.dp))
                        Text(
                            "classifying...",
                            style = MaterialTheme.typography.labelSmall,
                            fontStyle = FontStyle.Italic
                        )
                    }

                    Spacer(Modifier.height(10.dp))
                    Text(
                        DateFormat.getDateTimeInstance().format(Date(n.postedAt)),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun CardTopRow(
    n: CapturedNotification,
    onToggleExpand: () -> Unit,
) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
            DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(n.postedAt)),
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f),
        )
        ManualClassifyBadge(n)
        // ManualClassifyBadge shows the VERDICT (⭐/○/✗ from importance).
        // FeedbackBadge shows the SOURCE (tapped/swiped/manual). Both convey
        // different info — show both when present.
        FeedbackBadge(feedback = n.feedback, source = n.feedbackSource)
        if (n.category == "failed") {
            Spacer(Modifier.width(6.dp))
            AssistChip(
                onClick = onToggleExpand,
                label = {
                    Text(
                        "FAILED · tap to classify",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.White,
                    )
                },
                colors = AssistChipDefaults.assistChipColors(
                    containerColor = Color(0xFFC62828),
                    labelColor = Color.White,
                )
            )
        }
        if (n.hidden) {
            Spacer(Modifier.width(6.dp))
            AssistChip(
                onClick = {},
                label = {
                    Text(
                        "FILTERED",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.Black,
                    )
                },
                colors = AssistChipDefaults.assistChipColors(
                    containerColor = Color(0xFFE0E0E0),
                    labelColor = Color.Black,
                )
            )
        }
    }
}

@Composable
private fun ManualClassifyBadge(n: CapturedNotification) {
    if (n.category != "manual" || n.importance == null) return
    val (label, color) = when {
        n.importance >= 7 -> "⭐ IMPORTANT" to Color(0xFF2E7D32)
        n.importance >= 4 -> "○ NEUTRAL" to Color(0xFF6D6D6D)
        else -> "✗ JUNK" to Color(0xFFC62828)
    }
    AssistChip(
        onClick = {},
        label = {
            Text(
                label,
                style = MaterialTheme.typography.labelSmall,
                color = Color.White,
            )
        },
        colors = AssistChipDefaults.assistChipColors(
            containerColor = color,
            labelColor = Color.White,
        )
    )
}

@Composable
private fun FailedClassifyPanel(
    reason: String?,
    onMarkImportant: () -> Unit,
    onMarkNeutral: () -> Unit,
    onMarkJunk: () -> Unit,
) {
    Column {
        Text(
            "Failed to classify",
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.SemiBold,
            color = Color(0xFFC62828),
        )
        if (!reason.isNullOrBlank()) {
            Spacer(Modifier.height(2.dp))
            Text(
                reason,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Spacer(Modifier.height(8.dp))
        Text(
            "Classify manually:",
            style = MaterialTheme.typography.labelMedium,
        )
        Spacer(Modifier.height(6.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = onMarkImportant,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF2E7D32),
                    contentColor = Color.White,
                ),
            ) { Text("Important", style = MaterialTheme.typography.labelSmall) }
            OutlinedButton(onClick = onMarkNeutral) {
                Text("Neutral", style = MaterialTheme.typography.labelSmall)
            }
            Button(
                onClick = onMarkJunk,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFFC62828),
                    contentColor = Color.White,
                ),
            ) { Text("Junk", style = MaterialTheme.typography.labelSmall) }
        }
    }
}

@Composable
private fun FeedbackBadge(feedback: Int, source: String?) {
    if (feedback == 0) return
    val wanted = feedback == 1
    val label = buildString {
        append(if (wanted) "✓ WANT" else "✗ SKIP")
        when (source) {
            "click" -> append(" · tapped")
            "fast_swipe" -> append(" · swiped")
            "manual" -> append(" · manual")
            null -> Unit
            else -> append(" · ").append(source)
        }
    }
    Spacer(Modifier.width(6.dp))
    AssistChip(
        onClick = {},
        label = {
            Text(
                label,
                style = MaterialTheme.typography.labelSmall,
                color = Color.White,
            )
        },
        colors = AssistChipDefaults.assistChipColors(
            containerColor = if (wanted) Color(0xFF2E7D32) else Color(0xFFC62828),
            labelColor = Color.White,
        )
    )
}

@Composable
private fun PredictorBreakdown(payload: PredictionPayload) {
    Column {
        Text(
            "Predictor breakdown",
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.SemiBold,
        )
        Spacer(Modifier.height(4.dp))
        Text(
            buildString {
                append("source=").append(payload.source.lowercase())
                payload.pKnn?.let { append("   P_knn=").append("%.2f".format(it)) }
                payload.pHead?.let { append("   P_head=").append("%.2f".format(it)) }
                payload.alpha?.let { append("   α=").append("%.2f".format(it)) }
                append("   P_final=").append("%.2f".format(payload.pFinal))
            },
            style = MaterialTheme.typography.bodySmall,
        )
        if (payload.neighbours.isNotEmpty()) {
            Spacer(Modifier.height(8.dp))
            Text(
                "kNN neighbours",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(2.dp))
            payload.neighbours.forEachIndexed { idx, hit ->
                Row(
                    Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        "${idx + 1}. sim=${"%.2f".format(hit.similarity)}",
                        style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier.width(86.dp),
                    )
                    Text(
                        if (hit.label >= 0.5f) "✓" else "✗",
                        style = MaterialTheme.typography.labelSmall,
                        color = if (hit.label >= 0.5f) Color(0xFF2E7D32) else Color(0xFFC62828),
                        modifier = Modifier.width(20.dp),
                    )
                    Text(
                        hit.packageName,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.weight(1f),
                    )
                }
            }
        }
    }
}

@Composable
private fun ClassifyStrip(
    currentImportance: Int?,
    onClassify: (NotificationListViewModel.Choice) -> Unit,
) {
    val activeChoice = currentImportance?.let { imp ->
        when {
            imp >= 7 -> NotificationListViewModel.Choice.IMPORTANT
            imp >= 4 -> NotificationListViewModel.Choice.NEUTRAL
            else -> NotificationListViewModel.Choice.JUNK
        }
    }
    Column {
        Text(
            "Your verdict (trains the model):",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(6.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ClassifyButton(
                label = "Important",
                active = activeChoice == NotificationListViewModel.Choice.IMPORTANT,
                activeColor = Color(0xFF2E7D32),
                onClick = { onClassify(NotificationListViewModel.Choice.IMPORTANT) },
            )
            ClassifyButton(
                label = "Neutral",
                active = activeChoice == NotificationListViewModel.Choice.NEUTRAL,
                activeColor = Color(0xFF6D6D6D),
                onClick = { onClassify(NotificationListViewModel.Choice.NEUTRAL) },
            )
            ClassifyButton(
                label = "Junk",
                active = activeChoice == NotificationListViewModel.Choice.JUNK,
                activeColor = Color(0xFFC62828),
                onClick = { onClassify(NotificationListViewModel.Choice.JUNK) },
            )
        }
    }
}

@Composable
private fun ClassifyButton(
    label: String,
    active: Boolean,
    activeColor: Color,
    onClick: () -> Unit,
) {
    if (active) {
        Button(
            onClick = onClick,
            colors = ButtonDefaults.buttonColors(
                containerColor = activeColor,
                contentColor = Color.White,
            )
        ) { Text(label, style = MaterialTheme.typography.labelSmall) }
    } else {
        OutlinedButton(onClick = onClick) {
            Text(label, style = MaterialTheme.typography.labelSmall)
        }
    }
}

private fun importanceColor(importance: Int): Color = when {
    importance >= 8 -> Color(0xFFB9F6CA)
    importance >= 5 -> Color(0xFFFFF59D)
    else -> Color(0xFFFFCDD2)
}
