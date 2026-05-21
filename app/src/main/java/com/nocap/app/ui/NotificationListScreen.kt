package com.nocap.app.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
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

    // Group by package, sort groups by most-recent notification inside each.
    val groups = remember(items) {
        items
            .groupBy { it.packageName }
            .toList()
            .sortedByDescending { (_, rows) -> rows.maxOf { it.postedAt } }
    }

    // Per-app collapse state (default: expanded for the top app, collapsed for the rest).
    val appExpanded = remember { mutableStateMapOf<String, Boolean>() }
    // Per-notification expand state (default collapsed — show 2-line preview).
    val cardExpanded = remember { mutableStateMapOf<Long, Boolean>() }

    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(vertical = 8.dp)) {
        groups.forEachIndexed { idx, (pkg, rows) ->
            val expanded = appExpanded[pkg] ?: (idx == 0)

            item(key = "header-$pkg") {
                AppHeader(
                    packageName = pkg,
                    count = rows.size,
                    expanded = expanded,
                    onToggle = { appExpanded[pkg] = !expanded },
                )
            }

            if (expanded) {
                items(rows, key = { it.id }) { n ->
                    NotificationCard(
                        n = n,
                        expanded = cardExpanded[n.id] ?: false,
                        onToggleExpand = { cardExpanded[n.id] = !(cardExpanded[n.id] ?: false) },
                        onFeedback = { value -> vm.setFeedback(n.id, value) },
                        onManualClassify = { imp -> vm.manualClassify(n.id, imp) },
                    )
                }
            }
        }
    }
}

@Composable
private fun AppHeader(
    packageName: String,
    count: Int,
    expanded: Boolean,
    onToggle: () -> Unit,
) {
    val ctx = LocalContext.current
    val info = remember(packageName) {
        (ctx.applicationContext as NocapApp).appLabels.resolve(packageName)
    }
    Row(
        Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surfaceVariant)
            .clickable(onClick = onToggle)
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            if (expanded) "▾" else "▸",
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(end = 8.dp),
        )
        if (info.icon != null) {
            Image(
                bitmap = info.icon.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier
                    .size(28.dp)
                    .clip(RoundedCornerShape(6.dp))
                    .padding(end = 0.dp),
            )
            Spacer(Modifier.width(10.dp))
        }
        Column(Modifier.weight(1f)) {
            Text(
                info.label,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                packageName,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Text(
            "$count",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun NotificationCard(
    n: CapturedNotification,
    expanded: Boolean,
    onToggleExpand: () -> Unit,
    onFeedback: (Int) -> Unit,
    onManualClassify: (Int) -> Unit,
) {
    Card(
        Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp, horizontal = 8.dp)
            .alpha(if (n.hidden) 0.55f else 1f)
            .clickable(onClick = onToggleExpand)
    ) {
        Column(Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(n.postedAt)),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.weight(1f),
                )
                FeedbackBadge(feedback = n.feedback, source = n.feedbackSource)
                if (n.category == "failed") {
                    if (n.feedback != 0) Spacer(Modifier.width(6.dp))
                    AssistChip(
                        onClick = {},
                        label = {
                            Text(
                                "FAILED",
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
                    if (n.feedback != 0 || n.category == "failed") Spacer(Modifier.width(6.dp))
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
                    if (n.category != null && n.importance != null) {
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
                    } else if (n.category == "failed") {
                        Spacer(Modifier.height(8.dp))
                        FailedClassifyPanel(
                            reason = n.reason,
                            onMarkImportant = { onManualClassify(9) },
                            onMarkNeutral = { onManualClassify(5) },
                            onMarkJunk = { onManualClassify(1) },
                        )
                    } else {
                        Spacer(Modifier.height(6.dp))
                        Text(
                            "classifying...",
                            style = MaterialTheme.typography.labelSmall,
                            fontStyle = FontStyle.Italic
                        )
                    }

                    Spacer(Modifier.height(10.dp))
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            DateFormat.getDateTimeInstance().format(Date(n.postedAt)),
                            style = MaterialTheme.typography.labelSmall,
                            modifier = Modifier.weight(1f)
                        )
                        FeedbackButton(
                            label = "Want",
                            active = n.feedback == 1,
                            activeColor = Color(0xFF2E7D32),
                            onClick = { onFeedback(if (n.feedback == 1) 0 else 1) },
                        )
                        FeedbackButton(
                            label = "Skip",
                            active = n.feedback == -1,
                            activeColor = Color(0xFFC62828),
                            onClick = { onFeedback(if (n.feedback == -1) 0 else -1) },
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun FeedbackButton(
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
private fun PredictorBreakdown(payload: com.nocap.app.diag.PredictionPayload) {
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

