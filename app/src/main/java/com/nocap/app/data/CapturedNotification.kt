package com.nocap.app.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "captured_notifications")
data class CapturedNotification(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val packageName: String,
    val postedAt: Long,
    val title: String,
    val text: String,
    val notificationKey: String,
    val category: String? = null,
    val importance: Int? = null,
    val summary: String? = null,
    val reason: String? = null,
    val classifiedAt: Long? = null,
    val hidden: Boolean = false,
    /** 0 = no feedback, 1 = user wanted to see this, -1 = user didn't want to see this */
    val feedback: Int = 0,
    /**
     * Where the feedback signal came from. null when feedback=0.
     * One of: "click", "fast_swipe", "manual".
     */
    val feedbackSource: String? = null,
    /**
     * JSON-encoded diagnostic payload describing the hybrid predictor's reasoning:
     * source, P_knn, P_head, neighbours[]. Null for legacy LLM-only path.
     * Schema = com.nocap.app.diag.PredictionPayload.
     */
    val predictionJson: String? = null,
)
