package com.nocap.app.gemini

import kotlinx.serialization.Serializable

@Serializable
data class Classification(
    val category: String,
    val importance: Int,
    val summary: String,
    val reason: String,
)

/** One past notification + Samar's verdict, used as a few-shot example in future classifications. */
data class FeedbackExample(
    val packageName: String,
    val title: String,
    val body: String,
    /** true = Samar wanted to see this, false = Samar wanted it hidden */
    val wanted: Boolean,
)
