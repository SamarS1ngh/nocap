package com.nocap.onlinehead

import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin
import java.util.Calendar
import java.util.TimeZone

/**
 * Extracts a fixed-length feature vector describing a notification's "situation."
 * The vector is what online-head sees ALONGSIDE the 384-dim text embedding.
 *
 * Layout (51 dims total):
 *   [0..30]   31 dims  — package one-hot: top-30 known packages + "other" bucket
 *   [31..32]   2 dims  — hour-of-day sin/cos (cyclic encoding, no Monday-Sunday jump)
 *   [33..39]   7 dims  — day-of-week one-hot
 *   [40..41]   2 dims  — log(1 + titleLen), log(1 + bodyLen)
 *   [42..44]   3 dims  — has-digit, has-url, has-emoji (0/1)
 *   [45..46]   2 dims  — inline-reply-available, android-importance hint normalized
 *   [47..50]   4 dims  — sender-hash buckets for messaging apps (sparse one-hot)
 *
 * Total = 51. Concatenated with 384-dim embedding → 435-dim input to TwoLayerNet.
 *
 * The package whitelist is provided externally so the app can update it without
 * touching this module. Pass top-30 packageNames observed in store for stable identity.
 */
class StructuredFeatures(
    private val topPackages: List<String>,
) {

    init {
        require(topPackages.size <= TOP_K_PACKAGES) {
            "topPackages must be at most $TOP_K_PACKAGES, got ${topPackages.size}"
        }
    }

    private val packageIndex: Map<String, Int> = topPackages
        .withIndex()
        .associate { (i, p) -> p to i }

    private val urlRegex = Regex("https?://|www\\.")
    private val digitRegex = Regex("\\d")
    private val emojiRegex = Regex("[\\p{So}\\p{Sk}]|[\\uD83C-\\uDBFF\\uDC00-\\uDFFF]")

    data class Input(
        val packageName: String,
        val title: String,
        val body: String,
        val postedAtMs: Long,
        val timeZone: TimeZone = TimeZone.getDefault(),
        val hasInlineReply: Boolean = false,
        val androidImportance: Int = 0,  // -1..4 typically; normalize to [0,1]
        val senderHash: Int? = null,     // for messaging apps; null if unknown
    )

    fun extract(input: Input): FloatArray {
        val out = FloatArray(DIM)

        // [0..30] package one-hot
        val pkgIdx = packageIndex[input.packageName] ?: OTHER_PACKAGE_IDX
        out[pkgIdx] = 1f

        // [31..32] hour-of-day sin/cos
        val cal = Calendar.getInstance(input.timeZone).apply { timeInMillis = input.postedAtMs }
        val hour = cal.get(Calendar.HOUR_OF_DAY)
        val hourFrac = hour / 24.0
        out[HOUR_SIN_IDX] = sin(2 * Math.PI * hourFrac).toFloat()
        out[HOUR_COS_IDX] = cos(2 * Math.PI * hourFrac).toFloat()

        // [33..39] day-of-week one-hot. Sunday=1..Saturday=7 in Calendar.
        val dow = cal.get(Calendar.DAY_OF_WEEK) - 1
        out[DOW_BASE_IDX + dow] = 1f

        // [40..41] title/body length (log1p)
        out[TITLE_LEN_IDX] = ln(1.0 + input.title.length).toFloat()
        out[BODY_LEN_IDX] = ln(1.0 + input.body.length).toFloat()

        // [42..44] content flags
        val combined = input.title + " " + input.body
        out[HAS_DIGIT_IDX] = if (digitRegex.containsMatchIn(combined)) 1f else 0f
        out[HAS_URL_IDX] = if (urlRegex.containsMatchIn(combined)) 1f else 0f
        out[HAS_EMOJI_IDX] = if (emojiRegex.containsMatchIn(combined)) 1f else 0f

        // [45..46] interaction hints
        out[INLINE_REPLY_IDX] = if (input.hasInlineReply) 1f else 0f
        out[ANDROID_IMP_IDX] = (input.androidImportance.coerceIn(-1, 4) + 1) / 5f

        // [47..50] sender hash buckets
        if (input.senderHash != null) {
            val bucket = (input.senderHash.toLong() and 0xffffffffL).rem(SENDER_BUCKETS.toLong()).toInt()
            out[SENDER_BASE_IDX + bucket] = 1f
        }

        return out
    }

    companion object {
        const val DIM = 51

        const val TOP_K_PACKAGES = 30
        const val OTHER_PACKAGE_IDX = 30

        const val HOUR_SIN_IDX = 31
        const val HOUR_COS_IDX = 32

        const val DOW_BASE_IDX = 33  // Sunday..Saturday

        const val TITLE_LEN_IDX = 40
        const val BODY_LEN_IDX = 41

        const val HAS_DIGIT_IDX = 42
        const val HAS_URL_IDX = 43
        const val HAS_EMOJI_IDX = 44

        const val INLINE_REPLY_IDX = 45
        const val ANDROID_IMP_IDX = 46

        const val SENDER_BUCKETS = 4
        const val SENDER_BASE_IDX = 47
    }
}
