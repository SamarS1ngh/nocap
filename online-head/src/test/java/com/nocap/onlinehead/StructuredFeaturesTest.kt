package com.nocap.onlinehead

import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.TimeZone

class StructuredFeaturesTest {

    private val utc = TimeZone.getTimeZone("UTC")
    private val pkgs = listOf("com.reddit", "com.whatsapp", "com.bank")
    private val sf = StructuredFeatures(pkgs)

    @Test
    fun dimensionalityIsExactly51() {
        val v = sf.extract(
            StructuredFeatures.Input(
                packageName = "com.reddit",
                title = "hi",
                body = "",
                postedAtMs = 0L,
                timeZone = utc,
            )
        )
        assertEquals(51, v.size)
    }

    @Test
    fun knownPackageActivatesCorrectIndex() {
        val v = sf.extract(
            StructuredFeatures.Input(
                packageName = "com.whatsapp",
                title = "",
                body = "",
                postedAtMs = 0L,
                timeZone = utc,
            )
        )
        assertEquals(1f, v[1], 0f)  // com.whatsapp is index 1
        assertEquals(0f, v[0], 0f)  // com.reddit not active
        assertEquals(0f, v[StructuredFeatures.OTHER_PACKAGE_IDX], 0f)
    }

    @Test
    fun unknownPackageFallsToOtherBucket() {
        val v = sf.extract(
            StructuredFeatures.Input(
                packageName = "com.tiktok",
                title = "",
                body = "",
                postedAtMs = 0L,
                timeZone = utc,
            )
        )
        assertEquals(1f, v[StructuredFeatures.OTHER_PACKAGE_IDX], 0f)
    }

    @Test
    fun contentFlagsDetectDigitsUrlEmoji() {
        val v = sf.extract(
            StructuredFeatures.Input(
                packageName = "com.bank",
                title = "OTP 1234",
                body = "Visit https://example.com 🎉",
                postedAtMs = 0L,
                timeZone = utc,
            )
        )
        assertEquals(1f, v[StructuredFeatures.HAS_DIGIT_IDX], 0f)
        assertEquals(1f, v[StructuredFeatures.HAS_URL_IDX], 0f)
        assertEquals(1f, v[StructuredFeatures.HAS_EMOJI_IDX], 0f)
    }

    @Test
    fun hourEncodingIsCyclic() {
        // Midnight UTC = epoch 0
        val v0 = sf.extract(
            StructuredFeatures.Input(
                packageName = "x",
                title = "",
                body = "",
                postedAtMs = 0L,
                timeZone = utc,
            )
        )
        // sin(0)=0, cos(0)=1
        assertEquals(0f, v0[StructuredFeatures.HOUR_SIN_IDX], 1e-5f)
        assertEquals(1f, v0[StructuredFeatures.HOUR_COS_IDX], 1e-5f)
    }

    @Test
    fun rejectsTooManyTopPackages() {
        try {
            StructuredFeatures(List(31) { "pkg$it" })
            throw AssertionError("expected error")
        } catch (e: IllegalArgumentException) {
            // expected
        }
    }
}
