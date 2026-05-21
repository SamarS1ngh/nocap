package com.nocap.app.gemini

// Holds both OpenAI (primary) and Gemini (backup) call paths.
// classify() tries OpenAI first, falls back to Gemini on failure.

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

class GeminiClient {

    private val http = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    private val openAiLimiter = RateLimiter(maxPerMinute = 60)
    private val geminiLimiter = RateLimiter(maxPerMinute = 12)

    // ----- PUBLIC API -----

    suspend fun classify(
        openAiKey: String?,
        geminiKey: String?,
        packageName: String,
        title: String,
        body: String,
        feedbackExamples: List<FeedbackExample> = emptyList(),
    ): Result<Classification> = withContext(Dispatchers.IO) {
        val userPart = buildString {
            val wanted = feedbackExamples.filter { it.wanted }
            val unwanted = feedbackExamples.filterNot { it.wanted }
            if (wanted.isNotEmpty() || unwanted.isNotEmpty()) {
                append("Samar's past feedback (use as guidance — past tastes don't override rule 1 of direction, but DO influence importance scoring):\n")
                if (wanted.isNotEmpty()) {
                    append("\nSamar WANTED to see these:\n")
                    wanted.take(6).forEach { ex ->
                        append("- App: ").append(ex.packageName)
                            .append(" | Title: ").append(ex.title.take(40))
                            .append(" | Body: ").append(ex.body.take(60))
                            .append('\n')
                    }
                }
                if (unwanted.isNotEmpty()) {
                    append("\nSamar did NOT want to see these:\n")
                    unwanted.take(6).forEach { ex ->
                        append("- App: ").append(ex.packageName)
                            .append(" | Title: ").append(ex.title.take(40))
                            .append(" | Body: ").append(ex.body.take(60))
                            .append('\n')
                    }
                }
                append("\nNow classify this NEW notification:\n")
            }
            append("App: ").append(packageName).append('\n')
            append("Title: ").append(title.ifBlank { "(none)" }).append('\n')
            append("Body: ").append(body.ifBlank { "(none)" })
        }

        val errors = mutableListOf<String>()

        if (!openAiKey.isNullOrBlank()) {
            val raw = callOpenAi(openAiKey, SYSTEM_PROMPT, userPart, forceJson = true)
            raw.onSuccess { jsonText ->
                runCatching { json.decodeFromString<Classification>(jsonText) }
                    .onSuccess { return@withContext Result.success(it) }
                    .onFailure { errors += "OpenAI parse: ${it.message?.take(80)}" }
            }.onFailure { errors += "OpenAI: ${it.message?.take(100)}" }
        }

        if (!geminiKey.isNullOrBlank()) {
            val raw = callGemini(geminiKey, SYSTEM_PROMPT + "\n\n" + userPart)
            raw.onSuccess { jsonText ->
                runCatching { json.decodeFromString<Classification>(jsonText) }
                    .onSuccess { return@withContext Result.success(it) }
                    .onFailure { errors += "Gemini parse: ${it.message?.take(80)}" }
            }.onFailure { errors += "Gemini: ${it.message?.take(100)}" }
        }

        Result.failure(
            IOException(
                if (errors.isEmpty()) "no api keys configured"
                else errors.joinToString(" | ")
            )
        )
    }

    suspend fun pingOpenAi(apiKey: String): Result<String> = withContext(Dispatchers.IO) {
        callOpenAi(apiKey, systemPrompt = null, userPart = "Reply with the single word: pong", forceJson = false)
            .map { it.trim().take(40) }
    }

    suspend fun pingGemini(apiKey: String): Result<String> = withContext(Dispatchers.IO) {
        callGemini(apiKey, "Reply with the single word: pong")
            .map { it.trim().take(40) }
    }

    // ----- OPENAI PATH -----

    private suspend fun callOpenAi(
        apiKey: String,
        systemPrompt: String?,
        userPart: String,
        forceJson: Boolean,
    ): Result<String> {
        val messages = buildList {
            if (systemPrompt != null) add(ChatMessage(role = "system", content = systemPrompt))
            add(ChatMessage(role = "user", content = userPart))
        }
        val request = OpenAiRequest(
            model = OPENAI_MODEL,
            messages = messages,
            temperature = 0.0,
            topP = 0.1,
            seed = 42,
            responseFormat = if (forceJson) ResponseFormat(type = "json_object") else null,
        )
        val req = Request.Builder()
            .url(OPENAI_ENDPOINT)
            .header("Authorization", "Bearer $apiKey")
            .post(json.encodeToString(request).toRequestBody(JSON_MEDIA))
            .build()

        return runWithRetry(openAiLimiter, req) { bodyStr ->
            val parsed = json.decodeFromString<OpenAiResponse>(bodyStr)
            parsed.choices.firstOrNull()?.message?.content
        }
    }

    // ----- GEMINI PATH -----

    private suspend fun callGemini(apiKey: String, prompt: String): Result<String> {
        val request = GeminiRequest(
            contents = listOf(GeminiContent(parts = listOf(GeminiPart(text = prompt)))),
            generationConfig = GenerationConfig(responseMimeType = "application/json")
        )
        val req = Request.Builder()
            .url("$GEMINI_ENDPOINT?key=$apiKey")
            .post(json.encodeToString(request).toRequestBody(JSON_MEDIA))
            .build()

        return runWithRetry(geminiLimiter, req) { bodyStr ->
            val parsed = json.decodeFromString<GeminiResponse>(bodyStr)
            parsed.candidates.firstOrNull()?.content?.parts?.firstOrNull()?.text
        }
    }

    // ----- SHARED RETRY / RATE-LIMIT HARNESS -----

    private suspend fun runWithRetry(
        limiter: RateLimiter,
        req: Request,
        extractText: (String) -> String?,
    ): Result<String> {
        var attempt = 0
        while (true) {
            limiter.acquire()
            val outcome: CallOutcome = try {
                http.newCall(req).execute().use { resp ->
                    val bodyStr = resp.body?.string().orEmpty()
                    when {
                        resp.code == 429 -> CallOutcome.RateLimited(bodyStr)
                        !resp.isSuccessful -> CallOutcome.Failure(
                            IOException("HTTP ${resp.code}: ${bodyStr.take(200)}")
                        )
                        else -> {
                            val text = extractText(bodyStr)
                            if (text == null) CallOutcome.Failure(IOException("no content in response"))
                            else CallOutcome.Success(text)
                        }
                    }
                }
            } catch (t: Throwable) {
                CallOutcome.Failure(t)
            }

            when (outcome) {
                is CallOutcome.Success -> return Result.success(outcome.text)
                is CallOutcome.Failure -> return Result.failure(outcome.error)
                is CallOutcome.RateLimited -> {
                    if (attempt >= 2) {
                        return Result.failure(
                            IOException("rate-limited after retries: ${outcome.bodySnippet.take(120)}")
                        )
                    }
                    attempt++
                    val waitMs = 5_000L shl (attempt - 1)
                    delay(waitMs)
                }
            }
        }
        @Suppress("UNREACHABLE_CODE")
        return Result.failure(IOException("unreachable"))
    }

    private sealed class CallOutcome {
        data class Success(val text: String) : CallOutcome()
        data class Failure(val error: Throwable) : CallOutcome()
        data class RateLimited(val bodySnippet: String) : CallOutcome()
    }

    // ----- OPENAI WIRE TYPES -----

    @Serializable
    private data class OpenAiRequest(
        val model: String,
        val messages: List<ChatMessage>,
        val temperature: Double,
        @SerialName("top_p") val topP: Double,
        val seed: Int,
        @SerialName("response_format") val responseFormat: ResponseFormat? = null,
    )

    @Serializable
    private data class ChatMessage(val role: String, val content: String)

    @Serializable
    private data class ResponseFormat(val type: String)

    @Serializable
    private data class OpenAiResponse(val choices: List<OpenAiChoice> = emptyList())

    @Serializable
    private data class OpenAiChoice(val message: OpenAiMessageOut? = null)

    @Serializable
    private data class OpenAiMessageOut(val role: String, val content: String)

    // ----- GEMINI WIRE TYPES -----

    @Serializable
    private data class GeminiRequest(
        val contents: List<GeminiContent>,
        val generationConfig: GenerationConfig,
    )

    @Serializable
    private data class GeminiContent(val parts: List<GeminiPart>)

    @Serializable
    private data class GeminiPart(val text: String)

    @Serializable
    private data class GenerationConfig(
        val responseMimeType: String,
        val temperature: Double = 0.0,
        val topP: Double = 0.1,
        val seed: Int = 42,
    )

    @Serializable
    private data class GeminiResponse(val candidates: List<GeminiCandidate> = emptyList())

    @Serializable
    private data class GeminiCandidate(val content: GeminiContentOut? = null)

    @Serializable
    private data class GeminiContentOut(val parts: List<GeminiPart> = emptyList())

    companion object {
        private const val OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
        private const val OPENAI_MODEL = "gpt-4o-mini"
        private const val GEMINI_ENDPOINT =
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

        private val JSON_MEDIA = "application/json".toMediaType()

        private val SYSTEM_PROMPT = """
            You classify Android notifications for a single user named Samar.

            CRITICAL CONTEXT: Samar OWNS this phone. Every notification arrives ON his device.

            DIRECTION RULES (apply in order):

            1. DEFAULT for any messaging/social app (WhatsApp, Instagram, Telegram, Signal,
               Messenger, Discord, SMS, Snapchat, etc.): assume the notification is INCOMING
               to Samar. Classify as personal, importance 9-10.

            2. OVERRIDE to outgoing_message ONLY if the title OR body literally starts with
               one of these exact phrases (case-insensitive):
                 - "You replied", "You sent", "You posted", "You shared"
                 - "Message sent", "Reply sent", "Sent to ", "Delivered to "
               No other signal flips a message to outgoing.

            3. Sender identification (for the summary):
                 - WhatsApp / Signal / SMS: title is the sender's name. Body is just the message text.
                 - Instagram / Telegram / multi-account apps: body usually has "<name>: <message>"
                   format. That "<name>" is the sender. Title may be Samar's own account
                   handle for inbox routing — IGNORE the title in those apps.
                 - If unclear, use whichever field contains a person's name.

            4. Never decide direction by guessing if a name "looks like Samar". You do not
               know his usernames. Direction is determined ONLY by rule 2.

            Output STRICT JSON only. No prose, no markdown. Shape:
            { "category": string, "importance": integer 0-10, "summary": string (<= 80 chars), "reason": string (<= 25 words) }

            Categories (pick exactly one):
              world_news              - geopolitics, world finance, world politics
              indian_national_big     - MAJOR Indian national news affecting many lives
              indian_national_routine - routine Indian national news
              indian_local            - city / local news
              tech                    - tech, innovation, software, hardware, science
              meme                    - memes, jokes, humor
              reddit                  - any Reddit notification (regardless of subject)
              personal                - INCOMING direct message TO Samar, plus banking, calendar, alarm, contacts
              outgoing_message        - acknowledgement of Samar's OWN action (sent, replied, posted)
              social                  - non-DM social interactions: likes, follows, comments, reactions, mentions, story views
              promo                   - ads, promotions, marketing, sales
              other                   - anything else

            Importance scoring guidance (Samar's preferences):
              - personal (incoming DM, banking, calendar, alarm): 9-10
              - world_news affecting many people: 8-10
              - tech / innovation: 8-10
              - meme: 7-9
              - indian_national_big: 7-9
              - social (likes/follows/comments/mentions): 2-4
              - indian_national_routine: 2-4
              - indian_local: 1-3
              - outgoing_message: 1-2 (Samar already knows)
              - reddit: 1-3 (Samar wants Reddit hidden by default)
              - promo: 0-2
              - other: judge by content

            Summary format rules (be CONSISTENT across runs):
              - personal DM: exactly "DM from <sender>: <first ~40 chars of message>"
                where <sender> is the OTHER person's name/handle from the title.
                Example: "DM from John: hey are we still on for tonight"
              - outgoing_message: exactly "You sent <recipient>: <first ~40 chars>"
                or "You posted: <gist>"
              - social: "<actor> <action> your <thing>"
                Example: "john_doe liked your post"
              - news/tech/meme: 1 sentence capturing what HAPPENED, <=80 chars, not a rephrasing of the title
              - Always under 80 characters total

            Be honest in "reason" - explain the score in <= 25 words.
        """.trimIndent()
    }
}

private class RateLimiter(private val maxPerMinute: Int) {
    private val mutex = Mutex()
    private val recent = ArrayDeque<Long>()

    suspend fun acquire() {
        while (true) {
            val waitMs = mutex.withLock {
                val now = System.currentTimeMillis()
                while (recent.isNotEmpty() && now - recent.first() >= 60_000) {
                    recent.removeFirst()
                }
                if (recent.size < maxPerMinute) {
                    recent.addLast(now)
                    return
                }
                60_000L - (now - recent.first()) + 50
            }
            if (waitMs > 0) delay(waitMs)
        }
    }
}
