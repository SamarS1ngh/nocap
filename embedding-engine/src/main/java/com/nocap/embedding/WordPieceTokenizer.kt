package com.nocap.embedding

import android.content.Context
import java.io.BufferedReader
import java.io.InputStreamReader

/**
 * BERT-style WordPiece tokenizer matching the vocab shipped with sentence-transformers/all-MiniLM-L6-v2.
 *
 * Reads vocab.txt from assets. Each line is one wordpiece; the line number is the token id.
 * Implements:
 *   1. Basic whitespace + punctuation splitting and lowercasing (uncased model).
 *   2. Greedy longest-match WordPiece on each whitespace token (## continuation prefix).
 *   3. [CLS] ... [SEP] wrapping with attention mask and zero token-type ids.
 *
 * Special tokens (must exist in vocab.txt):
 *   [PAD] [UNK] [CLS] [SEP]
 */
class WordPieceTokenizer private constructor(
    private val vocab: Map<String, Int>,
) {

    val padId: Int = vocab.getValue("[PAD]")
    val unkId: Int = vocab.getValue("[UNK]")
    val clsId: Int = vocab.getValue("[CLS]")
    val sepId: Int = vocab.getValue("[SEP]")

    data class Encoded(
        val inputIds: IntArray,
        val attentionMask: IntArray,
        val tokenTypeIds: IntArray,
    )

    fun encode(text: String, maxSeqLen: Int = MAX_SEQ_LEN): Encoded {
        val pieces = wordpieces(basicTokenize(text))
        val truncated = pieces.take(maxSeqLen - 2)

        val ids = IntArray(maxSeqLen) { padId }
        val mask = IntArray(maxSeqLen)

        ids[0] = clsId
        mask[0] = 1
        for ((i, p) in truncated.withIndex()) {
            ids[i + 1] = vocab[p] ?: unkId
            mask[i + 1] = 1
        }
        ids[truncated.size + 1] = sepId
        mask[truncated.size + 1] = 1

        return Encoded(
            inputIds = ids,
            attentionMask = mask,
            tokenTypeIds = IntArray(maxSeqLen),
        )
    }

    private fun basicTokenize(text: String): List<String> {
        val out = mutableListOf<String>()
        val sb = StringBuilder()
        fun flush() {
            if (sb.isNotEmpty()) {
                out += sb.toString()
                sb.clear()
            }
        }
        for (ch in text.lowercase()) {
            when {
                ch.isWhitespace() -> flush()
                ch.isPunctuation() -> {
                    flush()
                    out += ch.toString()
                }
                else -> sb.append(ch)
            }
        }
        flush()
        return out
    }

    private fun wordpieces(tokens: List<String>): List<String> {
        val pieces = mutableListOf<String>()
        for (token in tokens) {
            if (token.length > MAX_WORD_LEN) {
                pieces += "[UNK]"
                continue
            }
            var start = 0
            var bad = false
            val subs = mutableListOf<String>()
            while (start < token.length) {
                var end = token.length
                var cur: String? = null
                while (start < end) {
                    val candidate = (if (start > 0) "##" else "") + token.substring(start, end)
                    if (candidate in vocab) {
                        cur = candidate
                        break
                    }
                    end--
                }
                if (cur == null) {
                    bad = true
                    break
                }
                subs += cur
                start = end
            }
            if (bad) pieces += "[UNK]" else pieces += subs
        }
        return pieces
    }

    private fun Char.isPunctuation(): Boolean {
        val cp = code
        return (cp in 33..47) || (cp in 58..64) || (cp in 91..96) || (cp in 123..126) ||
            Character.getType(this).let {
                it == Character.CONNECTOR_PUNCTUATION.toInt() ||
                    it == Character.DASH_PUNCTUATION.toInt() ||
                    it == Character.START_PUNCTUATION.toInt() ||
                    it == Character.END_PUNCTUATION.toInt() ||
                    it == Character.INITIAL_QUOTE_PUNCTUATION.toInt() ||
                    it == Character.FINAL_QUOTE_PUNCTUATION.toInt() ||
                    it == Character.OTHER_PUNCTUATION.toInt()
            }
    }

    companion object {
        const val MAX_SEQ_LEN = 128
        private const val MAX_WORD_LEN = 100

        internal fun fromVocab(map: Map<String, Int>): WordPieceTokenizer = WordPieceTokenizer(map)

        fun fromAssets(context: Context, vocabAsset: String = "vocab.txt"): WordPieceTokenizer {
            val map = HashMap<String, Int>(32000)
            context.assets.open(vocabAsset).use { stream ->
                BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).use { reader ->
                    var idx = 0
                    while (true) {
                        val line = reader.readLine() ?: break
                        map[line] = idx
                        idx++
                    }
                }
            }
            return WordPieceTokenizer(map)
        }
    }
}
