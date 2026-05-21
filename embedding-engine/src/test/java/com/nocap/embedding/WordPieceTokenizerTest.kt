package com.nocap.embedding

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class WordPieceTokenizerTest {

    private fun buildTokenizer(): WordPieceTokenizer {
        // Tiny vocab covering specials + a few wordpieces. Order = id.
        val vocab = listOf(
            "[PAD]", "[UNK]", "[CLS]", "[SEP]",
            "hello", "world",
            "play", "##ing", "##er",
            ".", ",", "?",
            "the", "cat",
        ).withIndex().associate { (i, v) -> v to i }
        return WordPieceTokenizer.fromVocab(vocab)
    }

    @Test
    fun specialTokensIdsAreCorrect() {
        val t = buildTokenizer()
        assertEquals(0, t.padId)
        assertEquals(1, t.unkId)
        assertEquals(2, t.clsId)
        assertEquals(3, t.sepId)
    }

    @Test
    fun encodeWrapsWithClsAndSep() {
        val t = buildTokenizer()
        val out = t.encode("hello world", maxSeqLen = 8)

        // [CLS] hello world [SEP] [PAD] [PAD] [PAD] [PAD]
        assertEquals(t.clsId, out.inputIds[0])
        assertEquals(4, out.inputIds[1])  // hello
        assertEquals(5, out.inputIds[2])  // world
        assertEquals(t.sepId, out.inputIds[3])
        for (i in 4 until 8) assertEquals(t.padId, out.inputIds[i])
    }

    @Test
    fun attentionMaskMatchesContent() {
        val t = buildTokenizer()
        val out = t.encode("hello", maxSeqLen = 6)
        // [CLS] hello [SEP] [PAD] [PAD] [PAD]
        assertArrayEquals(intArrayOf(1, 1, 1, 0, 0, 0), out.attentionMask)
    }

    @Test
    fun wordpieceContinuationUsesHashHashPrefix() {
        val t = buildTokenizer()
        val out = t.encode("playing", maxSeqLen = 6)
        // play + ##ing → 6, 7
        assertEquals(6, out.inputIds[1])
        assertEquals(7, out.inputIds[2])
    }

    @Test
    fun unknownWordFallsBackToUnk() {
        val t = buildTokenizer()
        val out = t.encode("xyznotinvocab", maxSeqLen = 4)
        assertEquals(t.unkId, out.inputIds[1])
    }

    @Test
    fun lowercasingApplied() {
        val t = buildTokenizer()
        val out = t.encode("HELLO", maxSeqLen = 4)
        assertEquals(4, out.inputIds[1])  // "hello"
    }

    @Test
    fun punctuationSplitFromWord() {
        val t = buildTokenizer()
        val out = t.encode("hello.", maxSeqLen = 6)
        // hello + .
        assertEquals(4, out.inputIds[1])
        assertEquals(9, out.inputIds[2])
    }

    @Test
    fun truncatesToMaxSeqLen() {
        val t = buildTokenizer()
        val long = (1..50).joinToString(" ") { "hello" }
        val out = t.encode(long, maxSeqLen = 8)
        // 8 - 2 = 6 content slots between CLS and SEP
        assertEquals(t.clsId, out.inputIds[0])
        assertEquals(t.sepId, out.inputIds[7])
        for (i in 1..6) assertEquals(4, out.inputIds[i])
    }
}
