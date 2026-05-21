# embedding-engine assets

Two files must be placed here before the module can run on device. They are intentionally **not committed** (model is ~90 MB; vocab is upstream-owned).

## Required files

| Filename | Purpose | Size |
|---|---|---|
| `minilm.tflite` | MiniLM-L6-v2 weights + graph | ~90 MB |
| `vocab.txt` | BERT WordPiece vocab | ~230 KB |

## Source

Both come from `sentence-transformers/all-MiniLM-L6-v2` on Hugging Face.

The PyTorch model must be converted to TFLite (the upstream repo doesn't ship one). Conversion script lives outside this module (see `tools/convert_minilm.py` in the project root once added).

Expected TFLite graph signature:
- inputs:
  - `input_ids` shape `(1, 128)` int32
  - `attention_mask` shape `(1, 128)` int32
  - `token_type_ids` shape `(1, 128)` int32
- outputs:
  - `token_embeddings` shape `(1, 128, 384)` float32 — raw token-level embeddings (pre-pool, pre-norm)

`EmbeddingEngine` handles mean-pool + L2-normalize in Kotlin. Do NOT bake pooling into the TFLite graph or shapes diverge.

## Why mean-pool + L2-norm in Kotlin (not in the graph)
Keeps the graph identical to upstream and lets us A/B between pooling strategies (mean / CLS / max) without re-converting the model.

## Bundling note
The build script sets `noCompress += "tflite"` so the model maps directly via `MappedByteBuffer`. Do not gzip the asset.

## APK size

90 MB asset will balloon the APK. Options when ready:
1. Asset Pack (install-time) — recommended for Play Store delivery.
2. Download on first run from a hosted URL, persist to internal storage, point Interpreter at file path.

Until then, the model lives in `assets/` for dev convenience.
