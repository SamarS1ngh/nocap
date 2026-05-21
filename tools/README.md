# tools/

One-shot scripts. Not built into the app. Run by hand on a dev machine.

## convert_minilm.py

Builds `embedding-engine/src/main/assets/minilm.tflite` and `vocab.txt` from the upstream HuggingFace PyTorch checkpoint of `sentence-transformers/all-MiniLM-L6-v2`.

### One-time setup
```bash
cd <repo root>
python -m venv .venv
source .venv/bin/activate
pip install -r tools/requirements.txt
```

(Python 3.10 / 3.11 recommended. TensorFlow doesn't always have wheels for the very latest Python.)

### Run
```bash
python tools/convert_minilm.py
```

Takes ~5-10 min. Downloads ~90MB on first run (cached by HuggingFace + Torch hubs).

Output:
- `embedding-engine/src/main/assets/minilm.tflite` (~90 MB float32)
- `embedding-engine/src/main/assets/vocab.txt` (~230 KB)

### Verify
The script verifies the signature itself before exiting:
```
inputs:
  serving_default_input_ids       shape=[1, 128] dtype=int32
  serving_default_attention_mask  shape=[1, 128] dtype=int32
  serving_default_token_type_ids  shape=[1, 128] dtype=int32
outputs:
  StatefulPartitionedCall:0       shape=[1, 128, 384] dtype=float32
```

If signature differs, the script aborts with `FATAL: ...`.

### Re-running
The script overwrites the existing `.tflite` and `vocab.txt`. Safe to re-run after model upgrades.

### Troubleshooting

- **`onnx2tf` errors about ops**: open `tools/build/saved_model/` (run with `--keep-intermediate`) and inspect. Usually fixable by bumping `opset_version` in `export_to_onnx`.
- **TensorFlow install fails**: pin to Python 3.10 or 3.11 in a fresh venv.
- **Apple Silicon / M1**: use `tensorflow-macos` in `requirements.txt` instead of `tensorflow`.
