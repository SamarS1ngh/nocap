#!/usr/bin/env python3
"""
Convert sentence-transformers/all-MiniLM-L6-v2 (PyTorch) to TFLite for nocap.

Pipeline:
    HuggingFace PyTorch
        → ONNX (via torch.onnx.export)
        → TF SavedModel (via onnx2tf)
        → TFLite (via tf.lite.TFLiteConverter)

Output:
    embedding-engine/src/main/assets/minilm.tflite
    embedding-engine/src/main/assets/vocab.txt

Graph signature produced (must match EmbeddingEngine.kt expectations):
    inputs:
        input_ids        (1, 128) int32
        attention_mask   (1, 128) int32
        token_type_ids   (1, 128) int32
    output:
        token_embeddings (1, 128, 384) float32  — raw token-level, pre-pool

Install once:
    pip install -r tools/requirements.txt

Run:
    python tools/convert_minilm.py
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "embedding-engine" / "src" / "main" / "assets"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEQ_LEN = 128
EMBED_DIM = 384

VOCAB_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main/vocab.txt"


def log(msg: str) -> None:
    print(f"[convert_minilm] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[convert_minilm] FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def check_deps() -> None:
    failures: list[tuple[str, str]] = []
    for mod in ("torch", "transformers", "onnx", "onnx2tf", "tensorflow"):
        try:
            __import__(mod)
        except Exception as e:  # ImportError + any module-level import-time exception
            failures.append((mod, f"{type(e).__name__}: {e}"))
    if failures:
        lines = ["dependency import failed:"]
        for name, msg in failures:
            lines.append(f"  - {name}: {msg}")
        lines.append("Install / repair with: pip install -r tools/requirements.txt")
        lines.append("If onnx2tf imports fail, also try: pip install tf-keras")
        die("\n".join(lines))


def export_to_onnx(onnx_path: Path) -> None:
    """Export the MiniLM PyTorch model to ONNX with fixed (1, 128) input shape."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    log(f"downloading {MODEL_ID} (PyTorch weights, ~90MB on first run)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    # Dummy input that pins the export to (batch=1, seq=128).
    dummy = tokenizer(
        "warmup",
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = dummy["input_ids"].to(torch.int32)
    attention_mask = dummy["attention_mask"].to(torch.int32)
    token_type_ids = dummy.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    else:
        token_type_ids = token_type_ids.to(torch.int32)

    class Wrapper(torch.nn.Module):
        """
        Wrap the encoder so the exported graph returns ONLY token_embeddings
        (last_hidden_state). Pooling + L2-norm happens in Kotlin so we keep
        the graph identical to upstream and can A/B pooling strategies.
        """

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(
                input_ids=input_ids.long(),
                attention_mask=attention_mask.long(),
                token_type_ids=token_type_ids.long(),
            ).last_hidden_state

    wrapped = Wrapper(model)
    wrapped.eval()

    log(f"exporting ONNX → {onnx_path}")
    torch.onnx.export(
        wrapped,
        (input_ids, attention_mask, token_type_ids),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["token_embeddings"],
        opset_version=14,
        do_constant_folding=True,
        dynamic_axes=None,  # fully static shapes — TFLite likes this
    )

    # Sanity check: shapes are what Kotlin expects.
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    expected_outputs = {"token_embeddings"}
    actual_outputs = {o.name for o in onnx_model.graph.output}
    if expected_outputs != actual_outputs:
        die(f"unexpected ONNX outputs: {actual_outputs}, expected {expected_outputs}")
    log("ONNX export OK")


def onnx_to_saved_model(onnx_path: Path, saved_model_dir: Path) -> None:
    """ONNX → TF SavedModel via onnx2tf CLI (more reliable than onnx-tf)."""
    log(f"converting ONNX → TF SavedModel → {saved_model_dir}")
    saved_model_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i",
        str(onnx_path),
        "-o",
        str(saved_model_dir),
        "-osd",          # output saved-model
        "-cotof",        # check output tensor for sanity
        "--non_verbose",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log("onnx2tf stdout:\n" + result.stdout)
        log("onnx2tf stderr:\n" + result.stderr)
        die(f"onnx2tf exited with code {result.returncode}")
    log("SavedModel conversion OK")


def saved_model_to_tflite(saved_model_dir: Path, tflite_path: Path) -> None:
    """TF SavedModel → TFLite (float32, no quantization for first pass)."""
    import tensorflow as tf

    log(f"converting SavedModel → TFLite → {tflite_path}")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    # Keep float32 for first ship. Add int8 quantization later if APK size hurts.
    converter.optimizations = []
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,  # safety net for ops TFLite lacks natively
    ]
    tflite_bytes = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_bytes)
    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    log(f"wrote {tflite_path.name}: {size_mb:.1f} MB")


def verify_tflite(tflite_path: Path) -> None:
    """Open the produced .tflite and verify signature matches what Kotlin expects."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    log("verifying TFLite signature")
    log(f"  inputs:  {len(in_details)}")
    for d in in_details:
        log(f"    {d['name']:30s} shape={list(d['shape'])} dtype={d['dtype'].__name__}")
    log(f"  outputs: {len(out_details)}")
    for d in out_details:
        log(f"    {d['name']:30s} shape={list(d['shape'])} dtype={d['dtype'].__name__}")

    # Sanity checks
    if len(in_details) != 3:
        die(f"expected 3 inputs, got {len(in_details)}")
    if len(out_details) != 1:
        die(f"expected 1 output, got {len(out_details)}")
    out_shape = list(out_details[0]["shape"])
    if out_shape != [1, SEQ_LEN, EMBED_DIM]:
        die(f"output shape mismatch: got {out_shape}, want [1, {SEQ_LEN}, {EMBED_DIM}]")
    log("signature OK")


def download_vocab(vocab_path: Path) -> None:
    log(f"downloading vocab.txt → {vocab_path}")
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(VOCAB_URL) as resp:
        vocab_path.write_bytes(resp.read())
    size_kb = vocab_path.stat().st_size / 1024
    log(f"wrote vocab.txt: {size_kb:.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-tflite",
        type=Path,
        default=ASSETS_DIR / "minilm.tflite",
        help="output path for the .tflite file",
    )
    parser.add_argument(
        "--out-vocab",
        type=Path,
        default=ASSETS_DIR / "vocab.txt",
        help="output path for vocab.txt",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="keep ONNX and SavedModel in a sibling dir for debugging",
    )
    args = parser.parse_args()

    check_deps()

    if args.keep_intermediate:
        scratch_dir = REPO_ROOT / "tools" / "build"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        tempdir_cleanup = None
    else:
        scratch_dir = Path(tempfile.mkdtemp(prefix="minilm-convert-"))
        tempdir_cleanup = scratch_dir

    try:
        onnx_path = scratch_dir / "minilm.onnx"
        saved_model_dir = scratch_dir / "saved_model"

        export_to_onnx(onnx_path)
        onnx_to_saved_model(onnx_path, saved_model_dir)
        saved_model_to_tflite(saved_model_dir, args.out_tflite)
        verify_tflite(args.out_tflite)
        download_vocab(args.out_vocab)

        log("done. drop the assets in place — run ./gradlew :app:installDebug next.")
    finally:
        if tempdir_cleanup is not None and tempdir_cleanup.exists():
            shutil.rmtree(tempdir_cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
