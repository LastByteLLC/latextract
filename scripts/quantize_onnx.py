"""Quantize TexTeller ONNX models for browser deployment.

Pulls FP32 ONNX from OleehyO/TexTeller (already published on HF) and produces:
  - int8/  : dynamic INT8 (deployable in ORT Web)
  - int4/  : MatMulNBits 4-bit weight-only (deployable in ORT Web with WebGPU/WASM)
  - int2/  : MatMulNBits 2-bit weight-only ("ternary" — Python-only, NOT in ORT Web)
  - fp32/  : original (baseline)

Usage:
  python scripts/quantize_onnx.py --out web/public/models
  python scripts/quantize_onnx.py --variants int8,int4 --out web/public/models
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

log = logging.getLogger("quantize_onnx")

REPO_ID = "OleehyO/TexTeller"
ONNX_FILES = ("encoder_model.onnx", "decoder_model_merged.onnx")
CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)


def fetch_fp32(cache_dir: Path) -> Path:
    """Download the FP32 ONNX + tokenizer from HF. Returns local snapshot path."""
    log.info("downloading %s (encoder + decoder_merged + tokenizer)", REPO_ID)
    local = snapshot_download(
        REPO_ID,
        cache_dir=str(cache_dir),
        allow_patterns=list(ONNX_FILES) + list(CONFIG_FILES),
    )
    return Path(local)


def copy_configs(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in CONFIG_FILES:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)


def copy_fp32(src: Path, dst: Path) -> None:
    onnx_dir = dst / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    for name in ONNX_FILES:
        shutil.copy2(src / name, onnx_dir / name)
    copy_configs(src, dst)


def quantize_int8(src: Path, dst: Path) -> None:
    """Dynamic INT8 quantization. Best ORT Web compatibility."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_dir = dst / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    for name in ONNX_FILES:
        src_path = src / name
        # transformers.js naming convention: <model>_quantized.onnx
        out_name = name.replace(".onnx", "_quantized.onnx")
        dst_path = onnx_dir / out_name
        log.info("INT8 quantize %s -> %s", name, out_name)
        quantize_dynamic(
            model_input=str(src_path),
            model_output=str(dst_path),
            weight_type=QuantType.QUInt8,
            per_channel=False,
            reduce_range=False,
            extra_options={"DefaultTensorType": 1},  # FLOAT
        )
    copy_configs(src, dst)


def quantize_nbits(src: Path, dst: Path, bits: int, block_size: int = 32) -> None:
    """Weight-only N-bit quantization via MatMulNBits.

    bits=4 → ORT Web compatible (with WebGPU/WASM).
    bits=2 → Python-only baseline (ORT Web does not implement INT2 MatMul).
    """
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer,
        DefaultWeightOnlyQuantConfig,
    )

    onnx_dir = dst / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    for name in ONNX_FILES:
        src_path = src / name
        out_name = name.replace(".onnx", f"_q{bits}.onnx")
        dst_path = onnx_dir / out_name
        log.info("INT%d quantize %s -> %s (block=%d)", bits, name, out_name, block_size)
        cfg = DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=True,
            bits=bits,
        )
        # Pass model path string so the quantizer wraps it in its ONNXModel
        # internally; output via ONNXModel.save_model_to_file (the public API).
        quantizer = MatMulNBitsQuantizer(str(src_path), algo_config=cfg)
        quantizer.process()
        # External-data format keeps any individual file under 2GB and is what
        # transformers.js / ORT Web load happily. The merged decoder needs it
        # because the saved proto can blow past the 2GB protobuf limit.
        use_ext = src_path.stat().st_size > 1_500_000_000
        quantizer.model.save_model_to_file(str(dst_path), use_external_data_format=use_ext)
    copy_configs(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/public/models", help="output root directory")
    ap.add_argument(
        "--variants", default="fp32,int8,int4,int2",
        help="comma-separated subset of: fp32,int8,int4,int2",
    )
    ap.add_argument("--cache", default=".hf_cache", help="HF cache dir for FP32 source")
    ap.add_argument("--block-size", type=int, default=32, help="MatMulNBits block size")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    out_root = Path(args.out)
    cache = Path(args.cache)
    variants = {v.strip() for v in args.variants.split(",") if v.strip()}

    fp32 = fetch_fp32(cache)
    log.info("FP32 source: %s", fp32)

    if "fp32" in variants:
        copy_fp32(fp32, out_root / "fp32")
    if "int8" in variants:
        quantize_int8(fp32, out_root / "int8")
    if "int4" in variants:
        quantize_nbits(fp32, out_root / "int4", bits=4, block_size=args.block_size)
    if "int2" in variants:
        quantize_nbits(fp32, out_root / "int2", bits=2, block_size=args.block_size)

    log.info("done. outputs in %s", out_root)
    for v in sorted(variants):
        d = out_root / v / "onnx"
        if d.exists():
            sizes = {f.name: f.stat().st_size for f in d.glob("*.onnx")}
            total = sum(sizes.values()) / 1e6
            log.info("  %s: %.1f MB total — %s", v, total, ", ".join(f"{n}={s/1e6:.0f}MB" for n, s in sizes.items()))


if __name__ == "__main__":
    main()
