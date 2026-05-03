"""Profile each quantized ONNX variant for size, speed, and accuracy.

Uses optimum.onnxruntime.ORTModelForVision2Seq, which handles the merged-decoder
KV-cache plumbing (the use_cache_branch + past_key_values dance) so we get a
faithful greedy decode that matches what transformers.js will run in the browser.

Outputs:
  runs/quant_profile.csv
  runs/quant_profile.md
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latextract.grammar.cfg import validate_grammar  # noqa: E402
from latextract.model.inference import _preprocess_vit, strip_artifacts  # noqa: E402

log = logging.getLogger("profile_quant")

# Quantize script writes files with these suffixes; map to optimum kwargs.
SUFFIX_BY_VARIANT = {
    "fp32": ("encoder_model.onnx", "decoder_model_merged.onnx"),
    "int8": ("encoder_model_quantized.onnx", "decoder_model_merged_quantized.onnx"),
    "int4": ("encoder_model_q4.onnx", "decoder_model_merged_q4.onnx"),
    "int2": ("encoder_model_q2.onnx", "decoder_model_merged_q2.onnx"),
}


@dataclass
class VariantResult:
    variant: str
    size_mb: float
    load_ms: float
    n_samples: int
    mean_latency_ms: float
    grammar_pass_pct: float
    exact_match_pct: float
    avg_tokens: float


def _wrap_for_grammar(body: str) -> str:
    body = body.strip()
    if body.startswith(("\\[", "\\begin{")):
        return body
    return f"\\[{body}\\]"


def profile_variant(variant_dir: Path, pairs: list[dict], image_size: int = 448) -> VariantResult:
    name = variant_dir.name
    enc_name, dec_name = SUFFIX_BY_VARIANT[name]
    onnx_dir = variant_dir / "onnx"
    enc_path = onnx_dir / enc_name
    dec_path = onnx_dir / dec_name
    if not enc_path.exists() or not dec_path.exists():
        raise FileNotFoundError(f"missing ONNX files for {name}: {enc_path}, {dec_path}")

    size_mb = (enc_path.stat().st_size + dec_path.stat().st_size) / 1e6
    log.info("[%s] loading (%.1f MB)", name, size_mb)

    # Optimum's loader has an awkward interaction: passing decoder_file_name
    # opts the model out of merged-decoder mode and demands a separate
    # decoder_with_past file. Sidestep by symlinking to the standard names
    # for the duration of the run.
    std_enc = onnx_dir / "encoder_model.onnx"
    std_dec = onnx_dir / "decoder_model_merged.onnx"
    created_links: list[Path] = []
    if not std_enc.exists():
        std_enc.symlink_to(enc_path.name)
        created_links.append(std_enc)
    if not std_dec.exists():
        std_dec.symlink_to(dec_path.name)
        created_links.append(std_dec)

    try:
        t0 = time.time()
        model = ORTModelForVision2Seq.from_pretrained(
            str(variant_dir),
            local_files_only=True,
            use_io_binding=False,
            provider="CPUExecutionProvider",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(variant_dir))
        load_ms = (time.time() - t0) * 1000
    finally:
        for link in created_links:
            try:
                link.unlink()
            except OSError:
                pass

    # TexTeller's ViT encoder is single-channel (grayscale). Read this from
    # the config rather than hardcoding so we don't break on retrained variants.
    enc_cfg = getattr(model.config, "encoder", model.config)
    num_channels = int(getattr(enc_cfg, "num_channels", 1))
    img_sz = getattr(enc_cfg, "image_size", image_size)
    if isinstance(img_sz, (list, tuple)):
        img_sz = img_sz[0]
    img_sz = int(img_sz)

    latencies: list[float] = []
    n_tokens: list[int] = []
    predictions: list[dict] = []
    for p in pairs:
        img = Image.open(ROOT / p["image"])
        px = _preprocess_vit(img, img_sz, num_channels=num_channels)
        t0 = time.time()
        try:
            with torch.inference_mode():
                out = model.generate(
                    pixel_values=px,
                    max_new_tokens=256,
                    num_beams=1,
                    do_sample=False,
                    no_repeat_ngram_size=4,
                    repetition_penalty=1.15,
                )
            latex = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
            latex = strip_artifacts(latex)
            n_tok = len(out[0])
        except Exception as e:
            log.warning("[%s] decode failed on %s: %s", name, p["id"], e)
            latex = ""
            n_tok = 0
        dt = (time.time() - t0) * 1000
        latencies.append(dt)
        n_tokens.append(n_tok)
        predictions.append({"id": p["id"], "ground_truth": p["body"], "pred": latex, "latency_ms": dt})
        log.info("[%s] %s — %.0fms — %r", name, p["id"], dt, latex[:80])

    samples = [_wrap_for_grammar(pr["pred"]) for pr in predictions if pr["pred"]]
    grammar_pct = validate_grammar(samples).pass_rate if samples else 0.0
    exact = sum(1 for pr in predictions if pr["pred"].strip() == pr["ground_truth"].strip())
    exact_pct = 100.0 * exact / max(1, len(predictions))

    # Save per-sample preds for later inspection
    pred_path = ROOT / "runs" / "quant" / f"preds_{name}.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w") as f:
        for pr in predictions:
            f.write(json.dumps(pr) + "\n")

    return VariantResult(
        variant=name,
        size_mb=size_mb,
        load_ms=load_ms,
        n_samples=len(pairs),
        mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        grammar_pass_pct=grammar_pct,
        exact_match_pct=exact_pct,
        avg_tokens=float(np.mean(n_tokens)) if n_tokens else 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="web/public/models")
    ap.add_argument("--manifest", default="data/rendered/manifest.jsonl")
    ap.add_argument("--out", default="runs/quant_profile")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--variants", default="", help="comma-separated subset (default: all dirs)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pairs = [
        json.loads(l) for l in (ROOT / args.manifest).read_text().splitlines() if l.strip()
    ]
    if args.limit:
        pairs = pairs[: args.limit]
    log.info("profiling on %d samples", len(pairs))

    models_root = Path(args.models)
    if args.variants:
        variants = [models_root / v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variants = sorted(d for d in models_root.iterdir() if d.is_dir() and (d / "onnx").exists())
    if not variants:
        log.error("no variants found in %s — run quantize_onnx.py first", models_root)
        sys.exit(1)

    results: list[VariantResult] = []
    for v in variants:
        try:
            results.append(profile_variant(v, pairs))
        except Exception as e:
            log.exception("[%s] profile failed: %s", v.name, e)

    if not results:
        log.error("no variants profiled successfully")
        sys.exit(1)

    out_csv = ROOT / f"{args.out}.csv"
    out_md = ROOT / f"{args.out}.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))

    lines = [
        "# Quantization profile",
        "",
        f"Profiled {len(results)} variants on {len(pairs)} eval samples (CPU, optimum-onnxruntime).",
        "",
        "| variant | size MB | load ms | mean latency ms | avg tokens | grammar % | exact % |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.variant} | {r.size_mb:.1f} | {r.load_ms:.0f} | {r.mean_latency_ms:.0f} | "
            f"{r.avg_tokens:.0f} | {r.grammar_pass_pct:.1f} | {r.exact_match_pct:.1f} |"
        )
    lines.append("")
    lines.append("**Note:** `int2` (~ternary) is included for accuracy reference; ONNX Runtime Web does not")
    lines.append("execute INT2 MatMul, so the browser demo only ships `int4` (or `int8` if size permits).")
    out_md.write_text("\n".join(lines))
    log.info("wrote %s and %s", out_csv, out_md)


if __name__ == "__main__":
    main()
