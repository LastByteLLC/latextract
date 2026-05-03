"""Run inference on the rendered dataset; save predictions to JSONL with SSIM verify."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn

from latextract.model.inference import LatexOCR
from latextract.data.render import render_latex, RenderError
from latextract.verify.ssim import compare

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
console = Console()
ROOT = Path(__file__).resolve().parents[1]


def main(model_id: str = "OleehyO/TexTeller", manifest: str = "data/rendered/manifest.jsonl",
         out_path: str = "data/eval/preds.jsonl", limit: int = 0, verify: bool = True):
    pairs = [json.loads(l) for l in (ROOT / manifest).read_text().splitlines() if l.strip()]
    if limit:
        pairs = pairs[:limit]
    console.print(f"loaded {len(pairs)} pairs; loading model {model_id}...")
    t0 = time.time()
    ocr = LatexOCR(model_id=model_id)
    console.print(f"model loaded in {time.time() - t0:.1f}s on {ocr.device}")

    out_full = ROOT / out_path
    out_full.parent.mkdir(parents=True, exist_ok=True)
    f = out_full.open("w")
    successes = 0
    matches = 0
    total_lat_ms = 0.0

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("ssim>=.95: {task.fields[match]}/{task.completed}"),
                  TimeElapsedColumn(), console=console) as prog:
        t = prog.add_task("inferring", total=len(pairs), match=0)
        for p in pairs:
            img_path = ROOT / p["image"]
            img = Image.open(img_path).convert("RGB")
            t1 = time.time()
            try:
                pred = ocr.predict(img)
            except Exception as e:
                console.print(f"[red]predict failed for {p['id']}: {e}")
                prog.advance(t); continue
            lat_ms = (time.time() - t1) * 1000
            total_lat_ms += lat_ms
            successes += 1
            ssim_score = None
            if verify:
                snippet = pred.latex if pred.latex.lstrip().startswith(("\\begin", "\\[")) else f"\\[ {pred.latex} \\]"
                try:
                    re_img = render_latex(snippet)
                    ssim_score = compare(img, re_img)
                    if ssim_score >= 0.95:
                        matches += 1
                except RenderError:
                    ssim_score = -1.0
            f.write(json.dumps({
                "id": p["id"], "ground_truth": p["body"], "prediction": pred.latex,
                "avg_logprob": pred.avg_logprob, "ssim": ssim_score, "latency_ms": lat_ms,
            }) + "\n")
            prog.advance(t, advance=1)
            prog.update(t, match=matches)

    f.close()
    avg_lat = total_lat_ms / max(1, successes)
    console.rule("[bold]Summary")
    console.print(f"predictions: {successes}/{len(pairs)}")
    console.print(f"render-match >= 0.95 SSIM: {matches}/{successes} ({100*matches/max(1,successes):.1f}%)")
    console.print(f"avg latency: {avg_lat:.0f} ms/image")
    console.print(f"output: {out_full}")


if __name__ == "__main__":
    import typer
    typer.run(main)
