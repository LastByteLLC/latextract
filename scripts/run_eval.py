"""End-to-end eval with optional verify-retry. Logs per-stage metrics."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from latextract.model.inference import LatexOCR
from latextract.verify.retry import extract_with_verify

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
console = Console()
ROOT = Path(__file__).resolve().parents[1]


def main(model_id: str = "OleehyO/TexTeller", manifest: str = "data/rendered/manifest.jsonl",
         out_path: str = "data/eval/results.jsonl", limit: int = 0, threshold: float = 0.95):
    pairs = [json.loads(l) for l in (ROOT / manifest).read_text().splitlines() if l.strip()]
    if limit:
        pairs = pairs[:limit]
    console.print(f"loaded {len(pairs)} pairs")
    ocr = LatexOCR(model_id=model_id)
    console.print(f"model on {ocr.device}, image_size={ocr.image_size}, channels={ocr.num_channels}")

    out = ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("w")
    n_greedy = 0
    n_with_retry = 0
    n_accepted = 0
    total_lat = 0.0
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("greedy:{task.fields[g]}  +retry:{task.fields[r]}  accepted:{task.fields[a]}"),
                  TimeElapsedColumn(), console=console) as prog:
        t = prog.add_task("eval", total=len(pairs), g=0, r=0, a=0)
        for p in pairs:
            img = Image.open(ROOT / p["image"]).convert("RGB")
            t0 = time.time()
            res = extract_with_verify(ocr, img, threshold=threshold, max_attempts=3)
            lat_ms = (time.time() - t0) * 1000
            total_lat += lat_ms
            # greedy = first candidate
            greedy_ssim = res.candidates[0].ssim
            best_ssim = res.ssim
            if greedy_ssim >= threshold:
                n_greedy += 1
            if best_ssim >= threshold and greedy_ssim < threshold:
                n_with_retry += 1
            if res.accepted:
                n_accepted += 1
            f.write(json.dumps({
                "id": p["id"],
                "ground_truth": p["body"],
                "greedy_latex": res.candidates[0].latex,
                "greedy_ssim": greedy_ssim,
                "best_latex": res.latex,
                "best_ssim": best_ssim,
                "attempts": res.attempts,
                "accepted": res.accepted,
                "latency_ms": lat_ms,
            }) + "\n")
            prog.advance(t, advance=1)
            prog.update(t, g=n_greedy, r=n_with_retry, a=n_accepted)
    f.close()

    n = len(pairs)
    tab = Table(title=f"Eval — {model_id} — threshold {threshold}")
    tab.add_column("metric"); tab.add_column("count"); tab.add_column("pct")
    tab.add_row("greedy SSIM>=th", f"{n_greedy}/{n}", f"{100*n_greedy/n:.1f}%")
    tab.add_row("rescued by retry", f"{n_with_retry}/{n}", f"{100*n_with_retry/n:.1f}%")
    tab.add_row("total accepted", f"{n_accepted}/{n}", f"{100*n_accepted/n:.1f}%")
    tab.add_row("avg latency", f"{total_lat / max(1,n):.0f} ms", "")
    console.print(tab)
    console.print(f"results -> {out}")


if __name__ == "__main__":
    import typer
    typer.run(main)
