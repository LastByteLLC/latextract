"""Shrink rendered LaTeX PNGs in place: grayscale -> 16-color palette -> oxipng.

Idempotent: a palette-mode PNG is treated as already-compressed and skipped.
Parallel across CPU cores. Safe to Ctrl-C; each file is rewritten atomically
via tempfile + replace.

Typical savings on tectonic+MuPDF math renders: ~30%.

Usage:
    python scripts/compress_rendered.py                       # data/rendered/
    python scripts/compress_rendered.py --dir data/rendered --colors 16
    python scripts/compress_rendered.py --colors 8            # more aggressive
"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()
ROOT = Path(__file__).resolve().parents[1]
OXIPNG = shutil.which("oxipng")


def _compress_one(args: tuple[str, int]) -> tuple[str, int, int, str]:
    """Returns (path, orig_bytes, new_bytes, status)."""
    path_str, colors = args
    path = Path(path_str)
    orig_size = path.stat().st_size
    try:
        im = Image.open(path)
        if im.mode == "P":
            return (path_str, orig_size, orig_size, "skip-palette")
        quantized = im.convert("L").quantize(colors, dither=Image.Dither.FLOYDSTEINBERG)
    except Exception as e:
        return (path_str, orig_size, orig_size, f"err-pil:{type(e).__name__}")

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=path.parent)
    tmp.close()
    try:
        quantized.save(tmp.name, "PNG", optimize=True)
        if OXIPNG:
            r = subprocess.run(
                [OXIPNG, "-o", "4", "--strip", "safe", "-q", tmp.name],
                capture_output=True,
            )
            if r.returncode != 0:
                return (path_str, orig_size, orig_size, "err-oxipng")
        new_size = os.path.getsize(tmp.name)
        if new_size >= orig_size:
            os.unlink(tmp.name)
            return (path_str, orig_size, orig_size, "skip-larger")
        os.replace(tmp.name, path)
        return (path_str, orig_size, new_size, "ok")
    except Exception as e:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        return (path_str, orig_size, orig_size, f"err:{type(e).__name__}")


def main(
    dir: str = "data/rendered",
    colors: int = 16,
    workers: int = 0,
):
    if OXIPNG is None:
        console.print("[yellow]oxipng not on PATH — running PIL-only pass[/yellow]")
    target_dir = (ROOT / dir).resolve()
    files = sorted(str(p) for p in target_dir.glob("*.png"))
    if not files:
        console.print(f"no .png files in {target_dir}")
        return
    workers = workers or max(1, mp.cpu_count() - 1)
    console.print(f"compressing {len(files)} PNGs in {target_dir} "
                  f"(palette-{colors}, {workers} workers)")

    t0 = time.time()
    total_orig = total_new = 0
    counts = {"ok": 0, "skip-palette": 0, "skip-larger": 0, "err": 0}
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"),
                  TextColumn("{task.fields[saved]}"),
                  TimeElapsedColumn(), console=console) as prog:
        task = prog.add_task("compress", total=len(files), saved="0 MB")
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futures = [ex.submit(_compress_one, (f, colors)) for f in files]
            for fut in as_completed(futures):
                _, orig, new, status = fut.result()
                total_orig += orig
                total_new += new
                if status == "ok":
                    counts["ok"] += 1
                elif status.startswith("skip-"):
                    counts[status] = counts.get(status, 0) + 1
                else:
                    counts["err"] += 1
                saved_mb = (total_orig - total_new) / 1e6
                prog.update(task, advance=1, saved=f"saved {saved_mb:.1f} MB")

    elapsed = time.time() - t0
    pct = 100 * (1 - total_new / max(1, total_orig))
    console.rule("[bold green]done")
    console.print(f"files: {counts}")
    console.print(f"size: {total_orig/1e6:.1f} MB -> {total_new/1e6:.1f} MB  "
                  f"({pct:.1f}% reduction)  in {elapsed:.0f}s")


if __name__ == "__main__":
    import typer
    typer.run(main)
