"""Re-compress all rendered PNGs in place with oxipng.

oxipng tries multiple zlib filter strategies and writes the smallest result
back to the same path. For 200 DPI math formulas (mostly white background,
sparse black ink), expected savings are 20-40% with no quality loss.

Idempotent: re-running on already-optimized files is a no-op (oxipng detects
the file is already minimal and skips).

Usage:
  python scripts/compress_pngs.py                    # data/rendered/*.png
  python scripts/compress_pngs.py --dir some/other   # custom dir
  python scripts/compress_pngs.py --level max        # max optimization (slow)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console

console = Console()
ROOT = Path(__file__).resolve().parents[1]


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.glob("*.png"))


def main(dir: str = "data/rendered", level: str = "2", threads: int = 8):
    if not shutil.which("oxipng"):
        console.print("[red]oxipng not found.[/red] Install with: brew install oxipng")
        sys.exit(1)
    target = ROOT / dir
    if not target.exists():
        console.print(f"[red]no such dir:[/red] {target}")
        sys.exit(1)
    pngs = list(target.glob("*.png"))
    if not pngs:
        console.print(f"no PNGs under {target}")
        return
    console.print(f"compressing {len(pngs)} PNGs in {target} (oxipng -r -o {level}, {threads} threads)")
    before = _dir_size(target)
    t0 = time.time()
    # Use -r so we don't blow past the OS's argv limit on large dirs.
    # -o is optimization level, --strip safe drops safe-to-remove metadata.
    cmd = ["oxipng", "-r", "-o", level, "--strip", "safe", "-q", "-t", str(threads), str(target)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        console.print(f"[red]oxipng failed:[/red] {res.stderr[-500:]}")
        sys.exit(res.returncode)
    after = _dir_size(target)
    elapsed = time.time() - t0
    saved = before - after
    pct = 100.0 * saved / max(1, before)
    console.print(f"  before: {before / 1024 / 1024:.1f} MB")
    console.print(f"  after:  {after / 1024 / 1024:.1f} MB")
    console.print(f"  saved:  [green]{saved / 1024 / 1024:.1f} MB ({pct:.1f}%)[/green]  in {elapsed:.1f}s")


if __name__ == "__main__":
    import typer
    typer.run(main)
