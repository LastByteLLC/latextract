"""Build a small (image, latex) dataset from arXiv math.DG."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from latextract.data.arxiv_fetch import search_recent, download_source
from latextract.data.extract_math import extract_envs
from latextract.data.render import render_latex, RenderError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build")
console = Console()

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RENDERED = ROOT / "data" / "rendered"
RENDERED.mkdir(parents=True, exist_ok=True)


def main(category: str = "math.DG", n_papers: int = 8, max_per_paper: int = 12, max_total: int = 80):
    console.rule(f"[bold]Building dataset: {category}")
    papers = search_recent(category, max_results=n_papers)
    console.print(f"got {len(papers)} paper records")

    out = ROOT / "data" / "rendered" / "manifest.jsonl"
    # Resume: skip papers already represented in the manifest, keep prior entries.
    seen_papers: set[str] = set()
    pair_count = 0
    if out.exists():
        with out.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen_papers.add(d.get("paper_id", ""))
                pair_count += 1
        console.print(f"resuming: {pair_count} existing pairs across {len(seen_papers)} papers")

    # Open in append mode so a Ctrl-C never loses already-rendered work.
    manifest_fp = out.open("a")
    try:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                      TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                      console=console) as prog:
            t = prog.add_task("papers", total=len(papers))
            for p in papers:
                arxiv_id = p.get_short_id().split("v")[0]
                if arxiv_id in seen_papers:
                    prog.advance(t); continue
                paper_dir = download_source(p, RAW)
                if paper_dir is None:
                    prog.advance(t); continue
                specs = extract_envs(arxiv_id, paper_dir, max_per_paper=max_per_paper)
                for i, spec in enumerate(specs):
                    if pair_count >= max_total:
                        break
                    try:
                        img = render_latex(spec.full_latex, dpi=200)
                    except RenderError:
                        continue
                    if img.width < 40 or img.height < 16:
                        continue
                    stem = f"{arxiv_id}_{i:03d}"
                    img_path = RENDERED / f"{stem}.png"
                    img.save(img_path)
                    manifest_fp.write(json.dumps({
                        "id": stem,
                        "paper_id": arxiv_id,
                        "env": spec.env,
                        "body": spec.body,
                        "full_latex": spec.full_latex,
                        "image": str(img_path.relative_to(ROOT)),
                        "width": img.width,
                        "height": img.height,
                    }) + "\n")
                    manifest_fp.flush()
                    pair_count += 1
                prog.advance(t)
                if pair_count >= max_total:
                    break
    finally:
        manifest_fp.close()

    console.rule(f"[bold green]Done")
    console.print(f"{pair_count} pairs in {out}")


if __name__ == "__main__":
    import typer
    typer.run(main)
