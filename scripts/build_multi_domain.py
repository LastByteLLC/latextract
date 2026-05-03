"""Pull formulas from multiple arXiv domains into separate manifests."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn)

from latextract.data.arxiv_fetch import download_source, search_recent
from latextract.data.extract_math import extract_envs
from latextract.data.render import RenderError, render_latex

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
console = Console()
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RENDERED = ROOT / "data" / "rendered"


def harvest(category: str, n_papers: int, max_per_paper: int, max_total: int) -> list[dict]:
    papers = search_recent(category, max_results=n_papers)
    pairs: list[dict] = []
    cat_slug = category.replace(".", "_").replace("-", "_")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                  console=console) as prog:
        t = prog.add_task(f"  {category}", total=len(papers))
        for p in papers:
            arxiv_id = p.get_short_id().split("v")[0]
            paper_dir = download_source(p, RAW)
            if paper_dir is None:
                prog.advance(t); continue
            specs = extract_envs(arxiv_id, paper_dir, max_per_paper=max_per_paper)
            for i, spec in enumerate(specs):
                if len(pairs) >= max_total:
                    break
                try:
                    img = render_latex(spec.full_latex, dpi=200)
                except RenderError:
                    continue
                if img.width < 40 or img.height < 16:
                    continue
                stem = f"{cat_slug}_{arxiv_id}_{i:03d}"
                img_path = RENDERED / f"{stem}.png"
                img.save(img_path)
                pairs.append({
                    "id": stem, "category": category, "paper_id": arxiv_id,
                    "env": spec.env, "body": spec.body, "full_latex": spec.full_latex,
                    "image": str(img_path.relative_to(ROOT)),
                    "width": img.width, "height": img.height,
                })
            prog.advance(t)
            if len(pairs) >= max_total:
                break
    return pairs


def main(categories: str = "math.AT,hep-th,cs.LG",
         n_papers: int = 6, max_per_paper: int = 8, max_total_per_cat: int = 30):
    cats = [c.strip() for c in categories.split(",")]
    all_pairs: list[dict] = []
    for c in cats:
        console.rule(f"[bold]{c}")
        all_pairs.extend(harvest(c, n_papers, max_per_paper, max_total_per_cat))

    out = RENDERED / "manifest_multi.jsonl"
    with out.open("w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    by_cat: dict[str, int] = {}
    for p in all_pairs:
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
    console.rule("[bold green]done")
    for c, n in by_cat.items():
        console.print(f"  {c}: {n}")
    console.print(f"manifest -> {out}")


if __name__ == "__main__":
    import typer
    typer.run(main)
