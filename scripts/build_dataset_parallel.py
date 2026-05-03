"""Parallel data generation: pull arXiv source, extract math envs, render
formulas in parallel across CPU cores. Designed to scale to 100K-1M samples.

Compared to scripts/build_dataset.py (sequential, single-domain), this:
  - sweeps multiple arXiv categories
  - uses ProcessPoolExecutor for parallel rendering (tectonic is CPU-bound)
  - dedupes by normalized-LaTeX hash so near-duplicates don't dominate
  - is fully resumable (skips paper IDs already in manifest)
  - writes manifest entries atomically per-row so Ctrl-C never loses work

Throughput on Apple M4 (10 CPU cores):
  ~7-8 renders/sec parallel (vs ~1 sequential), ~25K-30K samples/hour.
  500K target dataset: ~16-20 hours wall-clock. Safe to run overnight.

Storage estimate at 500K samples: ~2.5 GB images + ~100 MB manifest.
"""
from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.live import Live
from rich.table import Table

from latextract.data.arxiv_fetch import download_source, search_recent
from latextract.data.extract_math import extract_envs

# Use spawn context: macOS + fitz/PIL + fork state can corrupt workers.
MP_CTX = mp.get_context("spawn")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_parallel")
console = Console()
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RENDERED = ROOT / "data" / "rendered"
RENDERED.mkdir(parents=True, exist_ok=True)

# Default category sweep — math + ML + theoretical physics. Tuneable via --categories.
DEFAULT_CATEGORIES = (
    "math.DG", "math.AT", "math.AG", "math.CO", "math.CA", "math.NT",
    "math.PR", "math.GT", "math.RT", "math.FA",
    "hep-th", "hep-ph", "gr-qc", "cond-mat.stat-mech",
    "cs.LG", "cs.AI", "cs.CL", "cs.CV", "stat.ML",
)

WS_RE = re.compile(r"\s+")


def _normalize_for_hash(latex: str) -> str:
    """Aggressive whitespace normalization for dedup."""
    return WS_RE.sub("", latex).lower()


def _render_one(args: tuple[str, str, int, str, str, str, str]) -> dict | None:
    """Worker: render one (paper_id, env, idx, body, full_latex, category, root_str) -> manifest dict.

    Deferred imports: with the 'spawn' context, the worker re-executes this
    module on import. Keep heavy deps (fitz, PIL) inside the function so we
    don't pay their import cost on every job.
    """
    import sys
    paper_id, env, i, body, full_latex, category, root_str = args
    sys.path.insert(0, str(Path(root_str) / "src"))
    from latextract.data.render import render_latex, RenderError
    try:
        img = render_latex(full_latex, dpi=200, timeout=20)
    except RenderError as e:
        return {"_error": f"RenderError: {str(e)[:200]}"}
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {str(e)[:200]}"}
    rendered_dir = Path(root_str) / "data" / "rendered"
    if img.width < 40 or img.height < 16:
        return None
    cat_slug = category.replace(".", "_").replace("-", "_")
    stem = f"{cat_slug}_{paper_id}_{i:03d}"
    img_path = rendered_dir / f"{stem}.png"
    img.save(img_path)
    return {
        "id": stem,
        "category": category,
        "paper_id": paper_id,
        "env": env,
        "body": body,
        "full_latex": full_latex,
        "image": str(img_path.relative_to(Path(root_str))),
        "width": img.width,
        "height": img.height,
    }


@dataclass
class Stats:
    fetched_papers: int = 0
    extracted_envs: int = 0
    rendered: int = 0
    skipped_dup: int = 0
    skipped_render_fail: int = 0


def _stats_table(s: Stats, target: int, elapsed: float) -> Table:
    rate = s.rendered / max(0.1, elapsed)
    eta = (target - s.rendered) / max(0.1, rate) if rate > 0 else 0
    t = Table.grid(expand=True)
    t.add_column(); t.add_column(justify="right")
    t.add_row("rendered", f"{s.rendered}/{target} ({100*s.rendered/max(1,target):.1f}%)")
    t.add_row("rate", f"{rate:.1f}/s")
    t.add_row("eta", f"{eta/60:.1f} min")
    t.add_row("papers fetched", str(s.fetched_papers))
    t.add_row("envs extracted", str(s.extracted_envs))
    t.add_row("dedup skips", str(s.skipped_dup))
    t.add_row("render fails", str(s.skipped_render_fail))
    return t


def main(
    categories: str = ",".join(DEFAULT_CATEGORIES),
    papers_per_category: int = 100,
    max_per_paper: int = 30,
    target: int = 5000,
    workers: int = 8,
    manifest: str = "data/rendered/manifest_large.jsonl",
    seen_paper_log: str = "data/rendered/seen_papers.jsonl",
):
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    out = ROOT / manifest
    seen_log = ROOT / seen_paper_log

    # Resume state
    seen_papers: set[str] = set()
    seen_hashes: set[str] = set()
    pair_count = 0
    if out.exists():
        with out.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen_papers.add(d.get("paper_id", ""))
                seen_hashes.add(_normalize_for_hash(d.get("body", "")))
                pair_count += 1
        console.print(f"resuming with {pair_count} pairs across {len(seen_papers)} papers, "
                      f"{len(seen_hashes)} unique hashes")

    manifest_fp = out.open("a")
    seen_fp = seen_log.open("a")
    stats = Stats(rendered=pair_count)
    t0 = time.time()

    with Live(_stats_table(stats, target, 1), refresh_per_second=2, console=console) as live:
        for category in cats:
            if stats.rendered >= target:
                break
            try:
                papers = search_recent(category, max_results=papers_per_category)
            except Exception as e:
                log.warning("search failed for %s: %s", category, e)
                continue

            # Build the work queue: (paper_id, env, idx, body, full_latex, category)
            jobs: list[tuple[str, str, int, str, str, str]] = []
            for p in papers:
                arxiv_id = p.get_short_id().split("v")[0]
                if arxiv_id in seen_papers:
                    continue
                paper_dir = download_source(p, RAW)
                if paper_dir is None:
                    continue
                stats.fetched_papers += 1
                seen_papers.add(arxiv_id)
                seen_fp.write(json.dumps({"paper_id": arxiv_id, "category": category}) + "\n")
                seen_fp.flush()
                specs = extract_envs(arxiv_id, paper_dir, max_per_paper=max_per_paper)
                stats.extracted_envs += len(specs)
                for i, spec in enumerate(specs):
                    h = _normalize_for_hash(spec.body)
                    if h in seen_hashes:
                        stats.skipped_dup += 1
                        continue
                    seen_hashes.add(h)
                    jobs.append((arxiv_id, spec.env, i, spec.body, spec.full_latex, category, str(ROOT)))
                live.update(_stats_table(stats, target, time.time() - t0))
                if stats.rendered + len(jobs) >= target:
                    break

            # Render the queue in parallel
            if not jobs:
                continue
            with ProcessPoolExecutor(max_workers=workers, mp_context=MP_CTX) as ex:
                futures = [ex.submit(_render_one, j) for j in jobs]
                for f in as_completed(futures):
                    try:
                        result = f.result(timeout=30)
                    except Exception:
                        stats.skipped_render_fail += 1
                        continue
                    if result is None:
                        stats.skipped_render_fail += 1
                    elif "_error" in result:
                        stats.skipped_render_fail += 1
                        log.warning("worker err: %s", result["_error"])
                    else:
                        manifest_fp.write(json.dumps(result) + "\n")
                        manifest_fp.flush()
                        stats.rendered += 1
                    live.update(_stats_table(stats, target, time.time() - t0))
                    if stats.rendered >= target:
                        # Cancel remaining (can't actually cancel running, but stop submitting)
                        break

    manifest_fp.close()
    seen_fp.close()
    elapsed = time.time() - t0
    console.rule("[bold green]done")
    console.print(f"rendered: {stats.rendered}/{target}  in {elapsed/60:.1f} min  "
                  f"(rate {stats.rendered/max(1,elapsed):.1f}/s)")
    console.print(f"manifest: {out}")


if __name__ == "__main__":
    import typer
    typer.run(main)
