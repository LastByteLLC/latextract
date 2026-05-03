"""Pull arXiv papers (source tarballs) for a given category."""
from __future__ import annotations

import logging
import tarfile
from pathlib import Path

import arxiv
import requests

log = logging.getLogger(__name__)

UA = "latextract/0.0.1 (research; mailto:support@podlp.com)"


def search_recent(category: str, max_results: int = 20) -> list[arxiv.Result]:
    client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    return list(client.results(search))


def download_source(result: arxiv.Result, dest_root: Path) -> Path | None:
    """Download e-print tarball for a paper. Returns the unpacked dir or None on failure."""
    arxiv_id = result.get_short_id().split("v")[0]
    paper_dir = dest_root / arxiv_id
    if paper_dir.exists() and any(paper_dir.iterdir()):
        return paper_dir
    paper_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        r.raise_for_status()
    except Exception as e:
        log.warning("download failed for %s: %s", arxiv_id, e)
        return None
    tar_path = paper_dir / "source.tar.gz"
    tar_path.write_bytes(r.content)
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            tf.extractall(paper_dir, filter="data")
    except (tarfile.TarError, EOFError) as e:
        log.warning("tar extract failed for %s: %s (likely single .tex file)", arxiv_id, e)
        # arXiv sometimes serves a bare .tex; rename
        tex_dest = paper_dir / "main.tex"
        tex_dest.write_bytes(r.content)
    tar_path.unlink(missing_ok=True)
    return paper_dir
