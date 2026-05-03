"""Package a manifest + images into a Hugging Face Dataset and (optionally)
push to the Hub.

The HF dataset schema:
  - id (string)             — globally unique
  - category (string)       — arXiv category
  - paper_id (string)       — arXiv id (no version)
  - env (string)            — display-math env name
  - body (string)           — raw inner LaTeX
  - full_latex (string)     — wrapped form actually rendered
  - image (Image)           — PNG, embedded
  - width, height (int)     — pixel dimensions

Usage:

  # build the parquet shards locally first (fast, no upload)
  python scripts/publish_hf_dataset.py --manifest data/rendered/manifest_large.jsonl \\
                                       --out data/hf_dataset/

  # then push to the hub (requires `huggingface-cli login`)
  python scripts/publish_hf_dataset.py --manifest data/rendered/manifest_large.jsonl \\
                                       --out data/hf_dataset/ \\
                                       --push --repo LastByteLLC/latextract-arxiv-math
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("publish_hf")
console = Console()
ROOT = Path(__file__).resolve().parents[1]


DATASET_CARD = """\
---
license: cc-by-4.0
task_categories:
  - image-to-text
  - image-feature-extraction
language:
  - en
tags:
  - latex
  - mathematics
  - ocr
  - arxiv
size_categories:
  - 10K<n<100K
---

# latextract-arxiv-math

A multi-domain corpus of (formula image, LaTeX source) pairs harvested from
arXiv source tarballs. Each formula is rendered in isolation with `tectonic`
into a tightly-cropped PNG; the ground-truth LaTeX is the body of the original
display-math environment, normalized to remove `\\label{...}`.

## Provenance

- **Source**: arXiv `e-print/<id>` tarballs (publicly downloadable)
- **Renderer**: tectonic (LaTeX engine) + MuPDF (PDF → PNG @ 200 DPI)
- **Filtering**: papers using `\\newcommand`, `\\def`, `tikz`, or `array` envs are
  skipped to keep the LaTeX subset learnable. Formulas longer than 800 chars
  are dropped.
- **Dedup**: aggressive whitespace-normalized hash. Each unique formula appears
  at most once.

## Schema

| field | type | description |
| --- | --- | --- |
| id | string | unique sample id |
| category | string | arXiv category (math.DG, hep-th, cs.LG, ...) |
| paper_id | string | arXiv paper id (no version suffix) |
| env | string | original display-math env name |
| body | string | raw LaTeX inside the env |
| full_latex | string | wrapped form actually rendered |
| image | Image | rendered PNG |
| width, height | int | pixel dimensions |

## Intended use

- Training/fine-tuning math-OCR models (Pix2Tex, Nougat, TexTeller, ...)
- Benchmarking grammar-constrained decoding
- Robustness studies (font/DPI variation can be added at training time)

## Licensing

Each arXiv source tarball is governed by its individual license. Most arXiv
papers are submitted under permissive arXiv-default terms that allow research
use; a minority use Creative Commons or stricter licenses. **For commercial
use**, filter by `paper_id` against arXiv's metadata to retain only CC-BY /
CC-BY-SA / CC0 papers.

## Generation

Rebuilt with:

```bash
python scripts/build_dataset_parallel.py --target 50000 --workers 8
python scripts/publish_hf_dataset.py --manifest data/rendered/manifest_large.jsonl \\
                                     --push --repo <org>/<name>
```

See https://github.com/LastByteLLC/latextract for the full pipeline.
"""


def main(
    manifest: str,
    out: str = "data/hf_dataset",
    push: bool = False,
    repo: str = "",
    private: bool = False,
):
    try:
        from datasets import Dataset, Features, Image, Value
    except ImportError as e:
        raise RuntimeError("datasets not installed: uv pip install datasets") from e

    manifest_path = ROOT / manifest
    out_path = ROOT / out
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    with manifest_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            d["image"] = str(ROOT / d["image"])  # absolute path so HF Image feature loads it
            rows.append(d)
    console.print(f"loaded {len(rows)} rows from {manifest_path}")

    features = Features({
        "id": Value("string"),
        "category": Value("string"),
        "paper_id": Value("string"),
        "env": Value("string"),
        "body": Value("string"),
        "full_latex": Value("string"),
        "image": Image(),
        "width": Value("int32"),
        "height": Value("int32"),
    })

    ds = Dataset.from_list(rows, features=features)
    console.print(f"built dataset: {ds}")

    # Save parquet shards locally
    ds.save_to_disk(str(out_path))
    (out_path / "README.md").write_text(DATASET_CARD)
    console.print(f"saved to {out_path}")

    if push:
        if not repo:
            raise SystemExit("--push requires --repo <org>/<name>")
        console.print(f"pushing to {repo} (private={private})...")
        ds.push_to_hub(repo, private=private)
        # Push the README separately so the dataset card renders on the hub
        try:
            from huggingface_hub import HfApi
            HfApi().upload_file(
                path_or_fileobj=str(out_path / "README.md"),
                path_in_repo="README.md",
                repo_id=repo,
                repo_type="dataset",
            )
        except Exception as e:
            log.warning("README upload failed: %s", e)
        console.print(f"[green]pushed to https://huggingface.co/datasets/{repo}[/green]")


if __name__ == "__main__":
    import typer
    typer.run(main)
