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
display-math environment, normalized to remove `\label{...}`.

## Provenance

- **Source**: arXiv `e-print/<id>` tarballs (publicly downloadable)
- **Renderer**: tectonic (LaTeX engine) + MuPDF (PDF → PNG @ 200 DPI)
- **Filtering**: papers using `\newcommand`, `\def`, `tikz`, or `array` envs are
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
python scripts/publish_hf_dataset.py --manifest data/rendered/manifest_large.jsonl \
                                     --push --repo <org>/<name>
```

See https://github.com/LastByteLLC/latextract for the full pipeline.
