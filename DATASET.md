# Dataset generation

How to generate a large (image, LaTeX) corpus locally on Apple Silicon and publish it as an HF dataset that anyone (including a RunPod fine-tune) can pull.

## TL;DR

```bash
# 1. generate ~50K formulas across 19 arXiv categories (~10–14 hr on M4, fully resumable)
python scripts/build_dataset_parallel.py --target 50000 --workers 8

# 2. package + (optionally) push to the Hub
huggingface-cli login
python scripts/publish_hf_dataset.py \
    --manifest data/rendered/manifest_large.jsonl \
    --out data/hf_dataset \
    --push --repo <org>/<name>
```

## Throughput math (Apple M4, 10 CPU cores)

`tectonic` is single-threaded but each render is independent, so we parallelize at the process level. Measured throughput:

| workers | renders/sec | per hour | per overnight (10h) |
|---|---|---|---|
| 1 (sequential) | ~1.0 | ~3,600 | ~36,000 |
| 4 | ~3.5 | ~12,600 | ~126,000 |
| 8 (default) | **~7–8** | ~25,000–28,000 | ~250,000+ |
| 12 (oversubscribed) | ~8 | (same; CPU-bound) | (same) |

**Realistic targets:**
- 10K samples — 25 minutes
- 50K samples — 2 hours
- 250K samples — overnight
- 1M samples — multi-day; consider RunPod CPU instances ($0.05/hr per core class)

## Why parallel matters

The sequential `scripts/build_dataset.py` is fine for ≤200 samples. Beyond that, parallel is ~7× faster on M4 and the only realistic path to 100K+ corpora.

The parallel script is **fully resumable**: it appends to the manifest one row at a time and skips paper IDs already represented, so a Ctrl-C never loses work and you can run it in chunks.

## Storage budget

- ~5 KB per PNG (compressed, tightly cropped) — varies; long multi-line formulas can hit 30 KB
- ~250 B per manifest entry
- 50K samples ≈ **300 MB total**
- 500K samples ≈ **2.5 GB total**

Stays comfortable on a laptop SSD. For 1M+, plan an external drive or stream straight to the HF Hub via parquet shards.

## Domain coverage

`build_dataset_parallel.py` defaults to 19 categories spanning math, theoretical physics, and ML:

```
math.DG  math.AT  math.AG  math.CO  math.CA  math.NT
math.PR  math.GT  math.RT  math.FA
hep-th   hep-ph   gr-qc    cond-mat.stat-mech
cs.LG    cs.AI    cs.CL    cs.CV    stat.ML
```

Override with `--categories "math.AG,math.NT"` to focus a specialist corpus.

## Quality controls

The pipeline applies these filters in order:

1. **Source filtering** (`extract_math.py`): skip formulas containing `\newcommand`, `\def`, `\input`, `tikz`, or `array` envs. These either expand unpredictably or aren't easily renderable in isolation.
2. **Length filter**: drop formulas whose body exceeds 800 characters (the long tail is mostly multi-page derivations the model shouldn't try to learn end-to-end).
3. **Render verification**: if tectonic fails or times out (20s), drop the sample.
4. **Image-size filter**: drop renders narrower than 40 px or shorter than 16 px (almost always a malformed env).
5. **Dedup by normalized-LaTeX hash**: aggressive whitespace strip + lowercase. Each unique formula appears at most once.

The `seen_papers.jsonl` log records every paper we've fetched, so re-runs don't re-download arXiv tarballs.

## Publishing to HF

`scripts/publish_hf_dataset.py`:

1. Reads the manifest, attaches images via `datasets.Image()` feature
2. Saves as parquet shards under `data/hf_dataset/`
3. Writes a dataset card (`README.md`) with provenance, licensing, schema
4. Optionally `push_to_hub(...)` — needs `huggingface-cli login` first

The dataset card warns about per-paper arXiv licensing — for commercial use, filter `paper_id` against arXiv's metadata to retain only CC-BY/CC-BY-SA/CC0 papers.

## Using the published dataset for RunPod training

Once published, `scripts/train.py` can load it directly:

```python
from datasets import load_dataset
ds = load_dataset("LastByteLLC/latextract-arxiv-math", split="train")
```

Or via the Hub from the RunPod setup script:

```bash
HF_DATASET=LastByteLLC/latextract-arxiv-math bash scripts/runpod_setup.sh
```

(small follow-up: wire `HF_DATASET` into `runpod_setup.sh` to use this path instead of regenerating; trivial 5-line addition to `train.py` to switch between local manifest and HF dataset.)

## Roadmap

1. **First public dataset** (this PR): 19 categories × ~5K papers each via the e-print API, target 50K verified pairs. Suitable for a math-OCR fine-tune; not large enough to train from scratch.
2. **Bulk arXiv via S3 mirror**: arXiv hosts the full source corpus on the `arxiv-dataset` S3 bucket. Rewrite `arxiv_fetch.py` to pull from there for 100× the throughput and no rate limits. Target: 500K–2M pairs.
3. **Augmentation pipeline**: after rendering, layer 4–6 fonts (CM, LM, STIX, Times, Helvetica), DPI sweep 72–300, mild blur/JPEG noise. Multiplies the effective dataset by ~10× without re-fetching arXiv.
4. **Synthetic addendum**: generate formulas from a CFG (the same `latex_math.lark` we use for grammar-constrained decoding) to fill in coverage gaps the harvested corpus underrepresents.
