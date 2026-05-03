# RunPod first-pass plan

## Goal
Fine-tune the wrapped TexTeller baseline on a larger arXiv math.DG slice and re-run the eval, on CUDA with bf16 + (optional) FlashAttention. Budget: $5–10.

## Hardware

| pick | $/hr | why |
|---|---|---|
| **A100 80 GB PCIe** | ~$1.50 | enough VRAM for batch 8, cheapest sane choice |
| H100 80 GB | ~$3–4 | ~2× faster, only worth it for the bigger run |
| RTX 4090 24 GB | ~$0.50 | tight on VRAM, fine for batch 2 + grad-accum |

A100 80 GB is the right pick for this first pass.

## Cost estimate (A100 80 GB)

| step | wall | $ |
|---|---|---|
| spin-up + setup | 5 min | $0.13 |
| dataset gen (~1500 pairs from 80 papers) | 8 min | $0.20 |
| fine-tune 800 steps, batch 8 | 25 min | $0.63 |
| eval | 3 min | $0.08 |
| **total** | **~40 min** | **~$1.05** |

So a "first pass" is comfortably under $2.

## Launch — clone from GitHub then run setup

Prereqs on RunPod: a pod with a recent CUDA / PyTorch image (`pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime` or newer) and an attached persistent volume mounted at `/workspace` so the model weights and dataset survive pod restart.

### One-liner (zero-config, defaults baked in)

```bash
curl -fsSL https://raw.githubusercontent.com/LastByteLLC/latextract/main/scripts/runpod_setup.sh | bash
```

### Or: clone first, then customize

```bash
cd /workspace
git clone https://github.com/LastByteLLC/latextract.git
cd latextract

# defaults: 80 papers from math.DG, 800 steps, batch 8, lr 5e-5
N_PAPERS=80 MAX_STEPS=800 BATCH=8 LR=5e-5 bash scripts/runpod_setup.sh

# bigger run, more domains, longer training:
N_PAPERS=200 MAX_STEPS=1500 BATCH=16 LR=3e-5 bash scripts/runpod_setup.sh

# 24 GB card (RTX 4090): smaller batch + more accumulation
BATCH=2 GRAD_ACCUM=8 bash scripts/runpod_setup.sh
```

### What setup does (in order)

1. `apt install tectonic poppler-utils` — TeX renderer + PDF rasterizer
2. `pip install -e .` then attempt `pip install flash-attn` (best-effort; SDPA fallback is fine if it fails to build)
3. Warm tectonic's package cache with a trivial render so subsequent compiles are fast
4. Run `scripts/build_dataset.py` to pull `$N_PAPERS` papers from arXiv math.DG and render `~$N_PAPERS × 18` isolated formulas
5. Run `scripts/train.py` with bf16 + gradient checkpointing (set by the `MAX_STEPS / BATCH / GRAD_ACCUM / LR` env vars) and save to `runs/textteller-mathdg/final`
6. Re-run `scripts/run_eval.py` against the same manifest with the fine-tuned weights → `data/eval/results_finetuned.jsonl`

### Pulling the fine-tuned weights back

After training, the final checkpoint is at `runs/textteller-mathdg/final/`. Two options:

```bash
# zip and download from RunPod's web UI (volume browser)
tar -czf /workspace/textteller-ft.tar.gz -C /workspace/latextract/runs textteller-mathdg

# or push to a Hugging Face hub repo
huggingface-cli login
huggingface-cli upload <username>/textteller-mathdg-ft runs/textteller-mathdg/final
```

## What it does

1. installs tectonic + poppler + python deps
2. tries to install flash-attn (falls back to PyTorch SDPA if it fails — both are fine)
3. pulls 80 fresh math.DG papers, renders ~1000–1500 isolated formulas
4. fine-tunes TexTeller for 800 steps with bf16, saves to `runs/textteller-mathdg/final`
5. runs the eval harness against the same manifest with the fine-tuned model

## What we expect

The current wrapped baseline (no training) hits **91% accept** on the local 44-formula set. Bottlenecks observed:

- model wraps single expressions in spurious `\begin{array}` (rescued by retry today)
- model emits `\hskip ...(1)` equation-number artifacts (stripped today)
- multi-line `\begin{split}` is sometimes truncated

A fine-tune on properly-rendered display-math (no equation numbers, no array wraps) should suppress these failure modes at the source — pushing greedy-pass into the 90s and total-accept into the 95–98% range.

## Knobs

- `N_PAPERS=80` — train slice size; bump to 200 for a longer run
- `MAX_STEPS=800` — ~25 min on A100; 1500 doubles training time
- `BATCH=8` — fits in A100 80 GB; on a 24 GB card use `BATCH=2 GRAD_ACCUM=8`
- `LR=5e-5` — conservative for fine-tuning a pretrained encoder-decoder

## After it finishes

`data/eval/results_finetuned.jsonl` has per-formula greedy/best SSIM + accept flags. Run `python scripts/make_demo.py` again to regenerate `demo_grid.png` with the post-fine-tune numbers.

## Not in this first pass

- **Grammar-constrained decoding** (Outlines): the next big win, eliminates the failure class at the source instead of rescuing after the fact. Adds ~10 ms/token but bumps accept rate further.
- **PDF text-layer prior**: only relevant for `(pdf, bbox)` input mode.
- **Distillation to a smaller student** for on-device deployment (MLX export).
