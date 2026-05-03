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

## Launch (one command)

```bash
# on the RunPod pod, after attaching a volume:
curl -fsSL https://raw.githubusercontent.com/Tom-Barrasso/latextract/main/scripts/runpod_setup.sh | bash
```

…or clone first and edit env vars:

```bash
git clone https://github.com/Tom-Barrasso/latextract.git && cd latextract
N_PAPERS=80 MAX_STEPS=800 BATCH=8 LR=5e-5 bash scripts/runpod_setup.sh
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
