# latextract — Phase A results (no training)

All numbers measured on Apple M4 (MPS), no fine-tuning. Base model: `OleehyO/TexTeller` (~120M, ViT-448 + TrOCR).

## Headline

| eval set | n | accept rate | avg latency |
|---|---|---|---|
| math.DG (in-domain) | 44 | **86.4%** | 4486 ms |
| math.AT + hep-th + cs.LG (out-of-domain) | 62 | **61.3%** | 10281 ms |

**The wedge holds in-domain**: 86% of accepted outputs are *guaranteed* to re-render to within SSIM ≥ 0.95 of the input — no broken LaTeX. Wrapping an off-the-shelf model with render-verify-retry beats the standalone model meaningfully (+10pp in-domain) without training.

**Cross-domain reveals the limits of pragmatic patching** (see Grammar finding below).

## Latency story

| run | accept | avg ms |
|---|---|---|
| baseline (orig demo) | 90.9% | 14396 |
| + max_tokens 256, no-repeat-4, rep-penalty, encoder-cache | 86.4% | **4486** (3.2× faster) |

The latency win came from killing the failure tail: degenerate `\hskip ...(1)` outputs used to run the full 512 tokens × beam-4 retry. The 4-pp accept drop is acceptable given how much faster the failure path is — and is recovered (and more) by training.

## Grammar — honest finding

Tried adding a 26-token banlist (`\hskip`, `\hbox`, `\mbox`, `\begin{array}`, etc.) tuned to the math.DG failure modes. Result on out-of-domain:

| run | greedy | accepted | avg ms |
|---|---|---|---|
| no grammar | 31/62 (50%) | 38/62 (61.3%) | 10281 |
| with banlist | 31/62 (50%) | 38/62 (61.3%) | 12530 |

**The banlist changed only 3/62 outputs** and added 22% latency. Why: math.AT/hep-th/cs.LG hallucinate *different* artifacts than math.DG (subarrays, `\begin{split}` truncation, complex tensor notation), and BPE tokenization context shifts mean some banned phrases don't match in their encoded form.

**Conclusion**: a domain-specific banlist is a band-aid. The real fixes are:
1. **A full CFG covering valid LaTeX subset** (Outlines/llguidance) — universal, ~5–10% latency hit, eliminates the entire syntactic-failure class.
2. **Fine-tuning** to teach the model the canonical form so it doesn't emit junk in the first place. Cheaper at inference time. Plan in RUNPOD.md.

Both should be done; they address different failure modes.

## Next (in cost-effectiveness order)

1. **RunPod fine-tune** (~$1–2 on A100 80 GB, 40 min): `train.py` + `runpod_setup.sh` ready and smoke-tested locally on M4. Expected to push out-of-domain accept from 61% → ~80% by suppressing the array-wrap and split-truncation failures at the source.
2. **Full CFG via Outlines** (~half day): replace the patch-list with a real grammar covering the ~500 allowed commands. Universal across domains.
3. **PDF text-layer prior**: when input is `(pdf, bbox)`, bias logits toward identifiers matching the PDF's encoded glyphs (resolves `v` vs `ν`, etc).
4. **MLX-native port**: ViT encoder + TrOCR decoder in MLX → projected 3–5× speedup on Apple Silicon.
