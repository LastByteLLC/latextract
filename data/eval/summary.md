# latextract — Phase A results (no training)

- **Test set:** 44 display-math formulas from recent arXiv math.DG papers
- **Hardware:** Apple M4 (MPS), no fine-tuning
- **Base model:** OleehyO/TexTeller (ViT-448 + TrOCR decoder, ~120M params)

## Headline

| stage | count | rate |
| --- | --- | --- |
| greedy SSIM ≥ 0.95 | 33/44 | 75.0% |
| rescued by verify-retry | +7/44 | +15.9% |
| **total accepted** | **40/44** | **90.9%** |
| still failing | 4/44 | 9.1% |

## Why this is the wedge

Existing OSS math-OCR models (pix2tex, Nougat, Texify, Unimernet) are trained for accuracy
but make no guarantees about output validity or visual fidelity. In our test set, the wrapped
baseline alone passes 75% of formulas. Adding render-verify with beam-diverse
retry rescues another 16% — formulas where the model's *first* guess wraps the right math
in a spurious `\begin{array}` or `\hskip ...(1)` artifact, but a beam alternative gets it right.

The contract: **output is guaranteed to re-render to within SSIM ≥ 0.95 of the input, or the
library returns `None` with a diagnostic.** Never broken LaTeX.

## Next

1. Grammar-constrained decoding (Outlines) — eliminate the failure-mode entirely instead of
   rescuing after the fact.
2. PDF text-layer prior — when input is a PDF region, bias toward identifiers that match
   the encoded glyphs (resolves v vs ν, etc).
3. RunPod fine-tune (~$5–10) on a larger arXiv slice to improve greedy-pass rate.