# latextract

Deterministic, self-verifying formula-to-LaTeX library. Wraps an off-the-shelf math-OCR model with grammar-constrained decoding + render-and-SSIM verification, so every accepted output is guaranteed to re-render to within SSIM ≥ 0.95 of the input — never broken LaTeX.

## Quick start

```bash
git clone https://github.com/LastByteLLC/latextract.git
cd latextract
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .
brew install tectonic                        # macOS; apt on Linux

# extract one formula image
python -m latextract.cli extract path/to/formula.png

# build a small eval set + run the harness
python scripts/build_dataset.py --n-papers 12
python scripts/run_eval.py
```

See [`data/eval/summary.md`](data/eval/summary.md) for current results, [`RUNPOD.md`](RUNPOD.md) for cloud fine-tuning, [`MLX.md`](MLX.md) for Apple-Silicon native conversion.
