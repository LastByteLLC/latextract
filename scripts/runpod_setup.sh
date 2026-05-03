#!/usr/bin/env bash
# RunPod setup: clone repo, install deps, fetch a fresh dataset, run training.
# Tested target: PyTorch 2.5+ CUDA image, A100 80GB or H100 80GB.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Tom-Barrasso/latextract.git}"
WORKDIR="${WORKDIR:-/workspace/latextract}"
N_PAPERS="${N_PAPERS:-80}"
MAX_STEPS="${MAX_STEPS:-800}"
BATCH="${BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-5e-5}"

echo "==> setup at $WORKDIR"
mkdir -p "$WORKDIR" && cd "$WORKDIR"
if [[ ! -d .git ]]; then
  git clone "$REPO_URL" .
fi

echo "==> installing tectonic (apt)"
apt-get update -qq
apt-get install -y -qq tectonic poppler-utils >/dev/null

echo "==> python deps"
pip install --quiet -e .
pip install --quiet flash-attn --no-build-isolation || echo "(flash-attn install best-effort; SDPA fallback is fine)"

echo "==> warm tectonic cache"
echo '\documentclass[preview]{standalone}\usepackage{amsmath}\begin{document}$x$\end{document}' \
  | tectonic - >/dev/null

echo "==> build dataset (${N_PAPERS} papers from math.DG)"
python scripts/build_dataset.py --n-papers "$N_PAPERS" --max-per-paper 25 --max-total 1500

echo "==> fine-tune (${MAX_STEPS} steps, batch ${BATCH}, grad-accum ${GRAD_ACCUM}, lr ${LR})"
python scripts/train.py \
  --max-steps "$MAX_STEPS" \
  --per-device-batch-size "$BATCH" \
  --grad-accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --bf16 \
  --out runs/textteller-mathdg

echo "==> eval against held-out manifest"
python scripts/run_eval.py \
  --model-id "$(pwd)/runs/textteller-mathdg/final" \
  --out-path data/eval/results_finetuned.jsonl

echo "==> done. results in data/eval/results_finetuned.jsonl"
