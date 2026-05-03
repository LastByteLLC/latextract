#!/usr/bin/env bash
# RunPod setup: clone repo, install deps, fetch a fresh dataset, run training.
# Tested target: PyTorch 2.5+ CUDA image, A100 80GB or H100 80GB.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/LastByteLLC/latextract.git}"
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

echo "==> upgrading pip"
python -m pip install --upgrade pip

echo "==> update torch to 2.5.1 (with CUDA 12.4) for best performance; skip if already up-to-date"
pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1

echo "==> installing tectonic (static binary) + poppler"
# tectonic isn't in the default Ubuntu repos on most RunPod images; grab the
# official static binary instead. Skip apt-get update so we don't stall on a
# slow/unreachable PPA (e.g. deadsnakes) baked into the base image.
apt-get install -y -qq poppler-utils >/dev/null
if ! command -v tectonic >/dev/null 2>&1; then
  TECTONIC_VERSION="${TECTONIC_VERSION:-0.15.0}"
  TECTONIC_TARBALL="tectonic-${TECTONIC_VERSION}-x86_64-unknown-linux-musl.tar.gz"
  TECTONIC_URL="https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@${TECTONIC_VERSION}/${TECTONIC_TARBALL}"
  tmp="$(mktemp -d)"
  curl --proto '=https' --tlsv1.2 -fsSL "$TECTONIC_URL" -o "$tmp/tectonic.tar.gz"
  tar -xzf "$tmp/tectonic.tar.gz" -C "$tmp"
  install -m 0755 "$tmp/tectonic" /usr/local/bin/tectonic
  rm -rf "$tmp"
fi
tectonic --version

echo "==> python deps"
pip install --quiet -e .
pip install flash-attn --no-build-isolation || echo "(flash-attn install best-effort; SDPA fallback is fine)"

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
