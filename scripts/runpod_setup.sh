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

echo "==> ensure torch matches the host NVIDIA driver"
# RunPod base images ship torch+cu121. Upgrading to cu124 needs driver >= 550 or
# cuDNN init dies on the first conv with CUDNN_STATUS_NOT_INITIALIZED. Pick the
# wheel index from the actual driver, and only reinstall when the running torch
# disagrees with what the driver supports.
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || echo 0)"
if [[ "${DRIVER_MAJOR:-0}" -ge 550 ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu124"
  WANTED_CUDA="12.4"
else
  TORCH_INDEX="https://download.pytorch.org/whl/cu121"
  WANTED_CUDA="12.1"
fi
CURRENT_CUDA="$(python -c 'import torch,sys; sys.stdout.write(torch.version.cuda or "")' 2>/dev/null || echo "")"
if [[ "$CURRENT_CUDA" != "$WANTED_CUDA" ]]; then
  echo "    driver=$DRIVER_MAJOR -> reinstalling torch for cuda $WANTED_CUDA (was: ${CURRENT_CUDA:-none})"
  pip install --upgrade --index-url "$TORCH_INDEX" torch==2.5.1 torchvision==0.20.1
else
  echo "    driver=$DRIVER_MAJOR matches torch+cu$CURRENT_CUDA, no reinstall"
fi

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

# torch 2.5.1 pins nvidia-cudnn-cu12==9.1.0.70, which fails to initialize
# (CUDNN_STATUS_NOT_INITIALIZED on the first conv) on Ada cards like L40S.
# Bump to a newer cuDNN; the pin warning is cosmetic — torch dlopens the .so
# from site-packages at runtime regardless of version.
echo "==> upgrade cuDNN past torch's broken pin (fixes L40S init)"
pip install --quiet --upgrade "nvidia-cudnn-cu12>=9.5"

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
