"""SSIM-based render verification."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


@dataclass
class VerifyResult:
    ssim: float
    matched: bool
    rendered: Image.Image | None
    error: str | None = None


def _to_gray_array(img: Image.Image, target_h: int = 256) -> np.ndarray:
    g = img.convert("L")
    w, h = g.size
    if h != target_h:
        new_w = max(1, round(w * target_h / h))
        g = g.resize((new_w, target_h), Image.LANCZOS)
    return np.asarray(g, dtype=np.float32) / 255.0


def _pad_to_same(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])

    def pad(x):
        ph, pw = h - x.shape[0], w - x.shape[1]
        if ph or pw:
            x = np.pad(x, ((0, ph), (0, pw)), constant_values=1.0)
        return x

    return pad(a), pad(b)


def compare(reference: Image.Image, candidate: Image.Image) -> float:
    a = _to_gray_array(reference)
    b = _to_gray_array(candidate)
    a, b = _pad_to_same(a, b)
    win = min(7, a.shape[0] - 1 if a.shape[0] % 2 else a.shape[0] - 2, a.shape[1] - 1 if a.shape[1] % 2 else a.shape[1] - 2)
    win = max(3, win | 1)  # odd, >=3
    return float(ssim(a, b, data_range=1.0, win_size=win))
