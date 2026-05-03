"""Smoke: render a fixed formula, save to data/eval/smoke.png, and round-trip SSIM with itself."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latextract.data.render import render_latex
from latextract.verify.ssim import compare

snippet = r"\[ \int_0^\infty e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2} \]"
img = render_latex(snippet)
out = Path(__file__).resolve().parents[1] / "data" / "eval" / "smoke.png"
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out)
print(f"saved {out} size={img.size}")
print(f"self-SSIM={compare(img, img):.4f}")
