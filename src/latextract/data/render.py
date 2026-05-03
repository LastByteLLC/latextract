"""Render a single LaTeX math snippet to a tightly-cropped PNG via tectonic + MuPDF."""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz  # pymupdf
from PIL import Image

log = logging.getLogger(__name__)

TECTONIC = shutil.which("tectonic") or "/opt/homebrew/bin/tectonic"

DOC_TEMPLATE = r"""\documentclass[preview, border=4pt, varwidth]{standalone}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{mathtools}
\usepackage{bm}
\begin{document}
%s
\end{document}
"""


class RenderError(RuntimeError):
    pass


def render_latex(snippet: str, dpi: int = 200, timeout: int = 60) -> Image.Image:
    """Compile snippet (e.g. '\\[ x^2 \\]' or '\\begin{equation}...\\end{equation}') into a PIL.Image."""
    src = DOC_TEMPLATE % snippet
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tex_path = td_path / "doc.tex"
        tex_path.write_text(src, encoding="utf-8")
        try:
            result = subprocess.run(
                [TECTONIC, "-X", "compile", "--outdir", str(td_path), "--keep-logs", str(tex_path)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise RenderError(f"tectonic timeout: {e}") from e
        if result.returncode != 0:
            raise RenderError(f"tectonic failed: {result.stderr[-500:]}")
        pdf_path = td_path / "doc.pdf"
        if not pdf_path.exists():
            raise RenderError("tectonic produced no pdf")
        # Rasterize page 1 with MuPDF
        doc = fitz.open(pdf_path)
        try:
            page = doc[0]
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            mode = "RGB" if pix.n == 3 else "RGBA"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            return img.convert("RGB")
        finally:
            doc.close()
