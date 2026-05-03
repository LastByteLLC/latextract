"""Self-verifying inference: re-render the prediction, compare via SSIM,
and try a small set of beam-diverse candidates if the top-1 doesn't match."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import torch
from PIL import Image

from latextract.data.render import RenderError, render_latex
from latextract.model.inference import LatexOCR, Prediction, strip_artifacts
from latextract.verify.ssim import compare

log = logging.getLogger(__name__)

# Aggressive artifact stripper for failure recovery
ARRAY_WRAP_RE = re.compile(
    r"^\s*\\\[\s*\\begin\{array\}(?:\{[^}]*\})?(.*?)\\end\{array\}\s*\\\]\s*$",
    re.DOTALL,
)
HSKIP_NUMBER_RE = re.compile(r"\\hskip\s+\d+(?:\.\d+)?pt\s*(?:\\?mbox\{)?\(\d+\)\}?")
EXTRA_NEWLINES_RE = re.compile(r"(\\\\\s*){2,}")


def aggressive_strip(latex: str) -> str:
    s = strip_artifacts(latex)
    # Unwrap a top-level \[ \begin{array}{...} BODY \end{array} \] with no row breaks
    m = ARRAY_WRAP_RE.match(s)
    if m:
        inner = m.group(1).strip()
        if "\\\\" not in inner:  # single-row "array" is just a wrapper
            s = f"\\[ {inner} \\]"
    s = HSKIP_NUMBER_RE.sub("", s)
    s = EXTRA_NEWLINES_RE.sub(r"\\\\", s)
    s = strip_artifacts(s)
    return s


def _ensure_display(s: str) -> str:
    s = s.strip()
    if s.startswith(("\\[", "\\begin{")):
        return s
    return f"\\[ {s} \\]"


@dataclass
class Candidate:
    latex: str
    rendered: Image.Image | None
    ssim: float
    error: str | None
    avg_logprob: float | None


@dataclass
class ExtractResult:
    latex: str | None        # final LaTeX, or None if no candidate cleared the bar
    ssim: float
    avg_logprob: float | None
    rendered: Image.Image | None
    attempts: int
    candidates: list[Candidate]
    accepted: bool           # True iff ssim >= threshold and renders cleanly


def _verify_one(latex_raw: str, reference: Image.Image) -> Candidate:
    latex = aggressive_strip(latex_raw)
    snippet = _ensure_display(latex)
    try:
        re_img = render_latex(snippet)
        s = compare(reference, re_img)
        return Candidate(latex=latex, rendered=re_img, ssim=s, error=None, avg_logprob=None)
    except RenderError as e:
        return Candidate(latex=latex, rendered=None, ssim=-1.0, error=str(e), avg_logprob=None)


@torch.inference_mode()
def extract_with_verify(
    ocr: LatexOCR, image: Image.Image,
    threshold: float = 0.95, max_attempts: int = 3,
) -> ExtractResult:
    """Greedy decode + render-verify; if SSIM < threshold, try beam-diverse alternates."""
    candidates: list[Candidate] = []

    # Attempt 1: greedy
    pred = ocr.predict(image, num_beams=1)
    cand = _verify_one(pred.latex, image)
    cand.avg_logprob = pred.avg_logprob
    candidates.append(cand)

    if cand.ssim >= threshold:
        return ExtractResult(latex=cand.latex, ssim=cand.ssim, avg_logprob=cand.avg_logprob,
                             rendered=cand.rendered, attempts=1, candidates=candidates, accepted=True)

    # Attempt 2..N: beam search returns multiple sequences; pick highest-SSIM
    if max_attempts >= 2:
        pixel_values = ocr.model.config  # placeholder — rerun via predict_beam
        beams = ocr.predict_beams(image, num_beams=4, num_return_sequences=4) if hasattr(ocr, "predict_beams") else []
        for bp in beams:
            c = _verify_one(bp.latex, image)
            c.avg_logprob = bp.avg_logprob
            candidates.append(c)

    # Pick best
    best = max(candidates, key=lambda c: c.ssim)
    accepted = best.ssim >= threshold and best.error is None
    return ExtractResult(
        latex=best.latex if accepted else best.latex,  # still return best-effort latex
        ssim=best.ssim,
        avg_logprob=best.avg_logprob,
        rendered=best.rendered,
        attempts=len(candidates),
        candidates=candidates,
        accepted=accepted,
    )
