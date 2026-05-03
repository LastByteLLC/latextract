"""Build a presentation-ready demo asset:

  1. data/eval/demo_grid.png  — montage of representative cases
  2. data/eval/demo.html      — side-by-side LaTeX + images + metrics
  3. data/eval/summary.md     — markdown summary of metrics

Picks: 3 greedy-pass cases, all retry-rescue cases (up to 3), and all failure cases.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

from latextract.data.render import render_latex, RenderError

console = Console()
ROOT = Path(__file__).resolve().parents[1]


def _font(size: int) -> ImageFont.ImageFont:
    for cand in [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ]:
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _row(input_img: Image.Image, predicted_render: Image.Image | None, label: str,
         ssim: float, accepted: bool, latex: str, gt: str) -> Image.Image:
    """Compose: [input] [arrow] [predicted render] | text panel."""
    H = 140
    pad = 14
    scale_in = H / input_img.height if input_img.height > 0 else 1
    in_resized = input_img.resize((max(1, int(input_img.width * scale_in)), H), Image.LANCZOS)
    if predicted_render is not None:
        scale_out = H / predicted_render.height if predicted_render.height > 0 else 1
        pred_resized = predicted_render.resize(
            (max(1, int(predicted_render.width * scale_out)), H), Image.LANCZOS)
    else:
        pred_resized = Image.new("RGB", (300, H), (245, 230, 230))
        d = ImageDraw.Draw(pred_resized)
        d.text((20, H // 2 - 10), "render failed", fill=(180, 0, 0), font=_font(18))

    text_w = 720
    row_w = pad + in_resized.width + pad + 30 + pad + pred_resized.width + pad + text_w + pad
    row_h = H + 2 * pad + 16
    canvas = Image.new("RGB", (row_w, row_h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    x = pad
    canvas.paste(in_resized, (x, pad + 8))
    x += in_resized.width + pad
    d.text((x, row_h // 2 - 12), "→", fill=(50, 50, 50), font=_font(28))
    x += 30 + pad
    canvas.paste(pred_resized, (x, pad + 8))
    x += pred_resized.width + pad

    label_color = (0, 130, 0) if accepted else ((180, 100, 0) if ssim and ssim > 0 else (180, 0, 0))
    d.text((x, pad), f"{label}   SSIM={ssim:.3f}", fill=label_color, font=_font(15))

    pred_short = (latex[:200] + "…") if len(latex) > 200 else latex
    gt_short = (gt[:200] + "…") if len(gt) > 200 else gt
    d.text((x, pad + 26), f"pred: {pred_short}", fill=(0, 0, 0), font=_font(12))
    d.text((x, pad + 26 + 16 + 8 + 26), f"gt:   {gt_short}", fill=(60, 60, 60), font=_font(12))

    # divider
    d.rectangle([0, row_h - 1, row_w, row_h], fill=(220, 220, 220))
    return canvas


def main():
    results = [json.loads(l) for l in (ROOT / "data/eval/results.jsonl").read_text().splitlines() if l.strip()]

    greedy_pass = [r for r in results if r["greedy_ssim"] >= 0.95][:3]
    rescued = [r for r in results if r["greedy_ssim"] < 0.95 and r["accepted"]][:4]
    failed = [r for r in results if not r["accepted"]]

    n_total = len(results)
    n_greedy = sum(1 for r in results if r["greedy_ssim"] >= 0.95)
    n_rescued = sum(1 for r in results if r["greedy_ssim"] < 0.95 and r["accepted"])
    n_accepted = sum(1 for r in results if r["accepted"])
    n_failed = n_total - n_accepted

    rows: list[Image.Image] = []
    section_titles = []

    def section_header(text: str, w: int = 1500) -> Image.Image:
        h = 36
        img = Image.new("RGB", (w, h), (240, 240, 240))
        d = ImageDraw.Draw(img)
        d.text((14, 8), text, fill=(20, 20, 20), font=_font(18))
        return img

    def add_section(title: str, items: list[dict]):
        if not items:
            return
        for r in items:
            input_img = Image.open(ROOT / "data" / "rendered" / f"{r['id']}.png").convert("RGB")
            try:
                snippet = r["best_latex"] if r["best_latex"].lstrip().startswith(("\\[", "\\begin")) else f"\\[ {r['best_latex']} \\]"
                pred_render = render_latex(snippet)
            except RenderError:
                pred_render = None
            label = title.split(" — ")[0]
            rows.append(_row(input_img, pred_render, label,
                             ssim=r["best_ssim"], accepted=r["accepted"],
                             latex=r["best_latex"], gt=r["ground_truth"]))
        section_titles.append(title)

    section_imgs = []

    def add_with_header(title: str, items: list[dict]):
        if not items:
            return
        section_imgs.append(section_header(title))
        for r in items:
            input_img = Image.open(ROOT / "data" / "rendered" / f"{r['id']}.png").convert("RGB")
            try:
                snippet = r["best_latex"] if r["best_latex"].lstrip().startswith(("\\[", "\\begin")) else f"\\[ {r['best_latex']} \\]"
                pred_render = render_latex(snippet)
            except RenderError:
                pred_render = None
            tag = "PASS (greedy)" if r["greedy_ssim"] >= 0.95 else ("RESCUED" if r["accepted"] else "FAIL")
            section_imgs.append(_row(
                input_img, pred_render, tag,
                ssim=r["best_ssim"], accepted=r["accepted"],
                latex=r["best_latex"], gt=r["ground_truth"],
            ))

    add_with_header("Greedy pass — model gets it on first try", greedy_pass)
    add_with_header("Rescued by render-verify retry — wedge in action", rescued)
    add_with_header("Still failing — work to do", failed[:3])

    # Compose final montage
    width = max(im.width for im in section_imgs)
    height = sum(im.height for im in section_imgs) + 80
    montage = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(montage)
    title = (f"latextract — TexTeller wrapped + verify-retry  |  "
             f"greedy {n_greedy}/{n_total}  rescued +{n_rescued}  "
             f"accepted {n_accepted}/{n_total} ({100*n_accepted/n_total:.1f}%)")
    d.text((14, 12), title, fill=(0, 0, 0), font=_font(22))
    d.text((14, 50), "real arXiv math.DG  •  M4 MPS  •  SSIM ≥ 0.95 = accept",
           fill=(80, 80, 80), font=_font(15))

    y = 80
    for im in section_imgs:
        montage.paste(im, (0, y))
        y += im.height

    out_png = ROOT / "data" / "eval" / "demo_grid.png"
    montage.save(out_png)
    console.print(f"wrote {out_png} ({montage.size})")

    # Markdown summary
    md = [
        "# latextract — Phase A results (no training)",
        "",
        f"- **Test set:** {n_total} display-math formulas from recent arXiv math.DG papers",
        f"- **Hardware:** Apple M4 (MPS), no fine-tuning",
        f"- **Base model:** OleehyO/TexTeller (ViT-448 + TrOCR decoder, ~120M params)",
        "",
        "## Headline",
        "",
        "| stage | count | rate |",
        "| --- | --- | --- |",
        f"| greedy SSIM ≥ 0.95 | {n_greedy}/{n_total} | {100*n_greedy/n_total:.1f}% |",
        f"| rescued by verify-retry | +{n_rescued}/{n_total} | +{100*n_rescued/n_total:.1f}% |",
        f"| **total accepted** | **{n_accepted}/{n_total}** | **{100*n_accepted/n_total:.1f}%** |",
        f"| still failing | {n_failed}/{n_total} | {100*n_failed/n_total:.1f}% |",
        "",
        "## Why this is the wedge",
        "",
        "Existing OSS math-OCR models (pix2tex, Nougat, Texify, Unimernet) are trained for accuracy",
        "but make no guarantees about output validity or visual fidelity. In our test set, the wrapped",
        f"baseline alone passes {100*n_greedy/n_total:.0f}% of formulas. Adding render-verify with beam-diverse",
        f"retry rescues another {100*n_rescued/n_total:.0f}% — formulas where the model's *first* guess wraps the right math",
        "in a spurious `\\begin{array}` or `\\hskip ...(1)` artifact, but a beam alternative gets it right.",
        "",
        "The contract: **output is guaranteed to re-render to within SSIM ≥ 0.95 of the input, or the",
        "library returns `None` with a diagnostic.** Never broken LaTeX.",
        "",
        "## Next",
        "",
        "1. Grammar-constrained decoding (Outlines) — eliminate the failure-mode entirely instead of",
        "   rescuing after the fact.",
        "2. PDF text-layer prior — when input is a PDF region, bias toward identifiers that match",
        "   the encoded glyphs (resolves v vs ν, etc).",
        "3. RunPod fine-tune (~$5–10) on a larger arXiv slice to improve greedy-pass rate.",
    ]
    md_path = ROOT / "data" / "eval" / "summary.md"
    md_path.write_text("\n".join(md))
    console.print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
