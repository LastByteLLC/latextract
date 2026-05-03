"""latextract CLI."""
from __future__ import annotations

import sys
from pathlib import Path

import typer
from PIL import Image
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help="Deterministic, self-verifying formula-to-LaTeX extractor")
console = Console()


@app.command()
def extract(
    image: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    verify: bool = typer.Option(True, help="Re-render predicted LaTeX and verify via SSIM"),
    threshold: float = typer.Option(0.95, help="SSIM acceptance threshold"),
    model: str = typer.Option("OleehyO/TexTeller", help="HF model id"),
    show_candidates: bool = typer.Option(False, help="Print all retry candidates"),
):
    """Extract LaTeX from a formula image with optional render-verify retry."""
    from latextract.model.inference import LatexOCR
    from latextract.verify.retry import extract_with_verify

    img = Image.open(image).convert("RGB")
    ocr = LatexOCR(model_id=model)

    if verify:
        res = extract_with_verify(ocr, img, threshold=threshold, max_attempts=3)
    else:
        pred = ocr.predict(img)
        res = type("X", (), {})()
        res.latex = pred.latex; res.ssim = float("nan"); res.avg_logprob = pred.avg_logprob
        res.attempts = 1; res.accepted = True; res.candidates = [pred]

    t = Table(title="latextract", show_header=False)
    t.add_row("LaTeX", res.latex if res.latex else "[red]<no candidate accepted>[/red]")
    if hasattr(res, "ssim"):
        t.add_row("SSIM (render-match)", f"{res.ssim:.3f}")
    if getattr(res, "avg_logprob", None) is not None:
        t.add_row("Avg log-prob", f"{res.avg_logprob:.3f}")
    t.add_row("Attempts", str(getattr(res, "attempts", 1)))
    t.add_row("Accepted", "[green]yes[/green]" if getattr(res, "accepted", True) else "[red]no[/red]")
    console.print(t)

    if show_candidates and hasattr(res, "candidates"):
        for i, c in enumerate(res.candidates):
            console.print(f"  candidate {i}: ssim={c.ssim:.3f}  latex={c.latex[:120]}")

    if not getattr(res, "accepted", True):
        sys.exit(2)


@app.command()
def eval_set(
    manifest: Path = typer.Option(Path("data/rendered/manifest.jsonl"), help="JSONL of pairs"),
    out: Path = typer.Option(Path("data/eval/results.jsonl"), help="Where to write results"),
    threshold: float = typer.Option(0.95),
    model: str = typer.Option("OleehyO/TexTeller"),
    limit: int = typer.Option(0, help="Limit samples (0 = all)"),
):
    """Run the eval harness over a manifest of (image, latex) pairs."""
    import importlib
    sys.argv = ["run_eval", "--model-id", model, "--manifest", str(manifest),
                "--out-path", str(out), "--threshold", str(threshold), "--limit", str(limit)]
    runner = importlib.import_module("scripts.run_eval")  # noqa: F401
    # The above import is for clarity in source layout; the simpler path is to
    # call typer.run on the module's main directly:
    from scripts.run_eval import main as run_main
    typer.run(run_main)


if __name__ == "__main__":
    app()
