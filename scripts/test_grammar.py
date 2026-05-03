"""Validate the CFG against real arXiv math samples."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from latextract.grammar.cfg import load_grammar, validate_grammar

console = Console()
ROOT = Path(__file__).resolve().parents[1]


def _wrap_for_grammar(body: str) -> str:
    """The grammar's start production is a display_math env; wrap a bare body."""
    body = body.strip()
    if body.startswith("\\[") or body.startswith("\\begin{"):
        return body
    return f"\\[{body}\\]"


def main(manifest: str = "data/rendered/manifest.jsonl", limit: int = 0):
    pairs = [json.loads(l) for l in (ROOT / manifest).read_text().splitlines() if l.strip()]
    if limit:
        pairs = pairs[:limit]
    samples = [_wrap_for_grammar(p["body"]) for p in pairs]
    console.print(f"validating {len(samples)} samples against latex_math.lark")
    report = validate_grammar(samples)

    t = Table(title="grammar coverage")
    t.add_column("metric"); t.add_column("value")
    t.add_row("samples", str(report.total))
    t.add_row("parsed cleanly", f"{report.passed} ({report.pass_rate:.1f}%)")
    t.add_row("failed", str(report.total - report.passed))
    console.print(t)

    if report.failed_samples:
        console.print("\n[bold]Sample failures (first 10):[/bold]")
        for sample, err in report.failed_samples[:10]:
            console.print(f"  [red]✗[/red] {sample}")
            console.print(f"     [dim]{err}[/dim]")


if __name__ == "__main__":
    import typer
    typer.run(main)
