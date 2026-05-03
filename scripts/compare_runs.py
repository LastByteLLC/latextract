"""Compare two or more results JSONLs side-by-side."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

console = Console()
ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def summarize(rows: list[dict], label: str) -> dict:
    n = len(rows)
    if n == 0:
        return {"label": label, "n": 0}
    n_greedy = sum(1 for r in rows if r["greedy_ssim"] >= 0.95)
    n_acc = sum(1 for r in rows if r["accepted"])
    return {
        "label": label, "n": n,
        "greedy": n_greedy, "greedy_pct": 100 * n_greedy / n,
        "accept": n_acc, "accept_pct": 100 * n_acc / n,
        "rescued": n_acc - n_greedy, "rescued_pct": 100 * (n_acc - n_greedy) / n,
        "avg_ms": mean(r["latency_ms"] for r in rows),
    }


def main():
    paths_with_labels = []
    for arg in sys.argv[1:]:
        if "=" in arg:
            label, p = arg.split("=", 1)
        else:
            p = arg
            label = Path(p).stem
        paths_with_labels.append((label, Path(p)))

    if not paths_with_labels:
        console.print("usage: compare_runs.py [LABEL=]path1.jsonl [LABEL=]path2.jsonl ...")
        sys.exit(1)

    summaries = [summarize(load(p), label) for label, p in paths_with_labels]

    t = Table(title="run comparison")
    t.add_column("run"); t.add_column("n", justify="right")
    t.add_column("greedy", justify="right"); t.add_column("rescued", justify="right")
    t.add_column("accept", justify="right"); t.add_column("avg ms", justify="right")
    for s in summaries:
        if s["n"] == 0:
            t.add_row(s["label"], "0", "—", "—", "—", "—"); continue
        t.add_row(
            s["label"], str(s["n"]),
            f"{s['greedy']} ({s['greedy_pct']:.0f}%)",
            f"+{s['rescued']} ({s['rescued_pct']:.0f}%)",
            f"{s['accept']} ({s['accept_pct']:.1f}%)",
            f"{s['avg_ms']:.0f}",
        )
    console.print(t)


if __name__ == "__main__":
    main()
