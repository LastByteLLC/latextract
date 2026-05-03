#!/usr/bin/env bash
# Refresh demo: comparison table + updated summary.md + multi-domain grid.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "== comparison =="
python scripts/compare_runs.py \
  "math.DG (no grammar, optimized)=data/eval/results_optimized.jsonl" \
  "multi-domain (no grammar)=data/eval/results_multi.jsonl" \
  "multi-domain (with grammar)=data/eval/results_multi_grammar.jsonl"

echo
echo "== regenerating math.DG grid =="
python scripts/make_demo.py --results-path data/eval/results_optimized.jsonl

echo
echo "== summary.md update =="
python - <<'PY'
import json
from pathlib import Path
from statistics import mean

ROOT = Path(".")
def load(p): return [json.loads(l) for l in (ROOT / p).read_text().splitlines() if l.strip()]
def stats(rows):
    n = len(rows)
    if n == 0: return None
    g = sum(1 for r in rows if r["greedy_ssim"] >= 0.95)
    a = sum(1 for r in rows if r["accepted"])
    return dict(n=n, g=g, a=a, gp=100*g/n, ap=100*a/n, ms=mean(r["latency_ms"] for r in rows))

mdg = stats(load("data/eval/results_optimized.jsonl"))
mlt0 = stats(load("data/eval/results_multi.jsonl"))
mltg = stats(load("data/eval/results_multi_grammar.jsonl")) if Path("data/eval/results_multi_grammar.jsonl").exists() else None

lines = [
    "# latextract — Phase A results (no training)",
    "",
    "All numbers measured on Apple M4 (MPS), no fine-tuning. Base model: OleehyO/TexTeller (~120M, ViT-448 + TrOCR).",
    "",
    "## In-domain (math.DG)",
    "",
    "| stage | count | rate |",
    "| --- | --- | --- |",
    f"| greedy SSIM ≥ 0.95 | {mdg['g']}/{mdg['n']} | {mdg['gp']:.1f}% |",
    f"| **total accepted (incl. retry)** | **{mdg['a']}/{mdg['n']}** | **{mdg['ap']:.1f}%** |",
    f"| avg latency | {mdg['ms']:.0f} ms | |",
    "",
    "## Out-of-domain — math.AT, hep-th, cs.LG combined (62 formulas)",
    "",
    "| run | greedy | accepted | avg ms |",
    "| --- | --- | --- | --- |",
    f"| no grammar | {mlt0['g']}/{mlt0['n']} ({mlt0['gp']:.1f}%) | {mlt0['a']}/{mlt0['n']} ({mlt0['ap']:.1f}%) | {mlt0['ms']:.0f} |",
]
if mltg:
    lines.append(f"| **with banlist grammar** | **{mltg['g']}/{mltg['n']} ({mltg['gp']:.1f}%)** | **{mltg['a']}/{mltg['n']} ({mltg['ap']:.1f}%)** | {mltg['ms']:.0f} |")
    delta_g = mltg['gp'] - mlt0['gp']
    delta_a = mltg['ap'] - mlt0['ap']
    lines += [
        "",
        f"**Grammar delta**: greedy {delta_g:+.1f} pp, accept {delta_a:+.1f} pp on out-of-domain.",
    ]

lines += [
    "",
    "## Notes",
    "",
    "- The wedge: every accepted output is guaranteed to re-render to within SSIM ≥ 0.95 of the input.",
    "- Failure mode beaten by retry: model wraps single expressions in spurious `\\begin{array}` (rescued by beam-diverse alternates).",
    "- Failure mode beaten by banlist grammar: TexTeller's `\\hskip 142.26378pt(1)` equation-number artifact and friends are forbidden at decode time.",
    "- Failure mode beaten by neither (yet): max-tokens truncation on multi-line `\\begin{split}` envs — needs a small fine-tune (RUNPOD.md).",
    "",
    "## Next",
    "",
    "1. **RunPod fine-tune** (~$1–2 on A100 80 GB): train.py + runpod_setup.sh ready, smoke-tested locally.",
    "2. **Full CFG** (Outlines): replace banlist with a real ~500-command grammar — bigger guarantee surface, slight latency hit.",
    "3. **PDF text-layer prior**: when input is `(pdf, bbox)`, bias logits toward identifiers matching the PDF's encoded glyphs.",
    "4. **MLX-native port**: ViT encoder + TrOCR decoder in MLX → expected 3–5× speedup on Apple Silicon.",
]
Path("data/eval/summary.md").write_text("\n".join(lines))
print("wrote data/eval/summary.md")
PY

echo
echo "done."
