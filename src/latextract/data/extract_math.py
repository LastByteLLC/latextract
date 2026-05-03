"""Extract display-math environments from .tex sources."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Display-mode environments we accept. Ordered by specificity.
DISPLAY_ENVS = ("equation*", "equation", "align*", "align", "gather*", "gather", "multline*", "multline")

# Disallow formulas containing these (likely user macros, custom packages, complex layout)
BLOCKLIST_PATTERNS = (
    r"\\newcommand", r"\\renewcommand", r"\\def\b", r"\\let\b",
    r"\\input\b", r"\\include\b",
    r"\\begin\{tikzcd\}", r"\\begin\{tikzpicture\}", r"\\begin\{xy\}",
    r"\\begin\{array\}", r"\\begin\{matrix\}", r"\\begin\{pmatrix\}",  # arrays often need cell-wise rendering
)
BLOCK_RE = re.compile("|".join(BLOCKLIST_PATTERNS))

# Comment stripping (LaTeX % comments, but not \% escaped)
COMMENT_RE = re.compile(r"(?<!\\)%.*?$", re.MULTILINE)
# Strip \label{...} (noise; doesn't render)
LABEL_RE = re.compile(r"\\label\{[^}]*\}")


@dataclass
class MathSpec:
    paper_id: str
    env: str           # 'equation', '\\[', '$$', etc.
    body: str          # raw inner content
    full_latex: str    # what we'll render: e.g. \[ ... \]


def _strip_comments(s: str) -> str:
    return COMMENT_RE.sub("", s)


def _find_main_tex(paper_dir: Path) -> Path | None:
    cands = list(paper_dir.rglob("*.tex"))
    if not cands:
        return None
    # Prefer one that contains \documentclass
    for c in cands:
        try:
            if "\\documentclass" in c.read_text(encoding="utf-8", errors="ignore"):
                return c
        except Exception:
            continue
    return cands[0]


def _read_all_tex(paper_dir: Path) -> str:
    """Concatenate all .tex files. Crude but adequate for env extraction."""
    parts = []
    for tex in sorted(paper_dir.rglob("*.tex")):
        try:
            parts.append(tex.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
    return _strip_comments("\n".join(parts))


def extract_envs(paper_id: str, paper_dir: Path, max_per_paper: int = 50) -> list[MathSpec]:
    src = _read_all_tex(paper_dir)
    out: list[MathSpec] = []

    for env in DISPLAY_ENVS:
        # Match \begin{env} ... \end{env}, non-greedy, multi-line
        pat = re.compile(
            r"\\begin\{" + re.escape(env) + r"\}(.*?)\\end\{" + re.escape(env) + r"\}",
            re.DOTALL,
        )
        for m in pat.finditer(src):
            body = LABEL_RE.sub("", m.group(1)).strip()
            if not body or len(body) > 800:
                continue
            if BLOCK_RE.search(body):
                continue
            full = f"\\begin{{{env}}}{body}\\end{{{env}}}"
            out.append(MathSpec(paper_id=paper_id, env=env, body=body, full_latex=full))
            if len(out) >= max_per_paper:
                return out

    # Also: \[ ... \]
    pat = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
    for m in pat.finditer(src):
        body = LABEL_RE.sub("", m.group(1)).strip()
        if not body or len(body) > 800 or BLOCK_RE.search(body):
            continue
        full = f"\\[{body}\\]"
        out.append(MathSpec(paper_id=paper_id, env="\\[\\]", body=body, full_latex=full))
        if len(out) >= max_per_paper:
            break

    return out
