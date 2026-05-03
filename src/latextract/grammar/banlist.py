"""Targeted constrained decoding via a banlist of token sequences.

This is the pragmatic subset of grammar-constrained decoding for our specific
failure modes. We're NOT trying to enforce the full ~500-command CFG here;
that's the next iteration. We're forbidding the specific token sequences that
TexTeller emits as artifacts (equation-number layout commands, repetition
patterns, and identifiers our subset doesn't allow).

Each entry in BANNED_PATTERNS is a string. We tokenize it (without special
tokens) for both the bare form ("\\hskip") and a leading-space form (" \\hskip"),
because BPE often produces different token IDs depending on context. The result
is a list of token-id sequences suitable for `bad_words_ids=` on `model.generate()`.
"""
from __future__ import annotations

import logging
from typing import Sequence

log = logging.getLogger(__name__)

# Patterns we forbid in the output. Picked to surgically eliminate the failure
# modes observed on real arXiv math.DG (see data/eval/results_optimized.jsonl).
BANNED_PATTERNS: tuple[str, ...] = (
    # Equation-numbering layout artifacts that wrapped models tend to hallucinate
    "\\hskip",
    "\\hbox",
    "\\mbox",
    "\\vskip",
    "\\vspace",
    "\\hspace",
    # Empty-array wrapping is a frequent failure mode
    "\\begin{array}",
    "\\end{array}",
    # Tiny size hacks from training-data noise
    "\\tiny",
    "\\small",
    # Page-layout commands that don't belong in a math snippet
    "\\par",
    "\\noindent",
    "\\quad\\quad\\quad\\quad",  # repeated qquad-like padding
)


def _dedupe(seqs: list[list[int]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for s in seqs:
        t = tuple(s)
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(s)
    return out


def build_banned_token_ids(tokenizer, patterns: Sequence[str] = BANNED_PATTERNS) -> list[list[int]]:
    """Return a list of token-id sequences suitable for `bad_words_ids` on generate().

    Each pattern is encoded twice (with and without a leading space) because BPE
    tokenizers (RoBERTa-style) split differently in the two contexts.
    """
    out: list[list[int]] = []
    for p in patterns:
        for variant in (p, " " + p):
            ids = tokenizer(variant, add_special_tokens=False).input_ids
            if ids:
                out.append(ids)
    deduped = _dedupe(out)
    log.info("banlist: %d patterns -> %d unique token sequences", len(patterns), len(deduped))
    return deduped
