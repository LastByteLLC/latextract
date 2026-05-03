"""CFG-based constrained decoding via llguidance.

Status: STARTER. Tests pass on the smoke formula and a handful of arXiv math.DG
samples; the grammar (latex_math.lark) covers display math + common math
notation but has known gaps documented in the file.

Public surface:
  - load_grammar() -> str: returns the Lark grammar source
  - validate_grammar(samples) -> ValidationReport: try parsing samples
  - build_cfg_logits_processor(tokenizer) -> LogitsProcessor: HF-compatible
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

GRAMMAR_FILE = Path(__file__).with_name("latex_math.lark")


def load_grammar() -> str:
    return GRAMMAR_FILE.read_text(encoding="utf-8")


@dataclass
class ValidationReport:
    total: int
    passed: int
    failed_samples: list[tuple[str, str]] = field(default_factory=list)  # (sample, error)

    @property
    def pass_rate(self) -> float:
        return 100.0 * self.passed / max(1, self.total)


def validate_grammar(samples: Iterable[str], grammar: str | None = None) -> ValidationReport:
    """Parse each sample with Lark; report what fails. Use this to grow the
    grammar against real corpus instead of guessing."""
    try:
        from lark import Lark
        from lark.exceptions import LarkError
    except ImportError as e:
        raise RuntimeError("lark is required for grammar validation: uv pip install lark") from e

    g = grammar if grammar is not None else load_grammar()
    parser = Lark(g, start="start", parser="earley", lexer="dynamic")
    samples = list(samples)
    failed: list[tuple[str, str]] = []
    passed = 0
    for s in samples:
        try:
            parser.parse(s)
            passed += 1
        except LarkError as e:
            failed.append((s[:120], str(e).splitlines()[0][:160]))
    return ValidationReport(total=len(samples), passed=passed, failed_samples=failed[:25])


def build_cfg_logits_processor(tokenizer, grammar: str | None = None):
    """Build an HF-compatible LogitsProcessor that enforces the LaTeX-math CFG.

    Uses llguidance under the hood. The returned object can be appended to
    `model.generate(logits_processor=[...])`.

    Note: this requires the tokenizer's vocabulary to round-trip with the
    grammar's terminals. For RoBERTa-BPE tokenizers (TexTeller, BART), the
    `Ġ` space-marker bytes need to map cleanly. llguidance handles this when
    you supply the tokenizer correctly.
    """
    try:
        import llguidance as llg
    except ImportError as e:
        raise RuntimeError("llguidance is required: uv pip install llguidance") from e

    g = grammar if grammar is not None else load_grammar()

    # Compile Lark -> llguidance grammar IR
    compiled = llg.LarkCompiler(g).compile()

    # Wrap the HF tokenizer for llguidance
    tw = llg.TokenizerWrapper(tokenizer)
    ll_tok = llg.LLTokenizer(tw)
    interp = llg.LLInterpreter(ll_tok, compiled, enable_backtrack=False, enable_ff_tokens=False)

    return _LLGuidanceLogitsProcessor(interp, ll_tok)


class _LLGuidanceLogitsProcessor:
    """Adapter: llguidance interpreter -> HF LogitsProcessor protocol.

    HF calls processor(input_ids, scores). We feed the most recently generated
    token to the interpreter, then mask scores by the allowed-token set.
    """

    def __init__(self, interpreter, ll_tokenizer):
        self.interp = interpreter
        self.ll_tok = ll_tokenizer
        self._last_seen_len = 0

    def __call__(self, input_ids, scores):
        import torch
        # input_ids: (batch, seq). For batch>1 we'd need separate interpreters;
        # the starter only handles batch=1.
        if input_ids.shape[0] != 1:
            log.warning("CFG processor only supports batch=1 in the starter; passing through")
            return scores

        seq = input_ids[0].tolist()
        # Catch the interpreter up to the latest token(s)
        for tok in seq[self._last_seen_len:]:
            try:
                self.interp.consume_token(int(tok))
            except Exception as e:
                log.debug("interpreter rejected token %d: %s", tok, e)
        self._last_seen_len = len(seq)

        # Get the allowed-token mask for the next step
        try:
            mask = self.interp.compute_mask()
        except Exception as e:
            log.warning("compute_mask failed: %s; passing through", e)
            return scores

        # mask is a list/array of allowed token ids
        allowed = torch.zeros_like(scores, dtype=torch.bool)
        allowed[0, list(mask)] = True
        scores = scores.masked_fill(~allowed, float("-inf"))
        return scores
