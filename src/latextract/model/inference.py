"""Inference wrapper. Phase A: wrap an existing HF math-OCR model.

Default model: OleehyO/TexTeller (ViT-448 encoder + TrOCR decoder, Apache-2.0).
Swap via env var LATEXTRACT_MODEL or constructor arg.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("LATEXTRACT_MODEL", "OleehyO/TexTeller")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# TexTeller emits trailing equation-number artifacts: a run of \qquad/\quad followed by (N).
TRAILING_NUMBER_RE = re.compile(r"(?:\s*\\(?:q?quad|,|;|:|!|\s))+\s*\(\d+\)\s*\\?\]?\s*$")
DOLLAR_WRAP_RE = re.compile(r"^\s*\\\[\s*(.*?)\s*\\\]\s*$", re.DOTALL)


def strip_artifacts(latex: str) -> str:
    s = latex.strip()
    # Drop the trailing eq-number block before any closing \]
    if s.endswith("\\]"):
        inner = s[:-2].rstrip()
        inner = TRAILING_NUMBER_RE.sub("", inner).rstrip()
        s = inner + "\\]" if inner else s
    else:
        s = TRAILING_NUMBER_RE.sub("", s).rstrip()
    return s


@dataclass
class Prediction:
    latex: str
    tokens: list[int]
    avg_logprob: float | None = None


def _preprocess_vit(image: Image.Image, target_size: int, num_channels: int = 3) -> torch.Tensor:
    """Resize + normalize for a ViT encoder. Letterbox-pad to square (white).
    For num_channels=1, returns a grayscale tensor with mean=0.5/std=0.5 normalization.
    """
    img = image.convert("L" if num_channels == 1 else "RGB")
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    bg = 255 if num_channels == 1 else (255, 255, 255)
    canvas = Image.new(img.mode, (target_size, target_size), bg)
    canvas.paste(img, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    if num_channels == 1:
        arr = (arr - 0.5) / 0.5
        arr = arr[None, ...]  # CHW with C=1
    else:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(arr).unsqueeze(0)


class LatexOCR:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None,
                 use_grammar: bool = True):
        self.model_id = model_id
        self.device = device or ("mps" if torch.backends.mps.is_available() else
                                  ("cuda" if torch.cuda.is_available() else "cpu"))
        log.info("loading %s on %s (grammar=%s)", model_id, self.device, use_grammar)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device).eval()
        enc_cfg = self.model.config.encoder
        sz = getattr(enc_cfg, "image_size", 448)
        self.image_size = int(sz[0] if isinstance(sz, (list, tuple)) else sz)
        self.num_channels = int(getattr(enc_cfg, "num_channels", 3))
        self.use_grammar = use_grammar
        self.bad_words_ids: list[list[int]] | None = None
        if use_grammar:
            from latextract.grammar import build_banned_token_ids
            self.bad_words_ids = build_banned_token_ids(self.tokenizer)

    def _encode(self, image: Image.Image) -> torch.Tensor:
        """Run the encoder once and return the encoder hidden states (cacheable)."""
        pixel_values = _preprocess_vit(image, self.image_size, self.num_channels).to(self.device)
        with torch.inference_mode():
            enc_out = self.model.encoder(pixel_values=pixel_values, return_dict=True)
        return enc_out.last_hidden_state

    @torch.inference_mode()
    def predict(self, image: Image.Image, max_new_tokens: int = 256, num_beams: int = 1,
                encoder_hidden_states: torch.Tensor | None = None) -> Prediction:
        if encoder_hidden_states is None:
            encoder_hidden_states = self._encode(image)
        from transformers.modeling_outputs import BaseModelOutput
        enc_kwargs = dict(encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states))
        out = self.model.generate(
            **enc_kwargs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            no_repeat_ngram_size=4,
            repetition_penalty=1.15,
            bad_words_ids=self.bad_words_ids,
            return_dict_in_generate=True,
            output_scores=True,
        )
        seq = out.sequences[0]
        latex = self.tokenizer.decode(seq, skip_special_tokens=True)
        latex = strip_artifacts(latex)
        avg = None
        if out.scores:
            logp = []
            scored_tokens = seq[-len(out.scores):]
            for step, token_id in zip(out.scores, scored_tokens):
                lp = torch.log_softmax(step[0], dim=-1)
                logp.append(float(lp[token_id]))
            avg = sum(logp) / max(1, len(logp))
        return Prediction(latex=latex.strip(), tokens=seq.tolist(), avg_logprob=avg)

    @torch.inference_mode()
    def predict_beams(self, image: Image.Image, max_new_tokens: int = 256,
                      num_beams: int = 3, num_return_sequences: int = 3,
                      encoder_hidden_states: torch.Tensor | None = None) -> list[Prediction]:
        """Return up to N beam-diverse predictions sorted by sequence-score (best first)."""
        if encoder_hidden_states is None:
            encoder_hidden_states = self._encode(image)
        from transformers.modeling_outputs import BaseModelOutput
        n_ret = min(num_return_sequences, num_beams)
        out = self.model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=n_ret,
            do_sample=False,
            no_repeat_ngram_size=4,
            repetition_penalty=1.15,
            bad_words_ids=self.bad_words_ids,
            return_dict_in_generate=True,
            output_scores=True,
        )
        scores = out.sequences_scores.tolist() if hasattr(out, "sequences_scores") and out.sequences_scores is not None else [None] * n_ret
        preds: list[Prediction] = []
        for i in range(n_ret):
            seq = out.sequences[i]
            latex = self.tokenizer.decode(seq, skip_special_tokens=True)
            latex = strip_artifacts(latex)
            preds.append(Prediction(latex=latex.strip(), tokens=seq.tolist(), avg_logprob=scores[i]))
        return preds
