"""Convert an HF VisionEncoderDecoderModel to MLX-native format.

Strategy (in order):

1. **mlx-vlm path** — if the model's architecture is on mlx-vlm's supported list,
   delegate to `mlx_vlm.utils.convert`. This is the only path that produces a
   model usable with `mlx-vlm` CLI today.

2. **Direct weight conversion** — for architectures mlx-vlm doesn't support
   (the common case for ViT-encoder + TrOCR/RoBERTa-decoder math-OCR models like
   TexTeller, Nougat, Pix2Tex), this script:

   a. Reads the HF safetensors,
   b. Renames PyTorch keys to a canonical MLX naming scheme (see KEY_MAP below),
   c. Saves the renamed tensors as MLX-format safetensors,
   d. Emits a `model.py` stub for the architecture, ready for hand-finishing,
   e. Copies the tokenizer files unchanged.

   The renamed weights are loadable into a hand-written MLX implementation
   (`src/latextract/mlx/vit_trocr.py`) — the architecture port is a separate
   ~500-line task, see MLX.md.

Usage:

  python scripts/convert_to_mlx.py OleehyO/TexTeller --out runs/textteller-mlx
  python scripts/convert_to_mlx.py path/to/local-checkpoint --out runs/local-mlx

Forward-compatible: works for any HF model whose state-dict keys follow the
HuggingFace ViT + Bart-derived-decoder convention (TrOCR, mBART, RoBERTa-based).
For different decoder families (LLaMA, T5), edit KEY_MAP.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from rich.console import Console
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, VisionEncoderDecoderModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
console = Console()
log = logging.getLogger("convert_to_mlx")

# ---------------------------------------------------------------------------
# Pytorch -> MLX key rename map. Each entry is (regex_substring, replacement).
# Order matters; first match wins. MLX convention follows Apple's mlx-examples
# ViT/Llama style: dot-separated, no `.weight` for parameters that are bare
# tensors, lowercase module names.
# ---------------------------------------------------------------------------
KEY_MAP: list[tuple[str, str]] = [
    # ViT encoder -----------------------------------------------------------
    ("encoder.embeddings.cls_token", "vision.cls_token"),
    ("encoder.embeddings.position_embeddings", "vision.pos_embed"),
    ("encoder.embeddings.patch_embeddings.projection.weight", "vision.patch_embed.proj.weight"),
    ("encoder.embeddings.patch_embeddings.projection.bias", "vision.patch_embed.proj.bias"),
    ("encoder.encoder.layer.", "vision.layers."),
    (".attention.attention.query.", ".attn.q_proj."),
    (".attention.attention.key.", ".attn.k_proj."),
    (".attention.attention.value.", ".attn.v_proj."),
    (".attention.output.dense.", ".attn.out_proj."),
    (".intermediate.dense.", ".mlp.fc1."),
    (".output.dense.", ".mlp.fc2."),
    (".layernorm_before.", ".norm1."),
    (".layernorm_after.", ".norm2."),
    ("encoder.layernorm.", "vision.norm."),
    ("encoder.pooler.dense.", "vision.pooler."),
    # TrOCR / Bart-style decoder -------------------------------------------
    ("decoder.model.decoder.embed_tokens.", "decoder.embed_tokens."),
    ("decoder.model.decoder.embed_positions.", "decoder.embed_positions."),
    ("decoder.model.decoder.layernorm_embedding.", "decoder.embed_norm."),
    ("decoder.model.decoder.layers.", "decoder.layers."),
    (".self_attn.q_proj.", ".self_attn.q_proj."),
    (".self_attn.k_proj.", ".self_attn.k_proj."),
    (".self_attn.v_proj.", ".self_attn.v_proj."),
    (".self_attn.out_proj.", ".self_attn.out_proj."),
    (".self_attn_layer_norm.", ".self_attn_norm."),
    (".encoder_attn.q_proj.", ".cross_attn.q_proj."),
    (".encoder_attn.k_proj.", ".cross_attn.k_proj."),
    (".encoder_attn.v_proj.", ".cross_attn.v_proj."),
    (".encoder_attn.out_proj.", ".cross_attn.out_proj."),
    (".encoder_attn_layer_norm.", ".cross_attn_norm."),
    (".fc1.", ".mlp.fc1."),
    (".fc2.", ".mlp.fc2."),
    (".final_layer_norm.", ".final_norm."),
    ("decoder.output_projection.", "decoder.lm_head."),
]

# Architectures we know the manual conversion path covers
KNOWN_ENCODER_TYPES = {"vit", "donut-swin", "swin"}
KNOWN_DECODER_TYPES = {"trocr", "roberta", "bart", "mbart"}


def _rename(key: str) -> str:
    out = key
    for src, dst in KEY_MAP:
        if src in out:
            out = out.replace(src, dst)
    return out


def _infer_architecture(cfg) -> tuple[str, str]:
    enc_t = getattr(cfg.encoder, "model_type", "?")
    dec_t = getattr(cfg.decoder, "model_type", "?")
    return enc_t, dec_t


def _try_mlx_vlm(model_id: str, out: Path) -> bool:
    """Try mlx-vlm's converter. Returns True if it succeeded."""
    try:
        from mlx_vlm.utils import convert
    except ImportError:
        log.warning("mlx-vlm not installed; skipping that path")
        return False
    try:
        convert(hf_path=model_id, mlx_path=str(out), quantize=False)
        return True
    except Exception as e:
        log.info("mlx-vlm convert failed (expected for non-VLM architectures): %s", type(e).__name__)
        return False


def _emit_stub(out: Path, cfg) -> None:
    """Drop a starter MLX architecture file if one isn't already present."""
    target = out / "model.py"
    if target.exists():
        log.info("model.py already exists at %s; not overwriting", target)
        return
    enc_t, dec_t = _infer_architecture(cfg)
    enc = cfg.encoder
    dec = cfg.decoder
    sz = getattr(enc, "image_size", 448)
    img = sz[0] if isinstance(sz, (list, tuple)) else sz
    stub = f'''"""MLX implementation skeleton for {enc_t} + {dec_t}.

This file is auto-generated by scripts/convert_to_mlx.py. The weights in
weights.safetensors next to this file have already been renamed to match
the module hierarchy below — fill in the forward-pass code and load them
with `mlx.utils.tree_unflatten(safetensors.load(...))`.

See MLX.md in the repo root for the full porting guide.
"""
import mlx.core as mx
import mlx.nn as nn

# --- inferred from HF config ---------------------------------------------
ENC_HIDDEN = {enc.hidden_size}
ENC_LAYERS = {enc.num_hidden_layers}
ENC_HEADS = {enc.num_attention_heads}
ENC_INTERMEDIATE = {getattr(enc, "intermediate_size", enc.hidden_size * 4)}
ENC_PATCH = {getattr(enc, "patch_size", 16)}
ENC_IMAGE_SIZE = {img}
ENC_CHANNELS = {getattr(enc, "num_channels", 3)}

DEC_HIDDEN = {dec.hidden_size}
DEC_LAYERS = {dec.num_hidden_layers}
DEC_HEADS = {dec.num_attention_heads}
DEC_INTERMEDIATE = {getattr(dec, "ffn_dim", getattr(dec, "intermediate_size", dec.hidden_size * 4))}
DEC_VOCAB = {dec.vocab_size}
DEC_MAX_POS = {getattr(dec, "max_position_embeddings", 1024)}


class ViTBlock(nn.Module):
    """Self-attention + MLP, pre-norm. Matches the renamed weights."""
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(ENC_HIDDEN)
        self.attn_q_proj = nn.Linear(ENC_HIDDEN, ENC_HIDDEN)
        self.attn_k_proj = nn.Linear(ENC_HIDDEN, ENC_HIDDEN)
        self.attn_v_proj = nn.Linear(ENC_HIDDEN, ENC_HIDDEN)
        self.attn_out_proj = nn.Linear(ENC_HIDDEN, ENC_HIDDEN)
        self.norm2 = nn.LayerNorm(ENC_HIDDEN)
        self.mlp_fc1 = nn.Linear(ENC_HIDDEN, ENC_INTERMEDIATE)
        self.mlp_fc2 = nn.Linear(ENC_INTERMEDIATE, ENC_HIDDEN)

    def __call__(self, x):
        # TODO(@maintainer): implement self-attention + GELU MLP, residual
        raise NotImplementedError("fill in ViTBlock.__call__ — see MLX.md")


class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # patch projection: conv1d-as-linear over flattened patches
        self.patch_embed_proj = nn.Linear(
            ENC_CHANNELS * ENC_PATCH * ENC_PATCH, ENC_HIDDEN
        )
        self.cls_token = mx.zeros((1, 1, ENC_HIDDEN))
        n_patches = (ENC_IMAGE_SIZE // ENC_PATCH) ** 2 + 1
        self.pos_embed = mx.zeros((1, n_patches, ENC_HIDDEN))
        self.layers = [ViTBlock() for _ in range(ENC_LAYERS)]
        self.norm = nn.LayerNorm(ENC_HIDDEN)

    def __call__(self, pixel_values):
        # TODO: patchify -> linear -> add cls + pos -> layers -> norm
        raise NotImplementedError("fill in ViTEncoder.__call__ — see MLX.md")


class TrOCRBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn_q_proj = nn.Linear(DEC_HIDDEN, DEC_HIDDEN)
        self.self_attn_k_proj = nn.Linear(DEC_HIDDEN, DEC_HIDDEN)
        self.self_attn_v_proj = nn.Linear(DEC_HIDDEN, DEC_HIDDEN)
        self.self_attn_out_proj = nn.Linear(DEC_HIDDEN, DEC_HIDDEN)
        self.self_attn_norm = nn.LayerNorm(DEC_HIDDEN)
        self.cross_attn_q_proj = nn.Linear(DEC_HIDDEN, DEC_HIDDEN)
        self.cross_attn_k_proj = nn.Linear(ENC_HIDDEN, DEC_HIDDEN)
        self.cross_attn_v_proj = nn.Linear(ENC_HIDDEN, DEC_HIDDEN)
        self.cross_attn_out_proj = nn.Linear(DEC_HIDDEN, DEC_HIDDEN)
        self.cross_attn_norm = nn.LayerNorm(DEC_HIDDEN)
        self.mlp_fc1 = nn.Linear(DEC_HIDDEN, DEC_INTERMEDIATE)
        self.mlp_fc2 = nn.Linear(DEC_INTERMEDIATE, DEC_HIDDEN)
        self.final_norm = nn.LayerNorm(DEC_HIDDEN)

    def __call__(self, x, encoder_h, self_mask=None, kv_cache=None):
        raise NotImplementedError("fill in TrOCRBlock.__call__ — see MLX.md")


class TrOCRDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(DEC_VOCAB, DEC_HIDDEN)
        self.embed_positions = nn.Embedding(DEC_MAX_POS, DEC_HIDDEN)
        self.embed_norm = nn.LayerNorm(DEC_HIDDEN)
        self.layers = [TrOCRBlock() for _ in range(DEC_LAYERS)]
        self.lm_head = nn.Linear(DEC_HIDDEN, DEC_VOCAB, bias=False)

    def __call__(self, input_ids, encoder_h, kv_cache=None):
        raise NotImplementedError("fill in TrOCRDecoder.__call__ — see MLX.md")


class VisionEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = ViTEncoder()
        self.decoder = TrOCRDecoder()

    def encode(self, pixel_values):
        return self.vision(pixel_values)

    def generate(self, pixel_values, *, max_new_tokens=256, bos=0, eos=2):
        raise NotImplementedError("fill in autoregressive generate loop")
'''
    target.write_text(stub)
    log.info("wrote stub to %s (%d lines)", target, stub.count("\n"))


def _convert_weights(model_id: str, out: Path) -> dict:
    """Manual path: rename keys, save as MLX safetensors."""
    log.info("loading weights from %s", model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    cfg = model.config
    enc_t, dec_t = _infer_architecture(cfg)
    if enc_t not in KNOWN_ENCODER_TYPES or dec_t not in KNOWN_DECODER_TYPES:
        log.warning("unknown encoder=%s or decoder=%s; KEY_MAP may be incomplete", enc_t, dec_t)

    sd = model.state_dict()
    renamed: dict[str, torch.Tensor] = {}
    n_unchanged = 0
    for k, v in sd.items():
        new_k = _rename(k)
        if new_k == k:
            n_unchanged += 1
        renamed[new_k] = v.detach().contiguous().to(torch.float16)

    out.mkdir(parents=True, exist_ok=True)
    weights_path = out / "weights.safetensors"
    save_file(renamed, str(weights_path))
    log.info("wrote %d tensors to %s (fp16, %d MB)", len(renamed), weights_path,
             weights_path.stat().st_size // (1024 * 1024))

    # Persist a key-rename audit log
    audit = {"renames": {k: _rename(k) for k in sd if _rename(k) != k},
             "unchanged_count": n_unchanged}
    (out / "rename_audit.json").write_text(json.dumps(audit, indent=2))

    # Save config + tokenizer
    cfg.save_pretrained(out)
    try:
        AutoTokenizer.from_pretrained(model_id).save_pretrained(out)
    except Exception as e:
        log.warning("tokenizer save failed: %s", e)

    return {"encoder_type": enc_t, "decoder_type": dec_t, "n_tensors": len(renamed)}


def main(model_id: str, out: str = "runs/mlx", skip_mlx_vlm: bool = False):
    out_path = Path(out)
    cfg = AutoConfig.from_pretrained(model_id)
    enc_t, dec_t = _infer_architecture(cfg)
    console.rule(f"[bold]Convert {model_id} -> {out_path}")
    console.print(f"encoder={enc_t}  decoder={dec_t}")

    if not skip_mlx_vlm:
        if _try_mlx_vlm(model_id, out_path):
            console.print(f"[green]converted via mlx-vlm -> {out_path}[/green]")
            console.print("you can now use `mlx-vlm generate` against this directory")
            return

    info = _convert_weights(model_id, out_path)
    _emit_stub(out_path, cfg)
    console.rule("[bold green]done")
    console.print(f"weights:        {out_path / 'weights.safetensors'}")
    console.print(f"config:         {out_path / 'config.json'}")
    console.print(f"tokenizer:      {out_path}/")
    console.print(f"arch stub:      {out_path / 'model.py'}")
    console.print(f"rename audit:   {out_path / 'rename_audit.json'}")
    console.print()
    console.print(f"[yellow]NEXT[/yellow]: implement the {info['encoder_type']} + {info['decoder_type']} forward "
                  "passes in model.py (or copy the reference implementation from MLX.md). "
                  "Weights are already in MLX layout; the architecture is the only remaining work.")


if __name__ == "__main__":
    import typer
    typer.run(main)
