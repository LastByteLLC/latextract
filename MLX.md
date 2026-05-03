# MLX-native conversion

`scripts/convert_to_mlx.py` converts an HF `VisionEncoderDecoderModel` to MLX-native format. It works for any HF model whose state-dict follows the standard ViT + Bart-derived-decoder convention (TexTeller, Nougat, Pix2Tex, TrOCR, …).

## Run

```bash
python scripts/convert_to_mlx.py OleehyO/TexTeller --out runs/textteller-mlx
```

## What it does

1. **Tries `mlx_vlm.utils.convert` first.** This succeeds only for the modern VLM architectures mlx-vlm supports (LLaVA, Qwen-VL, Pixtral, …). Math-OCR ViT+TrOCR models are not on the list as of mlx-vlm 0.5, so this step normally falls through.
2. **Direct weight conversion.** Reads HF safetensors, renames PyTorch keys to MLX naming via `KEY_MAP` (top of the script), saves as `weights.safetensors` in fp16.
3. **Audit log.** `rename_audit.json` records every rename so you can verify nothing was dropped or misnamed.
4. **Architecture stub.** `model.py` is generated from the HF config — correct module hierarchy, correct dimensions, `__call__` bodies left as `NotImplementedError` for hand-finishing.
5. **Tokenizer + config** copied unchanged so the MLX side can use the same vocabulary.

Output for TexTeller (~120M params):
- `weights.safetensors` — 597 MB fp16
- `model.py` — 110-line skeleton (ViT block, ViT encoder, TrOCR block, TrOCR decoder, top-level VisionEncoderDecoder)
- `rename_audit.json` — every key renamed
- `config.json`, `tokenizer.json`, `vocab.json` — verbatim from HF

## What's still required

The script automates the **deterministic** part (weight conversion). What's left is the **forward-pass implementation** — about 300 lines of MLX code:

- `ViTBlock.__call__` — pre-norm self-attention + GELU MLP (≈40 lines)
- `ViTEncoder.__call__` — patchify + pos-embed + N blocks + final layernorm (≈25 lines)
- `TrOCRBlock.__call__` — self-attention + cross-attention to encoder + MLP, with KV-cache plumbing (≈80 lines)
- `TrOCRDecoder.__call__` — embed + N blocks + lm_head, with causal mask + cross-attention to encoder (≈40 lines)
- `VisionEncoderDecoder.generate` — autoregressive loop, KV cache, optional beam search (≈80 lines)

References for porting:
- [`mlx-examples/llms/llama`](https://github.com/ml-explore/mlx-examples/tree/main/llms/llama) — KV-cached autoregressive decoder pattern
- [`mlx-examples/clip`](https://github.com/ml-explore/mlx-examples/tree/main/clip) — ViT encoder pattern (closest analog to our ViT block)
- HF source for `transformers/models/vit/modeling_vit.py` and `transformers/models/trocr/modeling_trocr.py` — exact reference for what each parameter does

The renamed weights load straight into the stub modules with `mx.load(weights_path)` + `tree_unflatten` once the forward passes are implemented.

## Why bother

| backend | latency on M4 (estimate) |
|---|---|
| PyTorch MPS (today) | 4500 ms / image |
| MLX-native (projected) | **800–1500 ms / image** |
| MLX-native + 4-bit quant | **400–800 ms / image** |

MLX has Apple-Silicon-specific kernels (Metal Performance Shaders Graph) and shared-memory architecture that PyTorch MPS doesn't fully exploit. Real-world speedup ranges from 3× to 5× for transformer-style workloads on M-series chips.

## Forward-compat note

`KEY_MAP` is the only architecture-specific bit. To support a new model:

- Different decoder family (T5, LLaMA, GPT-style)? Add their key patterns to `KEY_MAP`.
- Different encoder (Swin, BEiT, ConvNeXt)? Same — append patterns. The script will warn if the encoder/decoder type isn't in `KNOWN_ENCODER_TYPES` / `KNOWN_DECODER_TYPES`, but it'll still attempt the conversion.

If you hit an unhandled key, look at `rename_audit.json`'s `unchanged_count` — keys that didn't match any rename remain under their original names. Either add the rename or accept the original name (and adjust `model.py` to match).
