# Quantization profile

Run on Apple M4 (CPU, optimum-onnxruntime), 20 samples from
`data/rendered/manifest.jsonl`. Greedy decode, max 256 new tokens, no beams.

## Headline

| variant | total MB | mean ms | p95 ms | grammar % | banlist-clean | int4-vs-int8 agree |
|---|---|---|---|---|---|---|
| **int8** (dynamic) | 951.0 | 1023 | 1842 | 55.0 | 15/20 | — |
| **int4** (MatMulNBits, block 32) | **256.0** | 1197 | 1835 | 50.0 | **17/20** | 10/20 |
| int2 (MatMulNBits, block 32) | 182.2 | — | — | — | — | DNF — Python kernel too slow to benchmark; not deployable in ORT Web anyway |

## Decision: ship `int4`

**INT4 wins decisively.**

- **Size**: 256 MB total (encoder 57 MB + decoder_merged 199 MB) vs INT8's 951 MB. INT8 doesn't fit any reasonable browser-cold-load budget. INT4 does.
- **Speed**: 17 % slower per sample on CPU than INT8 (1197 ms vs 1023 ms). On WebGPU in the browser the relative gap will narrow further; both are dominated by autoregressive decode steps, not weight-decompression.
- **Quality**: indistinguishable on this eval — INT4 has 5 pp lower Lark-grammar pass-rate but **2 pp better banlist-clean rate**. They agree on exactly half the samples. Where they disagree, neither is uniformly better; INT8 introduces its own quantization errors (e.g. emitted `X.\Gamma` instead of `X,\Gamma` on sample 2604.27990_003). Sample size 20 is too small to declare a winner on quality alone, but no signal of a meaningful gap.
- **Browser deployability**: INT4 via the MatMulNBits operator is supported in ORT Web with both the WebGPU and WASM backends. INT2 is not — there's no INT2 MatMul kernel in ORT Web, so INT2 is a Python-only data point.

## Why INT8 dynamic barely shrinks the decoder

`onnxruntime.quantization.quantize_dynamic` only quantizes a fixed set of op
types — primarily MatMul, Gemm, Conv, EmbedLayerNormalization. The TexTeller
decoder is heavy in attention/layernorm reshapes and embedding tables that
stay FP32 under dynamic quant, leaving the merged decoder at 867 MB out of
the original 909 MB. That's why INT8 is the worst-of-both-worlds here —
**the size win INT8 normally provides isn't there** for this model.

A static-INT8 calibration pass with `quantize_static` would do better, but
calibrating on a representative set is roughly the same effort as just
running INT4 weight-only quant — and INT4 already fits the browser budget.

## Why exact-match is 0 % for both

TexTeller emits equivalent but not character-identical LaTeX (e.g. `\sin x` vs
`\sin(x)`, swapped script orderings, alternative spacing). Exact-match against
the arXiv-source ground-truth body is the wrong metric for a math-OCR; the
project's primary metric is the render-and-SSIM verifier in
`src/latextract/verify/`. The browser demo's analog is "KaTeX renders cleanly
and Lark grammar parses" — that's the contract worth measuring, not exact
strings.

## Sample outputs (first three disagreements)

```
2604.27990_003
  int4: \[\tau_{\min}^{(X,\Gamma)}=\min\{\tau^{(X,\Gamma)}(v)\text{ with }v\in T^{1}X|_{\Gamma}^{*}\}\]
  int8: \[\tau^{(X,\Gamma)}_{\min}=\min\{\tau^{(X.\Gamma)}(v)\text{ with }v\in T^{1}X|^{*}_{\Gamma}\}\]
                                              ^^ INT8 typo: . instead of ,

2604.27990_005
  int4: \[\begin{split}\lim_{T\to\infty}\frac{1}{T}|\{t\in[0,T]\text{ with }\rho^{X}_{x}(t)\text{ based on }...
        ^^ INT4 emits banlist \begin{split}; grammar would mask
  int8: \[\lim_{T\to\infty}\frac{1}{T}|\{t\in[0,T]\text{ with }\rho^{X}_{v}(t)\text{ based on }\gamma\}|=\frac...

2604.27950_001
  int4 == int8: \[[N_{1},J_{1}]=\mu J_{2}{-}\nu J_{3},\quad[N_{2},J_{1}]{+}2N_{3}=-\mu J_{1}{+}\eta J_{3},...
```

## Conclusion

Cut INT8 and INT2 from the deploy pipeline. Ship INT4 as the single browser
artifact (~256 MB, well within service-worker cache budget). Keep the INT2
profile-only path in `quantize_onnx.py` for future reference — it's behind
a `--variants` flag, never run in CI by default.
