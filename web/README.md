# latextract — web demo

Browser-only LaTeX OCR. Wraps `OleehyO/TexTeller` in `@huggingface/transformers`
v3 with two extras:

- **Incremental grammar-constrained decoding** (Option B from the planning doc).
  At each decode step, the top-K candidate next tokens are filtered through a
  prefix-validity check. **Currently this is a heuristic** (brace balance +
  banlist on the running text) because `lark-js` only emits LALR parsers and
  our grammar is Earley-only — see "Grammar status" below. The Python side
  uses the full Lark CFG.
- **KaTeX render-verify** as the browser-side analog of the tectonic+SSIM
  contract: an output is "valid" iff the grammar parses it and KaTeX renders it
  without throwing.

## Architecture

```
image upload
  ↓ (canvas: letterbox 448×448, ImageNet normalize)
encoder_model.onnx (ViT)                    [ONNX Runtime Web — WebGPU or WASM]
  ↓ (encoder_hidden_states)
decoder_model_merged.onnx (TrOCR)           [autoregressive, KV-cached]
  ↓ logits → GrammarConstrainedLogitsProcessor (top-K Lark check + banlist mask)
  ↓
LaTeX string
  ↓ stripArtifacts (regex)
  ↓ KaTeX render → HTML preview
```

## Local development

```bash
# 1. quantize the FP32 ONNX from HF (requires onnx + onnxruntime + onnx_ir)
# INT4 is the only deployed variant — see ../runs/quant_profile.md.
# Pass --variants int8,int2 for additional experiments.
cd ..
python scripts/quantize_onnx.py --variants int4 --out web/public/models

# 2. install + serve
cd web
npm install
npm run dev
```

## Grammar status

The browser-side logits processor enforces grammar constraints via a
**heuristic prefix check** (brace balance + banlist), not the full Lark CFG.
This is a known gap. The Python `latex_math.lark` grammar is Earley-only
(rules `function` and `misc_command` overlap on `\Re`, `\Im`, etc., yielding
20+ reduce/reduce collisions in any LALR parser). `lark-js`, the natural
Lark→JS path, only emits LALR.

To unlock full CFG enforcement in the browser, the realistic options are:
1. **Port to nearley.js** — a JS Earley parser. Manual translation of
   `latex_math.lark` to `.ne` format, ~half day.
2. **Run Python Lark via Pyodide** — heavier (~10 MB Pyodide bundle) but
   no grammar work. Probably overkill for this use case.
3. **Refactor the grammar to be LALR-friendly** — risky, would change the
   Python-side accuracy that already works.

Until one of those lands, the heuristic catches the dominant failure modes
(unbalanced braces, banlist commands like `\hskip`/`\begin{array}`) which
covers the majority of model-quantization-induced errors observed in the
profile run. Real LaTeX produced by TexTeller is mostly grammar-valid; the
prefix check is a guard against the tail, not the bulk.

## Profiling quantizations

```bash
python scripts/profile_quant.py --models web/public/models --limit 30
# → runs/quant_profile.{md,csv}
```

Reports for each variant:
- on-disk size (encoder + decoder)
- cold-load time
- mean per-image latency on CPU
- grammar-pass % (Lark parse on output)
- exact-match % vs reference

**Note:** `int2` (~ternary) is included in the profile for accuracy reference
but is not deployable: ONNX Runtime Web has no INT2 MatMul kernel, and on the
Python side the kernel is ~10× slower than INT4 with no quality benefit. True
ternary in the browser would need BitNet-style training, not post-training
quantization. `int8` is also profile-only here because dynamic INT8 barely
shrinks this specific model's decoder (951 MB) — see `runs/quant_profile.md`.

## Deployment (GitHub Pages)

Two workflows in `.github/workflows/`:

1. **`quantize-models.yml`** — manually triggered. Downloads FP32 ONNX from HF,
   quantizes, uploads to a workflow artifact (and optionally to a HF Hub repo
   if `HF_TOKEN` secret + `HF_QUANT_REPO` variable are configured).

2. **`deploy-web.yml`** — runs on push to `main`. Pulls the latest quantized
   artifact, compiles the Lark grammar, builds the Vite bundle, deploys to
   Pages. If no quantized artifact exists yet, deploys without models and the
   UI shows a friendly "models not yet quantized" state.

### Repo-level setup (one-time)

1. Settings → Pages → Source = "GitHub Actions"
2. Run the **Quantize models** workflow manually (default: `int4`,
   ~5 min on a `ubuntu-latest` runner).
3. Push to `main`, or run **Deploy web demo** manually. The site appears
   under `https://<user>.github.io/<repo>/`.

## Service worker

`src/sw.ts` is a custom Workbox SW (injected via `vite-plugin-pwa`):

- ONNX + tokenizer JSON: cache-first, never expire while cache version matches.
- ORT runtime WASM: cache-first.
- App shell (HTML, JS, CSS): network-first so updates roll out fast.

The page can ask the SW for cache size via `postMessage({ type: 'model-cache-size' })`.

## Known limits

- **Cold load**: first visit downloads ~256 MB of INT4 model weights plus
  ~22 MB of ORT WASM. Service worker caches it; second visit ~2 s.
- **Mobile**: borderline. Tested INT4 fits in iOS Safari memory, but expect
  slow first decode on older devices. WebGPU on iOS 17+ helps significantly.
- **WebGPU**: significantly faster than WASM but unavailable in Firefox stable
  (as of writing) and inconsistent on iOS. The demo auto-disables the option
  when `navigator.gpu` is missing.
- **Grammar**: enforces against the failure modes catalogued in
  `latex_math.lark` (~500 most common commands, no user macros). Anything
  outside that surface will be rejected even if it would render fine in
  pdflatex.
