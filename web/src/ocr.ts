// Browser OCR pipeline: image → LaTeX, with optional grammar constraints.
//
// We use @huggingface/transformers v3, which wraps onnxruntime-web for ONNX
// inference. Models are hosted under /models/<variant>/ (served by the static
// site or a CDN); transformers.js loads encoder + merged-decoder ONNX +
// tokenizer.json + config.json from there.

import {
  AutoTokenizer,
  VisionEncoderDecoderModel,
  env,
  type PreTrainedTokenizer,
} from "@huggingface/transformers";

import { buildBanlistSingleTokens } from "./grammar/banlist";
import { GrammarConstrainedLogitsProcessor } from "./grammar/logits_processor";
import { warmGrammar } from "./grammar/incremental";

// INT4 is the only deployed variant — see runs/quant_profile.md for the
// benchmark that selected it over INT8 (951 MB → unshippable) and INT2
// (182 MB but no ORT Web kernel). The type stays a union so we can add
// future variants without a wide refactor.
export type Variant = "int4";
export type Backend = "webgpu" | "wasm";

export interface InitOptions {
  variant: Variant;
  backend: Backend;
  modelsBaseUrl?: string; // override for HF-Hub-hosted models
  onProgress?: (msg: string, frac?: number) => void;
}

export interface ExtractOptions {
  useGrammar: boolean;
  maxNewTokens?: number;
}

export interface ExtractResult {
  latex: string;
  rawLatex: string;
  durationMs: number;
  grammarApplied: boolean;
}

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

// TexTeller emits these trailing artifacts; mirror src/latextract/model/inference.py.
const TRAILING_NUMBER_RE = /(?:\s*\\(?:q?quad|,|;|:|!|\s))+\s*\(\d+\)\s*\\?\]?\s*$/;
const STRIP_RE = /^(.*?)\s*$/s;

export function stripArtifacts(latex: string): string {
  let s = latex.trim();
  if (s.endsWith("\\]")) {
    let inner = s.slice(0, -2).replace(/\s+$/, "");
    inner = inner.replace(TRAILING_NUMBER_RE, "").replace(/\s+$/, "");
    s = inner ? inner + "\\]" : s;
  } else {
    s = s.replace(TRAILING_NUMBER_RE, "").replace(/\s+$/, "");
  }
  return STRIP_RE.exec(s)?.[1] ?? s;
}

export class LatexOCR {
  private tokenizer!: PreTrainedTokenizer;
  private model!: VisionEncoderDecoderModel;
  private banlistTokens!: Set<number>;
  private imageSize = 448;
  private grammarReady = false;

  static async create(opts: InitOptions): Promise<LatexOCR> {
    const inst = new LatexOCR();
    await inst.init(opts);
    return inst;
  }

  private async init(opts: InitOptions): Promise<void> {
    const { variant, backend, modelsBaseUrl, onProgress } = opts;

    // Point transformers.js at our local model directory layout. The lib
    // string-concatenates `localModelPath + model_id + "/" + filename` in the
    // browser, so localModelPath must be the *parent* of the variant dir and
    // the variant name is passed as the model_id.
    //   localModelPath = "/latextract/models/"
    //   model_id       = "int4"
    //   → fetches "/latextract/models/int4/tokenizer.json"
    const modelsRoot = modelsBaseUrl ?? `${import.meta.env.BASE_URL}models/`;
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = modelsRoot;
    if (env.backends?.onnx?.wasm) {
      env.backends.onnx.wasm.numThreads = navigator.hardwareConcurrency ?? 4;
    }

    onProgress?.("loading tokenizer", 0.1);
    this.tokenizer = await AutoTokenizer.from_pretrained(variant, { local_files_only: true });

    onProgress?.(`loading ${variant} model (${backend})`, 0.3);
    // dtype tells transformers.js which ONNX file to pick within onnx/:
    // q4 → *_q4.onnx (MatMulNBits 4-bit weight-only).
    this.model = await VisionEncoderDecoderModel.from_pretrained(variant, {
      device: backend,
      dtype: "q4",
      local_files_only: true,
    });

    onProgress?.("warming grammar", 0.85);
    this.grammarReady = await warmGrammar();
    this.banlistTokens = buildBanlistSingleTokens(this.tokenizer);

    // Pull image_size from config when present.
    const cfg = (this.model as unknown as { config?: { encoder?: { image_size?: number | number[] } } }).config;
    const sz = cfg?.encoder?.image_size;
    if (typeof sz === "number") this.imageSize = sz;
    else if (Array.isArray(sz) && sz.length) this.imageSize = sz[0];

    onProgress?.("ready", 1.0);
  }

  /** Letterbox-resize to a square canvas, normalize with ImageNet stats, return CHW Float32. */
  private async preprocess(imageBitmap: ImageBitmap | HTMLImageElement): Promise<Float32Array> {
    const target = this.imageSize;
    const canvas = new OffscreenCanvas(target, target);
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, target, target);

    const w = (imageBitmap as ImageBitmap).width;
    const h = (imageBitmap as ImageBitmap).height;
    const scale = target / Math.max(w, h);
    const nw = Math.max(1, Math.round(w * scale));
    const nh = Math.max(1, Math.round(h * scale));
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(imageBitmap as CanvasImageSource, (target - nw) >> 1, (target - nh) >> 1, nw, nh);
    const { data } = ctx.getImageData(0, 0, target, target);

    // RGBA → CHW with ImageNet normalization
    const out = new Float32Array(3 * target * target);
    const planeSize = target * target;
    for (let i = 0; i < planeSize; i++) {
      const r = data[i * 4] / 255;
      const g = data[i * 4 + 1] / 255;
      const b = data[i * 4 + 2] / 255;
      out[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      out[planeSize + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      out[2 * planeSize + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
    return out;
  }

  async extract(image: ImageBitmap | HTMLImageElement, opts: ExtractOptions): Promise<ExtractResult> {
    const t0 = performance.now();
    const pixelData = await this.preprocess(image);
    const target = this.imageSize;

    // Build a Tensor of shape [1, 3, H, W] — RawImage-like pipelines do this for us
    // but we already have the right shape, so go straight to the model API.
    const { Tensor } = await import("@huggingface/transformers");
    const pixel_values = new Tensor("float32", pixelData, [1, 3, target, target]);

    const generationConfig: Record<string, unknown> = {
      max_new_tokens: opts.maxNewTokens ?? 256,
      num_beams: 1,
      do_sample: false,
      no_repeat_ngram_size: 4,
      repetition_penalty: 1.15,
    };

    const logitsProcessors: unknown[] = [];
    let grammarApplied = false;
    if (opts.useGrammar) {
      const proc = new GrammarConstrainedLogitsProcessor(this.tokenizer, {
        topK: 64,
        banlist: this.banlistTokens,
      });
      logitsProcessors.push(proc);
      grammarApplied = this.grammarReady; // false → fell back to heuristic
    }

    // generate() in transformers.js v3 takes inputs + generation_config + logits_processor.
    const output = await (this.model as unknown as {
      generate: (args: Record<string, unknown>) => Promise<{ sequences?: number[][] } | number[][]>;
    }).generate({
      inputs: pixel_values,
      generation_config: generationConfig,
      logits_processor: logitsProcessors,
    });

    const seq: number[] = Array.isArray(output)
      ? (output[0] as number[])
      : ((output as { sequences?: number[][] }).sequences?.[0] ?? []);

    const raw = this.tokenizer.decode(seq, { skip_special_tokens: true }) as string;
    const cleaned = stripArtifacts(raw);

    return {
      latex: cleaned,
      rawLatex: raw,
      durationMs: performance.now() - t0,
      grammarApplied,
    };
  }
}
