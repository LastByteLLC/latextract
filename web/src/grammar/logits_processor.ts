// Custom @huggingface/transformers LogitsProcessor that enforces the
// LaTeX-math CFG incrementally during generation.
//
// At each decode step we:
//   1. Decode the running output to a UTF-8 prefix.
//   2. Take the top-K candidate next tokens by logit.
//   3. For each, check whether `prefix + decode(token)` is a grammar prefix.
//   4. Mask the invalid ones to -Infinity.
//
// Tokens outside the top-K are left alone — masking the entire vocab would
// require ~50k decode+parse calls per step. Top-K=64 is enough to enforce
// against the failure modes we actually see (banlist words, malformed scripts,
// runaway array-wraps) without crippling latency.

import type { PreTrainedTokenizer, Tensor } from "@huggingface/transformers";
import { LogitsProcessor } from "@huggingface/transformers";
import { isValidPrefix } from "./incremental";

const NEG_INF = -1e9;

export interface GrammarOptions {
  /** How many top-logit candidates to grammar-check per step. Default 64. */
  topK?: number;
  /** If true, also mask single-token-banlist IDs at every step (cheap). */
  banlist?: ReadonlySet<number>;
  /** Stop applying the grammar after this many output tokens (escape hatch). */
  maxSteps?: number;
}

export class GrammarConstrainedLogitsProcessor extends LogitsProcessor {
  private tokenizer: PreTrainedTokenizer;
  private opts: Required<Omit<GrammarOptions, "banlist">> & { banlist: ReadonlySet<number> };
  private prefixCache = new Map<string, boolean>();
  private steps = 0;

  constructor(tokenizer: PreTrainedTokenizer, opts: GrammarOptions = {}) {
    super();
    this.tokenizer = tokenizer;
    this.opts = {
      topK: opts.topK ?? 64,
      maxSteps: opts.maxSteps ?? 512,
      banlist: opts.banlist ?? new Set<number>(),
    };
  }

  /** Reset internal state between generations. */
  reset(): void {
    this.prefixCache.clear();
    this.steps = 0;
  }

  /** transformers.js calls this with input_ids (per-batch token sequences) and
   *  the next-step logits tensor. We mutate logits in place and return it. */
  _call(input_ids: number[][] | bigint[][], logits: Tensor): Tensor {
    this.steps++;
    if (this.steps > this.opts.maxSteps) return logits;

    // Batch=1 only (the demo always generates a single sequence at a time).
    const seq = (input_ids[0] ?? []).map((x) => Number(x));

    // Decode the running prefix once per step.
    const prefix = this.tokenizer.decode(seq, { skip_special_tokens: true }) as string;

    // Get the raw logits buffer for this batch element.
    const data = logits.data as Float32Array;
    const vocab = logits.dims[logits.dims.length - 1];
    const start = (logits.dims.length === 3 ? logits.dims[1] - 1 : 0) * vocab;
    const row = data.subarray(start, start + vocab);

    // 1. Banlist mask (cheap, always applied).
    if (this.opts.banlist.size) {
      for (const id of this.opts.banlist) row[id] = NEG_INF;
    }

    // 2. Top-K grammar check.
    const topK = this.topKIndices(row, this.opts.topK);
    for (const id of topK) {
      const tokenStr = this.tokenizer.decode([id], { skip_special_tokens: true }) as string;
      if (!tokenStr) continue;
      const candidate = prefix + tokenStr;
      const cached = this.prefixCache.get(candidate);
      let ok: boolean;
      if (cached !== undefined) {
        ok = cached;
      } else {
        ok = isValidPrefix(candidate);
        // Bound cache to ~10k entries (LRU-ish: clear oldest on overflow).
        if (this.prefixCache.size > 10_000) this.prefixCache.clear();
        this.prefixCache.set(candidate, ok);
      }
      if (!ok) row[id] = NEG_INF;
    }

    return logits;
  }

  /** Indices of the top-K largest logits. O(V·log K) with a min-heap. */
  private topKIndices(row: Float32Array, k: number): number[] {
    if (k >= row.length) return Array.from(row.keys());
    // Min-heap of size k keyed by value
    const heap: { v: number; i: number }[] = [];
    const swap = (a: number, b: number) => {
      [heap[a], heap[b]] = [heap[b], heap[a]];
    };
    const siftDown = (n: number) => {
      let i = n;
      while (true) {
        const l = 2 * i + 1, r = 2 * i + 2;
        let s = i;
        if (l < heap.length && heap[l].v < heap[s].v) s = l;
        if (r < heap.length && heap[r].v < heap[s].v) s = r;
        if (s === i) break;
        swap(i, s);
        i = s;
      }
    };
    const siftUp = (n: number) => {
      let i = n;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (heap[p].v <= heap[i].v) break;
        swap(p, i);
        i = p;
      }
    };
    for (let i = 0; i < row.length; i++) {
      const v = row[i];
      if (heap.length < k) {
        heap.push({ v, i });
        siftUp(heap.length - 1);
      } else if (v > heap[0].v) {
        heap[0] = { v, i };
        siftDown(0);
      }
    }
    return heap.map((h) => h.i);
  }
}
