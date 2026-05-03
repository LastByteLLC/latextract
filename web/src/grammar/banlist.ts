// Port of src/latextract/grammar/banlist.py to JS.
// Patterns we forbid the decoder from emitting; tokenized in two contexts
// (bare and leading-space) because BPE splits depend on context.

import type { PreTrainedTokenizer } from "@huggingface/transformers";

export const BANNED_PATTERNS: readonly string[] = [
  "\\hskip",
  "\\hbox",
  "\\mbox",
  "\\vskip",
  "\\vspace",
  "\\hspace",
  "\\begin{array}",
  "\\end{array}",
  "\\tiny",
  "\\small",
  "\\par",
  "\\noindent",
  "\\quad\\quad\\quad\\quad",
];

export function buildBanlistTokenSequences(
  tokenizer: PreTrainedTokenizer,
  patterns: readonly string[] = BANNED_PATTERNS,
): number[][] {
  const seen = new Set<string>();
  const out: number[][] = [];
  for (const pat of patterns) {
    for (const variant of [pat, " " + pat]) {
      const ids = tokenizer.encode(variant, { add_special_tokens: false }) as number[];
      if (!ids.length) continue;
      const key = ids.join(",");
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(ids);
    }
  }
  return out;
}

/** Build a set of "single-token banned IDs" — tokens that decode into a banned
 *  pattern entirely. This is the cheap path; multi-token bans require trie matching. */
export function buildBanlistSingleTokens(
  tokenizer: PreTrainedTokenizer,
  patterns: readonly string[] = BANNED_PATTERNS,
): Set<number> {
  const out = new Set<number>();
  for (const seq of buildBanlistTokenSequences(tokenizer, patterns)) {
    if (seq.length === 1) out.add(seq[0]);
  }
  return out;
}
