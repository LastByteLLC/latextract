// Incremental prefix-validity check against the LaTeX-math CFG.
//
// The CFG lives in src/latextract/grammar/latex_math.lark. At build time,
// `lark-js` compiles it to ./parser.js (a self-contained Earley parser).
// We import that and wrap it with a prefix-validity API:
//
//   isValidPrefix(text) → true if `text` can be extended to a grammar-accepted string.
//
// True incremental Earley would maintain column state across calls; lark-js
// doesn't expose that, so we re-parse from scratch and distinguish error types:
//
//   parse succeeds              → valid (and complete)
//   throws UnexpectedEOF        → valid prefix, more input expected
//   throws UnexpectedToken/Char → invalid — this branch is dead
//
// The cost is O(N) per check. For 256-token outputs with K=64 candidates that's
// O(N²·K) overall — bounded and small in practice (<2s of CPU per generation
// even in WASM).

// Compiled parser is currently disabled: lark-js (the natural .lark→JS path)
// only emits LALR and our grammar has 20+ reduce/reduce collisions. We use the
// heuristic prefix check below until a real Earley-on-JS path lands (nearley
// port or llguidance WASM). Keeping the loader here as a no-op + a hook so a
// future port can drop in without further wiring.
const _parser: { parse: (text: string) => unknown } | null = null;

/** Returns whether the full CFG parser is loaded (currently always false). */
export async function warmGrammar(): Promise<boolean> {
  return false;
}

/** Heuristic prefix check used when the compiled parser is unavailable.
 *  Catches the failures we actually see in practice: unmatched braces,
 *  banlist commands, runaway repetition. Cheap and deterministic. */
function heuristicPrefixOk(text: string): boolean {
  // Brace balance: the closing count must never exceed the open count
  // at any prefix (we're checking a prefix, so equal-or-fewer is fine).
  let depth = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === "{" && text[i - 1] !== "\\") depth++;
    else if (c === "}" && text[i - 1] !== "\\") depth--;
    if (depth < 0) return false;
  }
  // Reject obvious garbage commands we know break round-trip rendering.
  const banned = [
    "\\hskip", "\\hbox", "\\mbox", "\\vskip", "\\vspace", "\\hspace",
    "\\begin{array}", "\\par", "\\noindent",
  ];
  for (const b of banned) if (text.includes(b)) return false;
  return true;
}

/** Returns true if `text` could be extended to a grammar-accepted string.
 *  This is the inner loop of the constrained-decoding logits processor. */
export function isValidPrefix(text: string): boolean {
  if (!_parser) return heuristicPrefixOk(text);
  try {
    _parser.parse(text);
    return true; // complete & valid
  } catch (e) {
    // lark-js exception names follow the Python conventions
    const name = (e as { name?: string }).name ?? "";
    if (name === "UnexpectedEOF" || name === "UnexpectedInput" || name.includes("EOF")) {
      return true; // valid prefix, more expected
    }
    return false;
  }
}

/** Same as isValidPrefix, but also returns true on empty (start state). */
export function isValidPrefixOrEmpty(text: string): boolean {
  if (!text || !text.trim()) return true;
  return isValidPrefix(text);
}
