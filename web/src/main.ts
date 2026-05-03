// UI wiring: drag-drop / paste / file input → OCR pipeline → KaTeX render.

import katex from "katex";
import { LatexOCR, type Backend, type Variant } from "./ocr";

declare const __COMMIT__: string;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const dropEl = $<HTMLElement>("drop");
const fileEl = $<HTMLInputElement>("file");
const pickBtn = $<HTMLButtonElement>("pick");
const variantSel = $<HTMLSelectElement>("variant");
const backendSel = $<HTMLSelectElement>("backend");
const grammarChk = $<HTMLInputElement>("grammar");
const statusEl = $<HTMLElement>("status");
const progressEl = $<HTMLProgressElement>("progress");
const statusText = $<HTMLElement>("status-text");
const resultEl = $<HTMLElement>("result");
const inputImg = $<HTMLImageElement>("input-img");
const renderedEl = $<HTMLElement>("rendered");
const latexOutEl = $<HTMLElement>("latex-out");
const verdictEl = $<HTMLElement>("verdict");
const copyBtn = $<HTMLButtonElement>("copy");
const commitEl = $<HTMLElement>("commit");

commitEl.textContent = `build ${__COMMIT__ ?? "dev"}`;

// When the SW activates a new version (it calls skipWaiting + clients.claim),
// reload once so the page picks up the new bundle. Without this, users keep
// running the old JS until they manually refresh — which is how stale
// precached bundles linger after a deploy.
if ("serviceWorker" in navigator) {
  let reloaded = false;
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (reloaded) return;
    reloaded = true;
    window.location.reload();
  });
}

let ocr: LatexOCR | null = null;
let currentVariant: Variant | null = null;
let currentBackend: Backend | null = null;

function setStatus(msg: string, frac = 0): void {
  statusEl.hidden = false;
  statusText.textContent = msg;
  progressEl.value = Math.round(frac * 100);
}

function hideStatus(): void {
  statusEl.hidden = true;
}

async function ensureOcr(): Promise<LatexOCR> {
  const variant = variantSel.value as Variant;
  const backend = backendSel.value as Backend;
  if (ocr && variant === currentVariant && backend === currentBackend) return ocr;
  setStatus("loading model…", 0);
  ocr = await LatexOCR.create({
    variant,
    backend,
    onProgress: (m, f) => setStatus(m, f ?? 0),
  });
  currentVariant = variant;
  currentBackend = backend;
  return ocr;
}

async function processFile(file: File): Promise<void> {
  const url = URL.createObjectURL(file);
  inputImg.src = url;
  resultEl.hidden = false;
  renderedEl.innerHTML = "";
  latexOutEl.textContent = "";
  verdictEl.textContent = "";
  verdictEl.className = "";

  const bitmap = await createImageBitmap(file);
  const inst = await ensureOcr();
  setStatus("decoding…", 0.5);

  const t0 = performance.now();
  const out = await inst.extract(bitmap, { useGrammar: grammarChk.checked });
  const dt = performance.now() - t0;

  latexOutEl.textContent = out.latex;

  // Validate by rendering with KaTeX. KaTeX-renders is our browser-side "verify".
  let katexOk = false;
  try {
    katex.render(out.latex, renderedEl, { throwOnError: true, displayMode: true });
    katexOk = true;
  } catch (e) {
    // Try one fallback: strip outer \[ \] and re-render
    const inner = out.latex.replace(/^\\\[\s*/, "").replace(/\s*\\\]$/, "");
    try {
      katex.render(inner, renderedEl, { throwOnError: true, displayMode: true });
      katexOk = true;
    } catch (e2) {
      renderedEl.textContent = `KaTeX failed: ${(e2 as Error).message.split("\n")[0]}`;
    }
  }

  const grammarSuffix = out.grammarApplied
    ? "grammar-constrained (CFG)"
    : grammarChk.checked
      ? "grammar-constrained (heuristic — compile parser.js for full CFG)"
      : "no grammar";

  if (katexOk) {
    verdictEl.className = "ok";
    verdictEl.textContent = `✓ valid LaTeX — KaTeX rendered cleanly · ${grammarSuffix} · ${dt.toFixed(0)} ms`;
  } else {
    verdictEl.className = "err";
    verdictEl.textContent = `✗ KaTeX rejected the output · ${grammarSuffix} · ${dt.toFixed(0)} ms`;
  }
  hideStatus();
}

// --- input handlers ---

pickBtn.addEventListener("click", () => fileEl.click());
fileEl.addEventListener("change", () => {
  const f = fileEl.files?.[0];
  if (f) void processFile(f);
});

dropEl.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropEl.classList.add("dragover");
});
dropEl.addEventListener("dragleave", () => dropEl.classList.remove("dragover"));
dropEl.addEventListener("drop", (e) => {
  e.preventDefault();
  dropEl.classList.remove("dragover");
  const f = e.dataTransfer?.files?.[0];
  if (f) void processFile(f);
});

document.addEventListener("paste", (e: ClipboardEvent) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const f = item.getAsFile();
      if (f) {
        void processFile(f);
        e.preventDefault();
        return;
      }
    }
  }
});

copyBtn.addEventListener("click", () => {
  const text = latexOutEl.textContent ?? "";
  void navigator.clipboard.writeText(text);
  const orig = copyBtn.textContent;
  copyBtn.textContent = "copied";
  setTimeout(() => (copyBtn.textContent = orig), 1200);
});

// Reload model when the variant or backend changes (we only support one
// loaded at a time; switching evicts the previous from the WebGPU device).
variantSel.addEventListener("change", () => {
  ocr = null;
  currentVariant = null;
});
backendSel.addEventListener("change", () => {
  ocr = null;
  currentBackend = null;
});

// Detect WebGPU availability and disable the option if unavailable.
if (!("gpu" in navigator)) {
  const opt = backendSel.querySelector('option[value="webgpu"]') as HTMLOptionElement | null;
  if (opt) {
    opt.disabled = true;
    opt.textContent += " (unavailable in this browser)";
  }
  backendSel.value = "wasm";
}
