// Service worker: cache-first for model files, network-first for app shell.
//
// This file is processed by vite-plugin-pwa with strategy=injectManifest, so
// the precache manifest (app-shell assets) is injected at __WB_MANIFEST.
// Models are NOT precached — we only cache them after the first explicit fetch
// because we don't want to download ~230 MB on first visit unless the user
// actually uses the app.

/// <reference lib="webworker" />

import { precacheAndRoute } from "workbox-precaching";
import { registerRoute } from "workbox-routing";
import { CacheFirst, NetworkFirst } from "workbox-strategies";
import { CacheableResponsePlugin } from "workbox-cacheable-response";
import { ExpirationPlugin } from "workbox-expiration";

declare const self: ServiceWorkerGlobalScope & { __WB_MANIFEST: { url: string }[] };

const APP_VERSION = "v1";
const MODEL_CACHE = `latextract-models-${APP_VERSION}`;
const ORT_CACHE = `latextract-ort-wasm-${APP_VERSION}`;

precacheAndRoute(self.__WB_MANIFEST ?? []);

// 1. ONNX model artifacts: cache-first, never expire while the cache name matches.
//    Path shape: /models/{int8,int4,fp32}/onnx/*.onnx and /models/.../tokenizer.json etc.
registerRoute(
  ({ url }) => url.pathname.includes("/models/") && /\.(onnx|json|txt)$/.test(url.pathname),
  new CacheFirst({
    cacheName: MODEL_CACHE,
    plugins: [
      new CacheableResponsePlugin({ statuses: [0, 200] }),
      new ExpirationPlugin({ maxEntries: 60, purgeOnQuotaError: true }),
    ],
  }),
);

// 2. ORT runtime WASM: cache-first too. These come from the @huggingface/transformers
//    CDN or are bundled locally; either way they don't change without an app upgrade.
registerRoute(
  ({ url }) => /ort-.*\.(wasm|mjs|js)$/.test(url.pathname) || url.pathname.includes("/onnx-"),
  new CacheFirst({
    cacheName: ORT_CACHE,
    plugins: [new CacheableResponsePlugin({ statuses: [0, 200] })],
  }),
);

// 3. App shell + everything else: network-first so updates roll out fast.
registerRoute(
  ({ request }) => request.destination === "document" || request.destination === "script" || request.destination === "style",
  new NetworkFirst({ cacheName: `latextract-shell-${APP_VERSION}` }),
);

// Clean old caches on activate.
self.addEventListener("activate", (event) => {
  const keep = new Set([MODEL_CACHE, ORT_CACHE, `latextract-shell-${APP_VERSION}`]);
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n.startsWith("latextract-") && !keep.has(n)).map((n) => caches.delete(n))),
    ),
  );
  void self.clients.claim();
});

// Allow the page to ask "how big is my model cache?" so the UI can show progress.
self.addEventListener("message", async (event) => {
  if ((event.data as { type?: string })?.type === "model-cache-size") {
    const cache = await caches.open(MODEL_CACHE);
    const reqs = await cache.keys();
    let total = 0;
    for (const req of reqs) {
      const resp = await cache.match(req);
      if (resp) {
        const blob = await resp.clone().blob();
        total += blob.size;
      }
    }
    event.source?.postMessage({ type: "model-cache-size", bytes: total, count: reqs.length });
  }
});
