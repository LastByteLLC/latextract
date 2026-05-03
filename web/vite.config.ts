import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

// Models live on a CDN/HF Hub, not in the bundle. The service worker
// caches them on first fetch (cache-first, networkFallback). We keep
// the SW simple and version it via the deployment commit SHA.
const COMMIT = process.env.GITHUB_SHA?.slice(0, 7) ?? "dev";

export default defineConfig({
  // GitHub Pages hosts under /<repo>/ — set base via env so CI can override.
  base: process.env.PAGES_BASE ?? "/",
  build: {
    target: "es2022",
    sourcemap: true,
    // transformers.js ships ONNX runtime WASM as separate files; let Vite see them.
    rollupOptions: {
      output: {
        manualChunks: {
          transformers: ["@huggingface/transformers"],
          katex: ["katex"],
        },
      },
    },
  },
  worker: { format: "es" },
  plugins: [
    VitePWA({
      registerType: "autoUpdate",
      strategies: "injectManifest",
      srcDir: "src",
      filename: "sw.ts",
      injectManifest: {
        // Don't precache models — they're big and we want explicit control.
        globPatterns: ["**/*.{js,css,html,wasm,svg,ico}"],
        maximumFileSizeToCacheInBytes: 50 * 1024 * 1024, // 50MB
      },
      manifest: {
        name: "latextract — image to LaTeX",
        short_name: "latextract",
        description: "Browser-only LaTeX OCR with grammar-constrained decoding.",
        theme_color: "#0b0d10",
        background_color: "#0b0d10",
        display: "standalone",
        icons: [],
      },
      devOptions: { enabled: false },
    }),
  ],
  define: {
    __COMMIT__: JSON.stringify(COMMIT),
  },
});
