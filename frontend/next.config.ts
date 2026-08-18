import type { NextConfig } from "next";

// GitHub Pages serves this repo's site from /aporia, not from the domain root,
// so every asset URL needs that prefix. It is opt-in via env rather than
// hardcoded because with it set, `next dev` on localhost:3000 would 404 —
// the same build has to work from a subpath and from a root.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  // Fully static export. The app has no server-side work of its own: it either
  // talks to the Python API in the browser, or reads the bundled demo JSON.
  // Static output means it deploys anywhere — Vercel, S3, GitHub Pages — and
  // costs nothing to host while the real backend is still being provisioned.
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
  ...(basePath ? { basePath, assetPrefix: basePath } : {}),
};

export default nextConfig;
