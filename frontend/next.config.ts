import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Fully static export. The app has no server-side work of its own: it either
  // talks to the Python API in the browser, or reads the bundled demo JSON.
  // Static output means it deploys anywhere — Vercel, S3, GitHub Pages — and
  // costs nothing to host while the real backend is still being provisioned.
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
