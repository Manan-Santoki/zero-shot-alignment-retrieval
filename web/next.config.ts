import type { NextConfig } from "next";

/**
 * API proxy — keeps the browser on a single origin in production.
 *
 *  Browser → https://your-host/api/…   (Next.js server rewrite)
 *         → http://api:8000/api/…       (internal Docker network)
 *
 * In local dev, API_UPSTREAM is unset and we fall through to
 * http://localhost:8000, which matches the local uvicorn server.
 *
 * Rewrites pass the response through unbuffered, so SSE event streams work
 * end-to-end without any extra plumbing.
 */
const API_UPSTREAM = process.env.API_UPSTREAM || "http://localhost:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_UPSTREAM}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
