# --- build stage ---
FROM node:20-alpine AS build

WORKDIR /app
COPY web/package.json web/pnpm-lock.yaml ./
RUN corepack enable && corepack prepare pnpm@10 --activate \
 && pnpm install --frozen-lockfile

COPY web/ ./

# Empty string → browser uses same-origin (/api/...) paths.
# Next.js server rewrites those to the internal api container via API_UPSTREAM.
ARG NEXT_PUBLIC_API_BASE=""
ENV NEXT_PUBLIC_API_BASE=${NEXT_PUBLIC_API_BASE}

RUN pnpm build

# --- runtime stage (Next standalone output) ---
FROM node:20-alpine AS runtime

WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000
# API_UPSTREAM is read by next.config.ts rewrites() at server runtime.
# Override via docker-compose's environment: block.
ENV API_UPSTREAM=http://api:8000

# Standalone build bundles only the deps next needs.
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public

EXPOSE 3000
CMD ["node", "server.js"]
