# ai-knot HTTP sidecar — a Python-free front door for Node/TS (and any HTTP client).
#
#   docker build -t ai-knot .
#   docker run -p 8000:8000 -v ai-knot-data:/data ai-knot
#
# Then, from TypeScript — no local Python, no MCP subprocess:
#
#   import { HttpKnowledgeBase } from "ai-knot";
#   const kb = new HttpKnowledgeBase({ baseUrl: "http://127.0.0.1:8000" });
#
# The container runs the deterministic ai-knot core over HTTP, so a Node/TS app
# never needs Python on its own host. Set AI_KNOT_SERVER_TOKEN to require an
# `Authorization: Bearer <token>` header on the /v1/* routes and /inspect.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    AI_KNOT_AGENT_ID=assistant \
    AI_KNOT_EMBED_URL=""

# Deterministic BM25-only recall by default — the sidecar makes no outbound
# connection (matching the air-gap guarantee). To enable the optional dense
# channel, set AI_KNOT_EMBED_URL to a reachable embeddings endpoint:
#   docker run -e AI_KNOT_EMBED_URL=http://host.docker.internal:11434 ...

WORKDIR /app

# Install from source so the image reflects this repo, not a pinned PyPI build.
# Copy only what the build backend needs first, so the dependency layer caches.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[server]"

# Run as a non-root user; SQLite lives on a writable volume so memory survives restarts.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /data \
 && chown -R appuser:appuser /data /app
VOLUME ["/data"]
USER appuser

EXPOSE 8000

# Liveness probe via the unauthenticated /health endpoint (no token required).
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

# Bind 0.0.0.0 so the sidecar is reachable from outside the container.
# AI_KNOT_AGENT_ID is an env so `docker run -e AI_KNOT_AGENT_ID=...` re-targets it.
CMD ["sh", "-c", "ai-knot --storage sqlite --data-dir /data serve \"$AI_KNOT_AGENT_ID\" --host 0.0.0.0 --port 8000"]
