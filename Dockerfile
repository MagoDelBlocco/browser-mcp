# syntax=docker/dockerfile:1

# Stage 1: Build stage
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Self-contained venv so the runtime stage inherits everything with one COPY
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# Lockfile + project definition first, so the dependency install is reproducible:
# uv.lock pins every transitive package, not just the three direct pins.
COPY pyproject.toml uv.lock ./

# MCP server + search/scrape stack, installed from uv.lock. --frozen asserts the
# lock matches pyproject.toml; --no-install-project because server.py is a script,
# not a package.
#
# uv sync ignores $VIRTUAL_ENV and targets the project environment, which defaults
# to /app/.venv. UV_PROJECT_ENVIRONMENT (set above) redirects it to /opt/venv so the
# packages land where the runtime stage copies from. Do NOT substitute a
# /app/.venv -> /opt/venv symlink: uv would bake that symlink path into every
# console-script shebang, and those scripts break in the runtime stage where only
# /opt/venv is copied (playwright & friends fail with a bare "not found").
RUN UV_NO_CACHE=1 uv sync --frozen --no-install-project

# Crawl4AI post-install, the supported setup path: creates ~/.crawl4ai, runs the
# DB migration and installs the Chromium build Crawl4AI drives, into
# PLAYWRIGHT_BROWSERS_PATH. Must run before any crawl is attempted.
RUN crawl4ai-setup

# Diagnostics only (it crawls a live site, so never let it gate the build)
RUN crawl4ai-doctor || true

# Stage 2: Runtime stage
FROM python:3.11-slim

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# Python environment (same path as the builder, so venv shebangs still resolve)
COPY --from=builder /opt/venv /opt/venv

# Chromium build managed by Playwright/Crawl4AI
COPY --from=builder /ms-playwright /ms-playwright

# Crawl4AI home (config + cache dirs created by crawl4ai-setup)
COPY --from=builder /root/.crawl4ai /root/.crawl4ai

# Shared libraries Chromium links against (fonts, X, GTK, audio, ...)
RUN apt-get update \
    && playwright install-deps chromium \
    && rm -rf /var/lib/apt/lists/*

# Copy server code (all modules: server + config/safety/chunking/crawl)
COPY *.py .

# Prove the browser stack works in the *final* image: launch Chromium, then run a
# real crawl through the server's own browser config. Fails the build if not.
RUN <<'PY' python
import asyncio
from crawl4ai import AsyncWebCrawler
from crawl import BROWSER_CFG, _run_config

async def main():
    async with AsyncWebCrawler(config=BROWSER_CFG) as crawler:
        res = await crawler.arun(
            "raw://<html><body><h1>Smoke</h1><p>" + "crawl4ai container smoke test. " * 20 + "</p></body></html>",
            config=_run_config(),
        )
    assert res.success, res.error_message
    assert "smoke test" in str(res.markdown).lower(), str(res.markdown)[:500]
    print("[build] crawl4ai smoke test OK")

asyncio.run(main())
PY

# Expose MCP port
EXPOSE 8000

# Start server
ENTRYPOINT ["python", "server.py"]
