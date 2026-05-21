# MCP Browser Server

Web search and content retrieval MCP server with two-stage reranking.

## Tools

- **`search`** — Google Custom Search with semantic reranking
- **`deep_fetch`** — Fetch full markdown from URLs, chunk + rerank by relevance

## Reranking Pipeline

1. **Embedding pre-filter** (`google/embeddinggemma-300m`) — bi-encoder cosine similarity to surface top 30 candidates
2. **Cross-encoder rerank** (`cross-encoder/ettin-reranker-400m-v1`) — precise pairwise scoring on candidates
3. **Score floor** — discard chunks below `RERANK_MIN_SCORE` (default: 0.0)
4. **Top-P sampling** — select chunks until cumulative normalized relevance reaches threshold

## Setup

```bash
# Copy and fill in credentials
cp .env.template .env
# Edit .env with your keys

# Build and run (pass HF_TOKEN as build arg for model download)
docker compose up -d --build
```

Server runs on port **13010** (MCP HTTP transport at `/mcp`).

## Configuration

| Env Var | Required | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | — |
| `GOOGLE_CX` | Yes | — |
| `HF_TOKEN` | Yes | — |
| `FIRECRAWL_URL` | No | `http://host.docker.internal:13002` |

## Tuning

Edit `pse_server.py` constants:

| Constant | Default | Description |
|---|---|---|
| `RERANK_TOP_CANDIDATES` | 30 | Chunks passed to cross-encoder |
| `RERANK_MIN_SCORE` | 0.0 | Minimum cross-encoder score to include |
| `TOP_P` | 4 | Cumulative normalized relevance threshold |
| `TOP_K` | 10 | Maximum chunks returned |
