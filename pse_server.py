from fastmcp import FastMCP
import asyncio
import json
import os
import re
import sys
import time
from typing import Annotated, Any
from urllib.parse import urlparse

import httpx

mcp = FastMCP("browser")

# ---- CONFIG ----
API_KEY = os.environ.get("GOOGLE_API_KEY")
CX = os.environ.get("GOOGLE_CX")
FIRECRAWL_BASE = os.environ.get("FIRECRAWL_URL", "http://host.docker.internal:13002")
MAX_CONTENT_LENGTH = 30000
TOP_P = 4
TOP_K = 10
RERANK_TOP_CANDIDATES = 30
RERANK_MIN_SCORE = 0.0

if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is required")
if not CX:
    raise RuntimeError("GOOGLE_CX is required")

GOOGLE_URL = "https://www.googleapis.com/customsearch/v1"

TIMEOUT = httpx.Timeout(20.0, connect=5.0)
LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10)

# ---- PERSISTENT WORKER ----
WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "worker.py")
WORKER_TIMEOUT = 120  # seconds
WORKER_IDLE_TIMEOUT = 120  # Kill worker after 2 min of inactivity


class PersistentWorker:
    """Keeps a worker subprocess alive between requests. Models stay in CPU RAM."""

    def __init__(self, name: str):
        self._name = name
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._idle_timer: asyncio.Task | None = None

    async def _ensure_running(self):
        if self._proc is not None and self._proc.returncode is None:
            return  # Already running
        # Start fresh worker — stderr=None inherits server stderr so tracebacks surface
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", WORKER_SCRIPT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            env=os.environ.copy(),
        )
        print(f"[{self._name}] worker started (pid={self._proc.pid})", flush=True)

        # Wait for READY signal, tolerating stdout noise from model loading
        while True:
            try:
                line = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=WORKER_TIMEOUT
                )
            except asyncio.TimeoutError:
                self._proc.kill()
                raise RuntimeError(f"Worker {self._name} timed out during startup")

            if not line:  # EOF — worker died
                self._proc = None
                raise RuntimeError(f"Worker {self._name} exited during startup")

            text = line.decode(errors="replace").strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                print(f"[{self._name}] startup noise (ignored): {text}", flush=True)
                continue
            if data.get("ready"):
                print(f"[{self._name}] worker ready", flush=True)
                break

    def _cancel_idle(self):
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _start_idle(self):
        self._cancel_idle()
        self._idle_timer = asyncio.create_task(self._idle_shutdown())

    async def _idle_shutdown(self):
        await asyncio.sleep(WORKER_IDLE_TIMEOUT)
        if self._proc is not None:
            print(f"[{self._name}] worker idle — shutting down", flush=True)
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    async def call(self, kind: str, query: str, texts: list[str]) -> list[float]:
        await self._ensure_running()
        assert self._proc is not None

        cmd = json.dumps({"kind": kind, "query": query, "texts": texts}) + "\n"
        self._proc.stdin.write(cmd.encode())
        await self._proc.stdin.drain()

        try:
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=WORKER_TIMEOUT)
        except asyncio.TimeoutError:
            self._proc.kill()
            raise RuntimeError(f"Worker {self._name} timed out after {WORKER_TIMEOUT}s")

        result = json.loads(line.decode())
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "worker returned error"))

        # Reset idle timer
        self._start_idle()
        return result["scores"]

    async def shutdown(self):
        self._cancel_idle()
        if self._proc is not None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except Exception:
                self._proc.kill()


_embed_worker = PersistentWorker("embed")
_rerank_worker = PersistentWorker("rerank")


async def worker_call(kind: str, query: str, texts: list[str]) -> list[float]:
    worker = _embed_worker if kind == "embed" else _rerank_worker
    return await worker.call(kind, query, texts)


# ---- HELPERS ----
def _sanitize_snippet(snippet: Any) -> str:
    if not snippet:
        return ""
    return str(snippet).replace("\n", " ").strip()


def _is_valid_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def keyword_overlap(q, text):
    q_terms = set(q.lower().split())
    t_terms = set(text.lower().split())
    return len(q_terms & t_terms) / (len(q_terms) + 1e-6)


def recursive_split(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    separators = ["\n\n", "\n", " ", ""]
    def _split_recursive(t: str, sep_idx: int) -> list[str]:
        if len(t) <= chunk_size:
            return [t]
        sep = separators[sep_idx]
        if not sep:
            chunks = []
            start = 0
            while start < len(t):
                chunks.append(t[start : start + chunk_size])
                start += max(chunk_size - overlap, 1)
            return chunks
        splits = t.split(sep)
        chunks = []
        current_chunk = ""
        for s in splits:
            if not s:
                continue
            if current_chunk and len(current_chunk) + len(sep) + len(s) <= chunk_size:
                current_chunk += sep + s
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(s) > chunk_size:
                    chunks.extend(_split_recursive(s, sep_idx + 1))
                else:
                    current_chunk = s
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
    return _split_recursive(text, 0)


def markdown_section_split(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    sections = re.split(r'(?=^#{1,6}\s)', text, flags=re.MULTILINE)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        header_match = re.match(r'^(#{1,6}\s.*?)(\n|$)', section)
        header = header_match.group(1) if header_match else ""
        body = section[len(header):].strip() if header_match else section
        if len(section) <= chunk_size:
            chunks.append({"text": section, "has_header": bool(header)})
            continue
        if body:
            sub_chunks = recursive_split(body, chunk_size, overlap)
            for sub in sub_chunks:
                chunks.append({"text": f"{header}\n{sub}", "has_header": bool(header)})
        else:
            chunks.append({"text": section, "has_header": bool(header)})
    return [c["text"].strip() if c["has_header"] else c["text"] for c in chunks if c["text"].strip()]


def merge_small_chunks(chunks: list[str], min_length: int = 200, max_length: int = 1200) -> list[str]:
    """Concatenates consecutive small chunks until they reach min_length, respecting max_length."""
    merged = []
    buffer = ""
    for chunk in chunks:
        if not chunk.strip():
            continue
        text = chunk.strip()

        if buffer and len(buffer) + len(text) > max_length:
            merged.append(buffer)
            buffer = text
        else:
            buffer = (buffer + "\n\n" + text).strip() if buffer else text

        if len(buffer) >= min_length:
            merged.append(buffer)
            buffer = ""

    if buffer:
        merged.append(buffer)
    return merged


# ---- SEARCH TOOL ----
@mcp.tool(description="Searches the web using Google Custom Search and reranks results semantically. Returns top relevant results with titles, URLs, snippets, and relevance scores.")
async def search(query: str, count: int = 5):
    query = (query or "").strip()
    if not query:
        return {"error": "query must not be empty"}

    count = max(1, min(int(count), 10))

    params = {
        "key": API_KEY,
        "cx": CX,
        "q": query,
        "num": count,
        "fields": "items(title,link,snippet)",
    }

    async with httpx.AsyncClient(timeout=TIMEOUT, limits=LIMITS) as client:
        try:
            resp = await client.get(GOOGLE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return {"error": "search_failed", "detail": str(e)}

    items = data.get("items", []) or []
    if not items:
        return {"query": query, "results": []}

    # ---- EMBEDDING RERANK via worker ----
    texts = [
        f"{item.get('title', '')}. {item.get('snippet', '')}"
        for item in items
    ]

    try:
        scores = await worker_call("embed", query, texts)
    except Exception as e:
        return {"error": f"embedding_failed", "detail": str(e)}

    scored = []
    for item, text, sim in zip(items, texts, scores):
        kw = keyword_overlap(query, text)
        final_score = 0.7 * sim + 0.3 * kw
        scored.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": _sanitize_snippet(item.get("snippet")),
            "score": float(final_score),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # normalize scores
    all_scores = [r["score"] for r in scored]
    min_s, max_s = min(all_scores), max(all_scores)
    print("Reranking scores:")
    print(all_scores)
    print("-------")

    results = []
    for i, r in enumerate(scored, 1):
        norm = (r["score"] - min_s) / (max_s - min_s + 1e-6)
        results.append({
            "id": f"S{i}",
            "title": r["title"],
            "url": r["url"],
            "snippet": r["snippet"],
            "score": r["score"],
            "score_norm": norm,
            "citation": f"[S{i}] {r['title']} - {r['url']}"
        })

    return {"results": results}


# ---- DEEP FETCH TOOL ----
@mcp.tool(description="Fetches full markdown content from provided URLs, splits it into semantic chunks, and reranks them based on query relevance. Returns the most relevant chunks (or full content if whole=True).")
async def deep_fetch(
    urls: Annotated[list[str], "List of URLs to fetch"],
    query: str = "",
    whole: bool = False
):
    valid_urls = [u for u in urls if isinstance(u, str) and _is_valid_http_url(u)]
    if not valid_urls:
        return {"error": "no valid URLs"}

    sem = asyncio.Semaphore(4)

    async def fetch_one(client: httpx.AsyncClient, url: str, idx: int):
        async with sem:
            payload = {"url": url, "formats": ["markdown"], "onlyMainContent": True}
            try:
                r = await client.post(
                    f"{FIRECRAWL_BASE}/v1/scrape",
                    json=payload,
                    timeout=httpx.Timeout(30.0, connect=5.0),
                )
                r.raise_for_status()
                data = r.json()
                return {"id": f"S{idx}", "url": url, "ok": True, "content": data.get("data", {}).get("markdown", "")}
            except Exception as e:
                return {"id": f"S{idx}", "url": url, "ok": False, "error": str(e)}

    t0 = time.perf_counter()

    async with httpx.AsyncClient(limits=LIMITS) as client:
        results = await asyncio.gather(*(fetch_one(client, u, i) for i, u in enumerate(valid_urls, 1)))

    t_fetch = time.perf_counter() - t0
    print(f"[deep_fetch] fetch: {t_fetch:.2f}s")

    # 1. WHOLE MODE: Bypasses all chunking/reranking
    if whole or not query:
        successful = [r for r in results if r.get("ok") and r.get("content")]
        if not successful:
            return {"error": "no content fetched from provided URLs"}

        out = []
        for r in successful:
            content = r["content"]
            if len(content) > MAX_CONTENT_LENGTH:
                out.append({
                    "url": r["url"],
                    "content": content[:MAX_CONTENT_LENGTH],
                    "truncated": True,
                    "message": f"Content truncated at {MAX_CONTENT_LENGTH} characters. This page is very large. Retry without whole=True to get chunked, reranked results instead.",
                })
            else:
                out.append({"url": r["url"], "content": content})
        return {"results": out}

    # 2. RERANKING MODE
    successful = [r for r in results if r.get("ok") and r.get("content")]
    if not successful:
        return {"error": "no content fetched from provided URLs"}

    CHUNK_SIZE = 1000
    OVERLAP = 200
    THRESHOLD = TOP_K * CHUNK_SIZE * 1.2
    all_chunks = []

    t_chunk = time.perf_counter()
    for res in successful:
        content = res["content"]
        if len(content) < THRESHOLD:
            all_chunks.append({"url": res["url"], "chunk_id": 0, "text": content, "embed_score": 1.0})
        else:
            chunks = markdown_section_split(content, CHUNK_SIZE, OVERLAP)
            merged_chunks = merge_small_chunks(chunks, min_length=200)
            for i, c in enumerate(merged_chunks):
                all_chunks.append({"url": res["url"], "chunk_id": i, "text": c})
    t_chunk = time.perf_counter() - t_chunk
    print(f"[deep_fetch] chunking: {t_chunk:.2f}s ({len(all_chunks)} chunks)")

    if not all_chunks:
        return {"error": "content fetched but contained no text"}

    # Stage 1: Embedding pre-filter via worker
    t_embed = time.perf_counter()
    try:
        embed_scores = await worker_call("embed", query, [c["text"] for c in all_chunks])
    except Exception as e:
        print(f"[deep_fetch] embedding failed: {e} — falling back to whole mode")
        out = []
        for r in successful:
            content = r["content"]
            if len(content) > MAX_CONTENT_LENGTH:
                out.append({
                    "url": r["url"],
                    "content": content[:MAX_CONTENT_LENGTH],
                    "truncated": True,
                    "message": f"Content truncated at {MAX_CONTENT_LENGTH} characters. Embedding failed; returned full content instead of reranked chunks.",
                })
            else:
                out.append({"url": r["url"], "content": content})
        return {"results": out}

    for chunk, score in zip(all_chunks, embed_scores):
        chunk["embed_score"] = score
    t_embed = time.perf_counter() - t_embed
    print(f"[deep_fetch] embed pre-filter: {t_embed:.2f}s ({len(all_chunks)} chunks)")

    # Sort by embedding score and take top candidates for cross-encoder
    all_chunks.sort(key=lambda x: x["embed_score"], reverse=True)
    candidates = all_chunks[:RERANK_TOP_CANDIDATES]

    # Stage 2: Cross-encoder reranking via worker
    t_rerank = time.perf_counter()
    try:
        rerank_scores = await worker_call("rerank", query, [c["text"] for c in candidates])
    except Exception as e:
        print(f"[deep_fetch] reranking failed: {e} — using embedding scores only")
        rerank_scores = [c["embed_score"] for c in candidates]
    for chunk, score in zip(candidates, rerank_scores):
        chunk["score"] = float(score)
    t_rerank = time.perf_counter() - t_rerank
    print(f"[deep_fetch] cross-encoder rerank: {t_rerank:.2f}s ({len(candidates)} candidates)")

    # Filter out low-scoring chunks before sampling
    relevant = [c for c in candidates if c["score"] >= RERANK_MIN_SCORE]
    if not relevant:
        return {"results": []}

    # Top-P Sampler (Cumulative Normalized Relevance) on reranked candidates
    scores = [c["score"] for c in relevant]
    min_s, max_s = min(scores), max(scores)
    norm_scores = [(s - min_s) / (max_s - min_s + 1e-6) for s in scores]

    selected = []
    cum_mass = 0.0
    for chunk, ns in zip(relevant, norm_scores):
        if len(selected) >= TOP_K:
            break
        cum_mass += ns
        selected.append(chunk)
        if cum_mass >= TOP_P:
            break

    # Clean output: strip internal scoring fields
    clean_results = [
        {"url": c["url"], "chunk_id": c["chunk_id"], "text": c["text"], "score": c["score"]}
        for c in selected
    ]

    return {"results": clean_results}


# ---- RUN ----
if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
