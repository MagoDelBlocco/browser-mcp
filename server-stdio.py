"""MCP web-search server: DuckDuckGo for discovery, Crawl4AI for retrieval.

Tools:
  search      — DuckDuckGo results, in engine rank order
  fetch       — Crawl4AI page fetch, chunked + BM25-ranked against a query
  research    — search + fetch + rank in one call

Ranking uses BM25 over the fetched chunks — no neural models required.

This module wires the pieces together. The pure chunking/ranking logic lives in
`chunking`, network-safety/credentials in `safety`, and the crawler/page-cache in
`crawl`; configuration is centralised in `config`.
"""
from fastmcp import Context, FastMCP
import asyncio
import contextlib
import json
import re
import sys
from typing import Annotated, Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from config import (
    AUTH_TOKEN,
    BIND_HOST,
    MAX_RESEARCH_DEPTH,
    MAX_RETURNED_CHUNKS,
    MAX_SEARCH_RESULTS,
    MAX_URLS_PER_CALL,
    PORT,
    SAMPLING_ENABLED,
    SAMPLING_TIMEOUT,
    SEARCH_BACKEND,
    SEARCH_REGION,
    SEARCH_RETRIES,
    SEARCH_SAFESEARCH,
)
from safety import _is_valid_http_url, _sanitize_snippet, _screen_urls
from crawl import _crawler_pool, _page_cache, _resource_key, crawl
from chunking import full_text_of, rank_chunks


# ---- SEARCH BACKEND (DuckDuckGo) ----
def _ddg_query(query: str, count: int, timelimit: str | None, kind: str) -> list[dict[str, Any]]:
    """Blocking DDG call — always run via asyncio.to_thread.

    The news index is a different corpus, not a filter over the web one: it carries
    real publish dates and is far better for anything time-sensitive.
    """
    client = DDGS()
    method = client.news if kind == "news" else client.text
    kwargs: dict[str, Any] = {
        "region": SEARCH_REGION,
        "safesearch": SEARCH_SAFESEARCH,
        "timelimit": timelimit,
        "max_results": count,
    }
    if kind != "news":
        kwargs["backend"] = SEARCH_BACKEND
    return method(query, **kwargs)


async def ddg_search(
    query: str, count: int, timelimit: str | None = None, kind: str = "web"
) -> list[dict[str, Any]]:
    """DDG search with backoff on rate limits. Raises on final failure."""
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(1, SEARCH_RETRIES + 1):
        try:
            return await asyncio.to_thread(_ddg_query, query, count, timelimit, kind)
        except (RatelimitException, TimeoutException) as e:
            last_err = e
            if attempt == SEARCH_RETRIES:
                break
            print(f"[search] {type(e).__name__} — retry {attempt}/{SEARCH_RETRIES} in {delay:.1f}s", flush=True, file=sys.stderr)
            await asyncio.sleep(delay)
            delay *= 2
        except DDGSException as e:
            raise RuntimeError(f"duckduckgo error: {e}") from e
    raise RuntimeError(f"duckduckgo unavailable after {SEARCH_RETRIES} attempts: {last_err}")


def _normalize_hits(raw: list[dict[str, Any]], limit: int) -> list[dict[str, str]]:
    """Map ddgs rows to {title,url,snippet}, deduplicated by URL, order preserved."""
    seen: set[str] = set()
    hits: list[dict[str, str]] = []
    for row in raw or []:
        if len(hits) >= limit:
            break
        url = (row.get("href") or row.get("url") or "").strip()
        if not url or not _is_valid_http_url(url) or url in seen:
            continue
        seen.add(url)
        hit = {
            "title": _sanitize_snippet(row.get("title")),
            "url": url,
            "snippet": _sanitize_snippet(row.get("body") or row.get("description")),
        }
        # News rows carry a real publish date; web rows almost never do
        for optional in ("date", "source"):
            value = _sanitize_snippet(row.get(optional))
            if value:
                hit[optional] = value
        hits.append(hit)
    return hits


def rank_hits(hits: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Annotate search hits with their DuckDuckGo engine rank.

    Deliberately reports `rank`, not a score: nothing here measures relevance, and
    a synthesised score reads to a model as if something did.
    """
    ranked = []
    for i, hit in enumerate(hits, 1):
        entry = {
            "id": f"S{i}",
            "title": hit["title"],
            "url": hit["url"],
            "snippet": hit["snippet"],
            "rank": i,
            "citation": f"[S{i}] {hit['title']} - {hit['url']}",
        }
        for optional in ("date", "source"):
            if hit.get(optional):
                entry[optional] = hit[optional]
        ranked.append(entry)
    return ranked


# ---- PAGE PRESENTATION ----
def _page_summaries(docs: list[dict[str, Any]], include_links: bool = True) -> list[dict[str, Any]]:
    """Per-page metadata to accompany the ranked chunks: what was read, and where it points."""
    pages = []
    for doc in docs:
        page: dict[str, Any] = {
            "url": doc["url"],
            "title": doc.get("title", ""),
            "content_type": doc.get("content_type", "html"),
        }
        for optional in ("published", "author"):
            if doc.get(optional):
                page[optional] = doc[optional]
        # Where to re-read the whole page without spending another fetch
        if _page_cache.ttl > 0 and doc.get("content"):
            page["resource"] = f"cache://page/{_resource_key(doc['url'])}"
        if doc.get("final_url"):
            page["final_url"] = doc["final_url"]
        if doc.get("cached"):
            page["cached"] = True
        if include_links and doc.get("links"):
            page["links"] = doc["links"]
        pages.append(page)
    return pages


# ---- SERVER ----
@contextlib.asynccontextmanager
async def _lifespan(_server):
    """Close the warm browser on shutdown, while its event loop is still running."""
    try:
        yield
    finally:
        await _crawler_pool.shutdown()


def _build_auth():
    """Optional static bearer token. No token configured → no auth (localhost only)."""
    if not AUTH_TOKEN:
        return None
    from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

    return StaticTokenVerifier(tokens={AUTH_TOKEN: {"client_id": "mcp-browser", "scopes": []}})


mcp = FastMCP("browser", auth=_build_auth(), lifespan=_lifespan)

# Every tool here reads the open web and changes nothing server-side.
READ_ONLY_WEB = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": True}


async def _progress(ctx, done: float, total: float, message: str) -> None:
    """Report progress when the client asked for it; a no-op otherwise.

    A `research` call can run for a minute or more. Suppressing errors keeps a client
    that never sent a progress token from turning a cosmetic feature into a failure.
    """
    if ctx is None:
        return
    with contextlib.suppress(Exception):
        await ctx.report_progress(done, total, message)


async def _sample_text(ctx, prompt: str, system: str, max_tokens: int = 200) -> str:
    """Borrow the client's model for a short reasoning step.

    Sampling means the server needs no API key or model of its own. Any client that
    does not support it simply returns nothing and every caller falls back to its
    deterministic path, so this is an enhancement and never a dependency.
    """
    if ctx is None or not SAMPLING_ENABLED:
        return ""
    try:
        result = await asyncio.wait_for(
            ctx.sample(prompt, system_prompt=system, max_tokens=max_tokens, temperature=0.3),
            timeout=SAMPLING_TIMEOUT,
        )
    except Exception as e:
        print(f"[sampling] unavailable ({type(e).__name__}), using fallback", flush=True, file=sys.stderr)
        return ""
    text = getattr(result, "text", None)
    if text is None:
        content = getattr(result, "content", None)
        text = getattr(content, "text", None) if content is not None else None
    return (text or "").strip()


def _parse_lines(text: str, limit: int) -> list[str]:
    """Numbered/bulleted list to a clean list of strings."""
    lines = []
    for raw in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw).strip().strip('"')
        if cleaned and len(cleaned) < 200:
            lines.append(cleaned)
        if len(lines) >= limit:
            break
    return lines


async def expand_query(ctx, query: str, limit: int = 2) -> list[str]:
    """Alternative phrasings of the query, via the client's model."""
    text = await _sample_text(
        ctx,
        f"Search query: {query}",
        system=(
            "You rewrite search queries. Give alternative phrasings that a relevant "
            f"document might actually use, one per line, at most {limit}. Vary the "
            "vocabulary — synonyms, technical terms, the words a document would use "
            "rather than the words a question uses. No numbering, no commentary."
        ),
        max_tokens=150,
    )
    variants = [v for v in _parse_lines(text, limit) if v.lower() != query.lower()]
    if variants:
        print(f"[sampling] query variants: {variants}", flush=True, file=sys.stderr)
    return variants


async def triage_results(ctx, query: str, hits: list[dict[str, Any]], take: int) -> list[dict[str, Any]]:
    """Choose which search results are worth the cost of fetching.

    Engine rank is a poor proxy for "answers this question" — falls back to it when
    sampling is unavailable or the reply cannot be parsed.
    """
    if len(hits) <= take:
        return hits[:take]
    listing = "\n".join(
        f"{i}. {h['title']} — {h['snippet'][:140]}" for i, h in enumerate(hits, 1)
    )
    text = await _sample_text(
        ctx,
        f"Question: {query}\n\nSearch results:\n{listing}",
        system=(
            f"Pick the {take} results most likely to answer the question. "
            "Reply with only their numbers, comma-separated, best first."
        ),
        max_tokens=60,
    )
    picked: list[dict[str, Any]] = []
    for token in re.findall(r"\d+", text):
        index = int(token) - 1
        if 0 <= index < len(hits) and hits[index] not in picked:
            picked.append(hits[index])
        if len(picked) >= take:
            break
    if picked:
        print(f"[sampling] triaged to {[h['id'] for h in picked]}", flush=True, file=sys.stderr)
        return picked
    return hits[:take]


async def assess_coverage(ctx, question: str, docs: list[dict[str, Any]]) -> str:
    """A follow-up search query if the question is not yet answered, else "".

    This is what separates research from a search wrapper: a single pass never
    notices that it missed. Requires sampling — without a model to judge coverage
    there is no non-arbitrary way to decide, so extra hops are simply skipped.
    """
    if not docs:
        return ""
    ranked = rank_chunks(docs, question, max_chunks=6)
    evidence = "\n\n".join(f"[{r['url']}] {r['text'][:400]}" for r in ranked.get("results", []))
    if not evidence:
        return ""
    text = await _sample_text(
        ctx,
        f"Question: {question}\n\nEvidence gathered so far:\n{evidence}",
        system=(
            "Judge whether the evidence answers the question. If it does, reply with "
            "exactly DONE. If something important is missing, reply with a single web "
            "search query that would fill that gap — the query only, no explanation."
        ),
        max_tokens=80,
    )
    cleaned = text.strip().strip('"')
    if not cleaned or cleaned.upper().startswith("DONE") or len(cleaned) > 200:
        return ""
    first_line = cleaned.splitlines()[0].strip()
    if first_line.lower() == question.lower():
        return ""
    print(f"[research] follow-up query: {first_line!r}", flush=True, file=sys.stderr)
    return first_line


def _build_sources(hits: list[dict[str, Any]], docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Citation list covering every hit seen, flagged with whether it was actually read."""
    fetched_urls = {d["url"] for d in docs}
    dates_by_url = {d["url"]: d.get("published") for d in docs if d.get("published")}
    sources = []
    for h in hits:
        source = {
            "id": h["id"],
            "title": h["title"],
            "url": h["url"],
            "snippet": h["snippet"],
            "fetched": h["url"] in fetched_urls,
            "citation": h["citation"],
        }
        # The page's own metadata beats the search index's guess at its date
        published = dates_by_url.get(h["url"]) or h.get("date")
        if published:
            source["published"] = published
        for optional in ("source", "hop"):
            if h.get(optional):
                source[optional] = h[optional]
        sources.append(source)
    return sources


@mcp.resource(
    "cache://pages",
    description="URLs currently held in the page cache, with their resource URIs.",
    mime_type="application/json",
)
def cached_pages() -> str:
    entries = [
        {"url": url, "resource": f"cache://page/{_resource_key(url)}",
         "title": record.get("title", ""), "chars": len(record.get("content", ""))}
        for url, (_, record) in _page_cache._entries.items()
    ]
    return json.dumps({"pages": entries, "ttl_seconds": _page_cache.ttl}, indent=2)


@mcp.resource(
    "cache://page/{key}",
    description="Full markdown of a previously fetched page, by cache key.",
    mime_type="text/markdown",
)
def cached_page(key: str) -> str:
    """Re-read a whole page without spending a fetch or filling the tool response.

    Lets a model pull the full text of something `fetch` already read, having seen
    only the ranked passages.
    """
    for url, (_, record) in _page_cache._entries.items():
        if _resource_key(url) == key:
            if _page_cache.get(url) is None:
                break  # expired between listing and read
            return record.get("content", "")
    return f"No cached page for key {key!r}. It may have expired; fetch the URL again."


# ---- TOOLS ----
@mcp.tool(
    description=(
        "Searches the web with DuckDuckGo. "
        "Returns titles, URLs, snippets and the engine's result rank — snippets only, no page content. "
        "Use kind='news' for time-sensitive topics: it queries the news index, which carries "
        "real publish dates. Follow up with `fetch` on the URLs worth reading, or use `research` "
        "to do both at once."
    ),
    annotations={**READ_ONLY_WEB, "idempotentHint": False, "title": "Web search"},
)
async def search(
    query: str,
    count: Annotated[int, "Number of results to return (1-20)"] = 5,
    timelimit: Annotated[str | None, "Recency filter: 'd' day, 'w' week, 'm' month, 'y' year"] = None,
    kind: Annotated[str, "'web' for general search, 'news' for dated news results"] = "web",
):
    query = (query or "").strip()
    if not query:
        return {"error": "query must not be empty"}

    kind = (kind or "web").strip().lower()
    if kind not in {"web", "news"}:
        return {"error": f"unknown kind {kind!r}, expected 'web' or 'news'"}
    count = max(1, min(int(count), MAX_SEARCH_RESULTS))

    try:
        raw = await ddg_search(query, count, timelimit, kind)
    except Exception as e:
        return {"error": "search_failed", "detail": str(e)}

    hits = _normalize_hits(raw, count)
    if not hits:
        return {"query": query, "kind": kind, "results": []}

    return {"query": query, "kind": kind, "results": rank_hits(hits)}


@mcp.tool(
    description=(
        "Fetches URLs with a real headless browser (JS-rendered) and returns clean markdown. "
        "Pages are split into chunks and the passages most relevant to the query are returned, "
        "ranked with BM25 and balanced across sources. Handles PDFs as well as HTML, returns "
        "each page's outbound links for following onward, and supports paging via offset."
    ),
    annotations={**READ_ONLY_WEB, "idempotentHint": True, "title": "Fetch pages"},
)
async def fetch(
    urls: Annotated[list[str], "List of URLs to fetch"],
    query: Annotated[str, "What you are looking for — drives chunk ranking"] = "",
    include_links: Annotated[bool, "Return each page's outbound links, for following onward"] = True,
    max_chunks: Annotated[int, "Passages to return (1-30)"] = MAX_RETURNED_CHUNKS,
    offset: Annotated[int, "Skip this many ranked passages — page deeper into the same result set"] = 0,
    full_text: Annotated[bool, "Return whole pages instead of ranked passages; ignores query"] = False,
    ctx: Context = None,
):
    if isinstance(urls, str):
        urls = [urls]
    candidates = [u for u in (urls or []) if isinstance(u, str) and _is_valid_http_url(u)]
    # Deduplicate, preserve order
    candidates = list(dict.fromkeys(candidates))[:MAX_URLS_PER_CALL]
    if not candidates:
        return {"error": "no valid URLs"}

    valid_urls, blocked = await _screen_urls(candidates)
    if not valid_urls:
        return {"error": "no fetchable URLs", "failed": blocked}

    await _progress(ctx, 0, 2, f"fetching {len(valid_urls)} page(s)")
    results = await crawl(valid_urls)
    await _progress(ctx, 1, 2, "ranking passages")
    successful = [r for r in results if r.get("ok") and r.get("content")]
    failed = blocked + [
        {
            "url": r["url"],
            "final_url": r.get("final_url"),
            "error": (
                f"redirected to {r['final_url']}; no content (likely consent/anti-bot wall)"
                if r.get("final_url") and r.get("final_url") != r["url"]
                else (r.get("error") or "empty content")
            ),
        }
        for r in results
        if not (r.get("ok") and r.get("content"))
    ]

    if not successful:
        return {"error": "no content fetched from provided URLs", "failed": failed}

    if full_text:
        ranked = full_text_of(successful)
    else:
        ranked = rank_chunks(
            successful,
            query.strip(),
            max_chunks=max(1, min(int(max_chunks), 30)),
            offset=max(0, int(offset)),
        )
    ranked["pages"] = _page_summaries(successful, include_links)
    if failed:
        ranked["failed"] = failed
    await _progress(ctx, 2, 2, "done")
    return ranked


@mcp.tool(
    description=(
        "One-shot web research: searches DuckDuckGo, fetches the top pages with a headless "
        "browser, and returns the passages most relevant to the query, with source citations. "
        "Use this when you want answers rather than a link list."
    ),
    annotations={**READ_ONLY_WEB, "idempotentHint": False, "title": "Research a question"},
)
async def research(
    query: str,
    max_results: Annotated[int, "How many search results to consider (1-20)"] = 6,
    fetch_top: Annotated[int, "How many of the top-ranked pages to actually read (1-10)"] = 3,
    kind: Annotated[str, "'web' for general search, 'news' for dated news results"] = "web",
    timelimit: Annotated[str | None, "Recency filter: 'd' day, 'w' week, 'm' month, 'y' year"] = None,
    depth: Annotated[int, "Search rounds (1-3). Above 1, gaps in the first round drive a follow-up search"] = 1,
    ctx: Context = None,
):
    query = (query or "").strip()
    if not query:
        return {"error": "query must not be empty"}

    kind = (kind or "web").strip().lower()
    if kind not in {"web", "news"}:
        return {"error": f"unknown kind {kind!r}, expected 'web' or 'news'"}
    max_results = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
    fetch_top = max(1, min(int(fetch_top), 10, max_results))
    depth = max(1, min(int(depth), MAX_RESEARCH_DEPTH))

    all_docs: list[dict[str, Any]] = []
    all_hits: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    hops: list[dict[str, Any]] = []
    hop_query = query
    total_steps = depth * 2 + 1

    for hop in range(1, depth + 1):
        await _progress(ctx, (hop - 1) * 2, total_steps, f"searching: {hop_query}")
        try:
            raw = await ddg_search(hop_query, max_results, timelimit, kind)
        except Exception as e:
            if all_docs:
                break  # keep what earlier hops found
            return {"error": "search_failed", "detail": str(e)}

        hits = _normalize_hits(raw, max_results)
        # Renumber across hops so citation ids stay unique
        ranked_hits = rank_hits(hits)
        for h in ranked_hits:
            h["id"] = f"S{len(all_hits) + 1}"
            h["citation"] = f"[{h['id']}] {h['title']} - {h['url']}"
            h["hop"] = hop
            if h["url"] not in seen_urls:
                all_hits.append(h)

        fresh = [h for h in ranked_hits if h["url"] not in seen_urls]
        fetchable, hop_blocked = await _screen_urls([h["url"] for h in fresh])
        blocked.extend(hop_blocked)
        candidates = [h for h in fresh if h["url"] in set(fetchable)]
        targets = await triage_results(ctx, hop_query, candidates, fetch_top)

        hops.append({"hop": hop, "query": hop_query, "read": [h["url"] for h in targets]})
        if targets:
            await _progress(ctx, (hop - 1) * 2 + 1, total_steps, f"reading {len(targets)} page(s)")
            docs = await crawl([h["url"] for h in targets])
            titles = {h["url"]: h["title"] for h in targets}
            for d in docs:
                seen_urls.add(d["url"])
                if not d.get("title"):
                    d["title"] = titles.get(d["url"], "")
                if d.get("ok") and d.get("content"):
                    all_docs.append(d)

        if hop >= depth:
            break
        # Ask whether anything is still missing; no answer means stop here
        follow_up = await assess_coverage(ctx, query, all_docs)
        if not follow_up:
            if depth > 1:
                print(f"[research] stopping after hop {hop} — coverage judged sufficient", flush=True, file=sys.stderr)
            break
        hop_query = follow_up

    sources = _build_sources(all_hits, all_docs)
    if not all_hits:
        return {"query": query, "results": [], "sources": []}
    if not all_docs:
        # Search worked, reading did not — snippets still beat nothing
        return {
            "query": query,
            "results": [],
            "sources": sources,
            "hops": hops,
            "note": "No page content could be fetched; falling back to search snippets.",
        }

    await _progress(ctx, total_steps - 1, total_steps, "ranking passages")
    variants = await expand_query(ctx, query)
    ranked = rank_chunks(all_docs, query, query_variants=variants)
    ranked["query"] = query
    ranked["sources"] = sources
    ranked["pages"] = _page_summaries(all_docs)
    if len(hops) > 1:
        ranked["hops"] = hops
    if blocked:
        ranked["failed"] = blocked
    await _progress(ctx, total_steps, total_steps, "done")
    return ranked


# ---- RUN ----
def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
