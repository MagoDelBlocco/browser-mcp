"""Retrieval layer: warm-browser crawling, PDF extraction, page cache.

This module owns everything that actually fetches a URL — the persistent
Chromium pool, the HTML/PDF/authenticated crawl paths, the per-domain
dispatcher, and the in-process page cache. It imports URL-safety and
credential helpers from `safety` and produces the `record` dicts that the
server's tools then chunk and rank.
"""
import asyncio
import contextlib
import hashlib
import re
import time
import urllib.request
from collections import OrderedDict
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urlparse

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    MemoryAdaptiveDispatcher,
    PruningContentFilter,
    RateLimiter,
)

from config import (
    ALLOW_PRIVATE_URLS,
    BROWSER_IDLE_TIMEOUT,
    CHECK_ROBOTS_TXT,
    CONTENT_TYPE_PROBE_TIMEOUT,
    FETCH_CONCURRENCY,
    FETCH_TIMEOUT,
    MAX_LINKS_PER_PAGE,
    MEMORY_THRESHOLD_PERCENT,
    PAGE_CACHE_MAX_ENTRIES,
    PAGE_CACHE_TTL,
    PAGE_TIMEOUT_MS,
    PDF_ENABLED,
    RATE_LIMIT_BACKOFF_CAP,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_MAX_DELAY,
    RATE_LIMIT_MIN_DELAY,
    USER_AGENT,
)
from safety import (
    _credentials,
    _credentials_for,
    _is_valid_http_url,
    _resolve_block_reason,
    _sanitize_snippet,
)


# ---- PERSISTENT BROWSER ----
class CrawlerPool:
    """Keeps one headless Chromium warm across requests, shutting it down when idle.

    Browser startup costs ~1-2s, so paying it per tool call dominates fast fetches.
    In-flight crawls hold a lease; the idle timer only arms once the last one finishes.
    """

    def __init__(self):
        self._crawler: AsyncWebCrawler | None = None
        self._leases = 0
        self._poisoned = False
        self._lock = asyncio.Lock()
        self._idle_timer: asyncio.Task | None = None

    def _cancel_idle(self):
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    async def _idle_shutdown(self):
        await asyncio.sleep(BROWSER_IDLE_TIMEOUT)
        async with self._lock:
            if self._leases == 0 and self._crawler is not None:
                print("[browser] idle — closing", flush=True)
                await self._close_locked()

    async def _close_locked(self):
        crawler, self._crawler = self._crawler, None
        self._poisoned = False
        if crawler is not None:
            with contextlib.suppress(Exception):
                await crawler.close()

    async def _release(self, failed: bool):
        async with self._lock:
            self._leases -= 1
            if failed:
                # A crashed browser poisons every later call, but other crawls may
                # still be using this instance — the last one out closes it.
                self._poisoned = True
            if self._leases == 0:
                self._cancel_idle()
                if self._poisoned:
                    await self._close_locked()
                else:
                    self._idle_timer = asyncio.create_task(self._idle_shutdown())

    @contextlib.asynccontextmanager
    async def lease(self):
        async with self._lock:
            self._cancel_idle()
            if self._poisoned and self._leases == 0:
                await self._close_locked()
            if self._crawler is None:
                print("[browser] launching chromium", flush=True)
                t0 = time.perf_counter()
                crawler = AsyncWebCrawler(config=BROWSER_CFG)
                await crawler.start()
                self._crawler = crawler
                print(f"[browser] ready in {time.perf_counter() - t0:.2f}s", flush=True)
            self._leases += 1
            crawler = self._crawler
        try:
            yield crawler
        except Exception:
            await self._release(failed=True)
            raise
        else:
            await self._release(failed=False)

    async def shutdown(self):
        async with self._lock:
            self._cancel_idle()
            await self._close_locked()


BROWSER_CFG = BrowserConfig(
    browser_type="chromium",
    headless=True,
    light_mode=True,   # disables background renderer features we never read
    text_mode=True,    # skips images/fonts — the single biggest fetch speedup
    user_agent=USER_AGENT,
    verbose=False,
    extra_args=[
        # Required when running as root inside a container
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-extensions",
        "--disable-background-networking",
        "--mute-audio",
    ],
)

_crawler_pool = CrawlerPool()


def _make_dispatcher() -> MemoryAdaptiveDispatcher:
    """Concurrency cap plus per-domain spacing, shared by the HTML and PDF paths.

    The rate limiter also backs off on 429/503, which is what keeps a batch from
    turning into a hammering loop against a host that is already asking us to stop.
    """
    rate_limiter = None
    if RATE_LIMIT_ENABLED:
        rate_limiter = RateLimiter(
            base_delay=(RATE_LIMIT_MIN_DELAY, RATE_LIMIT_MAX_DELAY),
            max_delay=RATE_LIMIT_BACKOFF_CAP,
            max_retries=2,
        )
    return MemoryAdaptiveDispatcher(
        memory_threshold_percent=MEMORY_THRESHOLD_PERCENT,
        max_session_permit=max(1, FETCH_CONCURRENCY),
        rate_limiter=rate_limiter,
    )


def _run_config(stream: bool = False) -> CrawlerRunConfig:
    """Main-content extraction tuned for LLM consumption (no nav/ads/link soup)."""
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.48, threshold_type="dynamic"),
            options={"ignore_links": True, "ignore_images": True, "body_width": 0},
        ),
        word_count_threshold=10,
        excluded_tags=["nav", "header", "footer", "aside", "form", "script", "style", "noscript"],
        exclude_all_images=True,
        # Links are collected, not rendered: `ignore_links` above keeps them out of the
        # markdown (no link soup for the model to wade through) while `result.links`
        # still gets populated. Excluding them here would zero out that list, leaving an
        # agent unable to see where a page points.
        exclude_external_links=False,
        exclude_social_media_links=False,
        remove_overlay_elements=True,
        wait_until="domcontentloaded",
        page_timeout=PAGE_TIMEOUT_MS,
        check_robots_txt=CHECK_ROBOTS_TXT,
        scan_full_page=False,
        stream=stream,
        verbose=False,
    )


# ---- FETCH RECORD HELPERS ----
def _markdown_of(result) -> str:
    """Prefer pruned main content, fall back to raw when pruning was too aggressive."""
    md = getattr(result, "markdown", None)
    if md is None:
        return ""
    fit = (getattr(md, "fit_markdown", None) or "").strip()
    raw = (getattr(md, "raw_markdown", None) or "").strip()
    if not fit and not raw:
        return str(md).strip()
    if fit and len(fit) >= max(200, 0.15 * len(raw)):
        return fit
    return raw or fit


# Ordered by trustworthiness: an explicit article timestamp beats a generic "date"
# meta tag, which beats a modification time.
_DATE_META_KEYS = (
    "article:published_time",
    "datepublished",
    "og:published_time",
    "citation_publication_date",
    "publishdate",
    "publish_date",
    "pubdate",
    "dc.date.issued",
    "dcterms.created",
    "dc.date",
    "date",
    "created",
    "article:modified_time",
    "og:updated_time",
    "lastmod",
)

_DATE_PATTERN = re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})")
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_URL_DATE_PATTERN = re.compile(
    r"/(20\d{2}|19\d{2})/(\d{1,2}|[A-Za-z]{3,9})(?:/(\d{1,2}))?(?:[/?#]|$)")
_JSONLD_PATTERN = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.S | re.I)


def _normalize_date(value: Any) -> str | None:
    """Best-effort ISO date (YYYY-MM-DD) from the many shapes metadata uses."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    # og:updated_time and friends are frequently raw epoch seconds (or millis)
    if text.isdigit() and len(text) in (10, 13):
        with contextlib.suppress(ValueError, OSError, OverflowError):
            seconds = int(text) / (1000 if len(text) == 13 else 1)
            return datetime.fromtimestamp(seconds, tz=UTC).date().isoformat()
    with contextlib.suppress(ValueError):
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    match = _DATE_PATTERN.search(text)
    if match:
        year, month, day = (int(g) for g in match.groups())
        with contextlib.suppress(ValueError):
            return date(year, month, day).isoformat()
    return None


def _date_from_jsonld(html: str) -> str | None:
    """datePublished out of JSON-LD, where most real article sites actually keep it.

    Crawl4AI's metadata only covers a handful of meta tags; arstechnica, the Guardian
    and similar expose nothing useful there but do carry schema.org blocks.
    Regex rather than a JSON parse: these blocks are frequently malformed or truncated.
    """
    if not html:
        return None
    blocks = _JSONLD_PATTERN.findall(html)[:10]
    for field in ("datePublished", "dateCreated", "uploadDate", "dateModified"):
        pattern = re.compile(rf'"{field}"\s*:\s*"([^"]{{4,40}})"', re.I)
        for block in blocks:
            match = pattern.search(block)
            if match:
                normalized = _normalize_date(match.group(1))
                if normalized:
                    return normalized
    return None


def _date_from_url(url: str) -> str | None:
    """Date embedded in the path, e.g. /2024/09/12/, /2024/Dec/31/ or /2024/09/.

    Last resort, but a reliable one: publishing platforms put the date there and it
    cannot drift the way a "last updated" banner does. Paths carrying only a year and
    month yield a month-precision "YYYY-MM" rather than a fabricated day.
    """
    match = _URL_DATE_PATTERN.search(url or "")
    if not match:
        return None
    year, raw_month, raw_day = match.groups()
    month = _MONTHS.get(raw_month[:3].lower()) if raw_month[0].isalpha() else int(raw_month)
    if not month or not 1 <= month <= 12:
        return None
    if raw_day is None:
        return f"{int(year):04d}-{month:02d}"
    with contextlib.suppress(ValueError):
        return date(int(year), month, int(raw_day)).isoformat()
    return None


def _published_date(metadata: dict[str, Any] | None, html: str = "", url: str = "") -> str | None:
    """Publish date for a page, or None when nothing credible is available.

    Version-sensitive answers hinge on this — without it a model cannot tell a 2019
    page from last week's. Sources are tried most to least trustworthy.
    """
    lowered = {str(k).lower(): v for k, v in (metadata or {}).items()}
    for key in _DATE_META_KEYS:
        normalized = _normalize_date(lowered.get(key))
        if normalized:
            return normalized
    return _date_from_jsonld(html) or _date_from_url(url)


def _extract_links(res, limit: int = None) -> list[dict[str, Any]]:
    """Deduplicated outbound links, internal first, http(s) only.

    Internal first because that is what an agent walking a documentation site or
    following pagination actually needs; `javascript:;` and `mailto:` are dropped.
    """
    limit = MAX_LINKS_PER_PAGE if limit is None else limit
    if limit <= 0:
        return []
    grouped = getattr(res, "links", None) or {}
    buckets: dict[str, list[dict[str, Any]]] = {"internal": [], "external": []}
    seen: set[str] = set()
    for kind in ("internal", "external"):
        for item in grouped.get(kind) or []:
            href = (item.get("href") or "").strip()
            if not href or href in seen or not _is_valid_http_url(href):
                continue
            seen.add(href)
            buckets[kind].append({
                "url": href,
                "text": _sanitize_snippet(item.get("text"))[:120],
                "internal": kind == "internal",
            })

    internal, external = buckets["internal"], buckets["external"]
    # Half the budget is reserved for external links. Filling internal-first starves
    # them completely on any real site — python.org/doc has 101 internal to 14
    # external, and the citations off-site are usually the interesting ones.
    ext_take = min(len(external), max(limit - len(internal), limit // 2))
    int_take = limit - ext_take
    if len(internal) < int_take:
        int_take = len(internal)
        ext_take = min(len(external), limit - int_take)
    return internal[:int_take] + external[:ext_take]


def _record_of(res) -> dict[str, Any]:
    ok = bool(getattr(res, "success", False))
    metadata = getattr(res, "metadata", None) or {}
    return {
        "url": getattr(res, "url", "") or "",
        "ok": ok,
        "status": getattr(res, "status_code", None),
        "title": (metadata.get("title") or "").strip(),
        "content": _markdown_of(res) if ok else "",
        "error": None if ok else getattr(res, "error_message", None),
        "content_type": "html",
        "links": _extract_links(res) if ok else [],
        "published": _published_date(metadata, getattr(res, "html", "") or "",
                                     getattr(res, "url", "") or "") if ok else None,
        "author": _sanitize_snippet(metadata.get("author")) or None,
    }


# ---- PAGE CACHE ----
class TTLCache:
    """Small LRU cache of fetched pages with a wall-clock TTL.

    Deliberately in-process rather than Crawl4AI's `CacheMode.ENABLED`: this covers
    the PDF path on the same terms as the HTML one, gives an explicit TTL, and does
    not grow a SQLite file on disk.
    """

    def __init__(self, ttl: int, max_entries: int):
        self.ttl = ttl
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        if self.ttl <= 0:
            return None
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        stored_at, record = entry
        if time.monotonic() - stored_at > self.ttl:
            del self._entries[key]
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return record

    def put(self, key: str, record: dict[str, Any]) -> None:
        if self.ttl <= 0:
            return
        self._entries[key] = (time.monotonic(), record)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()


_page_cache = TTLCache(PAGE_CACHE_TTL, PAGE_CACHE_MAX_ENTRIES)


def _resource_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


# ---- CONTENT TYPE ROUTING ----
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Redirects are not followed while probing — a 3xx target has not been screened."""

    def redirect_request(self, *args, **kwargs):
        return None


def _probe_content_type(url: str) -> str:
    """Content-Type via HEAD, or "" when the server will not say.

    A URL-suffix guess is not enough: arxiv serves PDFs from extensionless paths
    like /pdf/1706.03762, and plenty of ".pdf" links redirect to landing pages.
    """
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=CONTENT_TYPE_PROBE_TIMEOUT) as response:
            return (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    except Exception:
        return ""


def _looks_like_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


async def _classify_urls(urls: list[str]) -> tuple[list[str], list[str]]:
    """Split into (html_urls, pdf_urls). Probes run concurrently, off-thread."""
    if not PDF_ENABLED:
        return list(urls), []
    types = await asyncio.gather(*(asyncio.to_thread(_probe_content_type, u) for u in urls))
    html_urls, pdf_urls = [], []
    for url, ctype in zip(urls, types):
        is_pdf = ctype == "application/pdf" or (not ctype and _looks_like_pdf_url(url))
        (pdf_urls if is_pdf else html_urls).append(url)
    return html_urls, pdf_urls


async def _crawl_pdfs(urls: list[str]) -> dict[str, dict[str, Any]]:
    """Extract text from PDF URLs. No browser involved — these never touch the pool."""
    try:
        from crawl4ai.processors.pdf import PDFContentScrapingStrategy, PDFCrawlerStrategy
    except ImportError as e:
        return {u: {"url": u, "ok": False, "content": "",
                    "error": f"PDF support unavailable, install crawl4ai[pdf]: {e}"} for u in urls}

    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        scraping_strategy=PDFContentScrapingStrategy(),
        stream=False,
        verbose=False,
    )
    out: dict[str, dict[str, Any]] = {}
    try:
        async with AsyncWebCrawler(crawler_strategy=PDFCrawlerStrategy()) as crawler:
            results = await asyncio.wait_for(
                crawler.arun_many(urls, config=run_cfg, dispatcher=_make_dispatcher()),
                timeout=FETCH_TIMEOUT,
            )
    except Exception as e:
        print(f"[pdf] extraction failed: {e}", flush=True)
        return {u: {"url": u, "ok": False, "content": "", "error": str(e)} for u in urls}

    for res in results:
        content = _markdown_of(res)
        metadata = getattr(res, "metadata", None) or {}
        # crawl4ai 0.9.2 flags every PDF as "Blocked by anti-bot protection: Near-empty
        # content (33 bytes)" because it inspects the HTTP stub, not the extracted text.
        # A 42k-character paper arrives with success=False, so content is the real signal.
        ok = bool(content.strip())
        url = getattr(res, "url", "") or ""
        out[url] = {
            "url": url,
            "ok": ok,
            "status": getattr(res, "status_code", None),
            "title": (metadata.get("title") or "").strip(),
            "content": content,
            "error": None if ok else (getattr(res, "error_message", None) or "no text extracted"),
            "content_type": "pdf",
            "links": [],
            # PDF metadata dates arrive as datetime objects, not strings
            "published": _published_date(metadata, "", url),
            "author": _sanitize_snippet(metadata.get("author")) or None,
        }
    return out


async def _reject_redirected_to_private(records: list[dict[str, Any]]) -> None:
    """Drop content from pages that redirected into non-public address space.

    The pre-fetch screen only sees the URL we were given; a public host is free to
    302 into 169.254.169.254. Checked in place, after the fact — the fetch already
    happened, but the body never reaches the caller.
    """
    if ALLOW_PRIVATE_URLS:
        return
    suspect = [r for r in records if r.get("final_url") and r.get("ok")]
    if not suspect:
        return
    reasons = await asyncio.gather(
        *(asyncio.to_thread(_resolve_block_reason, r["final_url"]) for r in suspect)
    )
    for record, reason in zip(suspect, reasons):
        if reason is not None:
            print(f"[fetch] blocked redirect {record['url']} -> {record['final_url']}: {reason}", flush=True)
            record.update(ok=False, content="", error=f"redirected to blocked address: {reason}")


async def crawl(urls: list[str]) -> list[dict[str, Any]]:
    """Fetch URLs, returning one record per requested URL, in order.

    HTML goes through the warm browser; PDFs take a separate, browser-free path.
    The two run concurrently.
    """
    by_url: dict[str, dict[str, Any]] = {}
    to_fetch: list[str] = []
    for u in urls:
        hit = _page_cache.get(u)
        if hit is not None:
            by_url[u] = {**hit, "cached": True}
        else:
            to_fetch.append(u)
    if len(to_fetch) < len(urls):
        print(f"[cache] {len(urls) - len(to_fetch)}/{len(urls)} served from cache", flush=True)

    if to_fetch:
        html_urls, pdf_urls = await _classify_urls(to_fetch)
        if pdf_urls:
            print(f"[fetch] routing {len(pdf_urls)} url(s) to the PDF extractor", flush=True)

        # Credentialed hosts each need their own browser, so they are grouped and run
        # apart from the pooled anonymous fetches.
        public_urls: list[str] = []
        by_credential: dict[str, list[str]] = {}
        for url in html_urls:
            host, _ = _credentials_for(url)
            if host is None:
                public_urls.append(url)
            else:
                by_credential.setdefault(host, []).append(url)

        tasks = []
        if public_urls:
            tasks.append(_crawl_html(public_urls))
        if pdf_urls:
            tasks.append(_crawl_pdfs(pdf_urls))
        for host, host_urls in by_credential.items():
            print(f"[auth] fetching {len(host_urls)} url(s) for {host}", flush=True)
            tasks.append(_crawl_authenticated(host_urls, _credentials[host]))
        for result in await asyncio.gather(*tasks):
            by_url.update(result)

    out = []
    for u in urls:
        record = by_url.get(u)
        if record is None:
            out.append({"url": u, "ok": False, "content": "", "error": "no result returned (timeout or blocked)"})
        elif record["url"] != u:
            out.append({**record, "url": u, "final_url": record.get("final_url") or record["url"]})
        else:
            out.append(record)
    await _reject_redirected_to_private(out)
    # Cached after the redirect screen, so content that was just withheld for
    # pointing into private space cannot be served from cache next time.
    for record in out:
        if record.get("ok") and record.get("content") and not record.get("cached"):
            _page_cache.put(record["url"], record)
    return out


async def _crawl_authenticated(urls: list[str], headers: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Fetch one credentialed host's URLs in a throwaway browser.

    Crawl4AI takes extra headers only on BrowserConfig, at launch — there is no
    per-request header. Reusing the pooled browser would therefore attach these
    credentials to every unrelated fetch for as long as it stayed warm, so this pays
    a browser launch instead. Rare enough not to matter.
    """
    browser_cfg = BrowserConfig(
        browser_type="chromium",
        headless=True,
        light_mode=True,
        text_mode=True,
        user_agent=USER_AGENT,
        headers=dict(headers),
        verbose=False,
        extra_args=list(BROWSER_CFG.extra_args or []),
    )
    out: dict[str, dict[str, Any]] = {}
    try:
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            results = await asyncio.wait_for(
                crawler.arun_many(urls, config=_run_config(), dispatcher=_make_dispatcher()),
                timeout=FETCH_TIMEOUT,
            )
    except Exception as e:
        print(f"[auth] credentialed fetch failed: {e}", flush=True)
        return {u: {"url": u, "ok": False, "content": "", "error": str(e)} for u in urls}

    expected_hosts = {(urlparse(u).hostname or "").lower() for u in urls}
    for res in results:
        record = _record_of(res)
        redirected = getattr(res, "redirected_url", None)
        if redirected and redirected != record["url"]:
            record["final_url"] = redirected
        # A credentialed request that redirected off-host may have carried the
        # Authorization header along with it. Withhold rather than hand back a body
        # obtained under those conditions.
        final_host = (urlparse(record.get("final_url") or record["url"]).hostname or "").lower()
        if record["ok"] and final_host not in expected_hosts:
            print(f"[auth] {record['url']} redirected off-host to {final_host}; withholding", flush=True)
            record.update(ok=False, content="", error="credentialed request redirected off-host")
        record["authenticated"] = True
        out[record["url"]] = record
        if redirected:
            out.setdefault(redirected, record)
    return out


async def _crawl_html(urls: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch HTML pages concurrently through the warm browser, keyed by result URL."""
    run_cfg = _run_config(stream=True)
    dispatcher = _make_dispatcher()

    t0 = time.perf_counter()
    deadline = t0 + FETCH_TIMEOUT
    by_url: dict[str, dict[str, Any]] = {}
    timed_out = False
    try:
        async with _crawler_pool.lease() as crawler:
            # Streamed so a single slow page costs only itself: results already
            # delivered are kept when the batch deadline hits.
            results = await crawler.arun_many(urls, config=run_cfg, dispatcher=dispatcher)
            try:
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        res = await asyncio.wait_for(results.__anext__(), timeout=remaining)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        timed_out = True
                        break
                    record = _record_of(res)
                    redirected = getattr(res, "redirected_url", None)
                    if redirected and redirected != record["url"]:
                        # Recorded here, not just when the result URL differs from the
                        # requested one, so the post-fetch address check always sees it
                        record["final_url"] = redirected
                    by_url[record["url"]] = record
                    if redirected:
                        by_url.setdefault(redirected, record)
            finally:
                with contextlib.suppress(Exception):
                    await results.aclose()
    except Exception as e:
        print(f"[fetch] crawl failed: {e}", flush=True)
        return {u: {"url": u, "ok": False, "content": "", "error": str(e)} for u in urls}

    if timed_out:
        print(f"[fetch] batch deadline {FETCH_TIMEOUT}s hit — {len(by_url)}/{len(urls)} done", flush=True)
    print(f"[fetch] {len(urls)} url(s) in {time.perf_counter() - t0:.2f}s", flush=True)
    return by_url
