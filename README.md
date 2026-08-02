# MCP Browser Server

Web search and content retrieval MCP server. DuckDuckGo for discovery, Crawl4AI
(headless Chromium) for retrieval, BM25 relevance ranking on top. No search
API keys required.

Known limitations and library quirks are documented in [CAVEATS.md](CAVEATS.md).

## Tools

- **`search`** — DuckDuckGo results. Titles, URLs, snippets, engine rank. `kind='news'`
  queries the news index instead, which carries real publish dates.
- **`fetch`** — Fetch URLs with a real browser → clean markdown; chunked and BM25-ranked.
  Handles PDFs, returns outbound links, supports `offset` paging and `full_text`.
- **`research`** — search + fetch + rank in one call, with citations. `depth>1` runs
  follow-up searches to close gaps the first round left.

## Resources

- `cache://pages` — what is currently cached, with a resource URI per page.
- `cache://page/{key}` — full markdown of an already-fetched page, no re-fetch and no
  tool-response cost. Each `pages[]` entry carries its own `resource` URI.

## Pipeline

1. **Search** — DuckDuckGo via `ddgs`, with backoff on rate limits
2. **Screen** — URLs resolving to non-public addresses are refused (see Network safety)
3. **Route** — `application/pdf` (detected by HEAD, not by file extension) goes to a
   browser-free PDF extractor; everything else to the warm Chromium
4. **Chunk** — markdown-section split with overlap, small chunks merged
5. **Rank** — BM25, near-duplicates dropped, balanced across sources, within a budget

The browser is kept warm between calls and shut down after `BROWSER_IDLE_TIMEOUT`
seconds idle. Fetches are streamed, so a slow page that trips `FETCH_TIMEOUT` costs
only itself. Successful fetches are cached for `PAGE_CACHE_TTL` seconds.

### Ranking notes

Chunks are scored with Okapi BM25 (`k1=1.5`, `b=0.75`) over stemmed, stopword-filtered
tokens. Every document is chunked, including short ones, so passages compete at
comparable lengths. Selection is round-robin across sources, capped by
`MAX_CONTENT_LENGTH`. Near-duplicate passages are collapsed, and the survivor records
the other URLs it appeared on in `also_in` — duplication becomes a corroboration signal.

With query variants (from sampling), rankings are fused with RRF rather than averaged,
because BM25 scores from different phrasings are not on a common scale.

Being purely lexical, it inherits the usual limits: a passage consisting of the query's
own words repeated can still outrank a real answer. A repetition damper reduces this for
the boilerplate scraped markdown actually produces, but deliberate keyword stuffing is
not something a bag-of-words ranker resolves.

## Sampling

`research` can borrow the **client's** model — no API key here — to expand the query,
pick which results are worth fetching, and judge whether the evidence answers the
question. Clients without sampling support fall back to engine rank and plain BM25,
and `depth>1` collapses to a single hop. It is an enhancement, never a dependency.

## Network safety

`fetch` takes URLs chosen by the model and `research` takes them from search results,
so both are treated as untrusted input. Before fetching, each hostname is resolved and
refused if it maps to a loopback, private, link-local, or otherwise non-public address —
which is what keeps the server from being used to read cloud metadata endpoints
(`169.254.169.254`) or internal services. Redirects into that space are caught after the
fetch and the content withheld. Set `ALLOW_PRIVATE_URLS=true` only if reaching your
internal network is the point.

The compose file publishes to `127.0.0.1` only. If you expose it more widely, set
`AUTH_TOKEN` — the server logs a warning at startup if you do not.

**Credentials** for sites behind a login are configured on the server
(`FETCH_CREDENTIALS` / `FETCH_CREDENTIALS_FILE`), keyed by host, and are never accepted
as tool arguments — a model that could name its own headers, on a server that already
fetches arbitrary URLs, would be an exfiltration primitive. They are withheld from
plaintext `http://`, and a credentialed request that redirects off-host has its content
withheld. Credentialed hosts each get a throwaway browser, since Crawl4AI only accepts
extra headers at browser launch; reusing the pooled one would leak them into every
later fetch.

## Setup

```bash
cp .env.template .env

docker compose up -d --build
```

Server runs on **127.0.0.1:13010** (MCP HTTP transport at `/mcp`).

The build installs Chromium via `crawl4ai-setup` and verifies it with a real crawl
in the final image, so a broken browser stack fails the build rather than the first
request. `shm_size: 1gb` and `init: true` in the compose file are required for
Chromium stability inside containers.

## Configuration

All optional — see `.env.template` for the full list.

| Env Var | Default | Description |
|---|---|---|
| `SEARCH_REGION` | `us-en` | DuckDuckGo region |
| `SEARCH_SAFESEARCH` | `moderate` | `on` / `moderate` / `off` |
| `SEARCH_BACKEND` | `auto` | `ddgs` backend selection |
| `FETCH_CONCURRENCY` | `4` | Pages crawled in parallel |
| `PAGE_TIMEOUT_MS` | `25000` | Per-page navigation timeout |
| `FETCH_TIMEOUT` | `120` | Whole-batch fetch ceiling (s) |
| `CHECK_ROBOTS_TXT` | `false` | Honor robots.txt before fetching |
| `BROWSER_IDLE_TIMEOUT` | `300` | Close idle browser after (s) |
| `MEMORY_THRESHOLD_PERCENT` | `85` | Dispatcher backs off above this |
| `PAGE_CACHE_TTL` | `900` | Cache lifetime (s); `0` disables |
| `PDF_ENABLED` | `true` | Route PDFs to the PDF extractor |
| `MAX_LINKS_PER_PAGE` | `25` | Outbound links returned per page |
| `DEDUPE_THRESHOLD` | `0.8` | Near-duplicate cutoff; `1.0` disables |
| `RATE_LIMIT_ENABLED` | `true` | Per-domain spacing and backoff |
| `SAMPLING_ENABLED` | `true` | Allow borrowing the client's model |
| `MAX_RESEARCH_DEPTH` | `3` | Ceiling on `research(depth=…)` |
| `ALLOW_PRIVATE_URLS` | `false` | Permit fetching non-public addresses |
| `BIND_HOST` | `0.0.0.0` | Bind address *inside* the container |
| `AUTH_TOKEN` | *(unset)* | Bearer token; required beyond localhost |

`BIND_HOST` is namespaced deliberately — a bare `HOST` is already set in many shells
and CI images, which would silently break the bind.

`MEMORY_THRESHOLD_PERCENT` is measured by Crawl4AI against **host** RAM, not the
container's `mem_limit`. If you lower `mem_limit`, lower this too, or the dispatcher
will keep launching pages until the container is OOM-killed.

`CHECK_ROBOTS_TXT` defaults to `false`. Turning it on is the polite choice for
unattended crawling; it will also cause some sites to refuse.

## Tuning

Edit `config.py` constants:

| Constant | Default | Description |
|---|---|---|
| `MAX_RETURNED_CHUNKS` | 10 | Default passages per call (`max_chunks` overrides) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 200 | Chunking granularity |
| `MAX_CONTENT_LENGTH` | 30000 | Ceiling on total characters returned per call |
| `MAX_URLS_PER_CALL` | 20 | URLs accepted by one `fetch` |

`MAX_URLS_PER_CALL` at `FETCH_CONCURRENCY=4` and `PAGE_TIMEOUT_MS=25000` is a ~125s
worst case against a 120s `FETCH_TIMEOUT`. That no longer loses the batch — slow pages
are dropped individually — but raise `FETCH_TIMEOUT` if you routinely fetch 20 URLs.
Per-domain rate limiting adds further delay when a batch targets one host.
