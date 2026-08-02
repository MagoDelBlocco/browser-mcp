# Caveats

Known limitations and surprising behaviour. Verified against crawl4ai 0.9.2,
fastmcp 3.4.5, ddgs 9.14.4.

## Ranking

**Keyword stuffing beats real answers.** BM25 is lexical: a passage made of the
query's own words repeated can outrank one that answers it. `_repetition_penalty`
damps this ~2.5x for the boilerplate scraped markdown actually produces (link dumps,
nav labels), but deliberate stuffing is not solvable by a bag-of-words ranker. The
realistic case — a relevant passage buried in a long page — ranks correctly.

**Dedup is conservative.** `DEDUPE_THRESHOLD=0.8` merges near-identical passages, but
syndicated copies with light editorial edits (a swapped word per paragraph) survive as
separate results. Deliberate: failing to merge is a cheaper error than hiding distinct
content. Lower the threshold if duplicates are eating result slots.

## Dates

**JSON-LD extraction never fires in practice.** `_date_from_jsonld` is implemented and
unit-tested, but `res.html` arrives already stripped (~5KB for a full article), so the
`<script type="application/ld+json">` block is gone before we see it. Dates come from
meta tags, epoch timestamps, URL paths and PDF metadata instead. Recovering JSON-LD
would need a separate raw fetch or a fetch config that degrades markdown quality.

**Month precision is real.** A URL like `/2024/09/slug` yields `"2024-09"`, not a
fabricated day. Consumers must handle both `YYYY-MM` and `YYYY-MM-DD`.

**Coverage is partial.** Many pages genuinely publish no date; `published` is absent
rather than guessed. JS-heavy aggregators (MSN and similar) typically yield nothing.

## PDFs

**crawl4ai reports `success=False` for every PDF.** Its anti-bot check inspects the
HTTP stub, not the extracted text, so a 42k-character paper arrives flagged
`"Blocked by anti-bot protection: Near-empty content (33 bytes)"`. `_crawl_pdfs`
therefore judges success by extracted content length and ignores `res.success`.
If a future crawl4ai fixes this, that override becomes redundant but stays correct.

**Requires the `crawl4ai[pdf]` extra.** Without pypdf, PDF URLs return a clear error
rather than crashing. Detection is a HEAD probe, so it costs one extra request per
uncached URL and falls back to the `.pdf` suffix when a server refuses HEAD.

**No credentialed PDFs.** The PDF path does not carry `FETCH_CREDENTIALS` headers.

## Sampling and multi-hop

**`depth>1` needs a sampling-capable client.** Coverage assessment is what decides
whether another hop is warranted; with no model to judge, extra hops are skipped and
`research` runs a single pass. Same for query expansion (falls back to plain BM25) and
result triage (falls back to engine rank). All degrade silently by design — check the
`ranking` field (`"rrf"` vs `"bm25"`) and the presence of `hops` to tell which ran.

**Multi-hop multiplies cost.** `depth=3` can mean 3 searches, 3 fetch batches and 4
sampling round-trips. `MAX_RESEARCH_DEPTH` caps it.

## Network safety

**DNS rebinding is not fully closed.** `_resolve_block_reason` resolves the hostname,
then the browser resolves it again independently. A hostile DNS server can return a
public address to the first and a private one to the second. The post-fetch redirect
check withholds the body, but the request itself was made.

**Credentials are server-configured only**, never tool arguments — a model that could
name its own headers on a server that fetches arbitrary URLs is an exfiltration
primitive. Credentialed hosts each get a throwaway browser, because crawl4ai accepts
extra headers only at browser launch; reusing the pooled browser would attach them to
every unrelated fetch while it stayed warm. Cost: one browser launch (~1-2s) per
credentialed host per call.

**`CHECK_ROBOTS_TXT` defaults to `false`.** Turning it on is the polite choice for
unattended crawling and will also cause some sites to refuse.

## Operations

**`BIND_HOST`, not `HOST`.** A bare `HOST` is already set in many shells and CI images
(often to the machine name), which would silently break the bind.

**`MEMORY_THRESHOLD_PERCENT` reads host RAM, not the container limit.** Crawl4AI's
dispatcher is not cgroup-aware, so lowering `mem_limit` without lowering this invites
an OOM kill instead of backpressure.

**Timing budgets interact.** `MAX_URLS_PER_CALL=20` at `FETCH_CONCURRENCY=4` and
`PAGE_TIMEOUT_MS=25000` is a ~125s worst case against a 120s `FETCH_TIMEOUT`. Streaming
means slow pages are dropped individually rather than losing the batch, but per-domain
rate limiting adds further delay when a batch targets one host.

**The cache is in-process.** It does not survive a restart and is not shared across
replicas. `PAGE_CACHE_TTL=0` disables it.

## Verification status

34 automated checks pass, plus live tests against arxiv, python.org, arstechnica,
httpbin and DuckDuckGo news.

**The Docker build has not been run** since `crawl4ai[pdf]` was added. Pinned versions
(`fastmcp==3.4.5`, `ddgs==9.14.4`, `crawl4ai[pdf]==0.9.2`) match what was verified
locally, not what a built image reported. Run `docker compose up -d --build` before
trusting it.
