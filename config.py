"""Central configuration for the MCP browser server.

All env-derived constants live here so the server and its modules read one
source of truth instead of re-reading `os.environ` (and re-defining the small
`_env_int` / `_env_bool` helpers) in every file. Importing `config` has no
side effects beyond reading environment variables at import time.
"""
import os


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# ---- CONFIG ----
# Search (DuckDuckGo via ddgs)
SEARCH_REGION = os.environ.get("SEARCH_REGION", "us-en")
SEARCH_SAFESEARCH = os.environ.get("SEARCH_SAFESEARCH", "moderate")
SEARCH_BACKEND = os.environ.get("SEARCH_BACKEND", "auto")
SEARCH_RETRIES = _env_int("SEARCH_RETRIES", 3)
MAX_SEARCH_RESULTS = 20

# Fetch (Crawl4AI)
FETCH_CONCURRENCY = _env_int("FETCH_CONCURRENCY", 4)
PAGE_TIMEOUT_MS = _env_int("PAGE_TIMEOUT_MS", 25000, minimum=1000)
FETCH_TIMEOUT = _env_int("FETCH_TIMEOUT", 120)  # whole-batch ceiling, seconds
CHECK_ROBOTS_TXT = _env_bool("CHECK_ROBOTS_TXT", False)
BROWSER_IDLE_TIMEOUT = _env_int("BROWSER_IDLE_TIMEOUT", 300)
MEMORY_THRESHOLD_PERCENT = float(_env_int("MEMORY_THRESHOLD_PERCENT", 85, minimum=10))
MAX_URLS_PER_CALL = 20

# Identify the crawler rather than impersonating a browser, so operators who want
# to block or contact us can.
USER_AGENT = os.environ.get(
    "USER_AGENT",
    "mcp-browser/1.0 (+https://github.com/modelcontextprotocol; automated retrieval)",
)

# PDF handling. Requires the `crawl4ai[pdf]` extra (pypdf); without it PDF URLs
# report a clear error instead of crashing the call.
PDF_ENABLED = _env_bool("PDF_ENABLED", True)
CONTENT_TYPE_PROBE_TIMEOUT = _env_int("CONTENT_TYPE_PROBE_TIMEOUT", 8)

# Per-domain politeness. FETCH_CONCURRENCY alone is a global cap, so 20 URLs from one
# host still arrive 4-at-a-time with no spacing. Delays are per domain, so a batch
# spread across sites is unaffected.
RATE_LIMIT_ENABLED = _env_bool("RATE_LIMIT_ENABLED", True)
RATE_LIMIT_MIN_DELAY = float(os.environ.get("RATE_LIMIT_MIN_DELAY", "0.5"))
RATE_LIMIT_MAX_DELAY = float(os.environ.get("RATE_LIMIT_MAX_DELAY", "1.5"))
RATE_LIMIT_BACKOFF_CAP = float(os.environ.get("RATE_LIMIT_BACKOFF_CAP", "30"))

# Authenticated fetching. Credentials are configured on the server, keyed by host, and
# are never accepted as tool arguments — a model that could name its own headers, on a
# server that already fetches arbitrary URLs, is an exfiltration primitive.
#   FETCH_CREDENTIALS='{"docs.example.com": {"Authorization": "Bearer ..."}}'
#   FETCH_CREDENTIALS_FILE=/run/secrets/credentials.json
FETCH_CREDENTIALS_FILE = os.environ.get("FETCH_CREDENTIALS_FILE", "").strip()
FETCH_CREDENTIALS_RAW = os.environ.get("FETCH_CREDENTIALS", "").strip()
ALLOW_INSECURE_CREDENTIALS = _env_bool("ALLOW_INSECURE_CREDENTIALS", False)

# Sampling borrows the *client's* model for query expansion and result triage, so the
# server needs no API key. Every use degrades to a deterministic fallback.
SAMPLING_ENABLED = _env_bool("SAMPLING_ENABLED", True)
SAMPLING_TIMEOUT = _env_int("SAMPLING_TIMEOUT", 30)
MAX_RESEARCH_DEPTH = _env_int("MAX_RESEARCH_DEPTH", 3)

# Page cache. Agents re-read the same URL constantly — `research` reads a page, then
# the model calls `fetch` on it a turn later. 0 disables.
PAGE_CACHE_TTL = _env_int("PAGE_CACHE_TTL", 900, minimum=0)
PAGE_CACHE_MAX_ENTRIES = _env_int("PAGE_CACHE_MAX_ENTRIES", 256, minimum=1)

# Outbound links returned per page, so an agent can traverse. Capped because links
# are pure token cost to a model that does not intend to follow them.
MAX_LINKS_PER_PAGE = _env_int("MAX_LINKS_PER_PAGE", 25, minimum=0)

# Near-duplicate passage removal. Jaccard over 5-word shingles; 1.0 disables.
DEDUPE_THRESHOLD = float(os.environ.get("DEDUPE_THRESHOLD", "0.8"))
DEDUPE_CANDIDATES = _env_int("DEDUPE_CANDIDATES", 150)

# Network safety. Fetch targets come from an LLM (`fetch`) or from search results
# (`research`), so without this the server is an open proxy into whatever network
# it runs on — cloud metadata endpoints included.
ALLOW_PRIVATE_URLS = _env_bool("ALLOW_PRIVATE_URLS", False)

# Serving. Namespaced because a bare HOST is already set in many shells and CI
# images (often to the machine name), which would silently break the bind.
BIND_HOST = os.environ.get("BIND_HOST", "0.0.0.0")  # container-internal; publish narrowly in compose
PORT = _env_int("PORT", 8000)
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "").strip()

# Chunking / output budget
MAX_RETURNED_CHUNKS = 10
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CONTENT_LENGTH = 30000  # ceiling on total characters returned per call
MAX_FULL_TEXT_LENGTH = _env_int("MAX_FULL_TEXT_LENGTH", 60000)  # ceiling for full_text mode
