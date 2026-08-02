"""Network-safety primitives: URL validation, SSRF/redirect screening, credentials.

Fetch targets come from an LLM (`fetch`) or from search results (`research`), so
both are treated as untrusted. This module owns everything that decides whether a
URL may be fetched and how authenticated requests are handled — the parts of the
server that keep it from being an open proxy into private networks.
"""
import asyncio
import ipaddress
import json
import socket
import urllib.request
from typing import Any
from urllib.parse import urlparse

from config import (
    ALLOW_INSECURE_CREDENTIALS,
    ALLOW_PRIVATE_URLS,
    FETCH_CREDENTIALS_FILE,
    FETCH_CREDENTIALS_RAW,
)

_BLOCKED_HOST_SUFFIXES = ("localhost", ".localhost", ".local", ".internal")


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


def _ip_is_public(raw_ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(raw_ip)
    except ValueError:
        return False
    # ::ffff:169.254.169.254 must be judged as the IPv4 address it wraps
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _resolve_block_reason(url: str) -> str | None:
    """None if the URL is safe to fetch, else a human-readable reason.

    Blocking is by resolved IP, so DNS names pointing at private space are caught
    too. Resolution here and the browser's own lookup are separate, so a hostile
    DNS server could still rebind between them; the post-fetch check in `crawl()`
    is what stops that content from reaching the caller.
    """
    if not _is_valid_http_url(url):
        return "invalid url"
    if ALLOW_PRIVATE_URLS:
        return None

    host = (urlparse(url).hostname or "").strip().rstrip(".").lower()
    if not host:
        return "no host"
    if host == "localhost" or host.endswith(_BLOCKED_HOST_SUFFIXES):
        return "blocked host (local)"

    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return "dns resolution failed"
    if not infos:
        return "dns resolution failed"
    for info in infos:
        ip = info[4][0]
        if not _ip_is_public(ip):
            return f"blocked non-public address ({ip})"
    return None


async def _screen_urls(urls: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Split URLs into (allowed, blocked-with-reason). DNS lookups run off-thread."""
    reasons = await asyncio.gather(
        *(asyncio.to_thread(_resolve_block_reason, u) for u in urls)
    )
    allowed: list[str] = []
    blocked: list[dict[str, str]] = []
    for url, reason in zip(urls, reasons):
        if reason is None:
            allowed.append(url)
        else:
            blocked.append({"url": url, "error": reason})
    return allowed, blocked


# ---- CREDENTIALS ----
def _load_credentials() -> dict[str, dict[str, str]]:
    """Host -> extra request headers, from a file or an inline JSON env var.

    Hosts are matched exactly; a leading dot ('.example.com') also covers subdomains.
    """
    raw = ""
    if FETCH_CREDENTIALS_FILE:
        try:
            with open(FETCH_CREDENTIALS_FILE, encoding="utf-8") as handle:
                raw = handle.read()
        except OSError as e:
            print(f"[auth] cannot read FETCH_CREDENTIALS_FILE: {e}", flush=True)
            return {}
    elif FETCH_CREDENTIALS_RAW:
        raw = FETCH_CREDENTIALS_RAW
    if not raw.strip():
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[auth] credentials are not valid JSON: {e}", flush=True)
        return {}
    if not isinstance(parsed, dict):
        print("[auth] credentials must be an object of host -> headers", flush=True)
        return {}

    credentials: dict[str, dict[str, str]] = {}
    for host, headers in parsed.items():
        if not isinstance(headers, dict) or not all(isinstance(v, str) for v in headers.values()):
            print(f"[auth] ignoring malformed entry for {host!r}", flush=True)
            continue
        credentials[str(host).strip().lower()] = {str(k): v for k, v in headers.items()}
    if credentials:
        print(f"[auth] credentials configured for {len(credentials)} host(s)", flush=True)
    return credentials


_credentials = _load_credentials()


def _credentials_for(url: str) -> tuple[str | None, dict[str, str] | None]:
    """(matched_host, headers) for a URL, or (None, None).

    Credentials are withheld from plaintext http:// unless explicitly allowed —
    a bearer token on the wire is worse than a failed fetch.
    """
    if not _credentials:
        return None, None
    parsed = urlparse(url)
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return None, None

    match = None
    if host in _credentials:
        match = host
    else:
        for configured in _credentials:
            if configured.startswith(".") and (host.endswith(configured) or host == configured[1:]):
                match = configured
                break
    if match is None:
        return None, None
    if parsed.scheme != "https" and not ALLOW_INSECURE_CREDENTIALS:
        print(f"[auth] refusing to send credentials to {host} over {parsed.scheme}", flush=True)
        return None, None
    return match, _credentials[match]
