"""Pure chunking and ranking primitives.

Everything here is a deterministic function of its inputs — no I/O, no browser,
no network. It owns the passage-splitting, tokenization, BM25 scoring, RRF
fusion, near-duplicate removal, and diversity ordering that turn fetched page
text into the ranked passages returned to the caller.

Config constants referenced as default parameter values come from `config`.
"""
import math
import re
import time
from collections import Counter
from typing import Any

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEDUPE_CANDIDATES,
    DEDUPE_THRESHOLD,
    MAX_CONTENT_LENGTH,
    MAX_FULL_TEXT_LENGTH,
    MAX_RETURNED_CHUNKS,
)


_TOKEN_RE = re.compile(r"\w+")

_STOPWORDS = frozenset("""
a an and are as at be been but by can did do does for from had has have how i if in
into is it its of on or que so than that the their then there these they this to was
were what when where which who why will with you your
""".split())


_STEM_RULES = (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ion", ""), ("ed", ""), ("es", ""), ("s", ""))


def _stem(word: str) -> str:
    """Crude suffix stripping so "handles"/"handle" and "extraction"/"extract" unify.

    Skips non-alphabetic tokens, which keeps identifiers like "crawl4ai" and version
    numbers intact. Only consistency between query and document matters here.
    """
    if len(word) <= 3 or not word.isalpha():
        return word
    for suffix, replacement in _STEM_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= 3:
            word = word[: -len(suffix)] + replacement
            break
    if len(word) >= 4 and word.endswith("e"):
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase stemmed word tokens, stopwords dropped.

    Regex-based so trailing punctuation does not glue itself to terms — plain
    `.split()` makes "crawl4ai." fail to match "crawl4ai".
    """
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25 relevance of each document to the query.

    Replaces raw keyword overlap, which divided only by query length and so scored
    strictly higher for longer text — a long irrelevant chunk beat a short exact
    match. The `b` term normalises by document length; idf discounts common words.
    """
    q_terms = tokenize(query)
    if not q_terms or not documents:
        return [0.0] * len(documents)

    doc_tokens = [tokenize(d) for d in documents]
    n_docs = len(doc_tokens)
    avg_len = sum(len(d) for d in doc_tokens) / n_docs or 1.0

    doc_freq: Counter[str] = Counter()
    term_freqs: list[Counter[str]] = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        term_freqs.append(tf)
        doc_freq.update(tf.keys())

    scores = []
    for tf, tokens in zip(term_freqs, doc_tokens):
        doc_len = len(tokens)
        score = 0.0
        for term in set(q_terms):
            freq = tf.get(term, 0)
            if not freq:
                continue
            df = doc_freq[term]
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / avg_len))
        scores.append(score * _repetition_penalty(tf, doc_len))
    return scores


def _repetition_penalty(tf: Counter[str], doc_len: int, max_share: float = 0.10) -> float:
    """Damp chunks dominated by one endlessly repeated word.

    BM25's term-frequency saturation still rewards a passage that repeats a query
    term thirty times over one that answers it once. Scraped markdown is full of
    such text — link dumps, repeated nav labels, cookie banners.

    Measured as the share of the chunk taken by its single most frequent token,
    which is stable across chunk lengths. A unique/total ratio is not: it rises as
    chunks get shorter, which would quietly reintroduce a length bias. Ordinary
    prose sits near 0.02-0.05 and is left untouched.
    """
    if doc_len < 20 or not tf:
        return 1.0
    share = max(tf.values()) / doc_len
    return min(1.0, max_share / share) if share > max_share else 1.0


def recursive_split(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    separators = ["\n\n", "\n", " ", ""]

    def _carry_tail(text_so_far: str, size: int) -> str:
        """Trailing `size` chars, snapped forward to a word boundary."""
        if size <= 0 or not text_so_far:
            return ""
        tail = text_so_far[-size:]
        boundary = tail.find(" ")
        return tail[boundary + 1:] if boundary != -1 else tail

    def _split_recursive(t: str, sep_idx: int) -> list[str]:
        if len(t) <= chunk_size:
            return [t]
        sep = separators[sep_idx]
        if not sep:
            chunks = []
            start = 0
            step = max(chunk_size - overlap, 1)
            while start < len(t):
                chunks.append(t[start : start + chunk_size])
                start += step
            return chunks

        chunks: list[str] = []
        current_chunk = ""
        for piece in t.split(sep):
            if not piece:
                continue
            if len(piece) > chunk_size:
                # Oversized piece: flush the buffer first, and *clear* it — leaving it
                # in place re-emitted the same text in the next chunk.
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                chunks.extend(_split_recursive(piece, sep_idx + 1))
                continue

            candidate = f"{current_chunk}{sep}{piece}" if current_chunk else piece
            if len(candidate) <= chunk_size:
                current_chunk = candidate
                continue

            chunks.append(current_chunk)
            carry = _carry_tail(current_chunk, overlap)
            current_chunk = f"{carry}{sep}{piece}" if carry else piece
            if len(current_chunk) > chunk_size:
                current_chunk = piece
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    return [c for c in _split_recursive(text, 0) if c.strip()]


def markdown_section_split(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    sections = re.split(r'(?=^#{1,6}\s)', text, flags=re.MULTILINE)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        header_match = re.match(r'^(#{1,6}\s.*?)(\n|$)', section)
        header = header_match.group(1) if header_match else ""
        body = section[len(header):].strip() if header_match else section
        if len(section) <= chunk_size:
            chunks.append(section)
        elif body:
            # Repeat the header on every sub-chunk so each stays self-describing
            chunks.extend(f"{header}\n{sub}".strip() for sub in recursive_split(body, chunk_size, overlap))
        else:
            chunks.append(section)
    return [c for c in chunks if c.strip()]


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


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion: score = sum over rankings of 1 / (k + rank).

    Combines rankings from several query phrasings. Fusing on *rank* rather than
    score is the point — BM25 scores from different queries are not on a common
    scale, so averaging them lets whichever query happened to produce bigger
    numbers dominate. A passage that places well for several phrasings wins here.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


def _shingles(text: str, k: int = 5) -> frozenset[str]:
    """Set of k-word shingles over normalised tokens.

    Built from `tokenize`, so stemming and stopword removal already absorb the
    trivial edits — reformatting, boilerplate wrappers — that syndicated copies pick up.
    """
    tokens = tokenize(text)
    if len(tokens) < k:
        return frozenset(tokens)
    return frozenset(" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    return intersection / union if union else 0.0


def _dedupe_chunks(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop near-duplicate passages, keeping the highest-scoring copy.

    Syndicated articles and mirrored docs otherwise spend several of the ten result
    slots restating one thing. The surviving copy records the other URLs it appeared
    on, which turns the duplication into a corroboration signal instead of noise.

    Only the top `DEDUPE_CANDIDATES` by score are compared — anything below that
    cannot be selected anyway, and it keeps this quadratic pass bounded.
    """
    if DEDUPE_THRESHOLD >= 1.0 or len(chunks) < 2:
        return chunks, 0

    ranked = sorted(chunks, key=lambda c: c["score"], reverse=True)
    head, tail = ranked[:DEDUPE_CANDIDATES], ranked[DEDUPE_CANDIDATES:]

    kept: list[dict[str, Any]] = []
    kept_shingles: list[frozenset[str]] = []
    removed = 0
    for chunk in head:
        shingles = _shingles(chunk["text"])
        length = len(chunk["text"])
        duplicate_of = None
        for existing, existing_shingles in zip(kept, kept_shingles):
            # Length prefilter: texts differing by more than 2x cannot be near-copies,
            # and this skips most of the expensive set intersections.
            other_length = len(existing["text"])
            if not (0.5 <= length / other_length <= 2.0) if other_length else True:
                continue
            if _jaccard(shingles, existing_shingles) >= DEDUPE_THRESHOLD:
                duplicate_of = existing
                break
        if duplicate_of is not None:
            removed += 1
            if duplicate_of["url"] != chunk["url"]:
                duplicate_of.setdefault("also_in", [])
                if chunk["url"] not in duplicate_of["also_in"]:
                    duplicate_of["also_in"].append(chunk["url"])
            continue
        kept.append(chunk)
        kept_shingles.append(shingles)
    return kept + tail, removed


def _diversity_order(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Full ranking, round-robin across sources.

    Taking a plain global top-N lets one verbose page fill every slot while the
    other fetched sources contribute nothing. Ordering the whole list (rather than
    only the first page of it) is what makes `offset` paging well defined.
    """
    per_url: dict[str, list[dict[str, Any]]] = {}
    for chunk in sorted(scored, key=lambda c: c["score"], reverse=True):
        per_url.setdefault(chunk["url"], []).append(chunk)

    # Best-scoring source first, so ties in the round-robin favour the stronger page
    queues = sorted(per_url.values(), key=lambda q: q[0]["score"], reverse=True)

    ordered: list[dict[str, Any]] = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(queue.pop(0))
    return ordered


def _select_chunks(
    scored: list[dict[str, Any]],
    max_chunks: int = MAX_RETURNED_CHUNKS,
    max_chars: int = MAX_CONTENT_LENGTH,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Take one page of the diversity ordering, within a character budget.

    Returns (chunks, total_available). The budget applies to what is returned, not
    to what was skipped, so paging deeper does not shrink the window.
    """
    ordered = _diversity_order(scored)
    selected: list[dict[str, Any]] = []
    used_chars = 0
    for chunk in ordered[max(0, offset):]:
        if len(selected) >= max_chunks:
            break
        if selected and used_chars + len(chunk["text"]) > max_chars:
            continue  # oversized chunk skipped; a smaller later one may still fit
        selected.append(chunk)
        used_chars += len(chunk["text"])
    return sorted(selected, key=lambda c: c["score"], reverse=True), len(ordered)


def full_text_of(docs: list[dict[str, Any]], max_chars: int = None) -> dict[str, Any]:
    """Whole documents instead of ranked passages, for when the model needs everything.

    Budget is shared evenly, so one long page cannot crowd the others out entirely.
    """
    max_chars = MAX_FULL_TEXT_LENGTH if max_chars is None else max_chars
    if not docs:
        return {"error": "content fetched but contained no text"}
    per_doc = max(1000, max_chars // len(docs))
    results = []
    for doc in docs:
        content = doc["content"]
        truncated = len(content) > per_doc
        results.append({
            "url": doc["url"],
            "title": doc.get("title", ""),
            "text": content[:per_doc],
            "truncated": truncated,
            "total_chars": len(content),
        })
    return {"results": results, "mode": "full_text"}


def rank_chunks(
    docs: list[dict[str, Any]],
    query: str,
    max_chunks: int = MAX_RETURNED_CHUNKS,
    offset: int = 0,
    query_variants: list[str] | None = None,
) -> dict[str, Any]:
    """Chunk fetched documents and return the most relevant passages.

    With `query_variants`, each phrasing is ranked separately and the rankings are
    fused with RRF, which softens BM25's exact-match brittleness — "how to fix X"
    stops missing a passage that says "resolving X".
    """
    all_chunks: list[dict[str, Any]] = []

    t_chunk = time.perf_counter()
    for doc in docs:
        # Every document is chunked, including short ones. Letting short pages through
        # whole put 12k-char blobs in the same ranking pool as 1k-char passages.
        chunks = merge_small_chunks(
            markdown_section_split(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP), min_length=200
        )
        for i, text in enumerate(chunks):
            all_chunks.append({"url": doc["url"], "title": doc.get("title", ""), "chunk_id": i, "text": text})
    print(f"[rank] chunking: {time.perf_counter() - t_chunk:.2f}s ({len(all_chunks)} chunks)", flush=True)

    if not all_chunks:
        return {"error": "content fetched but contained no text"}

    texts = [c["text"] for c in all_chunks]
    variants = [v.strip() for v in (query_variants or []) if v and v.strip() and v.strip() != query]
    if variants:
        rankings = []
        for phrasing in [query, *variants]:
            phrasing_scores = bm25_scores(phrasing, texts)
            # Passages the phrasing does not match at all must not earn rank credit
            matched = [i for i in range(len(texts)) if phrasing_scores[i] > 0]
            matched.sort(key=lambda i: phrasing_scores[i], reverse=True)
            rankings.append(matched)
        fused = rrf_fuse(rankings)
        for i, chunk in enumerate(all_chunks):
            chunk["score"] = round(fused.get(i, 0.0), 5)
        ranking_method = "rrf"
    else:
        for chunk, score in zip(all_chunks, bm25_scores(query, texts)):
            chunk["score"] = round(float(score), 4)
        ranking_method = "bm25"

    deduped, removed = _dedupe_chunks(all_chunks)
    if removed:
        print(f"[rank] dropped {removed} near-duplicate chunk(s)", flush=True)

    selected, total = _select_chunks(deduped, max_chunks=max_chunks, offset=offset)
    results = []
    for c in selected:
        entry = {
            "url": c["url"], "title": c["title"], "chunk_id": c["chunk_id"],
            "text": c["text"], "score": c["score"],
        }
        if c.get("also_in"):
            entry["also_in"] = c["also_in"]
        results.append(entry)
    out = {
        "results": results,
        "total_chunks": total,
        "offset": offset,
        "has_more": offset + len(selected) < total,
        "ranking": ranking_method,
    }
    if variants:
        out["query_variants"] = variants
    if removed:
        out["duplicates_removed"] = removed
    return out
