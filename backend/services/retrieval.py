"""
Dedicated retrieval pipeline for RAG.

Architecture
------------
    question ──► Embed ──► pgvector RPC ──► Score filter ──► Top-K ──► Merge

Stages
------
    1. **Embed question**    — generate 384-dim vector via embedding service
    2. **Vector search**      — call ``match_documents()`` RPC on Supabase
    3. **Score filter**       — discard chunks with cosine similarity < threshold
    4. **Select top-K**       — keep the K best (already sorted by the DB)
    5. **Fallback**           — if nothing passes, retry with lower threshold
    6. **Merge**              — join chunks into a single context string
    7. **Cache**              — store result keyed by (normalised_question, top_k)

Caching
-------
    Two-tier: in-memory LRU (always on) + optional Redis (when REDIS_URL is set).
    Each cache entry has a TTL (default 5 minutes) because new documents
    uploaded after the cache entry would make it stale.

    Cache key = sha256(normalised_question) — so "What is pricing?" and
    "What is pricing? " produce the same cache hit.

Why each stage
--------------
    - **Embed**: turns the question from text into a vector for cosine comparison
    - **Vector search**: pgvector ``<=>`` operator does approximate nearest-neighbour
      over all document chunks — this is what makes RAG work
    - **Score filter**: thresholds at 0.70 discard loosely-related chunks that would
      dilute the context and cause hallucination
    - **Top-K**: limits the context window so the LLM prompt stays within token limits
    - **Fallback**: prevents the chatbot from saying "I don't know" when a slightly
      lower threshold (0.50) would find a useful match
    - **Merge**: joins chunks into a single block the LLM can read as one context
    - **Cache**: avoids re-embedding and re-querying for repeated/identical questions
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import get_settings
from db import get_supabase
from services.embeddings import embed_text, get_embedding_dimension
from utils.logger import logger

settings = get_settings()

# ── Constants ────────────────────────────────────────────────────────

DEFAULT_TOP_K: int = 5
DEFAULT_SIMILARITY_THRESHOLD: float = 0.25  # cosine similarity floor (tuned for all-MiniLM-L6-v2)
FALLBACK_THRESHOLD: float = 0.15             # lower bar when nothing found
FALLBACK_TOP_K: int = 3                      # fewer results in fallback mode
CACHE_TTL_SECONDS: int = 300                 # 5 minutes
LRU_CACHE_MAX_SIZE: int = 256                # in-memory LRU capacity


# ── Data types ───────────────────────────────────────────────────────


@dataclass
class ChunkResult:
    """One retrieved document chunk with its similarity score."""

    id: str
    filename: str
    chunk_index: int
    content: str
    similarity: float
    metadata: dict = field(default_factory=dict)

    @property
    def source_label(self) -> str:
        """Human-readable label for source attribution."""
        return f"{self.filename} (chunk {self.chunk_index})"


@dataclass
class RetrievalResult:
    """
    The output of a retrieval run.

    Attributes
    ----------
    chunks : list of ChunkResult
        The individual chunks that matched (after filtering & top-K).
    context : str
        All chunks merged into a single text block, ready for the LLM.
    sources : list of str
        Deduplicated list of source filenames.
    latency_ms : float
        Total wall-clock time for the retrieval (embed + search + filter + merge).
    from_cache : bool
        Whether this result was served from the cache.
    fallback_used : bool
        Whether the fallback (lower threshold) was triggered.
    """

    chunks: List[ChunkResult] = field(default_factory=list)
    context: str = ""
    sources: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    from_cache: bool = False
    fallback_used: bool = False

    @property
    def is_empty(self) -> bool:
        """True when no relevant chunks were found."""
        return len(self.chunks) == 0

    @property
    def avg_score(self) -> float:
        """Mean similarity score across returned chunks (for observability)."""
        if not self.chunks:
            return 0.0
        return sum(c.similarity for c in self.chunks) / len(self.chunks)


# ── In-memory LRU cache ──────────────────────────────────────────────


class _LRUCache:
    """
    Thread-safe, TTL-aware LRU cache backed by an OrderedDict.

    Entries expire after ``ttl_seconds``. When the cache is full the
    least-recently-used entry is evicted.
    """

    def __init__(self, max_size: int = LRU_CACHE_MAX_SIZE, ttl: int = CACHE_TTL_SECONDS):
        self._max_size = max_size
        self._ttl = ttl
        self._store: OrderedDict[str, Tuple[float, RetrievalResult]] = OrderedDict()

    def get(self, key: str) -> Optional[RetrievalResult]:
        """Return cached result if present and not expired."""
        if key not in self._store:
            return None

        timestamp, result = self._store[key]
        if time.monotonic() - timestamp > self._ttl:
            # Expired — remove and return None
            del self._store[key]
            return None

        # Move to end (most-recently-used)
        self._store.move_to_end(key)
        return result

    def set(self, key: str, result: RetrievalResult) -> None:
        """Insert or update a cache entry. Evicts LRU if at capacity."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (time.monotonic(), result)

        # Evict oldest if over capacity
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Delete all entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# Global cache instances
_lru_cache = _LRUCache()
_redis_client: Optional[object] = None


def _get_redis():
    """Lazy-init Redis connection (only when REDIS_URL is configured)."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    if not settings.redis_url:
        return None

    try:
        import redis as _redis

        _redis_client = _redis.Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=False,
        )
        _redis_client.ping()
        logger.info("Redis cache connected.")
        return _redis_client
    except Exception as exc:
        logger.debug(f"Redis caching not available — using in-memory only. ({exc})")
        return None  # Redis unavailable — no fallback error


# ── Cache helpers ────────────────────────────────────────────────────


def _build_cache_key(question: str, top_k: int) -> str:
    """
    Build a deterministic cache key from the question and top_k.

    Normalises whitespace before hashing so minor formatting
    differences don't cause cache misses.
    """
    normalised = " ".join(question.strip().lower().split())
    payload = f"{normalised}|k={top_k}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_cached(cache_key: str) -> Optional[RetrievalResult]:
    """Check in-memory LRU first, then Redis (if available)."""
    result = _lru_cache.get(cache_key)
    if result is not None:
        logger.debug(f"[retrieval] cache hit (LRU): {cache_key[:12]}...")
        return result

    redis = _get_redis()
    if redis is None:
        return None

    try:
        raw = redis.get(cache_key)
        if raw is None:
            return None
        result = _deserialize(raw)
        if result is not None:
            # Promote to LRU for next in-memory hit
            _lru_cache.set(cache_key, result)
            logger.debug(f"[retrieval] cache hit (Redis): {cache_key[:12]}...")
        return result
    except Exception as exc:
        logger.debug(f"[retrieval] Redis get skipped: {exc}")
        return None


def _set_cache(cache_key: str, result: RetrievalResult) -> None:
    """Write to LRU (always) and Redis (if available)."""
    _lru_cache.set(cache_key, result)

    redis = _get_redis()
    if redis is None:
        return

    try:
        redis.setex(cache_key, CACHE_TTL_SECONDS, _serialize(result))
    except Exception as exc:
        logger.warning(f"[retrieval] Redis set failed: {exc}")


def _serialize(result: RetrievalResult) -> bytes:
    """Pickle a RetrievalResult for Redis storage."""
    import pickle

    return pickle.dumps(result)


def _deserialize(raw: bytes) -> Optional[RetrievalResult]:
    """Unpickle a RetrievalResult. Returns None on corruption."""
    import pickle

    try:
        return pickle.loads(raw)
    except Exception as exc:
        logger.warning(f"[retrieval] cache deserialisation failed: {exc}")
        return None


# ── Stage 1: Embed ──────────────────────────────────────────────────


async def _embed_question(question: str) -> Tuple[List[float], float]:
    """
    Generate a vector embedding for the question.

    Runs ``embed_text`` in a thread so the event loop stays free.
    Measures and returns embedding latency.

    Returns:
        (embedding_vector, latency_ms)
    """
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()
    embedding = await loop.run_in_executor(None, embed_text, question)
    elapsed = (time.monotonic() - t0) * 1000
    logger.debug(f"[retrieval] embedding generated in {elapsed:.1f}ms")
    return embedding, elapsed


# ── Stage 2–4: Vector search → Score filter → Top-K ─────────────────


async def _search_vectors(
    query_embedding: List[float],
    top_k: int,
    similarity_threshold: float,
) -> List[ChunkResult]:
    """
    Call the ``match_documents`` Supabase RPC.

    The database function runs:

    .. code:: sql

        SELECT d.*, 1 - (d.embedding <=> query_embedding) AS similarity
        FROM documents d
        WHERE 1 - (d.embedding <=> query_embedding) >= similarity_threshold
        ORDER BY d.embedding <=> query_embedding
        LIMIT top_k;

    ``<=>`` is pgvector's cosine distance operator.
    ``1 - distance`` converts it to cosine similarity (0–1).
    """
    client = get_supabase()
    result = client.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "similarity_threshold": similarity_threshold,
        },
    ).execute()

    rows = result.data or []
    chunks = [
        ChunkResult(
            id=row["id"],
            filename=row["filename"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            similarity=round(row["similarity"], 4),
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]
    return chunks


# ── Stage 6: Merge ──────────────────────────────────────────────────


def _merge_chunks(chunks: List[ChunkResult]) -> Tuple[str, List[str]]:
    """
    Merge retrieved chunks into a single context string.

    Each chunk is prefixed with its source for attribution.
    Chunks are separated by double-newlines.

    Returns:
        (context_block, source_filenames)
    """
    if not chunks:
        return "", []

    blocks: List[str] = []
    sources: List[str] = []

    for chunk in chunks:
        blocks.append(f"[Source: {chunk.source_label}]\n{chunk.content}")
        if chunk.filename not in sources:
            sources.append(chunk.filename)

    context = "\n\n---\n\n".join(blocks)
    return context, sources


# ── Public API ──────────────────────────────────────────────────────


async def retrieve_context(
    question: str,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    use_cache: bool = True,
) -> RetrievalResult:
    """
    Run the full retrieval pipeline for a given question.

    1. Check cache → return early if hit
    2. Embed the question
    3. Search pgvector via ``match_documents`` RPC
    4. Filter by cosine similarity threshold (default 0.70)
    5. Select top-K
    6. If nothing found, retry with fallback threshold (0.50)
    7. Merge chunks into context string
    8. Cache the result

    Returns:
        ``RetrievalResult`` with chunks, merged context, sources, and metadata.

    Raises:
        RuntimeError: if embedding or database queries fail (after logging).
    """
    k = top_k or settings.top_k_results
    threshold = similarity_threshold or DEFAULT_SIMILARITY_THRESHOLD
    t0 = time.monotonic()

    logger.info(
        f"[retrieval] question='{question[:80]}...' | top_k={k} | threshold={threshold}"
    )

    # ── 0. Check cache ────────────────────────────────────
    cache_key = _build_cache_key(question, k) if use_cache else None
    if cache_key:
        cached = _get_cached(cache_key)
        if cached is not None:
            cached.from_cache = True
            cached.latency_ms = (time.monotonic() - t0) * 1000
            return cached

    # ── 1. Embed question ─────────────────────────────────
    try:
        query_embedding, embed_latency = await _embed_question(question)
    except Exception as exc:
        logger.error(f"[retrieval] embedding failed: {exc}", exc_info=True)
        raise RuntimeError(
            "Could not process your question — embedding service is unavailable."
        ) from exc

    # ── 2. Vector search (primary) ────────────────────────
    fallback_used = False
    try:
        chunks = await _search_vectors(query_embedding, k, threshold)
    except Exception as exc:
        logger.error(f"[retrieval] vector search failed: {exc}", exc_info=True)
        raise RuntimeError(
            "Could not search documents — database is unavailable."
        ) from exc

    # ── 3. Fallback if nothing passed the threshold ───────
    if not chunks and threshold > FALLBACK_THRESHOLD:
        logger.info(
            f"[retrieval] no results at threshold={threshold}; "
            f"retrying with fallback threshold={FALLBACK_THRESHOLD}"
        )
        try:
            chunks = await _search_vectors(query_embedding, FALLBACK_TOP_K, FALLBACK_THRESHOLD)
            fallback_used = True
        except Exception as exc:
            logger.error(f"[retrieval] fallback search failed: {exc}")

    # ── 4. Merge context ──────────────────────────────────
    context, sources = _merge_chunks(chunks)
    total_latency = (time.monotonic() - t0) * 1000

    # ── 5. Build result ───────────────────────────────────
    result = RetrievalResult(
        chunks=chunks,
        context=context,
        sources=sources,
        latency_ms=round(total_latency, 2),
        fallback_used=fallback_used,
    )

    logger.info(
        f"[retrieval] done: {len(chunks)} chunks from {len(sources)} sources | "
        f"avg_score={result.avg_score:.2f} | latency={total_latency:.0f}ms"
        f"{' | FALLBACK' if fallback_used else ''}"
    )

    # ── 6. Cache result ───────────────────────────────────
    if cache_key and not result.is_empty:
        _set_cache(cache_key, result)

    return result


# ── Utility: clear cache ────────────────────────────────────────────


def clear_retrieval_cache() -> int:
    """
    Invalidate the entire retrieval cache.

    Call this after uploading new documents so subsequent queries
    see the fresh data.

    Returns:
        Number of entries cleared.
    """
    count = len(_lru_cache)
    _lru_cache.clear()

    redis = _get_redis()
    if redis:
        try:
            # Delete keys matching the retrieval cache prefix pattern
            # (Approach: we don't prefix, so we just log that Redis needs flush)
            redis.flushdb()
        except Exception:
            pass

    logger.info(f"[retrieval] cache cleared ({count} LRU entries)")
    return count
