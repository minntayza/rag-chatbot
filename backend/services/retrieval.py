"""
Retrieval pipeline — semantic search over document chunks.

Architecture
------------
    question → Embed → pgvector RPC → Score filter → Top-K → Merge

Stages
------
    1. **Embed question** — 384-dim vector via embedding service
    2. **Vector search**  — cosine similarity via match_documents() RPC
    3. **Score filter**   — discard chunks below threshold
    4. **Fallback**       — retry with lower threshold if nothing found
    5. **Merge**          — join chunks into context block
    6. **Cache**          — store result for identical future queries
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config import get_settings
from db import get_supabase
from services.cache import cache, build_retrieval_cache_key
from services.embeddings import embed_text
from utils.logger import logger

settings = get_settings()

# ── Constants ────────────────────────────────────────────────────────

DEFAULT_SIMILARITY_THRESHOLD: float = 0.25
FALLBACK_THRESHOLD: float = 0.15
FALLBACK_TOP_K: int = 3


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
        return f"{self.filename} (chunk {self.chunk_index})"


@dataclass
class RetrievalResult:
    """Output of one retrieval run."""

    chunks: List[ChunkResult] = field(default_factory=list)
    context: str = ""
    sources: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    from_cache: bool = False
    fallback_used: bool = False

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    @property
    def avg_score(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(c.similarity for c in self.chunks) / len(self.chunks)


# ── Stage 1: Embed ──────────────────────────────────────────────────


async def _embed_question(question: str) -> Tuple[List[float], float]:
    """Generate embedding vector. Returns (embedding, latency_ms)."""
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()
    embedding = await loop.run_in_executor(None, embed_text, question)
    elapsed = (time.monotonic() - t0) * 1000
    logger.debug(f"[retrieval] embedding done in {elapsed:.1f}ms")
    return embedding, elapsed


# ── Stage 2: Vector search ──────────────────────────────────────────


async def _search_vectors(
    query_embedding: List[float],
    top_k: int,
    similarity_threshold: float,
) -> List[ChunkResult]:
    """Call match_documents() RPC on Supabase."""
    result = (
        get_supabase()
        .rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": top_k,
                "similarity_threshold": similarity_threshold,
            },
        )
        .execute()
    )

    return [
        ChunkResult(
            id=row["id"],
            filename=row["filename"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            similarity=round(row["similarity"], 4),
            metadata=row.get("metadata", {}),
        )
        for row in (result.data or [])
    ]


# ── Stage 3: Merge ──────────────────────────────────────────────────


def _merge_chunks(chunks: List[ChunkResult]) -> Tuple[str, List[str]]:
    """Join chunks into a context block with source attribution."""
    if not chunks:
        return "", []

    blocks: List[str] = []
    sources: List[str] = []

    for chunk in chunks:
        blocks.append(f"[Source: {chunk.source_label}]\n{chunk.content}")
        if chunk.filename not in sources:
            sources.append(chunk.filename)

    return "\n\n---\n\n".join(blocks), sources


# ── Public API ──────────────────────────────────────────────────────


async def retrieve_context(
    question: str,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    use_cache: bool = True,
) -> RetrievalResult:
    """
    Full retrieval pipeline for a single question.

    1. Check cache → early return on hit
    2. Embed the question
    3. Vector search with primary threshold
    4. Fallback search if nothing found
    5. Merge chunks into context string
    6. Cache the result

    Returns:
        RetrievalResult with chunks, merged context, sources, and metadata.
    """
    k = top_k or settings.top_k_results
    threshold = similarity_threshold or DEFAULT_SIMILARITY_THRESHOLD
    t0 = time.monotonic()

    logger.info(
        f"[retrieval] q='{question[:80]}...' | k={k} | threshold={threshold}"
    )

    # 0. Cache check
    cache_key = build_retrieval_cache_key(question, k) if use_cache else None
    if cache_key:
        cached = await cache.get(cache_key)
        if cached is not None:
            # Rebuild RetrievalResult from cached dict
            chunks = [
                ChunkResult(**c) for c in cached.get("chunks", [])
            ]
            result = RetrievalResult(
                chunks=chunks,
                context=cached.get("context", ""),
                sources=cached.get("sources", []),
                from_cache=True,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
            logger.debug(f"[retrieval] cache hit: {cache_key[:12]}...")
            return result

    # 1. Embed question
    try:
        query_embedding, _ = await _embed_question(question)
    except Exception as exc:
        logger.error(f"[retrieval] embedding failed: {exc}", exc_info=True)
        raise RuntimeError(
            "Could not process your question — embedding service unavailable."
        ) from exc

    # 2. Vector search
    fallback_used = False
    try:
        chunks = await _search_vectors(query_embedding, k, threshold)
    except Exception as exc:
        logger.error(f"[retrieval] search failed: {exc}", exc_info=True)
        raise RuntimeError(
            "Could not search documents — database unavailable."
        ) from exc

    # 3. Fallback
    if not chunks and threshold > FALLBACK_THRESHOLD:
        logger.info(
            f"[retrieval] no results at {threshold}; "
            f"retrying with {FALLBACK_THRESHOLD}"
        )
        try:
            chunks = await _search_vectors(
                query_embedding, FALLBACK_TOP_K, FALLBACK_THRESHOLD
            )
            fallback_used = True
        except Exception as exc:
            logger.error(f"[retrieval] fallback search failed: {exc}")

    # 4. Merge
    context, sources = _merge_chunks(chunks)
    total_latency = (time.monotonic() - t0) * 1000

    result = RetrievalResult(
        chunks=chunks,
        context=context,
        sources=sources,
        latency_ms=round(total_latency, 2),
        fallback_used=fallback_used,
    )

    logger.info(
        f"[retrieval] done: {len(chunks)} chunks from {len(sources)} sources | "
        f"avg_score={result.avg_score:.2f} | {total_latency:.0f}ms"
        f"{' | FALLBACK' if fallback_used else ''}"
    )

    # 5. Cache
    if cache_key and not result.is_empty:
        cacheable = {
            "chunks": [
                {
                    "id": c.id, "filename": c.filename,
                    "chunk_index": c.chunk_index, "content": c.content,
                    "similarity": c.similarity, "metadata": c.metadata,
                }
                for c in chunks
            ],
            "context": context,
            "sources": sources,
        }
        asyncio.create_task(cache.set(cache_key, cacheable))

    return result


# ── Cache invalidation ───────────────────────────────────────────────


async def clear_retrieval_cache() -> None:
    """Invalidate retrieval cache (call after document uploads)."""
    await cache.clear()
    logger.info("[retrieval] cache invalidated.")
