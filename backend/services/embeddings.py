"""
Embedding service — local & remote backends.

Supports two modes configured via ``EMBEDDING_MODE`` env var:

    local    — sentence-transformers (all-MiniLM-L6-v2, 384-dim, no API key)
    remote   — OpenAI-compatible /embedding endpoint (Mimo, OpenAI, etc.)

The ``embed_texts()`` function auto-selects the backend based on config.
Both backends support batch embedding.
"""

from __future__ import annotations

from typing import List

import httpx

from config import get_settings
from utils.logger import logger

settings = get_settings()

# ── Local model (lazy-loaded) ────────────────────────────────────────

_local_model = None
_LOCAL_MODEL_NAME = "all-MiniLM-L6-v2"
_LOCAL_DIMENSION = 384


def _get_local_model():
    """Lazy-load the sentence-transformers model (imported on first call)."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading local embedding model: {_LOCAL_MODEL_NAME}...")
        _local_model = SentenceTransformer(_LOCAL_MODEL_NAME)
        logger.info("Local embedding model loaded.")
    return _local_model


# ── Public API ───────────────────────────────────────────────────────


def embed_text(text: str) -> List[float]:
    """
    Embed a single text string.

    Uses the backend selected by ``EMBEDDING_MODE`` (default: local).
    """
    return embed_texts([text])[0]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed multiple texts in a single call.

    - **local mode**: synchronous batch encoding via sentence-transformers
    - **remote mode**: asynchronous HTTP POST to ``<EMBEDDING_BASE_URL>/embeddings``

    The remote call uses ``httpx.Client`` (sync) to stay compatible with
    the ingestion pipeline's synchronous helpers. For async callers, wrap
    in ``asyncio.to_thread()``.

    Returns:
        List of float lists, one per input text, in the same order.
    """
    mode = _resolve_mode()

    if mode == "local":
        model = _get_local_model()
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vectors]

    return _embed_remote(texts)


# ── Remote backend ───────────────────────────────────────────────────


def _resolve_mode() -> str:
    """
    Auto-detect embedding mode.

    - If ``EMBEDDING_API_KEY`` is set, use **remote**
    - Otherwise, use **local**
    """
    if settings.embedding_api_key:
        return "remote"
    return "local"


def _embed_remote(texts: List[str]) -> List[List[float]]:
    """
    Call a remote OpenAI-compatible ``/embeddings`` endpoint.

    Sends all texts in a single batch request. The server must support
    accepting an array of strings as the ``input`` parameter.
    """
    if not texts:
        return []

    headers = {
        "Authorization": f"Bearer {settings.embedding_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.embedding_model,
        "input": texts,
    }

    # Use a synchronous client here — the ingestion pipeline wraps
    # embedding calls in asyncio.to_thread for parallelism.
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{settings.embedding_base_url}/embeddings",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    # The response shape is OpenAI-compatible:
    # { "data": [ { "embedding": [...], "index": 0 }, ... ] }
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]


def get_embedding_dimension() -> int:
    """Return the dimension of the active embedding model."""
    mode = _resolve_mode()
    if mode == "local":
        return _LOCAL_DIMENSION
    return settings.embedding_dimension
