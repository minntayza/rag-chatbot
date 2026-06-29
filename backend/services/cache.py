"""
Unified caching service — single source of truth.

Deduplicates the two previously-separate cache implementations
(services/cache.py and services/retrieval.py's _LRUCache).

Architecture
------------
    - **Redis primary** (when configured and reachable)
    - **In-memory LRU fallback** (always available, TTL-aware)
    - **JSON serialisation** (safe, no pickle RCE risk)

Usage
-----
    from services.cache import cache

    await cache.get("key")          # returns value or None
    await cache.set("key", value)   # store with default TTL
    await cache.set("key", v, 60)   # store with 60s TTL
    await cache.delete("key")
    await cache.clear()             # flush all
"""

from __future__ import annotations

import asyncio
import hashlib
import json as _json
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import redis.asyncio as aioredis

from config import get_settings
from utils.logger import logger

settings = get_settings()

# ── JSON Encoder for complex types ───────────────────────────────────


class _CacheEncoder(_json.JSONEncoder):
    """Extends JSONEncoder to handle bytes, sets, and dataclasses."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, bytes):
            return obj.hex()
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, "__dataclass_fields__"):
            import dataclasses
            return dataclasses.asdict(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)


# ── In-memory LRU fallback ───────────────────────────────────────────


class _LRUFallback:
    """TTL-aware LRU cache. NOT thread-safe (single event loop)."""

    def __init__(self, max_size: int = 512, ttl: int = 300):
        self._store: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None
        ts, value = self._store[key]
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        t = ttl if ttl is not None else self._ttl
        self._store[key] = (time.monotonic(), value)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# ── Unified Cache Service ────────────────────────────────────────────


class CacheService:
    """
    Single cache service used by the entire application.

    Auto-detects Redis availability. Falls back to in-memory LRU.
    Uses JSON serialisation (no pickle).
    """

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._lru = _LRUFallback()
        self._redis_available: Optional[bool] = None
        self._lock = asyncio.Lock()

    async def _ensure_redis(self) -> Optional[aioredis.Redis]:
        if self._redis_available is False:
            return None
        if self._redis is not None:
            return self._redis

        async with self._lock:
            if self._redis is not None:
                return self._redis
            if self._redis_available is False:
                return None

            if not settings.redis_enabled or not settings.redis_url:
                self._redis_available = False
                return None

            try:
                self._redis = aioredis.from_url(
                    settings.redis_url,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    decode_responses=True,  # JSON is string-based
                )
                await self._redis.ping()
                self._redis_available = True
                logger.info("Redis cache connected.")
                return self._redis
            except Exception as exc:
                self._redis_available = False
                self._redis = None
                logger.debug(
                    f"Redis unavailable ({exc}) — using in-memory fallback."
                )
                return None

    async def get(self, key: str) -> Optional[Any]:
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                raw = await redis.get(key)
                if raw is None:
                    return None
                return _json.loads(raw)
            except Exception:
                pass

        return self._lru.get(key)

    async def set(
        self, key: str, value: Any, ttl: int | None = None
    ) -> None:
        expiry = ttl if ttl is not None else settings.redis_cache_ttl
        serialised = _json.dumps(value, cls=_CacheEncoder)

        redis = await self._ensure_redis()
        if redis is not None:
            try:
                await redis.setex(key, expiry, serialised)
            except Exception:
                pass

        self._lru.set(key, value, expiry)

    async def delete(self, key: str) -> None:
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                await redis.delete(key)
            except Exception:
                pass
        self._lru.delete(key)

    async def clear(self) -> None:
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                await redis.flushdb()
            except Exception:
                pass
        self._lru.clear()
        logger.info("Cache cleared (Redis + LRU).")

    async def health(self) -> Dict[str, Any]:
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                await redis.ping()
                return {
                    "backend": "redis",
                    "status": "up",
                    "lru_entries": len(self._lru),
                }
            except Exception as exc:
                return {
                    "backend": "redis",
                    "status": "down",
                    "error": str(exc),
                    "lru_entries": len(self._lru),
                }
        return {
            "backend": "lru",
            "status": "up (redis unavailable)",
            "lru_entries": len(self._lru),
        }


# ── Convenience: cache-key builder for retrieval ─────────────────────


def build_retrieval_cache_key(question: str, top_k: int = 5) -> str:
    """
    Deterministic cache key for RAG retrieval results.

    Normalises whitespace and case so minor variations produce the
    same key, maximising cache hit rate.
    """
    normalised = " ".join(question.strip().lower().split())
    payload = f"rag:{normalised}|k={top_k}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Singleton ────────────────────────────────────────────────────────

cache = CacheService()
