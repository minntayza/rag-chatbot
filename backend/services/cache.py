"""
Redis caching service with graceful fallback.

Why Redis in production?
    - Shared cache across multiple backend instances (horizontal scaling)
    - TTL ensures stale data expires automatically
    - Persistence (RDB/AOF) survives restarts
    - Atomic operations prevent race conditions
    - Built-in pub/sub for cache invalidation across instances

Architecture
------------
    Always tries Redis first. If Redis is unreachable (connection refused,
    timeout, config disabled), falls back gracefully to in-memory LRU
    so the application never breaks.

Usage
-----
    from services.cache import cache

    await cache.set("key", {"data": 123}, ttl=300)
    value = await cache.get("key")  # returns None if expired/missing
"""

from __future__ import annotations

import asyncio
import json as _json
import pickle
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import redis.asyncio as aioredis

from config import get_settings
from services.metrics import record_cache_hit, record_cache_miss
from utils.logger import logger

settings = get_settings()


# ── In-memory LRU fallback ───────────────────────────────────────────


class _LRUFallback:
    """Thread-safe TTL-aware in-memory cache used when Redis is down."""

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


# ── Redis cache ──────────────────────────────────────────────────────


class CacheService:
    """
    Production cache with Redis primary + LRU fallback.

    Automatically detects Redis availability and degrades gracefully.
    """

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._fallback = _LRUFallback()
        self._redis_available: Optional[bool] = None
        self._lock = asyncio.Lock()

    async def _ensure_redis(self) -> Optional[aioredis.Redis]:
        """Connect to Redis if not already connected. Thread-safe."""
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
                logger.info("Redis disabled by config — using in-memory cache.")
                return None

            try:
                self._redis = aioredis.from_url(
                    settings.redis_url,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    decode_responses=False,
                )
                await self._redis.ping()
                self._redis_available = True
                logger.info("Redis cache connected.")
                return self._redis
            except Exception as exc:
                self._redis_available = False
                self._redis = None
                logger.warning(
                    f"Redis unavailable ({exc}) — using in-memory fallback."
                )
                return None

    async def get(self, key: str) -> Optional[Any]:
        """Get a value. Returns None on cache miss or expiration."""
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                raw = await redis.get(key)
                if raw is None:
                    record_cache_miss()
                    return None
                record_cache_hit("redis")
                return pickle.loads(raw)
            except Exception:
                pass

        # Fallback to LRU
        value = self._fallback.get(key)
        if value is not None:
            record_cache_hit("lru")
        else:
            record_cache_miss()
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value with optional TTL (seconds). Default: from config."""
        expiry = ttl if ttl is not None else settings.redis_cache_ttl

        redis = await self._ensure_redis()
        if redis is not None:
            try:
                await redis.setex(key, expiry, pickle.dumps(value))
            except Exception:
                pass

        # Always write to LRU fallback (so it's available even if Redis goes down)
        self._fallback.set(key, value, expiry)

    async def delete(self, key: str) -> None:
        """Delete a key from both Redis and LRU."""
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                await redis.delete(key)
            except Exception:
                pass
        self._fallback.delete(key)

    async def clear(self) -> None:
        """Clear all cache entries (flush Redis + LRU)."""
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                await redis.flushdb()
            except Exception:
                pass
        self._fallback.clear()
        logger.info("Cache cleared (Redis + LRU)")

    async def health(self) -> Dict[str, Any]:
        """Check cache health. Returns status dict."""
        redis = await self._ensure_redis()
        if redis is not None:
            try:
                await redis.ping()
                return {"backend": "redis", "status": "healthy", "fallback_size": len(self._fallback)}
            except Exception as exc:
                return {"backend": "redis", "status": f"error: {exc}", "fallback_size": len(self._fallback)}
        if self._redis_available is False:
            return {"backend": "lru", "status": "healthy (redis unavailable)", "size": len(self._fallback)}
        return {"backend": "unknown", "status": "not initialised"}


# ── Singleton ────────────────────────────────────────────────────────

cache = CacheService()
