"""
FastAPI application entry-point — Enterprise-grade RAG Chatbot API.

Run with:
    uvicorn main:app --reload

Health check routes:
    GET /          — liveness probe (always 200)
    GET /health    — readiness probe (checks DB + cache)
    GET /metrics   — Prometheus scrape endpoint
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from db import close_db, get_supabase, init_db
from routers import chat, upload
from schemas import HealthResponse
from services.cache import cache
from services.security import get_security_headers
from utils.logger import logger

settings = get_settings()


# ── Lifespan (startup / shutdown) ────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, connect cache. Shutdown: clean up connections."""
    logger.info("Starting RAG Chatbot API...")
    logger.info(f"Environment: {settings.app_env}")

    await init_db()
    logger.info("Database connection verified.")

    # Warm the cache connection
    await cache._ensure_redis()

    # Set Prometheus app info
    if settings.prometheus_enabled:
        from services.metrics import app_info
        app_info.labels(
            version="1.0.0", environment=settings.app_env
        ).set(1)

    logger.info("Application startup complete.")
    yield
    logger.info("Shutting down...")
    await close_db()


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG FAQ Chatbot API",
    description="Enterprise-grade Retrieval-Augmented Generation chatbot.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.app_debug else None,
    redoc_url=None,
)


# ── CORS ─────────────────────────────────────────────────────────────
# In production, CORS_ORIGINS should be an explicit list, not wildcard.
# The default value in config.py is ["http://localhost:3000", "http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Security headers middleware ──────────────────────────────────────


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to every response."""
    response = await call_next(request)
    for header, value in get_security_headers().items():
        response.headers.setdefault(header, value)
    return response


# ── Prometheus middleware ────────────────────────────────────────────

if settings.prometheus_enabled:
    from services.metrics import PrometheusMiddleware, mount_metrics_endpoint

    app.add_middleware(PrometheusMiddleware)
    mount_metrics_endpoint(app)


# ── Exception handlers ───────────────────────────────────────────────


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning(f"ValueError: {exc}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    # NEVER return raw exception text in production
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


# ── Routers ──────────────────────────────────────────────────────────

app.include_router(chat.router)
app.include_router(upload.router)


# ── Health ───────────────────────────────────────────────────────────


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Liveness probe — always returns 200 if the process is alive."""
    return HealthResponse()


@app.get("/health", tags=["Health"])
async def health():
    """
    Readiness probe — checks all dependencies.

    Returns 200 only when DB, cache, and all services are healthy.
    Use this for Kubernetes readiness probes and load balancer health checks.

    NOTE: Internal error details are NOT exposed — only up/down status.
    """
    checks: dict = {"status": "ok", "version": "1.0.0", "checks": {}}

    # 1. Database check
    try:
        client = get_supabase()
        client.table("documents").select("id", count="exact").limit(1).execute()
        checks["checks"]["database"] = "up"
    except Exception:
        checks["checks"]["database"] = "down"
        checks["status"] = "degraded"

    # 2. Cache check
    try:
        cache_health = await cache.health()
        checks["checks"]["cache"] = "up" if "error" not in cache_health.get("status", "") else "down"
    except Exception:
        checks["checks"]["cache"] = "down"

    # 3. LLM connectivity check
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{settings.llm_base_url}/models",
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            )
            checks["checks"]["llm_api"] = "up" if response.status_code == 200 else "down"
    except Exception:
        checks["checks"]["llm_api"] = "down"

    status_code = 200 if checks["status"] == "ok" else 503
    return JSONResponse(content=checks, status_code=status_code)


# ── Run directly ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level="debug" if settings.app_debug else "info",
    )
