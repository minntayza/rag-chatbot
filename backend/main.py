"""
FastAPI application entry-point.

Responsibilities:
  - Create and configure the FastAPI app.
  - Register CORS middleware.
  - Register routers (chat, upload).
  - Wire up lifespan events (DB init / shutdown).
  - Add global exception handlers.
  - Serve health-check at GET /.

Run with:
    uvicorn main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from db import close_db, init_db
from routers import chat, upload
from schemas import HealthResponse
from utils.logger import logger

settings = get_settings()


# ── Lifespan (startup / shutdown) ────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run once on startup, then again on shutdown."""
    logger.info("Starting RAG Chatbot API...")
    logger.info(f"Environment: {settings.app_env}")
    await init_db()
    logger.info("Database connection verified.")
    yield
    logger.info("Shutting down...")
    await close_db()


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG FAQ Chatbot API",
    description="Enterprise-grade Retrieval-Augmented Generation chatbot for FAQ documents.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)


# ── CORS ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handlers ───────────────────────────────────────


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Catch validation / business-logic errors."""
    logger.warning(f"ValueError: {exc}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for unhandled exceptions.
    Logs the full traceback and returns a safe 500 response.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


# ── Routers ──────────────────────────────────────────────────────────

app.include_router(chat.router)
app.include_router(upload.router)


# ── Health check ─────────────────────────────────────────────────────


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Simple liveness probe."""
    return HealthResponse()


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Alternative health endpoint (for load balancers)."""
    return HealthResponse()


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
