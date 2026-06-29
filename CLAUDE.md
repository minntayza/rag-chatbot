# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Enterprise-grade RAG FAQ Chatbot. Users upload documents (PDF, TXT, CSV), the system indexes them with pgvector embeddings, and answers questions via LLM with real-time SSE streaming.

**Architecture flow:** User Question → Security Check → Embed → pgvector Cosine Search → Filter → Merge Context → LLM (Mimo 2.5 Pro) → Streamed Answer

## Commands

### Frontend (root directory)

```bash
npm run dev          # Vite dev server (frontend)
npm run build        # Production build
npm run lint         # ESLint + Prettier check
npm run format       # Prettier auto-format
```

### Backend (`backend/` directory)

```bash
# Start dev server (from backend/)
uvicorn main:app --reload --port 8000

# Tests (from backend/)
pytest                                    # all tests (99 total)
pytest tests/unit/ -v --tb=short          # unit tests only
pytest tests/integration/ -v --tb=short   # integration tests (requires Supabase secrets)
pytest tests/unit/test_ingestion.py -v    # single test file
pytest -k "test_name" -v                  # single test by name

# Linting (from backend/)
ruff check . --select E,F,I,N,W --ignore E501
mypy . --ignore-missing-imports --check-untyped-defs
```

### Docker

```bash
docker compose up -d                        # API + Redis
docker compose --profile monitoring up -d   # + Prometheus + Grafana
docker compose logs -f api                  # tail backend logs
```

### Full stack local dev

Run backend (`uvicorn main:app --reload --port 8000` from `backend/`) and frontend (`npm run dev` from root) in separate terminals. Frontend proxies to `http://localhost:8000`.

## Environment Setup

Copy `backend/.env.example` to `backend/.env`. Required variables:
- `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY` — Supabase project credentials
- `LLM_API_KEY` — Mimo API key
- `REDIS_URL` — optional, falls back to in-memory LRU cache

Settings are validated at startup via pydantic-settings in `backend/config.py`. Missing required vars cause an immediate startup failure.

## Backend Architecture

**Entry point:** `backend/main.py` — FastAPI app with lifespan (DB init + cache warm-up), CORS, security headers, and Prometheus middleware.

**Config:** `backend/config.py` — `Settings` class (pydantic-settings), cached singleton via `get_settings()`.

**Routers** (`backend/routers/`):
- `chat.py` — `/chat`, `/chat/stream` (SSE), `/chat/{session_id}`, `/chat/feedback`
- `upload.py` — POST/GET/DELETE `/upload`
- All endpoints are rate-limited (sliding window per-IP)

**Services** (`backend/services/`):
- `ingestion.py` — 7-stage document pipeline: validate → extract → clean → split → deduplicate → embed → store
- `retrieval.py` — pgvector cosine search with two-tier cache (LRU + Redis), fallback threshold
- `generation.py` — Prompt building, LLM streaming/non-streaming, retry with exponential backoff, token counting
- `rag.py` — Orchestrator combining retrieval + generation
- `cache.py` — Unified Redis + LRU cache with JSON serialization
- `security.py` — Rate limiter, prompt injection detection (6 regex patterns), document sanitization
- `metrics.py` — 20+ Prometheus gauges + request middleware
- `evaluation.py` — RAGAS quality scoring (heuristic mode, 10% sample rate)
- `embeddings.py` — Local sentence-transformers (`all-MiniLM-L6-v2`) or remote embedding backends

**Database:** Supabase PostgreSQL with pgvector. Tables: `documents`, `chat_history`, `feedback` (all RLS-enabled). Vector search via Supabase RPC function `match_documents()`. Migration SQL in `backend/supabase_migration.sql`.

**Models/Schemas:** `backend/models.py` (DB models), `backend/schemas.py` (Pydantic request/response schemas)

## Frontend Architecture

**Stack:** React 19 + TanStack Start (SSR) + TanStack Router (file-based routing) + TanStack React Query + Tailwind CSS v4 + shadcn/ui (new-york style).

**Entry:** `src/start.ts` → `src/router.tsx` → `src/routes/__root.tsx` (root layout with QueryClientProvider)

**Main route:** `src/routes/index.tsx` — Single-page chat UI with file upload, streaming responses, and RAG source debugging sidebar.

**API client:** `src/lib/api.ts` — Session-based (random UUID in sessionStorage). Functions: `sendMessage()`, `streamMessage()` (async generator for SSE), `getHistory()`, `submitFeedback()`, `uploadDocument()`, `listDocuments()`, `deleteDocument()`. All calls target `http://localhost:8000`.

**UI components:** 47 shadcn/ui components in `src/components/ui/`. Config in `components.json` (new-york style, slate base, lucide icons).

**SSR error handling:** `src/server.ts` wraps TanStack Start to catch h3 swallowed errors.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`): 3 sequential jobs — lint (ruff + mypy) → test (unit + integration with Supabase secrets) → build Docker image.

## Testing Patterns

- `backend/tests/unit/` — Mocked dependencies (Supabase, LLM, embeddings)
- `backend/tests/integration/` — Real Supabase connection (requires secrets in env)
- `backend/pytest.ini` — `asyncio_mode=auto`, strict markers
- `backend/tests/conftest.py` — Shared fixtures

## Key Conventions

- Backend uses `loguru` for logging, not stdlib `logging`
- All config comes from `backend/config.py` `get_settings()` — never hardcode values
- Backend uses Supabase Python SDK (REST), not direct psycopg2/SQLAlchemy connections for queries
- Frontend API client hardcodes `http://localhost:8000` — update `src/lib/api.ts` for production
- Embeddings default to local `all-MiniLM-L6-v2` (384 dimensions) — no API key needed unless overriding
