# RAG FAQ Chatbot

Enterprise-grade Retrieval-Augmented Generation (RAG) chatbot that answers questions from your uploaded documents. Supports **PDF, TXT, and CSV** files with real-time streaming responses.

![Chat Interface](screenshots/chat-interface.png)

## Architecture

```
User Question → Security Check → Embed → pgvector Search → Filter → Merge → LLM → Answer
                     │                │                                              │
                     │                └── Supabase PostgreSQL + pgvector ────────────┘
                     │
                     ├── Rate Limiting (sliding window per-IP)
                     ├── Prompt Injection Detection (6 regex patterns)
                     └── Input Sanitisation
```

## Features

### Core RAG
- 📄 **Multi-format ingestion** — PDF, TXT, CSV with auto-detection
- 🔍 **Semantic search** — pgvector cosine similarity across document chunks
- 🚫 **No hallucination** — Strict system prompt, refuses to guess
- ⚡ **Streaming responses** — Word-by-word SSE streaming via Mimo 2.5 Pro
- 💾 **Chat history** — Session-based persistence in Supabase
- 🔎 **RAG debugging** — Sidebar shows retrieved sources with metadata
- 📊 **CSV intelligence** — Auto-converts spreadsheet rows to natural language for embedding
- 🔄 **Duplicate detection** — SHA-256 content hashing prevents redundant chunks
- ⏱️ **Retry with backoff** — Exponential retry on transient LLM/embedding failures
- 📈 **Token counting** — tiktoken integration for usage monitoring
- 🗄️ **Row Level Security** — All tables RLS-enabled with granular policies

### Security
- 🛡️ **Rate limiting** — Sliding window per-IP (20 chat/60s, 5 stream/60s, 10 upload/120s)
- 🔒 **Prompt injection detection** — 6 regex patterns block system overrides, role impersonation, delimiter attacks, DAN/jailbreak
- 🧹 **Document sanitisation** — Strips control characters, HTML tags, null bytes before storage
- 🔐 **Security headers** — X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, HSTS
- 📝 **Input validation** — Pydantic schemas enforce types and constraints on all endpoints

### Enterprise
- 📊 **Prometheus metrics** — 20+ metrics: request counts, latencies, token usage, cache hits, RAGAS scores
- 📈 **Grafana dashboards** — Pre-configured panels for monitoring RAG pipeline health
- 🧪 **RAGAS evaluation** — Automated quality scoring (faithfulness, answer relevancy, context recall, context precision)
- 🗄️ **Redis caching** — Distributed cache with JSON serialisation (no pickle RCE risk)
- 🐳 **Docker** — Multi-stage production Dockerfile with non-root user and health checks
- 🔄 **CI/CD** — GitHub Actions: lint → test → build pipeline
- ✅ **99 tests** — 47 unit + 37 integration + 15 schema tests with pytest

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI (Python 3.12) |
| **Vector DB** | Supabase PostgreSQL + pgvector |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **LLM** | Mimo 2.5 Pro (OpenAI-compatible API) |
| **Frontend** | React 19 + TanStack Start + shadcn/ui |
| **Styling** | Tailwind CSS v4 |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter |
| **PDF Extraction** | PyMuPDF |
| **Caching** | Redis + in-memory LRU fallback |
| **Metrics** | Prometheus + Grafana |
| **Evaluation** | RAGAS (heuristic mode) |
| **Container** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Testing** | pytest (99 tests) |

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 20+
- Supabase project with pgvector enabled
- Docker & Docker Compose (optional, for production)

### 1. Clone and install

```bash
git clone https://github.com/minntayza/rag-chatbot.git
cd rag-chatbot
```

### 2. Backend setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Supabase and LLM credentials
```

### 3. Database setup

Run the migration SQL in your Supabase SQL Editor:

```bash
cat backend/supabase_migration.sql
```

This creates tables (`documents`, `chat_history`, `feedback`), indexes, RLS policies, and the `match_documents()` pgvector search function.

### 4. Start backend

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000
```

### 5. Start frontend

```bash
npm install
npm run dev
```

Open **http://localhost:8080** (or the port Vite assigns).

### 6. Docker (Production)

```bash
# Build and run with Docker Compose
docker compose up -d

# With monitoring stack (Prometheus + Grafana)
docker compose --profile monitoring up -d

# API: http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3001 (admin/admin)
```

### 7. Run tests

```bash
cd backend
pip install pytest pytest-asyncio pytest-cov
pytest -v --tb=short
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload PDF, TXT, or CSV (returns chunk count) |
| `GET` | `/upload` | List all uploaded documents |
| `DELETE` | `/upload/{filename}` | Delete all chunks for a file |
| `POST` | `/chat` | Send a question, get a RAG answer (non-streaming) |
| `POST` | `/chat/stream` | Same, with SSE word-by-word streaming |
| `GET` | `/chat/{session_id}` | Conversation history for a session |
| `POST` | `/chat/feedback` | Submit thumbs up/down on a response |
| `GET` | `/health` | Deep readiness probe (DB + cache + LLM) |
| `GET` | `/metrics` | Prometheus metrics endpoint |

![API Schema](screenshots/api-schema.png)

## How It Works

### 1. Document Ingestion (`services/ingestion.py`)

```
Upload → Validate → Extract text → Clean → Split → Deduplicate → Embed → Store
```

- **PDF**: PyMuPDF per-page extraction
- **TXT**: UTF-8 decode with fallback
- **CSV**: Rows converted to natural language sentences before embedding

### 2. Retrieval (`services/retrieval.py`)

```
Question → Embed → pgvector RPC → Score filter (≥0.25) → Top-K → Merge → Cache
```

- Cosine similarity via `match_documents()` database function
- Two-tier cache: in-memory LRU + optional Redis
- Fallback threshold (0.15) when primary returns nothing

### 3. Generation (`services/generation.py`)

```
Context + Prompt → Token count → LLM (stream/non-stream) → Retry → Store
```

- Prompt: "You are a professional customer support assistant. Answer ONLY from context."
- Retry: 429/5xx retried with exponential backoff (1s → 2s → 4s)
- Streaming: SSE `data: {"type":"token","token":"..."}` events
- Token estimation via tiktoken (cl100k_base)

## Database Schema

### `documents`
| Column | Type | Description |
|---|---|---|
| `id` | UUID | Primary key |
| `filename` | TEXT | Source filename |
| `chunk_index` | INT | Chunk order |
| `content` | TEXT | Chunk text |
| `embedding` | vector(384) | Cosine-searchable vector |
| `metadata` | JSONB | Source, hash, length |

### `chat_history`
| Column | Type | Description |
|---|---|---|
| `id` | UUID | Primary key |
| `session_id` | TEXT | Client-generated session |
| `role` | TEXT | `user` or `assistant` |
| `message` | TEXT | Message content |
| `timestamp` | TIMESTAMPTZ | Auto-generated |

### `feedback`
| Column | Type | Description |
|---|---|---|
| `id` | UUID | Primary key |
| `message_id` | UUID | FK → chat_history CASCADE |
| `rating` | FLOAT | +1.0 (up) / -1.0 (down) |
| `comment` | TEXT | Optional note |

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI entry point + security headers + metrics
│   ├── config.py            # pydantic-settings from .env
│   ├── db.py                # Supabase client
│   ├── models.py            # Data classes
│   ├── schemas.py           # Pydantic request/response models
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile           # Multi-stage production build
│   ├── Dockerfile.dev       # Development with hot-reload
│   ├── pytest.ini           # Test configuration
│   ├── supabase_migration.sql  # Full DB migration
│   ├── routers/
│   │   ├── chat.py          # /chat, /chat/stream, /chat/feedback + rate limits
│   │   └── upload.py        # /upload (CRUD) + rate limits
│   ├── services/
│   │   ├── cache.py         # Unified Redis + LRU cache (JSON serialisation)
│   │   ├── embeddings.py    # Local + remote embedding backends
│   │   ├── evaluation.py    # RAGAS quality metrics (heuristic mode)
│   │   ├── generation.py    # Prompt building + LLM + streaming + retry
│   │   ├── ingestion.py     # 7-stage document pipeline
│   │   ├── metrics.py       # Prometheus metrics (20+ gauges)
│   │   ├── rag.py           # End-to-end orchestrator
│   │   ├── retrieval.py     # pgvector search + fallback
│   │   └── security.py      # Rate limiter, injection detection, sanitisation
│   ├── tests/
│   │   ├── conftest.py      # Shared fixtures
│   │   ├── unit/
│   │   │   ├── test_ingestion.py    # 24 tests
│   │   │   ├── test_evaluation.py   # 7 tests
│   │   │   └── test_schemas.py      # 16 tests
│   │   └── integration/
│   │       ├── test_api.py          # 12 tests
│   │       └── test_pipeline.py     # 25 tests
│   └── utils/
│       └── logger.py        # Loguru configuration
├── docker-compose.yml       # API + Redis + Prometheus + Grafana
├── .github/workflows/
│   └── ci.yml               # Lint → Test → Build pipeline
├── src/
│   ├── lib/api.ts           # Frontend API client
│   ├── routes/
│   │   └── index.tsx        # Chat UI
│   └── components/ui/       # shadcn/ui components
└── package.json
```

## Configuration

All settings via `.env` (copy from `.env.example`):

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_ANON_KEY` | Supabase anon/public key |
| `LLM_API_KEY` | Mimo/OpenAI API key |
| `LLM_BASE_URL` | API base URL |
| `LLM_MODEL` | Model name (e.g. `mimo-v2.5-pro`) |
| `EMBEDDING_MODEL` | Local model name (default: `all-MiniLM-L6-v2`) |
| `EMBEDDING_DIMENSION` | Vector dimensions (default: 384) |
| `CHUNK_SIZE` | Text chunk size (default: 500) |
| `CHUNK_OVERLAP` | Chunk overlap (default: 50) |
| `TOP_K_RESULTS` | Chunks to retrieve (default: 5) |
| `SIMILARITY_THRESHOLD` | Minimum cosine score (default: 0.25) |
| `FALLBACK_THRESHOLD` | Lower bound for fallback (default: 0.15) |
| `REDIS_URL` | Redis connection string (optional) |
| `CORS_ORIGINS` | Comma-separated allowed origins |
| `PROMETHEUS_ENABLED` | Enable metrics endpoint (default: true) |
| `EVALUATION_SAMPLE_RATE` | RAGAS eval sample rate (default: 0.1) |

## Monitoring

### Prometheus Metrics

20+ metrics exposed at `/metrics`:

- **HTTP**: request counts, latencies, in-flight requests
- **RAG**: retrieval duration, generation duration, query counts
- **Tokens**: input/output token usage
- **Cache**: hits, misses
- **Documents**: uploads, chunks, embeddings
- **RAGAS**: faithfulness, answer relevancy, context recall, context precision

### Grafana Dashboards

Pre-configured panels for:
- Request rate and latency percentiles
- RAG pipeline performance (retrieval vs generation time)
- Token usage and cost estimation
- Cache hit rate
- RAGAS quality scores over time

### RAGAS Evaluation

Automated quality scoring on 10% of queries (configurable):

| Metric | Description | Target |
|---|---|---|
| Faithfulness | Answer grounded in context | ≥ 0.7 |
| Answer Relevancy | Answer matches question intent | ≥ 0.6 |
| Context Recall | Relevant chunks retrieved | ≥ 0.5 |
| Context Precision | Retrieved chunks are relevant | ≥ 0.5 |

## Security

### Rate Limiting

Sliding window per-IP limits:
- Chat: 20 requests / 60 seconds
- Streaming: 5 requests / 60 seconds
- Upload: 10 requests / 120 seconds
- Feedback: 30 requests / 60 seconds

### Prompt Injection Detection

6 regex patterns detect and block:
- System prompt overrides ("ignore previous instructions")
- Role impersonation ("you are now", "act as")
- Delimiter attacks ("```system", "---END---")
- DAN/jailbreak patterns

### Document Sanitisation

Uploaded documents are sanitised before storage:
- Control characters stripped
- HTML tags removed
- Null bytes removed
- Content normalised to NFC unicode

## License

MIT
