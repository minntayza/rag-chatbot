# RAG FAQ Chatbot

Enterprise-grade Retrieval-Augmented Generation (RAG) chatbot that answers questions from your uploaded documents. Supports **PDF, TXT, and CSV** files with real-time streaming responses.

![Chat Interface](screenshots/chat-interface.png)

## Architecture

```
User Question → Embed → pgvector Search → Filter → Merge → LLM → Answer
                     │                                              │
                     └── Supabase PostgreSQL + pgvector ────────────┘
```

## Features

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

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 20+
- Supabase project with pgvector enabled

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
| `GET` | `/health` | Health check |

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
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # pydantic-settings from .env
│   ├── db.py                # Supabase client
│   ├── models.py            # Data classes
│   ├── schemas.py           # Pydantic request/response models
│   ├── requirements.txt     # Python dependencies
│   ├── supabase_migration.sql  # Full DB migration
│   ├── routers/
│   │   ├── chat.py          # /chat, /chat/stream, /chat/feedback
│   │   └── upload.py        # /upload (CRUD)
│   ├── services/
│   │   ├── embeddings.py    # Local + remote embedding backends
│   │   ├── generation.py    # Prompt building + LLM + streaming + retry
│   │   ├── ingestion.py     # 7-stage document pipeline
│   │   ├── rag.py           # End-to-end orchestrator
│   │   └── retrieval.py     # pgvector search + caching + fallback
│   └── utils/
│       └── logger.py        # Loguru configuration
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
| `CHUNK_SIZE` | Text chunk size (default: 500) |
| `CHUNK_OVERLAP` | Chunk overlap (default: 50) |
| `TOP_K_RESULTS` | Chunks to retrieve (default: 5) |

## License

MIT
