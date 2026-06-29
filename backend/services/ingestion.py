"""
Document Ingestion Pipeline
===========================

Complete async pipeline for processing uploaded documents into RAG-ready chunks.

Pipeline stages
---------------
  1. **Validate**       — type, size, integrity
  2. **Extract text**   — PyMuPDF for PDF, direct read for TXT
  3. **Clean text**     — normalise whitespace, strip control chars, fix encoding
  4. **Split**          — RecursiveCharacterTextSplitter with configurable chunk/overlap
  5. **Deduplicate**    — content-hash check against existing Supabase rows
  6. **Embed + Store**  — batch-embed with retry, then batch-insert into Supabase

Each stage emits structured progress events so the caller can stream updates
to the frontend (e.g. via SSE or WebSocket).

Usage
-----
    from services.ingestion import ingest_document, ProgressEvent

    async for event in ingest_document(file_bytes, filename, content_type):
        print(f"{event.stage} — {event.message} — {event.percent}%")

Types
-----
    ProgressEvent   — (stage, message, percent, detail) sent after each major action
    IngestResult    — (filename, chunks_created, new_chunks, duplicates_skipped, duration_ms)
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io as _io
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF — synchronous but runs in a thread via run_in_executor
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from db import get_supabase
from services.embeddings import embed_texts
from services.cache import cache
from services.security import sanitise_document_text
from utils.logger import logger

settings = get_settings()

# ── Constants ────────────────────────────────────────────────────────

ALLOWED_MIME_TYPES: Set[str] = {
    "application/pdf",
    "text/plain",
    "text/csv",
    "application/csv",
}
MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024  # 20 MB
MAX_EMBEDDING_RETRIES: int = 3
EMBEDDING_RETRY_DELAY_SEC: float = 2.0  # base delay before exponential backoff
MAX_CHUNKS_PER_BATCH: int = 100           # Supabase insert batch ceiling


# ── Public data types ─────────────────────────────────────────────────


@dataclass
class ProgressEvent:
    """
    Emitted between pipeline stages so the caller can report progress.

    ``stage`` is one of: validating, extracting, cleaning, splitting,
    deduplicating, embedding, storing, done, error.
    """

    stage: str
    message: str
    percent: int = 0
    detail: Dict | None = None


@dataclass
class IngestResult:
    """Final result returned when ingestion completes."""

    filename: str
    chunks_created: int = 0
    new_chunks: int = 0
    duplicates_skipped: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None


# ── Shared helper: MIME type resolution ──────────────────────────────


def _resolve_content_type(content_type: str, filename: str) -> str:
    """
    Resolve the actual content type from MIME or file extension.

    Browsers sometimes send generic ``application/octet-stream`` —
    we fall back to extension-based detection.
    """
    if content_type and content_type != "application/octet-stream":
        return content_type

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    ext_map = {
        "pdf": "application/pdf",
        "txt": "text/plain",
        "csv": "text/csv",
    }
    return ext_map.get(ext, content_type or "application/octet-stream")


# ── Stage 1: Validate ────────────────────────────────────────────────


def _validate(
    filename: str,
    content_type: str,
    size_bytes: int,
) -> None:
    """
    Check that the uploaded file type and size are within allowed limits.

    Raises:
        ValueError: if the file fails validation.
    """
    detected = _resolve_content_type(content_type, filename)

    if detected not in ALLOWED_MIME_TYPES:
        raise ValueError(
            f"Unsupported file type '{content_type}'. Allowed: PDF, TXT, CSV."
        )

    if size_bytes > MAX_FILE_SIZE_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        raise ValueError(
            f"File too large ({size_mb:.1f} MB). Maximum allowed is 20 MB."
        )

    if size_bytes == 0:
        raise ValueError("File is empty.")


# ── Stage 2: Extract text ────────────────────────────────────────────


def _extract_text(file_bytes: bytes, content_type: str, filename: str = "") -> str:
    """
    Extract raw text from PDF, TXT, or CSV file.

    - **PDF**: PyMuPDF per-page extraction, joined with double-newlines.
    - **TXT**: UTF-8 decode (fallback latin-1).
    - **CSV**: rows converted to structured natural-language sentences.
    """
    detected = _resolve_content_type(content_type, filename)

    if detected == "application/pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            pages: List[str] = []
            for i, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    pages.append(page_text)
            return "\n\n".join(pages)
        finally:
            doc.close()

    if detected in ("text/csv", "application/csv"):
        return _extract_csv(file_bytes)

    # Plain-text: try UTF-8 first, fall back to latin-1
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed; falling back to latin-1.")
        return file_bytes.decode("latin-1", errors="replace")


def _extract_csv(file_bytes: bytes) -> str:
    """
    Convert CSV rows into natural-language text blocks.

    Strategy
    --------
    1. Detect delimiter (comma, tab, or pipe).
    2. Read the header row to get column names.
    3. Group rows by ``Doc_Type`` / ``Business_Unit`` so each chunk
       represents one coherent business transaction grouping.
    4. For each row, write a human-readable sentence:

           "Transaction TXN-2025-00001 on 13/10/2025: Paddy Trading
            purchased 2201.84 Baskets of Paddy from Ko Oo at 11600 MMK/unit.
            Total cost: 25541344 MMK. Payment via Cash. Status: ✅ OK."

    Why sentence form?
        Embedding models (all-MiniLM-L6-v2) capture meaning from sentences,
        not raw tabular data. Converting CSV to prose lets pgvector find
        the right rows when users ask "how much did we pay Ko Oo in October?"
    """
    text = file_bytes.decode("utf-8", errors="replace")
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return text  # header-only or empty

    # Detect delimiter
    delimiter = ","
    first = lines[0]
    if first.count("\t") > first.count(","):
        delimiter = "\t"
    elif first.count("|") > first.count(","):
        delimiter = "|"

    reader = csv.DictReader(_io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        return text

    # Normalise column names (strip whitespace, collapse spaces)
    cols = [c.strip().replace(" ", "_") for c in reader.fieldnames]

    # Identify key columns for sentence building
    id_col = next((c for c in cols if c.lower() in ("txn_id", "id", "transaction_id")), cols[0])
    date_col = next((c for c in cols if "date" in c.lower()), None)
    party_col = next((c for c in cols if c.lower() in ("party_name", "customer", "supplier", "vendor")), None)
    item_col = next((c for c in cols if c.lower() in ("item_name", "item", "product", "description")), None)
    qty_col = next((c for c in cols if c.lower() in ("qty", "quantity", "amount")), None)
    unit_col = next((c for c in cols if c.lower() in ("unit", "uom")), None)
    cost_col = next((c for c in cols if c.lower() in ("total_cost", "cost", "amount", "total_sales")), None)
    type_col = next((c for c in cols if c.lower() in ("doc_type", "type", "transaction_type")), None)
    bu_col = next((c for c in cols if c.lower() in ("business_unit", "unit", "department", "division")), None)
    cat_col = next((c for c in cols if c.lower() in ("category", "group")), None)
    status_col = next((c for c in cols if c.lower() in ("payment_status", "status")), None)
    method_col = next((c for c in cols if c.lower() in ("payment_method", "method")), None)
    cash_in_col = next((c for c in cols if c.lower() == "cash_in"), None)
    cash_out_col = next((c for c in cols if c.lower() == "cash_out"), None)

    # Build a mapping from original (possibly spaced) column names to normalised names
    name_map: Dict[str, str] = dict(zip(cols, reader.fieldnames))  # normalised → original

    # Group rows by doc type and business unit
    blocks: List[str] = []
    rows_processed = 0

    for row in reader:
        rows_processed += 1

        # Read values using the original fieldnames from the reader
        txn_id = row.get(name_map.get(id_col, id_col), row.get(id_col, ""))
        date = row.get(name_map.get(date_col, ""), "") if date_col else ""
        party = row.get(name_map.get(party_col, ""), "") if party_col else ""
        item = row.get(name_map.get(item_col, ""), "") if item_col else ""
        qty = row.get(name_map.get(qty_col, ""), "") if qty_col else ""
        unit = row.get(name_map.get(unit_col, ""), "") if unit_col else ""
        cost = row.get(name_map.get(cost_col, ""), "") if cost_col else ""
        doc_type = row.get(name_map.get(type_col, ""), "") if type_col else ""
        bu = row.get(name_map.get(bu_col, ""), "") if bu_col else ""
        category = row.get(name_map.get(cat_col, ""), "") if cat_col else ""
        status = row.get(name_map.get(status_col, ""), "") if status_col else ""
        method = row.get(name_map.get(method_col, ""), "") if method_col else ""
        cash_in = row.get(name_map.get(cash_in_col, ""), "") if cash_in_col else ""
        cash_out = row.get(name_map.get(cash_out_col, ""), "") if cash_out_col else ""

        # Build sentence
        parts: List[str] = []
        parts.append(f"Transaction {txn_id}")

        if date:
            parts.append(f" on {date}")
        if bu:
            parts.append(f" ({bu})")

        parts.append(f": {doc_type}" if doc_type else ":")

        if party:
            parts.append(f" — Party: {party}")
        if item:
            parts.append(f" | Item: {item}")
        if category and category != item:
            parts.append(f" ({category})")
        if qty:
            parts.append(f" | Quantity: {qty}")
            if unit:
                parts.append(f" {unit}")
        if cost and cost != "0":
            parts.append(f" | Cost: {cost} MMK")
        if cash_in and cash_in != "0":
            parts.append(f" | Cash In: {cash_in} MMK")
        if cash_out and cash_out != "0":
            parts.append(f" | Cash Out: {cash_out} MMK")
        if method:
            parts.append(f" | Payment Method: {method}")
        if status:
            parts.append(f" | Status: {status}")

        blocks.append("".join(parts))

    logger.info(
        f"[csv] extracted {len(blocks)} rows from {rows_processed} CSV rows"
    )

    # Group blocks into larger chunks for embedding (one per row is too expensive)
    group_size = max(1, len(blocks) // 50) if len(blocks) > 50 else 1
    if group_size > 1:
        grouped = []
        for i in range(0, len(blocks), group_size):
            group = blocks[i : i + group_size]
            grouped.append("\n".join(group))
        return "\n\n".join(grouped)

    return "\n".join(blocks)


# ── Stage 3: Clean text ──────────────────────────────────────────────


def _clean_text(raw_text: str) -> str:
    """
    Normalise and sanitise extracted text before chunking.

    Transformations (in order):
        1. Replace carriage returns (\\r) and windows line-endings (\\r\\n) → \\n
        2. Collapse sequences of 3+ newlines into exactly 2 (preserve paragraph breaks)
        3. Replace non-breaking spaces and zero-width characters with regular spaces
        4. Strip ASCII control characters except tab (\\t) and newline (\\n)
        5. Replace tab characters with a single space
        6. Collapse multiple spaces (3+) into a single space
        7. Strip leading/trailing whitespace from every line

    Returns:
        Cleaned, normalised text ready for chunking.
    """
    # 1. Normalise line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Collapse excessive blank lines (keep at most 2 consecutive newlines)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)  # second pass for edge case

    # 3. Replace non-breaking spaces and zero-width chars
    text = text.replace(" ", " ").replace("​", "").replace("‎", "").replace("‏", "")

    # 4. Strip ASCII control characters (0x00–0x1F), but keep \t (0x09) and \n (0x0A)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

    # 5. Replace tabs with a single space
    text = text.replace("\t", " ")

    # 6. Collapse >2 consecutive spaces
    text = re.sub(r" {3,}", " ", text)

    # 7. Clean per-line whitespace
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove any completely blank lines that remain after stripping
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── Stage 4: Split ───────────────────────────────────────────────────


def _split_text(text: str) -> List[str]:
    """
    Split cleaned text into overlapping chunks.

    Uses LangChain's **RecursiveCharacterTextSplitter** with these separators
    (tried in order until a chunk fits within ``chunk_size``):

        1. ``\\n\\n``  — paragraph break (preferred)
        2. ``\\n``     — line break
        3. ``. ``     — sentence boundary
        4. `` ``       — word boundary
        5. ``""``      — character boundary (last resort)

    Chunks shorter than 20 characters are discarded as noise.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)

    # Discard chunks that are too short to be useful
    chunks = [c for c in raw_chunks if len(c.strip()) >= 20]
    return chunks


# ── Stage 5: Deduplicate ─────────────────────────────────────────────


def _compute_content_hashes(chunks: List[str]) -> List[str]:
    """
    Compute a **SHA-256 hex digest** for each chunk.

    We hash the *normalised whitespace* version of the chunk so that
    semantically identical chunks with different whitespace still collide.

    Returns:
        A list of hex digests, one per chunk, in the same order.
    """
    return [
        hashlib.sha256(
            " ".join(chunk.split()).encode("utf-8")
        ).hexdigest()
        for chunk in chunks
    ]


async def _find_existing_hashes(
    hashes: List[str],
    filename: str,
) -> Set[str]:
    """
    Query Supabase for any chunks of *other files* that share these hashes.

    Only checks across *different* filenames — we allow re-uploading
    the same file to replace its own chunks.

    Returns:
        Set of hex digests that already exist in the database.
    """
    if not hashes:
        return set()

    client = get_supabase()
    existing: Set[str] = set()

    all_existing = (
        client.table("documents")
        .select("id, filename, content")
        .eq("filename", filename)
        .execute()
    )

    for row in (all_existing.data or []):
        h = hashlib.sha256(" ".join(row["content"].split()).encode("utf-8")).hexdigest()
        existing.add(h)

    # Cross-file check: sample from other files (limit 1000 to stay performant)
    cross = (
        client.table("documents")
        .select("content")
        .neq("filename", filename)
        .limit(1000)
        .execute()
    )

    for row in (cross.data or []):
        h = hashlib.sha256(" ".join(row["content"].split()).encode("utf-8")).hexdigest()
        existing.add(h)

    return existing


# ── Stage 6: Embed + Store ───────────────────────────────────────────


async def _embed_with_retry(
    chunks: List[str],
) -> List[List[float]]:
    """
    Batch-embed chunks with **exponential-backoff retry**.

    Uses ``asyncio.sleep`` (non-blocking) so the event loop stays free.
    First attempt gets extra time for model download on cold start.

    Raises:
        RuntimeError: if all ``MAX_EMBEDDING_RETRIES`` attempts fail.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(1, MAX_EMBEDDING_RETRIES + 1):
        try:
            embeddings = embed_texts(chunks)
            if attempt > 1:
                logger.info(f"Embedding succeeded on attempt {attempt}/{MAX_EMBEDDING_RETRIES}.")
            return embeddings
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_EMBEDDING_RETRIES:
                delay = EMBEDDING_RETRY_DELAY_SEC * (2 ** (attempt - 1))  # 2s → 4s → 8s
                logger.warning(
                    f"Embedding attempt {attempt}/{MAX_EMBEDDING_RETRIES} failed: {exc}. "
                    f"Retrying in {delay:.0f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"Embedding failed after {MAX_EMBEDDING_RETRIES} attempts: {exc}")

    raise RuntimeError(
        f"Embedding failed after {MAX_EMBEDDING_RETRIES} attempts."
    ) from last_exc


async def _store_chunks(
    chunks: List[str],
    embeddings: List[List[float]],
    content_hashes: List[str],
    filename: str,
    existing_hashes: Set[str],
) -> Tuple[int, int]:
    """
    Insert chunks into Supabase, skipping any whose hash already exists.

    Returns:
        (new_chunks_inserted, duplicates_skipped)
    """
    client = get_supabase()
    rows: List[Dict] = []
    duplicates = 0

    for idx, (chunk, embedding, chash) in enumerate(zip(chunks, embeddings, content_hashes)):
        if chash in existing_hashes:
            duplicates += 1
            continue

        rows.append({
            "id": str(uuid.uuid4()),
            "filename": filename,
            "chunk_index": idx,
            "content": chunk,
            "embedding": embedding,
            "metadata": {
                "source": filename,
                "chunk_index": idx,
                "content_hash": chash,
                "chunk_length": len(chunk),
            },
        })

    if not rows:
        logger.info(f"All {len(chunks)} chunks are duplicates — nothing to insert.")
        return 0, duplicates

    # Batch-insert in chunks of MAX_CHUNKS_PER_BATCH
    inserted = 0
    for start in range(0, len(rows), MAX_CHUNKS_PER_BATCH):
        batch = rows[start : start + MAX_CHUNKS_PER_BATCH]
        client.table("documents").insert(batch).execute()
        inserted += len(batch)
        logger.debug(f"Inserted batch {start // MAX_CHUNKS_PER_BATCH + 1}: {len(batch)} rows")

    return inserted, duplicates


# ── Public API: main orchestrator ────────────────────────────────────


async def ingest_document(
    file_bytes: bytes,
    filename: str,
    content_type: str,
) -> AsyncGenerator[ProgressEvent, None]:
    """
    Run the full ingestion pipeline, yielding progress events at each stage.

    This is an **async generator** so callers can stream progress:

        async for event in ingest_document(data, name, mime):
            send_to_frontend(event)

    Stage sequence:

        validating → extracting → cleaning → splitting
        → deduplicating → embedding → storing → done

    Yields:
        ``ProgressEvent`` at the start and end of each stage.

    Raises:
        ValueError:    validation failure (bad file)
        RuntimeError:  embedding exhaustion after retries
        Exception:     unexpected errors during extraction or storage
    """
    start_time = time.monotonic()

    try:
        # ── Stage 1: Validate ──────────────────────────────
        stage_start = time.monotonic()
        _validate(filename, content_type, len(file_bytes))
        logger.info(
            f"[ingestion] validating — {filename} | "
            f"type={content_type} | "
            f"size={len(file_bytes) / 1024:.1f} KB"
        )
        yield ProgressEvent(
            stage="validating",
            message=f"Validated '{filename}'",
            percent=5,
            detail={
                "size_kb": round(len(file_bytes) / 1024, 1),
                "type": content_type,
            },
        )

        # ── Stage 2: Extract text ──────────────────────────
        try:
            raw_text = _extract_text(file_bytes, content_type, filename)
        except Exception as exc:
            logger.error(f"[ingestion] extraction failed for '{filename}': {exc}")
            yield ProgressEvent(
                stage="error",
                message=f"Could not extract text: {exc}",
                percent=0,
            )
            return

        char_count = len(raw_text)
        logger.info(
            f"[ingestion] extracting — {char_count} chars extracted "
            f"({time.monotonic() - stage_start:.1f}s)"
        )
        yield ProgressEvent(
            stage="extracting",
            message=f"Extracted {char_count} characters",
            percent=15,
            detail={"char_count": char_count},
        )

        if not raw_text.strip():
            yield ProgressEvent(
                stage="error",
                message="File appears to contain no readable text.",
                percent=0,
            )
            return

        # ── Stage 3: Clean ──────────────────────────────────
        stage_start = time.monotonic()

        # Security: sanitise document text against RAG poisoning
        raw_text = sanitise_document_text(raw_text, filename)

        clean_text = _clean_text(raw_text)
        cleaned_chars = len(clean_text)
        removed = char_count - cleaned_chars
        logger.info(
            f"[ingestion] cleaning — {cleaned_chars} chars after cleaning "
            f"({removed} chars removed, {time.monotonic() - stage_start:.1f}s)"
        )
        yield ProgressEvent(
            stage="cleaning",
            message=f"Cleaned — {cleaned_chars} characters retained",
            percent=30,
            detail={"chars_before": char_count, "chars_after": cleaned_chars, "removed": removed},
        )

        if not clean_text.strip():
            yield ProgressEvent(
                stage="error",
                message="No text remained after cleaning.",
                percent=0,
            )
            return

        # ── Stage 4: Split ─────────────────────────────────
        stage_start = time.monotonic()
        chunks = _split_text(clean_text)
        chunk_count = len(chunks)
        avg_len = round(sum(len(c) for c in chunks) / max(chunk_count, 1))
        logger.info(
            f"[ingestion] splitting — {chunk_count} chunks "
            f"(avg {avg_len} chars, {time.monotonic() - stage_start:.1f}s)"
        )
        yield ProgressEvent(
            stage="splitting",
            message=f"Split into {chunk_count} chunks (avg {avg_len} chars)",
            percent=45,
            detail={
                "chunk_count": chunk_count,
                "avg_chunk_length": avg_len,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
        )

        if not chunks:
            yield ProgressEvent(
                stage="error",
                message="No chunks produced — text may be too short.",
                percent=0,
            )

            return

        # ── Stage 5: Deduplicate ────────────────────────────
        stage_start = time.monotonic()
        content_hashes = _compute_content_hashes(chunks)
        existing_hashes = await _find_existing_hashes(content_hashes, filename)
        dup_count = len(set(content_hashes) & existing_hashes)
        new_count = len(content_hashes) - dup_count
        logger.info(
            f"[ingestion] deduplicating — {new_count} new / {dup_count} duplicate "
            f"({time.monotonic() - stage_start:.1f}s)"
        )
        yield ProgressEvent(
            stage="deduplicating",
            message=f"Found {new_count} new chunks, {dup_count} duplicates",
            percent=60,
            detail={
                "total_chunks": chunk_count,
                "new_chunks": new_count,
                "duplicates_skipped": dup_count,
            },
        )

        if new_count == 0:
            # All chunks are duplicates — skip embedding entirely
            yield ProgressEvent(
                stage="done",
                message=f"All {chunk_count} chunks already exist — skipped.",
                percent=100,
                detail={"filename": filename, "chunks_created": 0, "duplicates_skipped": dup_count, "duration_ms": (time.monotonic() - start_time) * 1000},
            )
            return

        # ── Stage 6: Embed ─────────────────────────────────
        stage_start = time.monotonic()
        yield ProgressEvent(
            stage="embedding",
            message=f"Generating embeddings for {new_count} chunks...",
            percent=70,
            detail={"chunks_to_embed": new_count, "embedding_model": settings.embedding_model},
        )

        try:
            embeddings = await _embed_with_retry(chunks)
        except RuntimeError as exc:
            logger.error(f"[ingestion] embedding exhausted retries: {exc}")
            yield ProgressEvent(
                stage="error",
                message=f"Embedding failed after {MAX_EMBEDDING_RETRIES} retries: {exc}",
                percent=0,
            )
            return

        embed_time = time.monotonic() - stage_start
        logger.info(
            f"[ingestion] embedding — {len(embeddings)} vectors in {embed_time:.1f}s "
            f"({len(embeddings) / max(embed_time, 0.001):.0f} vec/s)"
        )
        yield ProgressEvent(
            stage="embedding",
            message=f"Generated {len(embeddings)} embeddings ({embed_time:.1f}s)",
            percent=85,
            detail={"vectors_generated": len(embeddings), "duration_s": round(embed_time, 1)},
        )

        # ── Stage 7: Store ──────────────────────────────────
        stage_start = time.monotonic()
        inserted, skipped = await _store_chunks(
            chunks, embeddings, content_hashes, filename, existing_hashes
        )
        store_time = time.monotonic() - stage_start
        total_duration = (time.monotonic() - start_time) * 1000

        logger.info(
            f"[ingestion] storing — {inserted} inserted, {skipped} skipped "
            f"({store_time:.1f}s) | total={total_duration:.0f}ms"
        )
        yield ProgressEvent(
            stage="storing",
            message=f"Stored {inserted} chunks, skipped {skipped} duplicates",
            percent=95,
            detail={"inserted": inserted, "skipped": skipped},
        )

        # Invalidate retrieval cache so next queries see new documents
        if inserted > 0:
            await cache.clear()

        # ── Done ─────────────────────────────────────────────
        yield ProgressEvent(
            stage="done",
            message=f"Ingestion complete — {inserted} chunks from '{filename}'",
            percent=100,
            detail={
                "filename": filename,
                "chunks_created": inserted,
                "duplicates_skipped": skipped,
                "total_chunks": chunk_count,
                "duration_ms": round(total_duration, 0),
            },
        )

    except Exception as exc:
        total_duration = (time.monotonic() - start_time) * 1000
        logger.error(f"[ingestion] unhandled error for '{filename}': {exc}", exc_info=True)
        yield ProgressEvent(
            stage="error",
            message=f"Ingestion failed: {exc}",
            percent=0,
            detail={"error": str(exc), "duration_ms": round(total_duration, 0)},
        )
