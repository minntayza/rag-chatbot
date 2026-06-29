"""
RAG (Retrieval-Augmented Generation) orchestrator.

Pipeline
--------
    1. Persist user message.
    2. Retrieve relevant context (→ services.retrieval).
    3. Generate answer via LLM (→ services.generation).
    4. Persist assistant response.
    5. Return answer + sources + pipeline metadata.

This module is the glue — it owns the end-to-end flow but delegates
retrieval and generation to dedicated services so each concern is
independently testable and replaceable.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from config import get_settings
from db import get_supabase
from models import ChatMessage
from services.generation import generate_answer, save_chat_message
from services.retrieval import RetrievalResult, retrieve_context
from utils.logger import logger

settings = get_settings()

# ── Fallback messages ────────────────────────────────────────────────

NO_DOCS_MESSAGE = (
    "I don't have any uploaded documents to search through. "
    "Please upload a PDF or TXT file first."
)

NO_MATCH_MESSAGE = (
    "I don't have enough information from the uploaded documents "
    "to answer that question. Try rephrasing or uploading documents "
    "that cover this topic."
)


# ── Public API ───────────────────────────────────────────────────────


async def query_rag(
    session_id: str,
    question: str,
) -> dict:
    """
    End-to-end RAG query — retrieve, generate, persist, return.

    Steps
    -----
    1. Persist the user's message.
    2. Call ``retrieve_context`` for relevant chunks.
    3. If chunks found → call ``generate_answer`` (with retry/timeout/token counting).
    4. Persist the assistant's response.
    5. Return the answer, sources, and pipeline metadata.

    Returns
    -------
    dict
        ``id``, ``session_id``, ``message``, ``sources``,
        ``retrieval_latency_ms``, ``fallback_used``,
        ``input_tokens``, ``output_tokens``, ``generation_latency_ms``,
        ``timestamp``.
    """
    logger.info(f"[rag] session={session_id} | q='{question[:80]}...'")

    # 1. Persist user message
    save_chat_message(session_id, "user", question)

    # 2. Retrieve context
    try:
        retrieval: RetrievalResult = await retrieve_context(question)
    except RuntimeError as exc:
        logger.error(f"[rag] retrieval failed: {exc}")
        raise

    # 3. Generate answer
    if retrieval.is_empty:
        answer = NO_DOCS_MESSAGE if not _has_any_documents() else NO_MATCH_MESSAGE
        sources: List[str] = []
        gen_input_tokens = 0
        gen_output_tokens = 0
        gen_latency_ms = 0.0
    else:
        try:
            gen_result = await generate_answer(
                context=retrieval.context,
                question=question,
            )
            answer = gen_result.answer
            gen_input_tokens = gen_result.input_tokens_estimate
            gen_output_tokens = gen_result.output_tokens_estimate
            gen_latency_ms = gen_result.latency_ms
        except RuntimeError:
            # LLM is down → return raw context as fallback
            logger.warning("[rag] LLM unavailable — returning raw context instead.")
            answer = (
                "⚠️ The AI service is temporarily unavailable. "
                "Here are the most relevant passages I found:\n\n"
                f"{retrieval.context[:2000]}"
            )
            gen_input_tokens = 0
            gen_output_tokens = 0
            gen_latency_ms = 0.0

        sources = retrieval.sources

    # 4. Persist assistant response
    assistant_msg = save_chat_message(session_id, "assistant", answer)

    logger.info(
        f"[rag] done | sources={sources} | "
        f"retrieval={retrieval.latency_ms:.0f}ms | "
        f"gen_latency={gen_latency_ms:.0f}ms | "
        f"tokens in={gen_input_tokens} out={gen_output_tokens}"
        f"{' | FALLBACK' if retrieval.fallback_used else ''}"
    )

    # 5. Return
    return {
        "id": assistant_msg.id,
        "session_id": session_id,
        "message": answer,
        "sources": sources,
        "retrieval_latency_ms": retrieval.latency_ms,
        "fallback_used": retrieval.fallback_used,
        "input_tokens": gen_input_tokens,
        "output_tokens": gen_output_tokens,
        "generation_latency_ms": gen_latency_ms,
        "timestamp": assistant_msg.timestamp,
    }


# ── Internal helpers ─────────────────────────────────────────────────


def _has_any_documents() -> bool:
    """Check if there is at least one document chunk in the database."""
    try:
        client = get_supabase()
        result = (
            client.table("documents")
            .select("id", count="exact")
            .limit(1)
            .execute()
        )
        return bool(result.count and result.count > 0)
    except Exception:
        return False
