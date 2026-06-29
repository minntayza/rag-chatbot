"""
Chat router — conversational Q&A, message history, and streaming.

Endpoints
---------
  POST /chat              → send a question, get a RAG answer (non-streaming)
  POST /chat/stream       → same, but streams the answer via SSE
  GET  /chat/{session_id} → full conversation history
  POST /chat/feedback     → submit user feedback on a response
"""

from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from db import get_supabase
from models import Feedback
from schemas import (
    ChatHistoryResponse,
    ChatMessageOut,
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from services.generation import generate_answer_stream, save_chat_message
from services.rag import query_rag
from services.retrieval import RetrievalResult, retrieve_context
from services.security import (
    check_rate_limit,
    sanitise_question,
    get_security_headers,
)
from utils.logger import logger

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── POST /chat (non-streaming) ───────────────────────────────────────


@router.post("", response_model=ChatResponse, status_code=201)
async def send_message(payload: ChatRequest, request: Request):
    """Accept a user question, run the full RAG pipeline, and return the answer."""
    # Rate limiting
    check_rate_limit(request, "chat")

    # Sanitise question
    sanitised, was_blocked = sanitise_question(payload.message)

    if was_blocked:
        # Refuse to process — don't even call the LLM
        return _refuse_response(payload.session_id, payload.message)

    try:
        result = await query_rag(
            session_id=payload.session_id,
            question=sanitised,
        )
        return ChatResponse(**result)
    except RuntimeError as exc:
        logger.error(f"RAG pipeline failed: {exc}")
        raise HTTPException(status_code=502, detail=str(exc))


# ── POST /chat/stream (SSE streaming) ────────────────────────────────


@router.post("/stream", status_code=201)
async def send_message_stream(payload: ChatRequest, request: Request):
    """Stream the RAG answer via Server-Sent Events (SSE)."""
    check_rate_limit(request, "stream")

    session_id = payload.session_id
    question = payload.message

    # Sanitise question
    sanitised, was_blocked = sanitise_question(question)

    save_chat_message(session_id, "user", sanitised)

    async def _event_stream():
        import json as _json

        if was_blocked:
            yield f"data: {_json.dumps({'type': 'error', 'message': 'Request blocked by security filter.'})}\n\n"
            return

        try:
            retrieval: RetrievalResult = await retrieve_context(sanitised)
        except RuntimeError as exc:
            yield f"data: {_json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return

        yield f"data: {_json.dumps({'type': 'status', 'message': 'Retrieving context...', 'chunks_found': len(retrieval.chunks), 'retrieval_latency_ms': retrieval.latency_ms})}\n\n"

        if retrieval.is_empty:
            from services.rag import _has_any_documents
            has_docs = _has_any_documents()
            fallback = (
                "I don't have any uploaded documents to search through. Please upload a PDF or TXT file first."
                if not has_docs
                else "I don't have enough information from the uploaded documents to answer that question. Try rephrasing or uploading documents that cover this topic."
            )
            yield f"data: {_json.dumps({'type': 'token', 'token': fallback})}\n\n"
            yield f"data: {_json.dumps({'type': 'done', 'sources': [], 'retrieval_latency_ms': retrieval.latency_ms, 'fallback_used': retrieval.fallback_used})}\n\n"
            save_chat_message(session_id, "assistant", fallback)
            return

        full_answer: List[str] = []
        try:
            async for token in generate_answer_stream(
                context=retrieval.context,
                question=sanitised,
            ):
                full_answer.append(token)
                yield f"data: {_json.dumps({'type': 'token', 'token': token})}\n\n"

            answer = "".join(full_answer)
            yield f"data: {_json.dumps({'type': 'done', 'sources': retrieval.sources, 'retrieval_latency_ms': retrieval.latency_ms, 'fallback_used': retrieval.fallback_used})}\n\n"
        except RuntimeError as exc:
            answer = (
                "⚠️ The AI service is temporarily unavailable. "
                "Here are the most relevant passages I found:\n\n"
                f"{retrieval.context[:2000]}"
            )
            yield f"data: {_json.dumps({'type': 'token', 'token': answer})}\n\n"
            yield f"data: {_json.dumps({'type': 'done', 'sources': retrieval.sources, 'error': str(exc)})}\n\n"

        save_chat_message(session_id, "assistant", answer)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            **get_security_headers(),
        },
    )


# ── GET /chat/{session_id} ───────────────────────────────────────────


@router.get("/{session_id}", response_model=ChatHistoryResponse)
async def get_history(session_id: str, request: Request):
    """Return the full conversation history for a session."""
    check_rate_limit(request, "chat")

    client = get_supabase()
    result = (
        client.table("chat_history")
        .select("*")
        .eq("session_id", session_id)
        .order("timestamp")
        .execute()
    )

    messages = result.data or []

    if not messages:
        raise HTTPException(
            status_code=404, detail="No history found for this session."
        )

    return ChatHistoryResponse(
        session_id=session_id,
        messages=[ChatMessageOut(**m) for m in messages],
    )


# ── POST /chat/feedback ──────────────────────────────────────────────


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(payload: FeedbackRequest, request: Request):
    """Record user feedback (thumbs up/down + optional comment)."""
    check_rate_limit(request, "feedback")

    client = get_supabase()

    msg_result = (
        client.table("chat_history")
        .select("id")
        .eq("id", str(payload.message_id))
        .execute()
    )
    if not msg_result.data:
        raise HTTPException(status_code=404, detail="Message not found.")

    fb = Feedback(
        id=str(uuid.uuid4()),
        message_id=str(payload.message_id),
        rating=payload.rating,
        comment=payload.comment,
    )
    client.table("feedback").insert(fb.to_dict()).execute()

    logger.info(
        f"Feedback recorded: message={payload.message_id} rating={payload.rating}"
    )

    return FeedbackResponse(
        id=uuid.UUID(fb.id),
        message_id=payload.message_id,
        rating=fb.rating,
        comment=fb.comment,
        created_at=fb.created_at,
    )


# ── Helpers ──────────────────────────────────────────────────────────


def _refuse_response(session_id: str, original_preview: str) -> ChatResponse:
    """Return a refusal when a question is blocked by the security filter."""
    logger.warning(
        f"[security] refused prompt injection: session={session_id} "
        f"preview={original_preview[:200]}"
    )

    # Persist the refusal so the user sees it in history
    msg = save_chat_message(
        session_id,
        "assistant",
        "I'm unable to process this request. Please rephrase your question.",
    )

    return ChatResponse(
        id=uuid.UUID(msg.id),
        session_id=session_id,
        message=msg.message,
        sources=[],
        retrieval_latency_ms=0,
        generation_latency_ms=0,
        input_tokens=0,
        output_tokens=0,
        fallback_used=False,
        timestamp=msg.timestamp,
    )
