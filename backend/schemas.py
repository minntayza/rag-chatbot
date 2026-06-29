"""
Pydantic schemas for request validation and response serialization.

Keeps API contracts explicit and independent of ORM models.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────


class Role(str, Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"


class Rating(float, Enum):
    """Feedback ratings (thumbs up / down)."""
    THUMBS_UP = 1.0
    THUMBS_DOWN = -1.0


# ── Chat ─────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Incoming chat message from the user."""
    session_id: str = Field(..., min_length=1, max_length=128, description="Client-generated session identifier")
    message: str = Field(..., min_length=1, max_length=10_000, description="User's question")


class ChatResponse(BaseModel):
    """Bot's answer plus pipeline metadata."""
    id: uuid.UUID
    session_id: str
    message: str
    sources: List[str] = Field(default_factory=list, description="Filenames used as context")
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    fallback_used: bool = False
    timestamp: datetime


class ChatMessageOut(BaseModel):
    """Single chat message returned in history."""
    id: uuid.UUID
    role: str
    message: str
    timestamp: datetime

    model_config = {"from_attributes": True}


class ChatHistoryResponse(BaseModel):
    """Full conversation history for a session."""
    session_id: str
    messages: List[ChatMessageOut]


# ── Upload ───────────────────────────────────────────────────────────


class UploadResponse(BaseModel):
    """Confirmation after a document is processed."""
    filename: str
    chunks_created: int
    duplicates_skipped: int = 0
    message: str = "Document uploaded and indexed successfully."


class IngestProgressEvent(BaseModel):
    """One progress event from the ingestion pipeline."""
    stage: str
    message: str
    percent: int


class IngestCompleteEvent(IngestProgressEvent):
    """Final summary emitted at the end of ingestion."""
    detail: dict | None = None


# ── Feedback ─────────────────────────────────────────────────────────


class FeedbackRequest(BaseModel):
    """User feedback on a bot response."""
    message_id: uuid.UUID
    rating: float = Field(..., ge=-1.0, le=1.0)
    comment: Optional[str] = Field(None, max_length=2_000)


class FeedbackResponse(BaseModel):
    """Confirmation that feedback was recorded."""
    id: uuid.UUID
    message_id: uuid.UUID
    rating: float
    comment: Optional[str]
    created_at: datetime


# ── Health ───────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Basic health-check payload."""
    status: str = "ok"
    version: str = "1.0.0"
