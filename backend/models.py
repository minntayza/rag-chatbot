"""
Data models for the application.

These are plain data classes — not ORM models. The Supabase client
returns dicts from the REST API, so we validate them with Pydantic
schemas instead of SQLAlchemy.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Document:
    """A chunk of text extracted from an uploaded document."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    chunk_index: int = 0
    content: str = ""
    embedding: list[float] = field(default_factory=list)
    metadata: dict | None = None
    created_at: str = field(default_factory=lambda: _utcnow().isoformat())

    def to_dict(self) -> dict:
        """Convert to dict for Supabase insert."""
        return {
            "id": self.id,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }


@dataclass
class ChatMessage:
    """One message in a conversation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    role: str = ""
    message: str = ""
    timestamp: str = field(default_factory=lambda: _utcnow().isoformat())

    def to_dict(self) -> dict:
        """Convert to dict for Supabase insert."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "message": self.message,
        }


@dataclass
class Feedback:
    """User feedback on a specific assistant message."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_id: str = ""
    rating: float = 0.0
    comment: Optional[str] = None
    created_at: str = field(default_factory=lambda: _utcnow().isoformat())

    def to_dict(self) -> dict:
        """Convert to dict for Supabase insert."""
        return {
            "id": self.id,
            "message_id": self.message_id,
            "rating": self.rating,
            "comment": self.comment,
        }
