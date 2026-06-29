"""
Unit tests for Pydantic schemas — request validation.

Run:  pytest tests/unit/test_schemas.py -v
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from schemas import (
    ChatRequest,
    FeedbackRequest,
    ChatResponse,
    UploadResponse,
)


class TestChatRequest:
    """POST /chat payload validation."""

    def test_valid_request(self):
        req = ChatRequest(session_id="abc-123", message="Hello world")
        assert req.session_id == "abc-123"
        assert req.message == "Hello world"

    def test_session_id_required(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="Hello")

    def test_message_required(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id="abc-123")

    def test_message_too_long(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id="abc", message="x" * 10_001)

    def test_empty_session_id(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id="", message="Hi")

    def test_empty_message(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id="abc", message="")


class TestFeedbackRequest:
    """POST /chat/feedback payload validation."""

    def test_valid_thumbs_up(self):
        req = FeedbackRequest(message_id=uuid.uuid4(), rating=1.0)
        assert req.rating == 1.0

    def test_valid_thumbs_down(self):
        req = FeedbackRequest(message_id=uuid.uuid4(), rating=-1.0)
        assert req.rating == -1.0

    def test_rating_out_of_range_high(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(message_id=uuid.uuid4(), rating=2.0)

    def test_rating_out_of_range_low(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(message_id=uuid.uuid4(), rating=-2.0)

    def test_comment_too_long(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(
                message_id=uuid.uuid4(), rating=1.0, comment="x" * 2001
            )

    def test_optional_comment(self):
        req = FeedbackRequest(message_id=uuid.uuid4(), rating=1.0)
        assert req.comment is None


class TestChatResponse:
    """Chat response serialization."""

    def test_minimal_response(self):
        resp = ChatResponse(
            id=uuid.uuid4(),
            session_id="abc",
            message="Hello!",
            timestamp="2025-01-01T00:00:00Z",
        )
        assert resp.sources == []
        assert resp.fallback_used is False

    def test_full_response(self):
        resp = ChatResponse(
            id=uuid.uuid4(),
            session_id="abc",
            message="Based on the FAQ...",
            sources=["faq.pdf", "pricing.txt"],
            retrieval_latency_ms=450.0,
            generation_latency_ms=3500.0,
            input_tokens=200,
            output_tokens=150,
            fallback_used=False,
            timestamp="2025-01-01T00:00:00Z",
        )
        assert resp.retrieval_latency_ms == 450.0
        assert resp.input_tokens == 200


class TestUploadResponse:
    """Upload response serialization."""

    def test_success(self):
        resp = UploadResponse(
            filename="doc.pdf", chunks_created=12, duplicates_skipped=3
        )
        assert resp.filename == "doc.pdf"
        assert resp.chunks_created == 12
        assert resp.duplicates_skipped == 3
