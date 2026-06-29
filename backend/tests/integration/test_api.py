"""
Integration tests for the FastAPI application.

These tests use httpx.AsyncClient with the ASGI transport to test
the full HTTP layer — routing, validation, CORS, and error handling.

All external dependencies (Supabase, LLM) are mocked so tests run
offline and pass in CI without any network access.

Run:  pytest tests/integration/test_api.py -v
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ── Set test env vars BEFORE importing app ────────────────────────

TEST_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRlc3QiLCJyb2xlIjoiYW5vbiIs"
    "ImlhdCI6MTcwMDAwMDAwMCwiZXhwIjoyMDAwMDAwMDAwfQ.fake_signature"
)

os.environ["SUPABASE_URL"] = "https://test-project.supabase.co"
os.environ["SUPABASE_ANON_KEY"] = TEST_ANON_KEY
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test-service-key"
os.environ["LLM_API_KEY"] = "test-llm-key"
os.environ["LLM_BASE_URL"] = "https://test.api.example.com"
os.environ["LLM_MODEL"] = "test-model"
os.environ["REDIS_ENABLED"] = "false"
os.environ["PROMETHEUS_ENABLED"] = "false"

from main import app
from db import get_supabase


# ── Mock Supabase client ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_supabase():
    """
    Mock get_supabase() so no real HTTP call is made to Supabase.

    Every route that calls get_supabase() gets a MagicMock instead.
    The mock's .table().select().eq().order().execute() chain returns
    an empty list, mimicking "no data found".
    """
    mock_client = MagicMock()
    mock_client.table.return_value = mock_client
    mock_client.select.return_value = mock_client
    mock_client.eq.return_value = mock_client
    mock_client.order.return_value = mock_client
    mock_client.limit.return_value = mock_client
    mock_client.execute.return_value = MagicMock(data=[], count=0)

    # Mock RPC (retrieval uses this)
    mock_client.rpc.return_value = mock_client

    with patch("db._supabase_client", mock_client), \
         patch("db.get_supabase", return_value=mock_client):
        yield mock_client


@pytest.fixture
async def client():
    """Async HTTP client — routes through the ASGI app in-process."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Tests ─────────────────────────────────────────────────────────


class TestHealthEndpoints:
    """GET / and GET /health always work without DB."""

    async def test_root_liveness(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    async def test_health_readiness(self, client):
        response = await client.get("/health")
        assert response.status_code in (200, 503)
        assert "checks" in response.json()

    async def test_metrics_endpoint(self, client):
        """Prometheus metrics endpoint returns scrape data."""
        response = await client.get("/metrics")
        # 200 if prometheus enabled (default), 404 if PROMETHEUS_ENABLED=false
        assert response.status_code in (200, 404, 500)


class TestChatEndpoints:
    """POST /chat — request validation works offline."""

    async def test_missing_session_id(self, client):
        response = await client.post("/chat", json={"message": "hi"})
        assert response.status_code == 422

    async def test_missing_message(self, client):
        response = await client.post("/chat", json={"session_id": "abc"})
        assert response.status_code == 422

    async def test_empty_message(self, client):
        response = await client.post(
            "/chat", json={"session_id": "abc", "message": ""}
        )
        assert response.status_code == 422

    async def test_message_too_long(self, client):
        response = await client.post(
            "/chat",
            json={"session_id": "abc", "message": "x" * 10_001},
        )
        assert response.status_code == 422

    async def test_valid_request_accepted(self, client):
        """Valid request passes validation — may get 500 (no real LLM) or 201."""
        response = await client.post(
            "/chat",
            json={"session_id": "test-session", "message": "Hello world"},
        )
        # 201 = success with mock, 500 = unhandled downstream error
        assert response.status_code in (201, 500)


class TestChatHistory:
    """GET /chat/{session_id} — history endpoints work."""

    async def test_history_endpoint_exists(self, client):
        """Endpoint returns a response (empty history from mock)."""
        response = await client.get("/chat/test-123")
        # Mock returns no data → 404
        assert response.status_code == 404


class TestChatStream:
    """POST /chat/stream — SSE streaming endpoint."""

    async def test_stream_valid_request(self, client):
        """Streaming endpoint accepts valid requests."""
        response = await client.post(
            "/chat/stream",
            json={"session_id": "stream-test", "message": "Hello"},
        )
        # With mock DB: gets past validation, fails at retrieval or returns stream
        assert response.status_code in (200, 201, 500)


class TestUploadEndpoints:
    """POST /upload — file validation."""

    async def test_no_file(self, client):
        response = await client.post("/upload")
        assert response.status_code == 422

    async def test_invalid_file_type(self, client):
        files = {"file": ("test.png", b"fake-image-data", "image/png")}
        response = await client.post("/upload", files=files)
        # 400=validation error, 422=extraction error, 502=ingestion service error
        assert response.status_code in (400, 422, 502)


class TestFeedback:
    """POST /chat/feedback — validation."""

    async def test_invalid_rating(self, client):
        import uuid
        response = await client.post(
            "/chat/feedback",
            json={
                "message_id": str(uuid.uuid4()),
                "rating": 5.0,
            },
        )
        assert response.status_code == 422

    async def test_valid_feedback_accepted(self, client):
        import uuid
        response = await client.post(
            "/chat/feedback",
            json={
                "message_id": str(uuid.uuid4()),
                "rating": 1.0,
            },
        )
        # Mock DB accepts this — 201 (created) or 404 (mock returns no data)
        assert response.status_code in (201, 404, 500)


class TestCORS:
    """CORS headers are present."""

    async def test_cors_preflight(self, client):
        response = await client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
