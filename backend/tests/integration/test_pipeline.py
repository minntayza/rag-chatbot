"""
Integration tests for the RAG pipeline components.

Tests the full retrieval and generation pipeline with mocked dependencies.
These tests verify that the pipeline stages connect correctly without
requiring a real database or LLM.

Run:  pytest tests/integration/test_pipeline.py -v
"""

from __future__ import annotations

import os

# Set test environment before importing app modules
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-key")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_BASE_URL", "https://test.example.com")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("REDIS_ENABLED", "false")
os.environ.setdefault("PROMETHEUS_ENABLED", "false")

import pytest

from services.generation import (
    build_prompt,
    _estimate_tokens,
    _estimate_messages_tokens,
)
from services.retrieval import (
    _merge_chunks,
    ChunkResult,
)
from services.cache import build_retrieval_cache_key
from services.ingestion import (
    _validate,
    _clean_text,
)


class TestPromptBuilding:
    """System and user prompt construction."""

    def test_builds_with_context(self):
        system, user = build_prompt(
            context="This is some context.",
            question="What is it?",
        )
        assert "customer support assistant" in system.lower()
        assert "This is some context." in system
        assert "What is it?" in user
        assert "Answer ONLY" in system

    def test_builds_without_context(self):
        system, user = build_prompt(
            context="",
            question="Hello?",
        )
        assert "no context" in system.lower()
        assert "upload documents" in system.lower()

    def test_user_question_in_user_prompt(self):
        _, user = build_prompt(context="Ctx", question="How much?")
        assert "How much?" in user

    def test_context_injected_into_system_prompt(self):
        long_context = "The answer is 42. " * 10
        system, _ = build_prompt(context=long_context, question="?")
        assert long_context in system


class TestTokenEstimation:
    """Token counting — character-based and tiktoken."""

    def test_empty_string(self):
        assert _estimate_tokens("") == 1  # minimum 1

    def test_short_text(self):
        assert _estimate_tokens("hello") == 1

    def test_typical_question(self):
        tokens = _estimate_tokens(
            "What is the return policy for items purchased online?"
        )
        assert 5 <= tokens <= 30

    def test_long_text(self):
        text = "word " * 1000
        tokens = _estimate_tokens(text)
        assert tokens > 100

    def test_message_tokens_includes_overhead(self):
        tokens = _estimate_messages_tokens("System here", "User here")
        assert tokens >= len("System here") / 4 + len("User here") / 4

    def test_message_tokens_grows_with_input(self):
        small = _estimate_messages_tokens("Hi", "?")
        large = _estimate_messages_tokens("Hi " * 500, "? " * 500)
        assert large > small * 10


class TestChunkMerging:
    """Merging retrieved chunks into context string."""

    def test_empty_chunks(self):
        context, sources = _merge_chunks([])
        assert context == ""
        assert sources == []

    def test_single_chunk(self):
        chunk = ChunkResult(
            id="1", filename="faq.pdf", chunk_index=0,
            content="Hello world.", similarity=0.85,
        )
        context, sources = _merge_chunks([chunk])
        assert "Hello world." in context
        assert "[Source:" in context
        assert sources == ["faq.pdf"]

    def test_multiple_chunks_same_file(self):
        chunks = [
            ChunkResult(id="1", filename="faq.pdf", chunk_index=0,
                        content="First.", similarity=0.9),
            ChunkResult(id="2", filename="faq.pdf", chunk_index=1,
                        content="Second.", similarity=0.8),
        ]
        context, sources = _merge_chunks(chunks)
        assert "First." in context
        assert "Second." in context
        assert sources == ["faq.pdf"]  # deduplicated

    def test_multiple_files(self):
        chunks = [
            ChunkResult(id="1", filename="faq.pdf", chunk_index=0,
                        content="From FAQ.", similarity=0.9),
            ChunkResult(id="2", filename="pricing.txt", chunk_index=0,
                        content="From pricing.", similarity=0.8),
        ]
        context, sources = _merge_chunks(chunks)
        assert "From FAQ." in context
        assert "From pricing." in context
        assert len(sources) == 2
        assert "faq.pdf" in sources
        assert "pricing.txt" in sources

    def test_separator_between_chunks(self):
        chunks = [
            ChunkResult(id="1", filename="a.pdf", chunk_index=0,
                        content="A.", similarity=0.9),
            ChunkResult(id="2", filename="b.pdf", chunk_index=0,
                        content="B.", similarity=0.8),
        ]
        context, _ = _merge_chunks(chunks)
        assert "---" in context  # separator between chunks


class TestCacheKey:
    """Cache key generation — deterministic and normalised."""

    def test_same_question_same_key(self):
        k1 = build_retrieval_cache_key("What is pricing?", 5)
        k2 = build_retrieval_cache_key("What is pricing?", 5)
        assert k1 == k2

    def test_whitespace_normalised(self):
        k1 = build_retrieval_cache_key("what is pricing", 5)
        k2 = build_retrieval_cache_key("  What   is   pricing  ", 5)
        assert k1 == k2

    def test_case_normalised(self):
        k1 = build_retrieval_cache_key("WHAT IS PRICING", 5)
        k2 = build_retrieval_cache_key("what is pricing", 5)
        assert k1 == k2

    def test_different_top_k_different_key(self):
        k1 = build_retrieval_cache_key("question", 5)
        k2 = build_retrieval_cache_key("question", 10)
        assert k1 != k2

    def test_different_questions_different_key(self):
        k1 = build_retrieval_cache_key("pricing", 5)
        k2 = build_retrieval_cache_key("shipping", 5)
        assert k1 != k2


class TestValidation:
    """Upload validation checks."""

    def test_all_valid_types(self):
        for ct in ("application/pdf", "text/plain", "text/csv", "application/csv"):
            _validate("test", ct, 1024)  # should not raise

    def test_rejects_bad_types(self):
        with pytest.raises(ValueError):
            _validate("test.jpg", "image/jpeg", 1024)

    def test_rejects_zero_bytes(self):
        with pytest.raises(ValueError, match="empty"):
            _validate("empty.txt", "text/plain", 0)


class TestTextCleaning:
    """Text cleaning pipeline."""

    def test_preserves_paragraphs(self):
        text = "Para 1 line 1.\nPara 1 line 2.\n\nPara 2 line 1."
        cleaned = _clean_text(text)
        assert "Para 1 line 1." in cleaned
        assert "Para 2 line 1." in cleaned

    def test_handles_binary_garbage(self):
        text = "Good text.\x01\x02\x03More text."
        cleaned = _clean_text(text)
        assert "Good text." in cleaned
        assert "More text." in cleaned
        assert "\x01" not in cleaned
