"""
Unit tests for the document ingestion pipeline.

Tests every stage independently with deterministic inputs.
No database or external API calls — pure function tests.

Run:  pytest tests/unit/test_ingestion.py -v
"""

from __future__ import annotations

import pytest

from services.ingestion import (
    _validate,
    _extract_text,
    _clean_text,
    _compute_content_hashes,
)


# ── Validation ───────────────────────────────────────────────────────


class TestValidate:
    """Validate stage — checks file type, size, and emptiness."""

    def test_allows_pdf(self):
        _validate("doc.pdf", "application/pdf", 1024)  # should not raise

    def test_allows_txt(self):
        _validate("doc.txt", "text/plain", 1024)

    def test_allows_csv(self):
        _validate("data.csv", "text/csv", 1024)

    def test_rejects_image(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _validate("photo.png", "image/png", 1024)

    def test_rejects_video(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _validate("video.mp4", "video/mp4", 1024)

    def test_rejects_empty_file(self):
        with pytest.raises(ValueError, match="empty"):
            _validate("empty.txt", "text/plain", 0)

    def test_rejects_oversized_file(self):
        too_big = 21 * 1024 * 1024  # 21 MB
        with pytest.raises(ValueError, match="too large"):
            _validate("big.pdf", "application/pdf", too_big)


# ── Text extraction ──────────────────────────────────────────────────


class TestExtractText:
    """Extract stage — PDF and TXT parsing."""

    def test_extracts_txt_utf8(self):
        text = _extract_text(b"Hello World", "text/plain")
        assert text == "Hello World"

    def test_extracts_txt_unicode(self):
        text = _extract_text("ជំរាបសួរ".encode("utf-8"), "text/plain")
        assert "ជំរាបសួរ" in text

    def test_extracts_txt_latin1_fallback(self):
        # Bytes that are valid latin-1 but not utf-8
        text = _extract_text(b"Hello \xff World", "text/plain")
        assert "Hello" in text

    def test_extracts_csv_content(self):
        csv_bytes = b"Txn_ID,Date,Party_Name,Item_Name,Total_Cost\nTXN-001,2025-01-01,Alice,Widget,30000\nTXN-002,2025-01-02,Bob,Gadget,25000"
        text = _extract_text(csv_bytes, "text/csv")
        # CSV conversion creates sentences with row data
        assert "Alice" in text
        assert "Bob" in text
        assert "Widget" in text

    def test_extracts_empty_pdf_triggers_no_crash(self):
        # Minimal valid PDF bytes
        pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
        )
        text = _extract_text(pdf, "application/pdf")
        assert isinstance(text, str)


# ── Text cleaning ───────────────────────────────────────────────────


class TestCleanText:
    """Clean stage — whitespace normalisation and control char removal."""

    def test_normalises_windows_line_endings(self):
        text = "Line1\r\nLine2\r\nLine3"
        cleaned = _clean_text(text)
        assert "\r" not in cleaned
        assert "Line1\nLine2\nLine3" == cleaned

    def test_collapses_excessive_blank_lines(self):
        text = "Paragraph one\n\n\n\n\nParagraph two"
        cleaned = _clean_text(text)
        assert cleaned == "Paragraph one\n\nParagraph two"

    def test_removes_null_bytes(self):
        text = "Hello\x00World"
        cleaned = _clean_text(text)
        assert "\x00" not in cleaned
        assert "HelloWorld" in cleaned

    def test_normalises_non_breaking_spaces(self):
        text = "Hello World"  # non-breaking space
        cleaned = _clean_text(text)
        assert " " not in cleaned
        assert "Hello World" in cleaned

    def test_strips_trailing_whitespace_per_line(self):
        text = "  hello   \n  world   "
        cleaned = _clean_text(text)
        assert cleaned == "hello\nworld"

    def test_collapses_multiple_spaces(self):
        text = "hello    world     test"
        cleaned = _clean_text(text)
        assert cleaned == "hello world test"

    def test_removes_zero_width_chars(self):
        text = "hello​world"  # zero-width space
        cleaned = _clean_text(text)
        assert "​" not in cleaned

    def test_returns_empty_string_for_empty_input(self):
        assert _clean_text("") == ""
        assert _clean_text("   \n\n\n   ") == ""


# ── Deduplication ───────────────────────────────────────────────────


class TestContentHashes:
    """Deduplicate stage — SHA-256 hashing."""

    def test_same_content_produces_same_hash(self):
        h1 = _compute_content_hashes(["hello world"])
        h2 = _compute_content_hashes(["hello world"])
        assert h1 == h2

    def test_different_content_produces_different_hash(self):
        h1 = _compute_content_hashes(["hello world"])
        h2 = _compute_content_hashes(["goodbye world"])
        assert h1 != h2

    def test_whitespace_normalised(self):
        """Different whitespace but same words → same hash."""
        h1 = _compute_content_hashes(["hello    world"])
        h2 = _compute_content_hashes(["hello world"])
        assert h1 == h2

    def test_multiple_chunks(self):
        hashes = _compute_content_hashes(["chunk1", "chunk2", "chunk3"])
        assert len(hashes) == 3
        assert len(set(hashes)) == 3  # all unique

    def test_case_sensitive(self):
        """Intentional: 'Hello' and 'hello' are different chunks."""
        h1 = _compute_content_hashes(["Hello World"])
        h2 = _compute_content_hashes(["hello world"])
        assert h1 != h2
