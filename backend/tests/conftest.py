"""Pytest configuration — shared fixtures and settings."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_document_text() -> str:
    """A realistic FAQ document for testing."""
    return """Frequently Asked Questions

Q: What is the return policy?
A: We offer a 30-day return policy. Items must be in original packaging with proof of purchase.

Q: What payment methods do you accept?
A: We accept Visa, Mastercard, American Express, and PayPal.

Q: How long does shipping take?
A: Standard shipping takes 3-5 business days. Expedited shipping is 1-2 business days.
"""


@pytest.fixture
def sample_question() -> str:
    return "What payment methods do you accept?"


@pytest.fixture
def sample_contexts() -> list[str]:
    return [
        "We accept Visa, Mastercard, American Express, and PayPal as payment methods.",
        "All transactions are encrypted with TLS 1.3 for your security.",
    ]
