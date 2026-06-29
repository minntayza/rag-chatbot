"""
Unit tests for RAGAS evaluation metrics.

Tests heuristic evaluation (no external LLM needed).
Run:  pytest tests/unit/test_evaluation.py -v
"""

from __future__ import annotations

import pytest

from services.evaluation import (
    EvaluationScores,
    _heuristic_evaluate,
    EvaluationSample,
)


class TestHeuristicEvaluation:
    """Heuristic evaluation produces reasonable scores without LLM calls."""

    def test_faithfulness_high_for_factual_answer(self):
        """An answer that mirrors the context should score high."""
        sample = EvaluationSample(
            question="What is the return policy?",
            answer="The return policy allows returns within 30 days with proof of purchase.",
            contexts=["We offer a 30-day return policy. Items must have proof of purchase."],
            sources=["faq.pdf"],
            session_id="test",
            timestamp="2025-01-01T00:00:00Z",
        )
        scores = _heuristic_evaluate(sample)
        assert scores.faithfulness >= 0.5, f"Expected high faithfulness, got {scores.faithfulness}"

    def test_faithfulness_perfect_for_refusal(self):
        """Refusing to answer when no context is 100% faithful."""
        sample = EvaluationSample(
            question="What is the meaning of life?",
            answer="I don't have enough information from the uploaded documents to answer that question.",
            contexts=["Some unrelated text about kitchen appliances."],
            sources=["unrelated.pdf"],
            session_id="test",
            timestamp="2025-01-01T00:00:00Z",
        )
        scores = _heuristic_evaluate(sample)
        assert scores.faithfulness == 1.0

    def test_answer_relevancy_correlates_with_question(self):
        """Answer that mentions question keywords scores higher."""
        relevant = EvaluationSample(
            question="What payment options exist?",
            answer="Payment options include Visa, Mastercard, and PayPal.",
            contexts=["We accept credit cards and PayPal."],
            sources=["faq.pdf"],
            session_id="test",
            timestamp="2025-01-01T00:00:00Z",
        )
        irrelevant = EvaluationSample(
            question="What payment options exist?",
            answer="Our office is open Monday through Friday from 9 to 5.",
            contexts=["Office hours: Monday-Friday 9am-5pm."],
            sources=["office.pdf"],
            session_id="test",
            timestamp="2025-01-01T00:00:00Z",
        )
        rel_score = _heuristic_evaluate(relevant).answer_relevancy
        irr_score = _heuristic_evaluate(irrelevant).answer_relevancy
        assert rel_score >= irr_score, f"Relevant ({rel_score}) should score >= irrelevant ({irr_score})"

    def test_context_precision_filters_noise(self):
        """Chunks related to the question should yield higher precision."""
        good = EvaluationSample(
            question="What is your shipping policy?",
            answer="Shipping takes 3-5 business days.",
            contexts=[
                "Shipping policy: Standard orders arrive in 3-5 business days.",
                "Shipping policy: Expedited shipping takes 1-2 business days.",
            ],
            sources=["shipping.pdf"],
            session_id="test",
            timestamp="2025-01-01T00:00:00Z",
        )
        bad = EvaluationSample(
            question="What is your shipping policy?",
            answer="I don't have enough information.",
            contexts=[
                "Our cafeteria serves lunch from noon to 2pm.",
                "Parking is free on weekends.",
            ],
            sources=["misc.pdf"],
            session_id="test",
            timestamp="2025-01-01T00:00:00Z",
        )
        good_precision = _heuristic_evaluate(good).context_precision
        bad_precision = _heuristic_evaluate(bad).context_precision
        assert good_precision > bad_precision

    def test_empty_contexts_scores_zero(self):
        sample = EvaluationSample(
            question="Anything",
            answer="Something",
            contexts=[],
            sources=[],
            session_id="test",
            timestamp="2025-01-01T00:00:00Z",
        )
        scores = _heuristic_evaluate(sample)
        assert scores.faithfulness == 1.0  # refusal = faithful
        assert scores.context_precision == 0.0  # no chunks = no precision

    def test_overall_is_average(self):
        scores = EvaluationScores(
            faithfulness=0.8,
            answer_relevancy=0.6,
            context_recall=0.7,
            context_precision=0.5,
        )
        assert 0.6 < scores.overall < 0.7

    def test_healthy_threshold(self):
        healthy = EvaluationScores(
            faithfulness=0.9, answer_relevancy=0.8,
            context_recall=0.7, context_precision=0.6,
        )
        assert healthy.is_healthy

        unhealthy = EvaluationScores(
            faithfulness=0.3, answer_relevancy=0.4,
            context_recall=0.2, context_precision=0.1,
        )
        assert not unhealthy.is_healthy
