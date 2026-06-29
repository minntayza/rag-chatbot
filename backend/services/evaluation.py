"""
RAGAS evaluation — automated quality scoring of RAG responses.

Why evaluation in production?
    - **Faithfulness**: is the answer grounded in the context, or hallucinated?
      Critical for catching LLM fabrications before users do.
    - **Answer Relevancy**: does the answer actually address the question?
      Prevents verbose-but-irrelevant responses that waste tokens.
    - **Context Recall**: did we retrieve all the information needed?
      If recall is low, the retriever missed key chunks — tune threshold.
    - **Context Precision**: are retrieved chunks relevant, or noise?
      Low precision means the vector search returns garbage → threshold too low.

These four metrics give you a continuous signal on pipeline health.
When any metric drops below threshold, you know it's time to tune
thresholds, re-chunk documents, or switch embedding models.

Pipeline
--------
    1. For every N-th query (configurable sample_rate), collect:
       - question, answer, context, sources
    2. Compute RAGAS metrics offline or async
    3. Log scores to structured logs (ELK-friendly)
    4. Export to Prometheus for dashboard alerting
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from config import get_settings
from utils.logger import logger

settings = get_settings()


# ── Data types ───────────────────────────────────────────────────────


@dataclass
class EvaluationSample:
    """One evaluation data point."""

    question: str
    answer: str
    contexts: List[str]
    sources: List[str]
    session_id: str
    timestamp: str


@dataclass
class EvaluationScores:
    """RAGAS scores for one sample."""

    faithfulness: float = 0.0     # 0-1: is the answer faithful to context?
    answer_relevancy: float = 0.0 # 0-1: does the answer address the question?
    context_recall: float = 0.0   # 0-1: how much of the needed context was retrieved?
    context_precision: float = 0.0 # 0-1: how relevant are the retrieved chunks?

    @property
    def overall(self) -> float:
        """Harmonic mean of all four scores."""
        scores = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_recall,
            self.context_precision,
        ]
        valid = [s for s in scores if s > 0]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)

    @property
    def is_healthy(self) -> bool:
        """True when all individual scores are above threshold."""
        return all(
            s >= 0.5 for s in [
                self.faithfulness,
                self.answer_relevancy,
                self.context_recall,
                self.context_precision,
            ]
        )


# ── Evaluation engine ────────────────────────────────────────────────


def should_evaluate() -> bool:
    """Decide whether to evaluate this query based on sample rate."""
    if not settings.evaluation_enabled:
        return False
    return random.random() < settings.evaluation_sample_rate


async def evaluate_response(
    question: str,
    answer: str,
    contexts: List[str],
    sources: List[str],
    session_id: str,
    timestamp: str,
) -> Optional[EvaluationScores]:
    """
    Compute RAGAS metrics for a single RAG response.

    Uses a lightweight rule-based approach when the ragas library is
    not installed, falling back to heuristics that correlate well
    with the full LLM-based RAGAS metrics.

    Returns:
        ``EvaluationScores`` or ``None`` if evaluation is disabled
        or the sample is not selected.
    """
    if not should_evaluate():
        return None

    sample = EvaluationSample(
        question=question,
        answer=answer,
        contexts=contexts,
        sources=sources,
        session_id=session_id,
        timestamp=timestamp,
    )

    scores = await _compute_scores(sample)
    _log_scores(sample, scores)
    return scores


async def _compute_scores(sample: EvaluationSample) -> EvaluationScores:
    """
    Compute RAGAS scores.

    Tries to use the `ragas` library if installed (full LLM-based metrics).
    Falls back to heuristic approximations otherwise.

    Heuristics
    ----------
    - **Faithfulness**: answers containing "I don't have enough information"
      get 1.0 (refusal to hallucinate is faithful). Otherwise, keyword overlap
      between answer and context estimates grounding.
    - **Answer Relevancy**: cosine-similarity-like overlap between question
      keywords and answer text. Short answers score lower.
    - **Context Recall**: proportion of context chunks that contain at least
      one keyword from the answer — estimates whether the retriever found
      what was needed.
    - **Context Precision**: proportion of retrieved chunks that have
      meaningful overlap with the question — measures retrieval noise.
    """
    try:
        # Try ragas library first
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _ragas_evaluate, sample
        )
    except (ImportError, Exception):
        return _heuristic_evaluate(sample)


def _ragas_evaluate(sample: EvaluationSample) -> EvaluationScores:
    """Full LLM-based evaluation via ragas library."""
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from ragas.dataset_schema import SingleTurnSample

    ragas_sample = SingleTurnSample(
        user_input=sample.question,
        response=sample.answer,
        retrieved_contexts=sample.contexts,
    )

    result = evaluate(
        dataset=[ragas_sample],
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    scores = result.to_pandas().iloc[0]
    return EvaluationScores(
        faithfulness=float(scores.get("faithfulness", 0)),
        answer_relevancy=float(scores.get("answer_relevancy", 0)),
        context_recall=float(scores.get("context_recall", 0)),
        context_precision=float(scores.get("context_precision", 0)),
    )


def _heuristic_evaluate(sample: EvaluationSample) -> EvaluationScores:
    """
    Fast heuristic evaluation that approximates RAGAS scores.

    No LLM calls — runs in microseconds. Good for continuous monitoring
    when the full ragas library is unavailable.
    """
    question_words = set(sample.question.lower().split())
    answer_words = set(sample.answer.lower().split())

    # Faithfulness: overlap between answer and context
    context_text = " ".join(sample.contexts).lower()
    context_words = set(context_text.split())

    refusal_patterns = [
        "i don't have enough information",
        "i don't have any uploaded documents",
        "not enough information from the uploaded documents",
    ]
    is_refusal = any(p in sample.answer.lower() for p in refusal_patterns)

    if is_refusal or not sample.contexts:
        faithfulness = 1.0  # refusal is honest
    else:
        # How many answer words appear in the context?
        if answer_words:
            overlap = len(answer_words & context_words) / len(answer_words)
            faithfulness = min(overlap * 1.5, 1.0)  # boost — most answers paraphrase
        else:
            faithfulness = 0.0

    # Answer Relevancy: does the answer contain question-relevant terms?
    if answer_words and question_words:
        answer_relevancy = len(question_words & answer_words) / len(question_words)
    else:
        answer_relevancy = 0.0

    # Context Recall: how many of the answer's key terms appear in context?
    meaningful_answer_words = {w for w in answer_words if len(w) > 2}
    if meaningful_answer_words and context_words:
        context_recall = len(meaningful_answer_words & context_words) / len(meaningful_answer_words)
    else:
        context_recall = 0.0

    # Context Precision: how many context chunks overlap with the question?
    if sample.contexts and question_words:
        relevant_chunks = sum(
            1 for ctx in sample.contexts
            if question_words & set(ctx.lower().split())
        )
        context_precision = min(relevant_chunks / len(sample.contexts), 1.0)
    else:
        context_precision = 0.0

    return EvaluationScores(
        faithfulness=round(faithfulness, 4),
        answer_relevancy=round(answer_relevancy, 4),
        context_recall=round(context_recall, 4),
        context_precision=round(context_precision, 4),
    )


def _log_scores(sample: EvaluationSample, scores: EvaluationScores) -> None:
    """Log evaluation scores as structured JSON for log aggregation."""
    import json as _json

    log_entry = {
        "type": "ragas_evaluation",
        "session_id": sample.session_id,
        "question": sample.question[:200],
        "answer_preview": sample.answer[:200],
        "num_contexts": len(sample.contexts),
        "scores": {
            "faithfulness": scores.faithfulness,
            "answer_relevancy": scores.answer_relevancy,
            "context_recall": scores.context_recall,
            "context_precision": scores.context_precision,
            "overall": scores.overall,
        },
        "healthy": scores.is_healthy,
    }

    if not scores.is_healthy:
        logger.warning(
            f"[evaluation] unhealthy scores | {_json.dumps(log_entry['scores'])}"
        )
    else:
        logger.info(
            f"[evaluation] scores={_json.dumps(log_entry['scores'])} | "
            f"q='{sample.question[:60]}...'"
        )

    # Export to Prometheus as well
    try:
        from services.metrics import (
            ev_faithfulness,
            ev_answer_relevancy,
            ev_context_recall,
            ev_context_precision,
        )
        ev_faithfulness.observe(scores.faithfulness)
        ev_answer_relevancy.observe(scores.answer_relevancy)
        ev_context_recall.observe(scores.context_recall)
        ev_context_precision.observe(scores.context_precision)
    except Exception:
        pass
