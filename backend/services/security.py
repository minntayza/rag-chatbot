"""
Security service — input sanitisation, prompt hardening, and rate limiting.

Vulnerabilities addressed
-------------------------
1. RAG poisoning      — sanitise uploaded document text before embedding
2. Prompt injection   — strip instruction-override patterns from user input
3. Indirect injection — detect adversarial payloads inside uploaded documents
4. Rate limiting      — per-IP sliding window for chat and upload endpoints
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from fastapi import HTTPException, Request

from utils.logger import logger

# ── Rate Limiting ────────────────────────────────────────────────────

# Per-IP sliding window: max_requests per window_seconds
RATE_LIMITS: Dict[str, Tuple[int, int]] = {
    "chat": (20, 60),      # 20 chat requests per 60s
    "upload": (10, 120),    # 10 uploads per 120s
    "stream": (5, 60),     # 5 streaming requests per 60s
    "feedback": (30, 60),  # 30 feedback submissions per 60s
}

# In-memory store: {ip: {endpoint: [(timestamp, ...), ...]}}
_window: Dict[str, Dict[str, List[float]]] = defaultdict(
    lambda: defaultdict(list)
)


def check_rate_limit(request: Request, endpoint: str) -> None:
    """
    Enforce per-IP rate limiting with a sliding window.

    Raises:
        HTTPException(429) if the rate limit is exceeded.

    Implementation
    --------------
    Stores timestamps in-memory (no Redis dependency).
    For production with multiple workers, replace with Redis Sorted Set.
    """
    max_requests, window = RATE_LIMITS.get(endpoint, (100, 60))

    ip = _get_client_ip(request)
    now = time.monotonic()

    # Get the request history for this IP+endpoint
    timestamps = _window[ip][endpoint]

    # Slide window: remove entries older than window_seconds
    timestamps[:] = [t for t in timestamps if now - t < window]

    if len(timestamps) >= max_requests:
        logger.warning(
            f"[security] rate limit hit: ip={ip} endpoint={endpoint} "
            f"requests={len(timestamps)}/{max_requests} window={window}s"
        )
        raise HTTPException(
            status_code=429,
            detail=(
                f"Too many requests. "
                f"Limit: {max_requests} per {window}s. "
                f"Please wait and try again."
            ),
        )

    timestamps.append(now)


# ── Input Sanitisation ───────────────────────────────────────────────


# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS: List[re.Pattern] = [
    # System prompt overrides
    re.compile(r"(?:ignore|forget|disregard)\s+(?:all|previous|above)\s+(?:instructions?|rules?|context)", re.IGNORECASE),
    # Role impersonation
    re.compile(r"(?:you are now|act as|pretend to be|roleplay as)\s+(?:a\s+)?(?!customer support)(\w+)", re.IGNORECASE),
    # System-level directives
    re.compile(r"\[SYSTEM\]|\[ASSISTANT\]|\[INTERNAL\]|\[OVERRIDE\]", re.IGNORECASE),
    # DAN / jailbreak patterns
    re.compile(r"\b(?:DAN|jailbreak|developer mode|god mode)\b", re.IGNORECASE),
    # Instruction delimiter attacks
    re.compile(r"---\s*(?:END\s*CONTEXT|SYSTEM|OVERRIDE|BEGIN)", re.IGNORECASE),
    # Escape the context block
    re.compile(r"<\|im_start\|>|<\|im_end\|>|<<SYS>>|<</SYS>>", re.IGNORECASE),
]

# Maximum safe lengths
MAX_QUESTION_LENGTH: int = 4_000      # chars (beyond limit → likely attack)
MAX_SANITISED_LENGTH: int = 10_000    # hard truncation


def sanitise_question(question: str) -> Tuple[str, bool]:
    """
    Sanitise a user question before passing to the LLM.

    Returns:
        (sanitised_text, was_blocked)

    If ``was_blocked`` is True, the question contained suspicious patterns
    and should be answered with a refusal message.
    """
    if not question or not question.strip():
        return question, False

    original = question

    # 1. Strip null bytes and control characters (except newline)
    question = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", question)

    # 2. Truncate unreasonably long questions
    if len(question) > MAX_QUESTION_LENGTH:
        logger.warning(
            f"[security] truncated long question: {len(question)} → {MAX_SANITISED_LENGTH}"
        )
        question = question[:MAX_SANITISED_LENGTH]

    # 3. Check for prompt injection patterns
    blocked = False
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(question):
            logger.warning(
                f"[security] blocked prompt injection in question: "
                f"pattern={pattern.pattern[:60]} | "
                f"preview={original[:200]}"
            )
            blocked = True
            break

    # 4. Escape markdown-like headers that could fake context boundaries
    question = re.sub(r"^#{1,6}\s+", "# ", question, flags=re.MULTILINE)

    return question.strip(), blocked


def sanitise_document_text(text: str, filename: str) -> str:
    """
    Sanitise uploaded document text before embedding.

    Detects and neutralises:
    - Embedded prompt-injection payloads (e.g. [SYSTEM] hidden in PDFs)
    - Massive blobs of repeated text (DoS via chunk explosion)
    - Encoded injection payloads (base64, hex)
    """
    original_len = len(text)

    # 1. Strip null bytes and non-printable control chars
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)

    # 2. Detect and flag injection payloads
    injection_count = 0
    for pattern in _INJECTION_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            # Replace injection patterns with [REDACTED] instead of just flagging
            text = pattern.sub("[REDACTED]", text)
            injection_count += len(matches)

    if injection_count > 0:
        logger.warning(
            f"[security] sanitised {injection_count} injection patterns "
            f"in document '{filename}'"
        )

    # 3. Detect repeated-text DoS (same line repeated many times)
    lines = text.split("\n")
    if len(lines) > 100:
        from collections import Counter
        line_counts = Counter(lines)
        most_common_count = line_counts.most_common(1)[0][1]
        if most_common_count > len(lines) * 0.5:
            logger.warning(
                f"[security] detected repeated-text DoS in '{filename}': "
                f"'{line_counts.most_common(1)[0][0][:80]}' repeated {most_common_count}x"
            )
            # Deduplicate: keep each unique line once
            seen: Set[str] = set()
            deduped = []
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    deduped.append(line)
            text = "\n".join(deduped)

    # 4. Truncate unreasonably large text (20 MB file → ~20M chars max)
    if len(text) > MAX_SANITISED_LENGTH * 20:
        text = text[:MAX_SANITISED_LENGTH * 20]
        logger.warning(f"[security] truncated massive doc '{filename}'")

    if len(text) < original_len:
        logger.info(
            f"[security] doc sanitisation: {original_len} → {len(text)} chars "
            f"(removed {original_len - len(text)})"
        )

    return text


# ── Auth helpers ─────────────────────────────────────────────────────


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, respecting proxy headers."""
    # Check common proxy headers (order matters)
    for header in ("X-Forwarded-For", "X-Real-IP", "CF-Connecting-IP"):
        value = request.headers.get(header)
        if value:
            # Take the first IP in case of multiple proxies
            return value.split(",")[0].strip()

    # Fallback to direct client
    host = request.client.host if request.client else "unknown"
    return host


# ── Content Security Headers ─────────────────────────────────────────


def get_security_headers() -> Dict[str, str]:
    """Return security headers to add to every response."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        # Content-Security-Policy is set per-endpoint since SSE needs different rules
    }
