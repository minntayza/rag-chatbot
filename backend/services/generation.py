"""
Response generation pipeline — LLM interaction layer.

Uses Mimo 2.5 Pro via an OpenAI-compatible chat completions endpoint.

Architecture
------------
    {context + question + system prompt}
                │
                ▼
    ┌───────────────────────┐
    │  1. Build prompt      │   — inject context into system template
    │  2. Count tokens       │   — estimate input token count
    │  3. Call LLM           │   — POST /chat/completions (stream or non-stream)
    │  4. Accumulate tokens  │   — measure output token estimate
    │  5. Retry on failure   │   — exponential backoff, max 3 attempts
    │  6. Return response    │   — answer text + token counts + latency
    └───────────────────────┘

Why each step
-------------
    - **Build prompt**: merges retrieved context into the system template.
      Without context the LLM has nothing to base its answer on.
    - **Count tokens**: helps monitor usage, detect prompt bloat, and
      stay within model context limits.
    - **Call LLM (streaming)**: ``stream=true`` returns tokens as they're
      generated, so the frontend can display the answer word-by-word
      instead of waiting for the full response.
    - **Timeout**: prevents hung requests from blocking the event loop.
      Default 60s, configurable.
    - **Retry**: transient API errors (429 rate-limit, 503 overload) are
      retried with exponential backoff (1s → 2s → 4s).
    - **Logging**: every call logs input tokens, output tokens, latency,
      and whether streaming was used.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from config import get_settings
from db import get_supabase
from models import ChatMessage
from utils.logger import logger

settings = get_settings()

# ── Constants ────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a professional customer support assistant. "
    "Answer user questions based strictly on the provided context."
)

SYSTEM_PROMPT_WITH_CONTEXT = """\
You are a professional customer support assistant. Follow these rules strictly:

1. Answer ONLY from the provided context below.
2. If the answer is not in the context, say exactly:
   "I don't have enough information from the uploaded documents to answer that question."
3. Never hallucinate, guess, or use external knowledge.
4. Return concise, accurate answers.
5. Cite the source document when possible.
6. Use a professional, helpful tone.

--- CONTEXT ---
{context}
--- END CONTEXT ---"""

USER_PROMPT = "Question: {question}"

# Retry configuration
MAX_RETRIES: int = 3
RETRY_BASE_DELAY_SEC: float = 1.0    # 1s → 2s → 4s
RETRYABLE_STATUSES: set[int] = {429, 500, 502, 503, 504}

# Timeout
DEFAULT_TIMEOUT_SEC: int = 60
STREAM_TIMEOUT_SEC: int = 120         # streaming can take longer

# Token estimation (character-based fallback when tiktoken is unavailable)
CHARS_PER_TOKEN: float = 4.0          # rough estimate: ~4 chars ≈ 1 token
MAX_OUTPUT_TOKENS: int = 1024
MAX_CONTEXT_TOKENS: int = 6000        # warn if input exceeds this


# ── Data types ───────────────────────────────────────────────────────


@dataclass
class GenerationResult:
    """
    The output of one generation run.

    Attributes
    ----------
    answer : str
        The generated response text.
    input_tokens_estimate : int
        Estimated number of tokens in the prompt (system + context + user).
    output_tokens_estimate : int
        Estimated number of tokens in the generated answer.
    latency_ms : float
        Wall-clock time from first API byte to full completion.
    model : str
        The model that produced this response (e.g. mimo-v2.5-pro).
    attempt : int
        Which retry attempt succeeded (1 = first try).
    streamed : bool
        Whether streaming mode was used.
    """

    answer: str = ""
    input_tokens_estimate: int = 0
    output_tokens_estimate: int = 0
    latency_ms: float = 0.0
    model: str = ""
    attempt: int = 1
    streamed: bool = False


# ── Token estimation ─────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count for a string.

    Tries to use ``tiktoken`` with the cl100k_base encoding (used by most
    modern models). Falls back to a character-based heuristic if tiktoken
    isn't installed.

    The character-based estimate divides by CHARS_PER_TOKEN (4.0), which
    is a reasonable average for English text. It will undercount for code
    and overcount for very short strings, but is accurate enough for
    monitoring and cost estimation.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except (ImportError, Exception):
        return max(1, int(len(text) / CHARS_PER_TOKEN))


def _estimate_messages_tokens(
    system_prompt: str,
    user_message: str,
) -> int:
    """
    Estimate total input tokens for a chat completion request.

    Adds a small overhead (~12 tokens) for message formatting that the
    API adds internally (role labels, separators, etc.).
    """
    overhead = 12  # role wrapper tokens
    return (
        overhead
        + _estimate_tokens(system_prompt)
        + _estimate_tokens(user_message)
    )


# ── Prompt builder ───────────────────────────────────────────────────


def build_prompt(context: str, question: str) -> Tuple[str, str]:
    """
    Build the system and user messages for the LLM.

    If ``context`` is provided, it is injected into the system prompt
    template. If empty, the system prompt instructs the model to look
    for evidence in context.

    Returns
    -------
        (system_message, user_message)

        - ``system_message``: the full system prompt with injected context
        - ``user_message``: the user's question, formatted
    """
    if context:
        system_msg = SYSTEM_PROMPT_WITH_CONTEXT.format(context=context)
    else:
        system_msg = (
            SYSTEM_PROMPT
            + " There is no context available — tell the user to upload documents."
        )

    user_msg = USER_PROMPT.format(question=question)
    return system_msg, user_msg


# ── Non-streaming call (with retry) ──────────────────────────────────


async def generate_answer(
    context: str,
    question: str,
) -> GenerationResult:
    """
    Generate an answer without streaming.

    Calls the LLM synchronously and returns the complete response.

    Retry logic
    -----------
    Transient errors (429, 5xx) are retried up to ``MAX_RETRIES`` times
    with exponential backoff: 1s → 2s → 4s.
    4xx errors (except 429) are NOT retried — they indicate a bad request
    that would fail on every attempt.
    """
    system_msg, user_msg = build_prompt(context, question)
    input_tokens = _estimate_messages_tokens(system_msg, user_msg)

    if input_tokens > MAX_CONTEXT_TOKENS:
        logger.warning(
            f"[generation] input tokens ({input_tokens}) exceeds "
            f"max recommended ({MAX_CONTEXT_TOKENS}). Consider reducing chunks."
        )

    logger.info(
        f"[generation] starting | input_tokens≈{input_tokens} | model={settings.llm_model}"
    )

    last_error: Optional[Exception] = None
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await _post_completion(
                headers=headers,
                system_msg=system_msg,
                user_msg=user_msg,
                stream=False,
                timeout=DEFAULT_TIMEOUT_SEC,
            )
            result.input_tokens_estimate = input_tokens
            result.model = settings.llm_model
            result.attempt = attempt
            result.streamed = False

            logger.info(
                f"[generation] done | attempt={attempt} | "
                f"in≈{input_tokens} out≈{result.output_tokens_estimate} | "
                f"latency={result.latency_ms:.0f}ms"
            )
            return result

        except httpx.HTTPStatusError as exc:
            last_error = exc
            status = exc.response.status_code

            if status not in RETRYABLE_STATUSES:
                logger.error(
                    f"[generation] non-retryable error {status}: "
                    f"{exc.response.text[:300]}"
                )
                raise RuntimeError(f"LLM request failed (HTTP {status})") from exc

            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))  # 1s → 2s → 4s
                logger.warning(
                    f"[generation] retryable error {status} | "
                    f"attempt={attempt}/{MAX_RETRIES} | retrying in {delay:.0f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"[generation] exhausted {MAX_RETRIES} retries | "
                    f"last error: {status}"
                )
                raise RuntimeError(
                    f"LLM unavailable after {MAX_RETRIES} attempts (HTTP {status})"
                ) from exc

        except (httpx.TimeoutException, httpx.ReadTimeout) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
                logger.warning(
                    f"[generation] timeout | attempt={attempt}/{MAX_RETRIES} | "
                    f"retrying in {delay:.0f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"[generation] exhausted retries after timeouts")
                raise RuntimeError("LLM request timed out") from exc

        except httpx.RequestError as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
                logger.warning(
                    f"[generation] network error ({exc}) | "
                    f"attempt={attempt}/{MAX_RETRIES}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"[generation] network error exhausted retries: {exc}")
                raise RuntimeError("Could not reach the LLM service") from exc

    # Should be unreachable, but guard:
    raise RuntimeError("LLM request failed") from (last_error or Exception("unknown"))


# ── Streaming call ───────────────────────────────────────────────────


async def generate_answer_stream(
    context: str,
    question: str,
) -> AsyncGenerator[str, None]:
    """
    Generate an answer with **Server-Sent Event streaming**.

    Each yielded value is a partial text chunk as it arrives from the LLM.
    The caller can accumulate these into the full response.

    Usage::

        async for token in generate_answer_stream(context, question):
            yield f"data: {token}\\n\\n"  # SSE format for the frontend

    Does NOT include retry logic — streaming endpoints cannot be replayed
    because the stream has already been partially consumed. If streaming
    fails, the caller should fall back to ``generate_answer()``.
    """
    system_msg, user_msg = build_prompt(context, question)
    input_tokens = _estimate_messages_tokens(system_msg, user_msg)

    logger.info(
        f"[generation] streaming start | input_tokens≈{input_tokens} | "
        f"model={settings.llm_model}"
    )

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": True,
    }

    char_count = 0
    t0 = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=STREAM_TIMEOUT_SEC) as client:
            async with client.stream(
                "POST",
                f"{settings.llm_base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    # SSE format: "data: {...json...}"
                    if not line.startswith("data: "):
                        continue
                    if line.strip() == "data: [DONE]":
                        break

                    try:
                        import json as _json

                        chunk_data = _json.loads(line[6:])  # strip "data: " prefix
                        delta = chunk_data["choices"][0].get("delta", {})
                        content = delta.get("content", "")

                        if content:
                            char_count += len(content)
                            yield content
                    except (KeyError, IndexError, _json.JSONDecodeError):
                        # Skip malformed chunks silently
                        continue

    except httpx.HTTPStatusError as exc:
        logger.error(
            f"[generation] streaming HTTP {exc.response.status_code}: "
            f"{exc.response.text[:300]}"
        )
        raise RuntimeError("LLM streaming failed") from exc
    except (httpx.TimeoutException, httpx.ReadTimeout):
        logger.error("[generation] streaming timed out")
        raise RuntimeError("LLM streaming timed out") from None
    except httpx.RequestError as exc:
        logger.error(f"[generation] streaming network error: {exc}")
        raise RuntimeError("Could not reach the LLM service") from exc

    elapsed = (time.monotonic() - t0) * 1000
    output_tokens = _estimate_tokens_by_chars(char_count)
    logger.info(
        f"[generation] streaming done | "
        f"in≈{input_tokens} out≈{output_tokens} | "
        f"chars={char_count} | latency={elapsed:.0f}ms"
    )


# ── Internal helpers ─────────────────────────────────────────────────


async def _post_completion(
    headers: Dict[str, str],
    system_msg: str,
    user_msg: str,
    stream: bool,
    timeout: int,
) -> GenerationResult:
    """
    Post a chat completion request and parse the response.

    Shared by the non-streaming and streaming codepaths.
    """
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": stream,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        t0 = time.monotonic()
        response = await client.post(
            f"{settings.llm_base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        elapsed = (time.monotonic() - t0) * 1000

    # raise_for_status is called by the caller (to handle retry logic)
    response.raise_for_status()
    data = response.json()

    try:
        answer: str = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.error(f"[generation] unexpected response shape: {data}")
        raise RuntimeError("Unexpected LLM response format") from exc

    # If the API returns usage, use it. Otherwise estimate.
    usage = data.get("usage", {})
    if usage:
        output_tokens = usage.get("completion_tokens", _estimate_tokens_by_chars(len(answer)))
    else:
        output_tokens = _estimate_tokens_by_chars(len(answer))

    return GenerationResult(
        answer=answer,
        input_tokens_estimate=0,   # set by caller
        output_tokens_estimate=output_tokens,
        latency_ms=round(elapsed, 2),
        model=settings.llm_model,
    )


def _estimate_tokens_by_chars(char_count: int) -> int:
    """Estimate token count from character count."""
    return max(1, int(char_count / CHARS_PER_TOKEN))


# ── Chat history persistence ─────────────────────────────────────────


def save_chat_message(
    session_id: str,
    role: str,
    message: str,
) -> ChatMessage:
    """
    Persist a message to the ``chat_history`` table and return the row.

    This is the single point where chat messages are stored.
    Both user messages and assistant responses go through here.
    """
    client = get_supabase()
    msg = ChatMessage(
        id=str(uuid.uuid4()),
        session_id=session_id,
        role=role,
        message=message,
    )
    client.table("chat_history").insert(msg.to_dict()).execute()
    logger.debug(f"[generation] saved {role} message | session={session_id}")
    return msg
