"""Provider dispatch.

A `ChatSession` is the unified surface. Each provider returns one when
`open_chat_session()` is called. The session preserves full multi-turn
history natively — subsequent `send(...)` calls re-use the accumulated
context.

Adding a new provider requires implementing two things:
  1. A `ChatSession` subclass (in provider module).
  2. Registration below.
"""

from __future__ import annotations

import time
from typing import Protocol

from ..types import ChatTurnResponse


class ChatSession(Protocol):
    """Multi-turn chat handle. Implementations must persist history."""

    def send(self, message: str) -> ChatTurnResponse: ...


def open_chat_session(
    provider: str,
    *,
    model: str,
    system_prompt: str,
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
) -> ChatSession:
    if provider == "gemini":
        from . import gemini as _g
        return _g.open_session(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    if provider == "openai":
        from . import openai as _o
        return _o.open_session(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    if provider == "ollama":
        from . import ollama as _ol
        return _ol.open_session(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    raise ValueError(f"Unknown provider: {provider!r}")


def send_with_retry(
    session: ChatSession,
    message: str,
    *,
    retry_attempts: int = 3,
    retry_backoff_sec: float = 5.0,
) -> ChatTurnResponse:
    """Wrap session.send() with linear backoff on transient errors.

    Note: a failed turn here does NOT roll back the chat history on the
    underlying client — providers append the request to history before
    they respond. Retries hit the same model with the same context.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retry_attempts + 1):
        try:
            return session.send(message)
        except Exception as exc:
            last_exc = exc
            if attempt == retry_attempts:
                break
            time.sleep(retry_backoff_sec * attempt)
    assert last_exc is not None
    raise last_exc
