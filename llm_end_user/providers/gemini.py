"""Gemini chat-session wrapper (google-genai SDK).

Uses `client.chats.create(...)` so the SDK maintains the full multi-turn
history server-side-with-locally-replayed-context across all 56 turns
of one experiment run.

Reads GEMINI_API_KEY from the environment (.env loaded at startup).
"""

from __future__ import annotations

import os

from ..types import ChatTurnResponse
from ._parsing import parse_structured_output


class GeminiChatSession:
    def __init__(self, client, chat, model: str) -> None:
        # Keep a strong reference to the client — its httpx transport closes
        # when the client is garbage collected, breaking subsequent turns.
        self._client = client
        self._chat = chat
        self._model = model

    def send(self, message: str) -> ChatTurnResponse:
        response = self._chat.send_message(message)
        raw = getattr(response, "text", "") or ""
        action, rel = parse_structured_output(raw)
        return ChatTurnResponse(
            raw_text=raw,
            action=action,
            relevant_strings_dict=rel,
            meta={"model": self._model, "provider": "gemini"},
        )


def open_session(
    *,
    model: str,
    system_prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> GeminiChatSession:
    from google import genai
    from google.genai import types as gtypes

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=api_key)

    config = gtypes.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
    )
    chat = client.chats.create(model=model, config=config)
    return GeminiChatSession(client=client, chat=chat, model=model)
