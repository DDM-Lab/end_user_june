"""OpenAI chat-session wrapper.

Maintains the full message history locally and re-sends it on every turn.

Reads OPENAI_API_KEY from the environment.
"""

from __future__ import annotations

import os

from ..types import ChatTurnResponse
from ._parsing import parse_structured_output


class OpenAIChatSession:
    def __init__(
        self,
        *,
        client,
        model: str,
        system_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def send(self, message: str) -> ChatTurnResponse:
        self._messages.append({"role": "user", "content": message})
        completion = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_output_tokens,
            response_format={"type": "json_object"},
            messages=self._messages,
        )
        raw = completion.choices[0].message.content or ""
        self._messages.append({"role": "assistant", "content": raw})
        action, rel = parse_structured_output(raw)
        return ChatTurnResponse(
            raw_text=raw,
            action=action,
            relevant_strings_dict=rel,
            meta={"model": self._model, "provider": "openai"},
        )


def open_session(
    *,
    model: str,
    system_prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> OpenAIChatSession:
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAIChatSession(
        client=OpenAI(),
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
