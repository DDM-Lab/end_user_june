"""Ollama chat-session wrapper for local open-source models."""

from __future__ import annotations

from ..types import ChatTurnResponse
from ._parsing import parse_structured_output


class OllamaChatSession:
    def __init__(
        self,
        *,
        model: str,
        system_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        self._model = model
        self._options = {
            "temperature": temperature,
            "num_predict": max_output_tokens,
        }
        self._messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def send(self, message: str) -> ChatTurnResponse:
        import ollama

        self._messages.append({"role": "user", "content": message})
        response = ollama.chat(
            model=self._model,
            messages=self._messages,
            options=self._options,
            format="json",
        )
        raw = response.get("message", {}).get("content", "") or ""
        self._messages.append({"role": "assistant", "content": raw})
        action, rel = parse_structured_output(raw)
        return ChatTurnResponse(
            raw_text=raw,
            action=action,
            relevant_strings_dict=rel,
            meta={"model": self._model, "provider": "ollama"},
        )


def open_session(
    *,
    model: str,
    system_prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> OllamaChatSession:
    return OllamaChatSession(
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
