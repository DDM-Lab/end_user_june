"""Shared parsing for model JSON output.

Every provider asks the model for:
    {"action": "Mark as Spam", "relevant_strings_dict": {"<str>": "<why>", ...}}
"""

from __future__ import annotations

import json
import re

from ..types import EMAIL_RESPONSE_LIST


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_structured_output(text: str) -> tuple[str | None, dict[str, str]]:
    """Extract (action, relevant_strings_dict) from raw model text.

    Tolerates leading/trailing prose around the JSON object — we grab the
    first {...} span and parse that. Returns (None, {}) on failure.
    """
    if not text:
        return None, {}
    m = _JSON_BLOCK.search(text)
    if not m:
        return None, {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, {}

    action = data.get("action")
    if isinstance(action, str):
        action = _canonicalize_action(action)
    else:
        action = None

    raw_dict = data.get("relevant_strings_dict") or {}
    if not isinstance(raw_dict, dict):
        raw_dict = {}
    rel: dict[str, str] = {}
    for k, v in raw_dict.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        rel[k] = str(v)

    return action, rel


def _canonicalize_action(raw: str) -> str | None:
    cleaned = raw.strip().strip(".").strip('"').strip("'")
    for canonical in EMAIL_RESPONSE_LIST:
        if cleaned.lower() == canonical.lower():
            return canonical
    norm = re.sub(r"[^a-z]", "", cleaned.lower())
    for canonical in EMAIL_RESPONSE_LIST:
        if norm == re.sub(r"[^a-z]", "", canonical.lower()):
            return canonical
    return None
