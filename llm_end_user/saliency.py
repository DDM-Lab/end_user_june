"""Build a 0/1 saliency vector from a model's self-reported relevant strings.

The protocol's primary saliency measure for LLM subjects (Task 2) is the
`relevant_strings_dict` the model returns alongside its action. We map
each highlighted substring back to character spans in the email's
plain-text body, then mark each token (word or punctuation run) as
1 if it falls inside any covered span, else 0.

The body the saliency vector is computed against MUST be the same
plain-text rendering shown to the model in the user message — otherwise
the substrings won't be locatable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def map_saliency(email_text: str, relevant_strings: dict[str, str]) -> list[int]:
    """Mark each token 1 if any model-highlighted string overlaps it."""
    tokens = tokenize(email_text)
    if not tokens or not relevant_strings:
        return [0] * len(tokens)

    covered: list[tuple[int, int]] = []
    lowered = email_text.lower()
    for needle in relevant_strings:
        if not needle:
            continue
        n = needle.strip().lower()
        if not n:
            continue
        start = 0
        while True:
            i = lowered.find(n, start)
            if i == -1:
                break
            covered.append((i, i + len(n)))
            start = i + max(1, len(n))

    vec = [0] * len(tokens)
    if not covered:
        return vec
    for idx, tok in enumerate(tokens):
        for span_start, span_end in covered:
            if tok.start < span_end and tok.end > span_start:
                vec[idx] = 1
                break
    return vec
