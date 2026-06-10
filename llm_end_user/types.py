"""Core types shared across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


EMAIL_RESPONSE_LIST: list[str] = [
    "Archive",
    "Mark as Spam",
    "Delete",
    "Reply",
    "Download Attachments",
    "Click Link",
]

# Compliance with a phishing request = "hit".
HIT_ACTIONS: frozenset[str] = frozenset({"Reply", "Download Attachments", "Click Link"})


class Bias(str, Enum):
    CONFORMITY = "conformity"
    AUTHORITY = "authority"
    URGENCY = "urgency"
    OVERCONFIDENCE = "overconfidence"
    FAMILIARITY = "familiarity"
    SCARCITY = "scarcity"
    ANCHORING = "anchoring"


class Author(str, Enum):
    HUMAN = "human"
    LLM = "llm"


class Legitimacy(str, Enum):
    HAM = "ham"
    PHISHING = "phishing"


@dataclass(frozen=True)
class TrialSpec:
    """One row in a sequence file — metadata only, no email body."""

    trial_order: int
    email_id: str
    bias: Bias
    author: Author
    legitimacy: Legitimacy


@dataclass(frozen=True)
class EmailContent:
    """The full content of one email — what goes into the prompt."""

    email_id: str
    sender_name: str
    sender_address: str
    subject: str
    body_text: str          # readable plain-text rendering of the HTML body
    body_html: str          # original HTML source (kept for reference / image)


@dataclass
class ChatTurnResponse:
    """A model's reply to one email turn."""

    raw_text: str
    action: Optional[str] = None
    relevant_strings_dict: dict[str, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


@dataclass
class TrialResult:
    """One row appended to dataset.csv."""

    subject_id: str
    provider: str
    model: str
    sequence_id: int
    trial_order: int
    email_id: str
    bias: str
    author: str
    legitimacy: str
    action: str
    hit: int
    saliency_strings: dict[str, str]
    saliency_vector: list[int]
    raw_response: str
