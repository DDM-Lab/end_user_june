"""Parse a sequence_N.txt file into ordered TrialSpec records.

Expected line format (TSV, one trial per line):

    <email_id>\t<bias>\t<author>\t<legitimacy>

Blank lines and lines beginning with `#` are ignored.
"""

from __future__ import annotations

from pathlib import Path

from .types import Author, Bias, Legitimacy, TrialSpec


def _normalize_author(raw: str) -> Author:
    r = raw.strip().lower()
    if r.startswith("human"):
        return Author.HUMAN
    return Author.LLM


def _normalize_legitimacy(raw: str) -> Legitimacy:
    r = raw.strip().lower()
    if r in ("phishing", "spam"):
        return Legitimacy.PHISHING
    return Legitimacy.HAM


def load_sequence(path: Path) -> list[TrialSpec]:
    trials: list[TrialSpec] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(
                    f"{path}:{lineno}: expected 4 tab-separated fields, got {len(parts)}: {line!r}"
                )
            email_id, bias_str, author_str, legit_str = (p.strip() for p in parts)
            trials.append(
                TrialSpec(
                    trial_order=len(trials) + 1,
                    email_id=email_id,
                    bias=Bias(bias_str.lower()),
                    author=_normalize_author(author_str),
                    legitimacy=_normalize_legitimacy(legit_str),
                )
            )
    return trials


def resolve_sequence_path(sequences_dir: Path, sequence_id: int) -> Path:
    candidate = sequences_dir / f"sequence_{sequence_id}.txt"
    if not candidate.exists():
        raise FileNotFoundError(f"Sequence file not found: {candidate}")
    return candidate
