"""Append-only writer for dataset.csv. Header is written on first row."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .types import TrialResult


_FIELDS = [
    "subject_id",
    "provider",
    "model",
    "sequence_id",
    "trial_order",
    "email_id",
    "bias",
    "author",
    "legitimacy",
    "action",
    "hit",
    "saliency_strings",
    "saliency_vector",
    "raw_response",
]


class DatasetWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._needs_header = not self.path.exists() or self.path.stat().st_size == 0

    def append(self, row: TrialResult) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FIELDS)
            if self._needs_header:
                writer.writeheader()
                self._needs_header = False
            writer.writerow(
                {
                    "subject_id": row.subject_id,
                    "provider": row.provider,
                    "model": row.model,
                    "sequence_id": row.sequence_id,
                    "trial_order": row.trial_order,
                    "email_id": row.email_id,
                    "bias": row.bias,
                    "author": row.author,
                    "legitimacy": row.legitimacy,
                    "action": row.action,
                    "hit": row.hit,
                    "saliency_strings": json.dumps(row.saliency_strings, ensure_ascii=False),
                    "saliency_vector": json.dumps(row.saliency_vector),
                    "raw_response": row.raw_response,
                }
            )
