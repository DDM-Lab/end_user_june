"""Load email content from data/Emails.csv keyed by EmailId.

The CSV stores the email body as raw HTML. We expose both a human-readable
plain-text rendering (built by stripping HTML, preserving hyperlinks as
[anchor text](href) so phishing cues like link-mismatch survive) and the
original HTML source.
"""

from __future__ import annotations

import csv
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

from .types import EmailContent


class _BodyExtractor(HTMLParser):
    """Convert HTML to readable plain text while preserving link targets.

    Strategy: only emit text once we have entered `<body>` and we are NOT
    currently inside a `<style>` or `<script>` block. This sidesteps the
    void-element trap (meta/link/img/br have no end tag, so naive
    skip-depth bookkeeping under `<head>` runs forever).

    Links are rendered as `anchor [-> href]` so an LLM can spot mismatches
    between the visible anchor text and the actual destination — a key
    phishing tell that is otherwise lost in a naive strip.
    """

    _BLOCK_TAGS = {"p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
    _DROP_BLOCK_TAGS = {"style", "script"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._in_body = False
        self._drop_depth = 0
        self._link_href: str | None = None
        self._link_text_start: int | None = None

    def _emit(self, s: str) -> None:
        if self._in_body and not self._drop_depth:
            self._parts.append(s)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "body":
            self._in_body = True
            return
        if tag in self._DROP_BLOCK_TAGS:
            self._drop_depth += 1
            return
        if not self._in_body or self._drop_depth:
            return
        if tag in self._BLOCK_TAGS or tag == "br":
            self._parts.append("\n")
        elif tag == "a":
            href = next((v for k, v in attrs if k.lower() == "href"), None)
            self._link_href = href
            self._link_text_start = len(self._parts)

    def handle_endtag(self, tag: str) -> None:
        if tag == "body":
            self._in_body = False
            return
        if tag in self._DROP_BLOCK_TAGS:
            self._drop_depth = max(0, self._drop_depth - 1)
            return
        if not self._in_body or self._drop_depth:
            return
        if tag == "a" and self._link_href and self._link_text_start is not None:
            anchor = "".join(self._parts[self._link_text_start :]).strip()
            if anchor and self._link_href not in ("", "#") and self._link_href != anchor:
                self._parts.append(f" [-> {self._link_href}]")
            self._link_href = None
            self._link_text_start = None
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_body and not self._drop_depth:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()


def _extract_body_text(html: str) -> str:
    p = _BodyExtractor()
    p.feed(html)
    return p.get_text()


class EmailLoader:
    """Load emails on demand from a single Emails.csv."""

    def __init__(self, csv_path: Path) -> None:
        csv.field_size_limit(sys.maxsize)
        self._path = csv_path
        self._rows: dict[str, dict[str, str]] = {}
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self._rows[row["EmailId"]] = row

    def __contains__(self, email_id: str) -> bool:
        return email_id in self._rows

    def load(self, email_id: str) -> EmailContent:
        try:
            row = self._rows[email_id]
        except KeyError as e:
            raise KeyError(
                f"Email id {email_id!r} not found in {self._path}"
            ) from e
        html = row.get("New Email Body", "") or ""
        return EmailContent(
            email_id=email_id,
            sender_name=row.get("Sender Name", "") or "",
            sender_address=row.get("Sender", "") or "",
            subject=row.get("Subject", "") or "",
            body_text=_extract_body_text(html),
            body_html=html,
        )
