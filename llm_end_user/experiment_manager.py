"""Core orchestrator.

Opens a single chat session per subject, sends 56 emails through it one
at a time (preserving full conversation memory), parses each reply,
maps the self-reported saliency back to a per-token 0/1 vector, and
appends one row per trial to dataset.csv.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .dataset_writer import DatasetWriter
from .email_loader import EmailLoader
from .prompts import SYSTEM_PROMPT, build_email_user_message
from .providers import open_chat_session, send_with_retry
from .saliency import map_saliency, tokenize
from .sequence_loader import load_sequence, resolve_sequence_path
from .types import (
    EMAIL_RESPONSE_LIST,
    HIT_ACTIONS,
    EmailContent,
    Legitimacy,
    TrialResult,
    TrialSpec,
)


log = logging.getLogger(__name__)


@dataclass
class Config:
    run_sequence: int
    run_subject_id: str
    run_dataset_path: Path
    run_sequences_dir: Path
    run_emails_csv: Path

    provider_name: str
    provider_model: str
    provider_temperature: float
    provider_max_tokens: int
    provider_retry_attempts: int
    provider_retry_backoff_sec: float

    @classmethod
    def load(cls, path: Path) -> "Config":
        with path.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh)
        run = data.get("run", {}) or {}
        provider = data.get("provider", {}) or {}
        return cls(
            run_sequence=int(run.get("sequence", 1)),
            run_subject_id=run.get("subject_id") or _new_subject_id(),
            run_dataset_path=Path(run.get("dataset_path", "data/dataset.csv")),
            run_sequences_dir=Path(run.get("sequences_dir", "data/sequences")),
            run_emails_csv=Path(run.get("emails_csv", "data/Emails.csv")),
            provider_name=str(provider.get("name", "gemini")),
            provider_model=str(provider.get("model", "gemini-2.5-flash")),
            provider_temperature=float(provider.get("temperature", 0.0)),
            provider_max_tokens=int(provider.get("max_output_tokens", 2048)),
            provider_retry_attempts=int(provider.get("retry_attempts", 3)),
            provider_retry_backoff_sec=float(provider.get("retry_backoff_sec", 5)),
        )


def _new_subject_id() -> str:
    return f"llm-{uuid.uuid4().hex[:12]}"


def _compute_hit(action: str | None, legitimacy: Legitimacy) -> int:
    if action is None or legitimacy != Legitimacy.PHISHING:
        return 0
    return 1 if action in HIT_ACTIONS else 0


def run_experiment(
    *,
    config_path: Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    load_dotenv()
    cfg = Config.load(config_path)
    _apply_overrides(cfg, overrides or {})

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log.info(
        "Starting subject=%s provider=%s model=%s sequence=%d",
        cfg.run_subject_id,
        cfg.provider_name,
        cfg.provider_model,
        cfg.run_sequence,
    )

    sequence_path = resolve_sequence_path(cfg.run_sequences_dir, cfg.run_sequence)
    trials: list[TrialSpec] = load_sequence(sequence_path)
    log.info("Loaded %d trials from %s", len(trials), sequence_path)

    loader = EmailLoader(cfg.run_emails_csv)
    writer = DatasetWriter(cfg.run_dataset_path)
    session = open_chat_session(
        cfg.provider_name,
        model=cfg.provider_model,
        system_prompt=SYSTEM_PROMPT,
        temperature=cfg.provider_temperature,
        max_output_tokens=cfg.provider_max_tokens,
    )

    for trial in trials:
        if trial.email_id not in loader:
            log.error("Email id %s missing in %s; skipping", trial.email_id, cfg.run_emails_csv)
            continue
        email = loader.load(trial.email_id)
        log.info(
            "Trial %d/%d email_id=%s bias=%s author=%s legit=%s",
            trial.trial_order,
            len(trials),
            trial.email_id,
            trial.bias.value,
            trial.author.value,
            trial.legitimacy.value,
        )
        result = _run_trial(cfg, session, trial, email)
        writer.append(result)

    log.info("Done. Wrote %d rows to %s", len(trials), cfg.run_dataset_path)
    return cfg.run_dataset_path


def _run_trial(
    cfg: Config,
    session,
    trial: TrialSpec,
    email: EmailContent,
) -> TrialResult:
    user_message = build_email_user_message(trial.trial_order, email)
    response = send_with_retry(
        session,
        user_message,
        retry_attempts=cfg.provider_retry_attempts,
        retry_backoff_sec=cfg.provider_retry_backoff_sec,
    )
    if response.action is None:
        log.warning(
            "Trial %d: could not parse action from response; raw=%r",
            trial.trial_order, response.raw_text[:200],
        )

    saliency_vector = map_saliency(email.body_text, response.relevant_strings_dict)
    expected_len = len(tokenize(email.body_text))
    assert len(saliency_vector) == expected_len, (
        f"saliency vector length {len(saliency_vector)} != tokens {expected_len}"
    )

    action = response.action if response.action in EMAIL_RESPONSE_LIST else ""
    return TrialResult(
        subject_id=cfg.run_subject_id,
        provider=cfg.provider_name,
        model=cfg.provider_model,
        sequence_id=cfg.run_sequence,
        trial_order=trial.trial_order,
        email_id=trial.email_id,
        bias=trial.bias.value,
        author=trial.author.value,
        legitimacy=trial.legitimacy.value,
        action=action,
        hit=_compute_hit(response.action, trial.legitimacy),
        saliency_strings=response.relevant_strings_dict,
        saliency_vector=saliency_vector,
        raw_response=response.raw_text,
    )


def _apply_overrides(cfg: Config, overrides: dict[str, Any]) -> None:
    mapping = {
        "sequence": "run_sequence",
        "subject_id": "run_subject_id",
        "provider": "provider_name",
        "model": "provider_model",
        "dataset_path": "run_dataset_path",
    }
    for cli_key, attr in mapping.items():
        if cli_key in overrides and overrides[cli_key] is not None:
            value = overrides[cli_key]
            if attr == "run_dataset_path":
                value = Path(value)
            setattr(cfg, attr, value)
