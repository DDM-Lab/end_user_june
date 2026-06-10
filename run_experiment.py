"""Bootstrap script for one LLM-subject run.

Usage:
    python run_experiment.py
    python run_experiment.py --sequence 2 --provider gemini --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_end_user.experiment_manager import run_experiment


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one LLM-subject pass of the phishing experiment.")
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--sequence", type=int, choices=range(1, 6))
    p.add_argument("--subject-id", type=str)
    p.add_argument("--provider", choices=["gemini", "openai", "ollama"])
    p.add_argument("--model", type=str)
    p.add_argument("--dataset-path", type=Path)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    overrides = {
        "sequence": args.sequence,
        "subject_id": args.subject_id,
        "provider": args.provider,
        "model": args.model,
        "dataset_path": args.dataset_path,
    }
    out_path = run_experiment(config_path=args.config, overrides=overrides)
    print(f"Wrote results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
