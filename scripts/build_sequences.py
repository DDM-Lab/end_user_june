"""Build sequence_{1..N}.txt from data/Emails.csv + annotations CSV.

Each sequence is a randomized ordering of the same 56 emails, where the
56 emails fill the 7 (bias) x 2 (author) x 2 (legitimacy) = 28 cells
with 2 emails each.

Selection criterion: for every (bias, author, legitimacy) cell, take the
2 emails with the highest annotator vote share for that bias (i.e. the
emails most annotators agreed represent the bias). Each email is used
in exactly one cell.

Output format (TSV, one trial per line):

    email_id<TAB>bias<TAB>author<TAB>legitimacy

Lines beginning with `#` and blank lines are ignored by the loader.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

BIASES = [
    "conformity",
    "authority",
    "urgency",
    "overconfidence",
    "familiarity",
    "scarcity",
    "anchoring",
]
AUTHORS = ["human", "llm"]
LEGITS = ["phishing", "ham"]


def _normalize_bias(raw: str) -> str:
    return (
        raw.replace("bias:", "")
        .replace("-trigger", "")
        .replace("-effect", "")
        .replace("-bias", "")
    )


def _load_emails(path: Path) -> dict[str, dict]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["EmailId"]: row for row in csv.DictReader(fh)}


def _load_annotations(path: Path) -> dict[str, dict[str, set[str]]]:
    """email_id -> {user_id -> {bias_names}} (excluding zero-biases)."""
    csv.field_size_limit(sys.maxsize)
    ub: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            b = _normalize_bias(row["selected_bias"])
            if b == "zeroes":
                continue
            ub[row["email_id"]][row["user_id"]].add(b)
    return ub


def _select_56(emails: dict[str, dict], user_bias: dict) -> list[dict]:
    cells = [(b, a, t) for b in BIASES for a in AUTHORS for t in LEGITS]
    candidates: dict[tuple, list[tuple[float, str]]] = {c: [] for c in cells}

    for eid, users in user_bias.items():
        if eid not in emails:
            continue
        e = emails[eid]
        author = "human" if e["Author"] == "Human" else "llm"
        legit = e["Type"]
        n_users = len(users)
        per_bias_votes: Counter[str] = Counter()
        for biases in users.values():
            for b in biases:
                per_bias_votes[b] += 1
        for bias, votes in per_bias_votes.items():
            candidates[(bias, author, legit)].append((votes / n_users, eid))

    for c in candidates:
        candidates[c].sort(reverse=True)

    used: set[str] = set()
    selection: list[dict] = []
    # Fill rarest cells first so we don't starve them.
    for c in sorted(cells, key=lambda c: len(candidates[c])):
        bias, author, legit = c
        picked = 0
        for _score, eid in candidates[c]:
            if eid in used:
                continue
            selection.append(
                {
                    "email_id": eid,
                    "bias": bias,
                    "author": author,
                    "legitimacy": legit,
                }
            )
            used.add(eid)
            picked += 1
            if picked == 2:
                break
        if picked < 2:
            print(
                f"WARNING: cell {c} only filled with {picked}/2 emails",
                file=sys.stderr,
            )
    return selection


def _write_sequence(path: Path, ordering: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# Auto-generated from data/Emails.csv + annotations.\n")
        fh.write("# Format: email_id<TAB>bias<TAB>author<TAB>legitimacy\n")
        for row in ordering:
            fh.write(
                f"{row['email_id']}\t{row['bias']}\t{row['author']}\t{row['legitimacy']}\n"
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--emails-csv", type=Path, default=Path("data/Emails.csv"))
    p.add_argument(
        "--annotations-csv",
        type=Path,
        default=Path("data/cognitive-phishing-annotations-latest.csv"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/sequences"))
    p.add_argument("--num-sequences", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260610)
    args = p.parse_args()

    emails = _load_emails(args.emails_csv)
    user_bias = _load_annotations(args.annotations_csv)
    base_selection = _select_56(emails, user_bias)
    print(f"Selected {len(base_selection)} emails covering 28 cells x 2 each.")

    rng = random.Random(args.seed)
    for i in range(1, args.num_sequences + 1):
        ordering = list(base_selection)
        rng.shuffle(ordering)
        out_path = args.out_dir / f"sequence_{i}.txt"
        _write_sequence(out_path, ordering)
        print(f"Wrote {out_path} ({len(ordering)} trials)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
