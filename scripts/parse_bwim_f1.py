#!/usr/bin/env python3
"""Parse BWIM run logs and compute block-level F1 for each round.

Extracts expected and actual block sets from feedback lines,
computes precision, recall, and F1, and prints a summary table.

Usage:
    python parse_bwim_f1.py <logfile> [<logfile2> ...]
    python parse_bwim_f1.py logs/transcripts/*/console.log
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RoundResult:
    round_num: int
    seed: int
    correct: bool
    round_score: int
    total_score: int
    expected: set[tuple[str, int, int, int]]
    predicted: set[tuple[str, int, int, int]]
    precision: float
    recall: float
    f1: float
    questions: int


def parse_blocks(block_str: str) -> set[tuple[str, int, int, int]]:
    """Parse a semicolon-separated block string into a set of (Color,x,y,z) tuples."""
    blocks = set()
    if not block_str or not block_str.strip():
        return blocks
    for item in block_str.split(";"):
        item = item.strip().rstrip(",")
        if not item:
            continue
        parts = item.split(",")
        if len(parts) != 4:
            continue
        try:
            color = parts[0].strip()
            x = int(parts[1].strip())
            y = int(parts[2].strip())
            z = int(parts[3].strip())
            blocks.add((color, x, y, z))
        except (ValueError, IndexError):
            continue
    return blocks


def compute_f1(
    expected: set[tuple[str, int, int, int]],
    predicted: set[tuple[str, int, int, int]],
) -> tuple[float, float, float]:
    """Compute block-level precision, recall, F1."""
    if not expected and not predicted:
        return 1.0, 1.0, 1.0

    intersection = expected & predicted
    n_match = len(intersection)
    n_pred = len(predicted)
    n_exp = len(expected)

    precision = n_match / n_pred if n_pred > 0 else 0.0
    recall = n_match / n_exp if n_exp > 0 else 0.0

    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# Regex patterns for feedback lines
_CORRECT_RE = re.compile(
    r"User input: Feedback: Correct structure built! \+10 points\. "
    r"(.*?) \| Round score: ([+-]?\d+) \| Total score: ([+-]?\d+)"
)
_INCORRECT_RE = re.compile(
    r"User input: Feedback: Incorrect structure\. -10 points\. "
    r"Expected: (.*?), but got: (.*?) \| Round score: ([+-]?\d+) \| Total score: ([+-]?\d+)"
)
_SEED_RE = re.compile(r"Starting trial (\d+)/\d+ with seed (\d+)")
_QUESTION_RE = re.compile(r"\[ASK\]")
_AGENT_MSG_RE = re.compile(r"^\S+:")


def parse_log(path: str) -> list[RoundResult]:
    """Parse a single BWIM log file and return per-round results."""
    results = []
    current_seed = 0
    round_counter = 0
    questions_this_round = 0

    with open(path) as f:
        for line in f:
            # Track seed transitions
            m = _SEED_RE.search(line)
            if m:
                current_seed = int(m.group(2))
                continue

            # Count questions (ASK responses from any agent)
            if _QUESTION_RE.search(line) and _AGENT_MSG_RE.search(line):
                questions_this_round += 1
                continue

            # Correct round
            m = _CORRECT_RE.search(line)
            if m:
                round_counter += 1
                block_str = m.group(1)
                round_score = int(m.group(2))
                total_score = int(m.group(3))
                expected = parse_blocks(block_str)
                # On correct rounds, predicted == expected
                prec, rec, f1 = 1.0, 1.0, 1.0
                results.append(RoundResult(
                    round_num=round_counter,
                    seed=current_seed,
                    correct=True,
                    round_score=round_score,
                    total_score=total_score,
                    expected=expected,
                    predicted=expected,
                    precision=prec,
                    recall=rec,
                    f1=f1,
                    questions=questions_this_round,
                ))
                questions_this_round = 0
                continue

            # Incorrect round
            m = _INCORRECT_RE.search(line)
            if m:
                round_counter += 1
                expected_str = m.group(1)
                got_str = m.group(2)
                round_score = int(m.group(3))
                total_score = int(m.group(4))
                expected = parse_blocks(expected_str)
                predicted = parse_blocks(got_str)
                prec, rec, f1 = compute_f1(expected, predicted)
                results.append(RoundResult(
                    round_num=round_counter,
                    seed=current_seed,
                    correct=False,
                    round_score=round_score,
                    total_score=total_score,
                    expected=expected,
                    predicted=predicted,
                    precision=prec,
                    recall=rec,
                    f1=f1,
                    questions=questions_this_round,
                ))
                questions_this_round = 0
                continue

    return results


def print_summary(results: list[RoundResult], label: str = "") -> None:
    """Print a summary table of round results."""
    if not results:
        print(f"No rounds found{' in ' + label if label else ''}.")
        return

    header = f"{'Rnd':>4} {'Seed':>4} {'OK':>3} {'RndScr':>6} {'TotScr':>6} {'Qst':>3} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Exp':>4} {'Pred':>4} {'Match':>5}"
    print(f"\n{'=' * 70}")
    if label:
        print(f"  {label}")
        print(f"{'=' * 70}")
    print(header)
    print("-" * len(header))

    for r in results:
        n_match = len(r.expected & r.predicted)
        ok = "Y" if r.correct else "N"
        print(
            f"{r.round_num:>4} {r.seed:>4} {ok:>3} "
            f"{r.round_score:>+6} {r.total_score:>+6} {r.questions:>3} "
            f"{r.precision:>6.3f} {r.recall:>6.3f} {r.f1:>6.3f} "
            f"{len(r.expected):>4} {len(r.predicted):>4} {n_match:>5}"
        )

    print("-" * len(header))

    # Aggregate stats
    n_total = len(results)
    n_correct = sum(1 for r in results if r.correct)
    n_incorrect = n_total - n_correct
    total_questions = sum(r.questions for r in results)
    final_score = results[-1].total_score if results else 0

    all_f1 = [r.f1 for r in results]
    incorrect_f1 = [r.f1 for r in results if not r.correct]
    mean_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    mean_incorrect_f1 = sum(incorrect_f1) / len(incorrect_f1) if incorrect_f1 else 0.0

    print(f"\n  Rounds:     {n_total}")
    print(f"  Correct:    {n_correct} ({100 * n_correct / n_total:.1f}%)")
    print(f"  Incorrect:  {n_incorrect}")
    print(f"  Questions:  {total_questions} ({total_questions / n_total:.2f}/round)")
    print(f"  Final score: {final_score:+d}")
    print(f"  Mean F1 (all):       {mean_f1:.4f}")
    if incorrect_f1:
        print(f"  Mean F1 (incorrect): {mean_incorrect_f1:.4f}")

        # Categorize failures
        near_miss = [r for r in results if not r.correct and r.f1 >= 0.5]
        catastrophic = [r for r in results if not r.correct and r.f1 < 0.5]
        empty = [r for r in results if not r.correct and len(r.predicted) == 0]
        print(f"  Near-miss (F1>=0.5): {len(near_miss)}")
        print(f"  Catastrophic (F1<0.5): {len(catastrophic)}")
        print(f"  Empty builds:        {len(empty)}")

    # Per-seed breakdown
    seeds = sorted(set(r.seed for r in results))
    if len(seeds) > 1:
        print(f"\n  Per-seed breakdown:")
        print(f"  {'Seed':>4} {'Rounds':>6} {'Correct':>7} {'Acc%':>6} {'MeanF1':>7} {'Score':>6} {'Qst':>4}")
        for seed in seeds:
            seed_results = [r for r in results if r.seed == seed]
            sc = sum(1 for r in seed_results if r.correct)
            sf1 = sum(r.f1 for r in seed_results) / len(seed_results)
            sq = sum(r.questions for r in seed_results)
            # Find the last total_score for this seed
            last_score = seed_results[-1].total_score
            print(
                f"  {seed:>4} {len(seed_results):>6} {sc:>7} "
                f"{100 * sc / len(seed_results):>5.1f}% {sf1:>7.4f} "
                f"{last_score:>+6} {sq:>4}"
            )

    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_bwim_f1.py <logfile> [<logfile2> ...]")
        sys.exit(1)

    all_results = []
    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} not found, skipping.")
            continue
        results = parse_log(str(path))
        label = path.name
        print_summary(results, label)
        all_results.extend(results)

    if len(sys.argv) > 2 and all_results:
        print_summary(all_results, "COMBINED (all logs)")


if __name__ == "__main__":
    main()
