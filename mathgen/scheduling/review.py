"""A finished review turned into an outcome, FSRS memory and a due date.

Built for a stage ladder in which each review asks a few questions and ends in
one outcome: "advance" or "maintenance_pass" (no misses), "hold" (some misses,
fewer than the reset threshold) or "reset" (too many misses).
`outcome_from_marks` is that rule, applied to the review's graded answers. The
outcome is the FSRS rating: reset is Again, hold is Hard, a clean review is
Good. Easy is unused: a fixed-length review gives no signal that separates it
from Good.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from fractions import Fraction

from . import fsrs

# The public names: see docs/api.md. Anything else here may change in any release.
__all__ = [
    "OUTCOME_RATING",
    "DEFAULT_DESIRED_RETENTION",
    "DEFAULT_MAXIMUM_INTERVAL_DAYS",
    "DEFAULT_RESET_MISSES",
    "DEFAULT_RESET_GAP_DAYS",
    "outcome_from_marks",
    "memory_from_history",
    "score",
]

OUTCOME_RATING = {"reset": fsrs.AGAIN, "hold": fsrs.HARD, "advance": fsrs.GOOD, "maintenance_pass": fsrs.GOOD}

DEFAULT_DESIRED_RETENTION = 0.9
DEFAULT_MAXIMUM_INTERVAL_DAYS = 365
DEFAULT_RESET_MISSES = 2
DEFAULT_RESET_GAP_DAYS = 1.0


def _miss(result) -> Fraction:
    """How much one graded answer counts as a miss: 0 for full marks, 1/2 for at
    least half the marks, 1 for less than half. `result` is a `GradeResult`
    (multi-part results are read from `marks`/`max_marks`, single answers from
    `correct`) or an `(earned, available)` pair."""
    if isinstance(result, tuple):
        earned, available = result
    elif getattr(result, "max_marks", None):
        earned, available = result.marks, result.max_marks
    else:
        earned, available = (1 if result.correct else 0), 1
    earned, available = Fraction(earned), Fraction(available)
    if available <= 0 or not 0 <= earned <= available:
        raise ValueError(f"marks must satisfy 0 <= earned <= available and available > 0, got {earned}/{available}")
    if earned == available:
        return Fraction(0)
    return Fraction(1, 2) if earned * 2 >= available else Fraction(1)


def outcome_from_marks(results, *, maintenance: bool = False, reset_misses: int = DEFAULT_RESET_MISSES) -> str:
    """The outcome of one finished review, from the graded answers to its questions.

    Each answer counts as a miss by its marks: none for full marks, half a miss
    for at least half the marks, a whole miss for less than half (see `_miss`).
    No misses advances the skill ("maintenance_pass" when `maintenance`); misses
    adding up to `reset_misses` or more reset it; anything in between holds it.
    With whole marks this is the plain ladder: one wrong answer holds, two
    reset. Two half-right answers add up to one miss, so they hold too.
    `results` must not be empty."""
    results = list(results)
    if not results:
        raise ValueError("a review needs at least one graded answer")
    if reset_misses < 1:
        raise ValueError(f"reset_misses must be at least 1, got {reset_misses}")
    misses = sum((_miss(r) for r in results), Fraction(0))
    if misses == 0:
        return "maintenance_pass" if maintenance else "advance"
    return "reset" if misses >= reset_misses else "hold"


def days_between(earlier: datetime, later: datetime) -> float:
    return (later - earlier).total_seconds() / 86400


def memory_from_history(
    log_rows: list[dict], stage: int, stage_delays: list[float]
) -> tuple[fsrs.Memory | None, datetime | None]:
    """FSRS memory for a skill that has none yet, rebuilt by replaying its review
    history: rows with `outcome` and `created_at`, in any order (they are
    replayed oldest first). Rows with an outcome FSRS does not rate are skipped.

    Returns (memory, time of the last replayed review). With no usable history
    but a `stage` above 0, stability is seeded from `stage_delays[stage - 1]`,
    the gap the learner already came back from to reach that stage (capped at
    the last delay for maintenance). A stage whose delay is not positive gives
    no seed: the skill starts fresh."""
    memory = None
    last = None
    for row in sorted(log_rows, key=lambda row: row["created_at"]):
        rating = OUTCOME_RATING.get(row["outcome"])
        if rating is None:
            continue
        at = row["created_at"]
        memory, _ = fsrs.step(memory, rating, days_between(last, at) if last else 0.0)
        last = at
    if memory is None and stage > 0 and stage_delays:
        delay = stage_delays[min(stage, len(stage_delays)) - 1]
        if delay > 0:
            memory = fsrs.Memory(float(delay), fsrs.initial_difficulty(fsrs.GOOD))
    return memory, last


def score(
    outcome: str,
    memory: fsrs.Memory | None,
    last_review_at: datetime | None,
    now: datetime,
    *,
    desired_retention: float = DEFAULT_DESIRED_RETENTION,
    maximum_interval_days: int = DEFAULT_MAXIMUM_INTERVAL_DAYS,
    reset_gap_days: float = DEFAULT_RESET_GAP_DAYS,
) -> dict:
    """Score one finished review.

    `memory`/`last_review_at` are the skill's FSRS state before this review
    (None for a first review). Returns {"memory", "rating", "retrievability",
    "elapsed_days", "interval_days", "due_at"}.

    A reset is due after `reset_gap_days` (a day by default; 0 brings it
    straight back), so the retry tests what the learner kept rather than what
    they saw a minute ago. Any other outcome is due when predicted recall falls
    to `desired_retention`, rounded to whole days, at least one and at most
    `maximum_interval_days`. A hold never lengthens the gap: it is due no later
    than the whole days since the last review, so a partly wrong review cannot
    push the skill further away than a clean one before it did."""
    if reset_gap_days < 0:
        raise ValueError(f"reset_gap_days must not be negative, got {reset_gap_days}")
    rating = OUTCOME_RATING[outcome]
    elapsed = days_between(last_review_at, now) if last_review_at else 0.0
    new_memory, r = fsrs.step(memory, rating, elapsed)
    if rating == fsrs.AGAIN:
        days = float(reset_gap_days)
    else:
        raw = fsrs.interval_days(new_memory.stability, desired_retention)
        days = float(min(maximum_interval_days, max(1, round(raw))))
        if rating == fsrs.HARD and last_review_at is not None:
            days = min(days, float(max(1, math.floor(elapsed))))
    return {
        "memory": new_memory,
        "rating": rating,
        "retrievability": r,
        "elapsed_days": elapsed,
        "interval_days": days,
        "due_at": now + timedelta(days=days),
    }
