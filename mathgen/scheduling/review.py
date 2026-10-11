"""A finished review turned into FSRS memory and a due date.

Built for a stage ladder in which each review asks a few questions and ends in
one outcome: "advance" or "maintenance_pass" (no misses), "hold" (one miss) or
"reset" (too many misses). The outcome is the FSRS rating: reset is Again, hold
is Hard, a clean review is Good. Easy is unused: a fixed-length review gives no
signal that separates it from Good.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from . import fsrs

OUTCOME_RATING = {"reset": fsrs.AGAIN, "hold": fsrs.HARD, "advance": fsrs.GOOD, "maintenance_pass": fsrs.GOOD}

DEFAULT_DESIRED_RETENTION = 0.9
DEFAULT_MAXIMUM_INTERVAL_DAYS = 365


def days_between(earlier: datetime, later: datetime) -> float:
    return (later - earlier).total_seconds() / 86400


def memory_from_history(
    log_rows: list[dict], stage: int, stage_delays: list[float]
) -> tuple[fsrs.Memory | None, datetime | None]:
    """FSRS memory for a skill that has none yet, rebuilt by replaying its review
    history: rows with `outcome` and `created_at`, oldest first. Rows with an
    outcome FSRS does not rate are skipped.

    Returns (memory, time of the last replayed review). With no usable history
    but a `stage` above 0, stability is seeded from `stage_delays[stage - 1]`,
    the gap the learner already came back from to reach that stage (capped at
    the last delay for maintenance)."""
    memory = None
    last = None
    for row in log_rows:
        rating = OUTCOME_RATING.get(row["outcome"])
        if rating is None:
            continue
        at = row["created_at"]
        memory, _ = fsrs.step(memory, rating, days_between(last, at) if last else 0.0)
        last = at
    if memory is None and stage > 0 and stage_delays:
        delay = stage_delays[min(stage, len(stage_delays)) - 1]
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
) -> dict:
    """Score one finished review.

    `memory`/`last_review_at` are the skill's FSRS state before this review
    (None for a first review). Returns {"memory", "rating", "retrievability",
    "elapsed_days", "interval_days", "due_at"}. A reset is due at once, so a
    failed skill comes straight back; any other outcome is due when predicted
    recall falls to `desired_retention`, rounded to whole days, at least one
    and at most `maximum_interval_days`."""
    rating = OUTCOME_RATING[outcome]
    elapsed = days_between(last_review_at, now) if last_review_at else 0.0
    new_memory, r = fsrs.step(memory, rating, elapsed)
    if rating == fsrs.AGAIN:
        days = 0.0
    else:
        raw = fsrs.interval_days(new_memory.stability, desired_retention)
        days = float(min(maximum_interval_days, max(1, round(raw))))
    return {
        "memory": new_memory,
        "rating": rating,
        "retrievability": r,
        "elapsed_days": elapsed,
        "interval_days": days,
        "due_at": now + timedelta(days=days),
    }
