"""FSRS memory model for spaced review of skills.

FSRS (Free Spaced Repetition Scheduler) is the open scheduling algorithm Anki
ships. It tracks two numbers per learner and skill: *stability* S, the number
of days until recall probability falls to 90%, and *difficulty* D in [1, 10].
Each completed review is rated (Again, Hard, Good, Easy) and moves both. The
next review is set for the day predicted recall drops to the desired retention.

This module is the FSRS-5 model with its published default weights; it matches
the reference implementation, py-fsrs 5.1.3. It is pure: no database, no
wall-clock reads, no settings lookups. `mathgen.scheduling.review` maps a
finished review onto it; an app stores the `Memory` it returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The public names: see docs/api.md. Anything else here may change in any release.
__all__ = [
    "AGAIN",
    "HARD",
    "GOOD",
    "EASY",
    "DEFAULT_WEIGHTS",
    "Memory",
    "retrievability",
    "interval_days",
    "initial_stability",
    "initial_difficulty",
    "next_difficulty",
    "step",
]

AGAIN, HARD, GOOD, EASY = 1, 2, 3, 4

# FSRS-5 default parameters, w[0]..w[18].
DEFAULT_WEIGHTS: tuple[float, ...] = (
    0.40255,
    1.18385,
    3.173,
    15.69105,
    7.1949,
    0.5345,
    1.4604,
    0.0046,
    1.54575,
    0.1192,
    1.01925,
    1.9395,
    0.11,
    0.29605,
    2.2698,
    0.2315,
    2.9898,
    0.51655,
    0.6621,
)

DECAY = -0.5
FACTOR = 19 / 81  # chosen so retrievability(S, S) == 0.9 exactly

MIN_STABILITY = 0.01
MIN_DIFFICULTY, MAX_DIFFICULTY = 1.0, 10.0


@dataclass(frozen=True)
class Memory:
    stability: float
    difficulty: float


def _clamp_d(d: float) -> float:
    return min(MAX_DIFFICULTY, max(MIN_DIFFICULTY, d))


def retrievability(elapsed_days: float, stability: float) -> float:
    """Predicted probability of recall `elapsed_days` after the last review."""
    if stability <= 0:
        raise ValueError(f"stability must be positive, got {stability!r}")
    elapsed_days = max(0.0, elapsed_days)
    return (1 + FACTOR * elapsed_days / stability) ** DECAY


def interval_days(stability: float, desired_retention: float) -> float:
    """Days until retrievability falls to `desired_retention` (unrounded)."""
    if not 0 < desired_retention < 1:
        raise ValueError(f"desired_retention must be between 0 and 1, got {desired_retention!r}")
    return stability / FACTOR * (desired_retention ** (1 / DECAY) - 1)


def initial_stability(rating: int, w=DEFAULT_WEIGHTS) -> float:
    return max(MIN_STABILITY, w[rating - 1])


def initial_difficulty(rating: int, w=DEFAULT_WEIGHTS) -> float:
    return _clamp_d(w[4] - math.exp(w[5] * (rating - 1)) + 1)


def next_difficulty(d: float, rating: int, w=DEFAULT_WEIGHTS) -> float:
    delta = -w[6] * (rating - 3)
    damped = d + delta * (10 - d) / 9  # linear damping: changes shrink near 10
    mean_reverted = w[7] * initial_difficulty(EASY, w) + (1 - w[7]) * damped
    return _clamp_d(mean_reverted)


def _recall_stability(d: float, s: float, r: float, rating: int, w) -> float:
    hard_penalty = w[15] if rating == HARD else 1.0
    easy_bonus = w[16] if rating == EASY else 1.0
    growth = math.exp(w[8]) * (11 - d) * s ** (-w[9]) * (math.exp(w[10] * (1 - r)) - 1) * hard_penalty * easy_bonus
    return s * (1 + growth)


def _forget_stability(d: float, s: float, r: float, w) -> float:
    s_forget = w[11] * d ** (-w[12]) * ((s + 1) ** w[13] - 1) * math.exp(w[14] * (1 - r))
    # A lapse never leaves memory more stable than a same-day Again would.
    return min(s_forget, s / math.exp(w[17] * w[18]))


def _short_term_stability(s: float, rating: int, w) -> float:
    return s * math.exp(w[17] * (rating - 3 + w[18]))


def step(memory: Memory | None, rating: int, elapsed_days: float, w=DEFAULT_WEIGHTS) -> tuple[Memory, float | None]:
    """Apply one rated review. Returns (new memory, retrievability before the
    review). The retrievability is None for a first review, which has no
    prediction to score.

    `elapsed_days` is counted in whole days, rounded down, as the reference
    implementation counts it: a review 4.75 days after the last one is a
    4-day gap, and anything under a day is a same-day review."""
    if isinstance(rating, bool) or rating not in (AGAIN, HARD, GOOD, EASY):
        raise ValueError(f"rating must be 1..4, got {rating!r}")
    if memory is None:
        return Memory(initial_stability(rating, w), initial_difficulty(rating, w)), None
    s, d = memory.stability, memory.difficulty
    elapsed_days = math.floor(max(0.0, elapsed_days))
    r = retrievability(elapsed_days, s)
    if elapsed_days < 1:
        new_s = _short_term_stability(s, rating, w)
    elif rating == AGAIN:
        new_s = _forget_stability(d, s, r, w)
    else:
        new_s = _recall_stability(d, s, r, rating, w)
    return Memory(max(MIN_STABILITY, new_s), next_difficulty(d, rating, w)), r
