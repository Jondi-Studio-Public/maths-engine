"""Exam question: name the likelihood of a compound event, using a stated convention.

Two independent events have stated probabilities (as percentages in steps of 5). Part (a) finds the probability that
both occur, part (b) the probability that at least one occurs (inclusion-exclusion, using the answer to (a)), and
part (c) names that probability with the four-term convention restated in the question. The convention is the one the
skill uses: possible for 0 < P <= 0.3, probable for 0.3 < P <= 0.55, likely for 0.55 < P < 0.9, certain for P >= 0.9.

Because the stated percentages are multiples of 5, both numeric answers are exact to 2 decimal places, and a value
can land exactly on a range edge (30%, 55%), where the closed end of each range decides the term.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator
from .exd_x_r2k_support import retry_build, number_part, word_part

_EVENTS = (
    ("it rains on Saturday", "it rains on Sunday"),
    ("the first delivery arrives on time", "the second delivery arrives on time"),
    ("the home team wins its first match", "the home team wins its second match"),
    ("the first smoke alarm sounds in a test", "the second smoke alarm sounds in a test"),
    ("a seedling in the first tray survives", "a seedling in the second tray survives"),
    ("the morning train is on time", "the evening train is on time"),
)
_CONVENTION = (
    "A probability P is described as 'possible' for 0 < P ≤ 0.3, 'probable' for 0.3 < P ≤ 0.55, 'likely' for "
    "0.55 < P < 0.9, and 'certain' for P ≥ 0.9."
)
_TERMS = ("possible", "probable", "likely", "certain")


def term(p: Fraction) -> str:
    """The convention's term for a probability p (exact)."""
    if p <= Fraction(3, 10):
        return "possible"
    if p <= Fraction(11, 20):
        return "probable"
    if p < Fraction(9, 10):
        return "likely"
    return "certain"


def _build_k_probability_language(rng: random.Random) -> dict | None:
    """P(both), P(at least one), then the convention's term for it."""
    p = rng.randrange(5, 100, 5)
    q = rng.randrange(5, 100, 5)
    both = Fraction(p * q, 100)  # percent
    either = p + q - both
    check(either == 100 - Fraction((100 - p) * (100 - q), 100), "probability language: union disagrees with complement")
    key = term(either / 100)
    ev_a, ev_b = rng.choice(_EVENTS)
    stem = (
        f"The probability that {ev_a} is {p}% and the probability that {ev_b} is {q}%. The two events are "
        f"independent. {_CONVENTION}"
    )
    pa = number_part(
        "Find the probability, as a percentage, that both events occur.",
        both,
        2,
        1,
        [f"P(both) = {p}% x {q}% = {float(both):.2f}%."],
        [
            (Fraction(p + q), "adds-instead-of-multiplies") if p + q < 100 else (None, ""),
            (Fraction(p + q, 2), "averaged-instead-of-multiplied"),
            (Fraction(min(p, q)), "took-smaller-probability"),
            (Fraction(p * (100 - q), 100), "used-one-event-failing"),
        ],
        sensible_range=(0, 100),
    )
    pb = number_part(
        "Hence find the probability, as a percentage, that at least one of the events occurs.",
        either,
        2,
        2,
        [f"P(at least one) = {p}% + {q}% - {float(both):.2f}% = {float(either):.2f}%."],
        [
            (Fraction(p + q) if p + q < 100 else None, "added-probabilities-for-at-least-one"),
            (100 - both, "gave-complement-of-both"),
            (both, "all-instead-of-at-least-one"),
            (Fraction(p + q, 2), "averaged-instead-of-multiplied"),
        ],
        follow_through=f"{p} + {q} - part_a",
        sensible_range=(0, 100),
    )
    if pa is None or pb is None:
        return None

    edges = {Fraction(3, 10): ("possible", "probable"), Fraction(11, 20): ("probable", "likely")}
    on_edge = (either / 100) in edges
    adjacent = {_TERMS[i]: [_TERMS[j] for j in (i - 1, i + 1) if 0 <= j < 4] for i in range(4)}
    wrongs = []
    for option in _TERMS:
        if option == key:
            continue
        if on_edge and option in edges[either / 100]:
            tag = "ignores-boundary-inclusivity"
        elif option in adjacent[key]:
            tag = "uses-connotation-not-stated-ranges"
        elif option == term(both / 100):
            tag = "reuses-earlier-part-value"
        else:
            tag = None
        wrongs.append((option, tag))
    pc = word_part(
        "Using the convention above, which term describes the probability that at least one of the events occurs? "
        "Answer with one of: possible, probable, likely, certain.",
        key,
        wrongs,
        1,
        [f"{float(either / 100):.4g} falls in the range for '{key}'."],
    )
    return dict(text=stem, parts=[pa, pb, pc])


@generator("exd_x_r2k_probability_language", level=11, presets={"exam": dict(tier="exam")})
def exd_x_r2k_probability_language(rng: random.Random, tier: str = "exam") -> dict:
    """P(both), P(at least one), then the convention's term for it."""
    check(tier == "exam", "exd_x_r2k_probability_language: only the exam tier exists")
    return retry_build(rng, _build_k_probability_language, "exd_x_r2k_probability_language")
