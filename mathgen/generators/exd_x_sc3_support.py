"""Small shared helpers for the `exd_x_sc3_*` multi-part exam generators (not a generator itself).

Every part in those questions is a single number. These helpers state an exact value at a fixed number of decimal
places, build a part dictionary, and turn a list of `(value, misconception id)` candidates into tagged distractors
that the number grader cannot accept.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

from ..answers import Distractor
from ..core import (
    GRADER_TOLERANCE,
    VerificationError,
    attempts,
    check,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)


def _on_grid(value: Fraction, places: int) -> tuple[Fraction, int]:
    """`value` as it will be shown. A decimal answer is shown at the `places` the question asks for. A whole-number
    answer (`places` of 0) keeps a whole-number wrong answer as it is and shows any other to one decimal place."""
    value = Fraction(value)
    if places > 0 or round_half_up(value, 0) == value:
        return round_half_up(value, places), places
    return round_half_up(value, 1), 1


def num_text(value: Fraction, places: int) -> str:
    """`value` at exactly `places` decimal places, or one more when it does not sit on that grid."""
    shown, p = _on_grid(value, places)
    return stated_precision_string(shown, p)


def tagged(
    key: Fraction,
    candidates: list[tuple[Fraction, str]],
    places: int,
    *,
    lo=None,
    hi=None,
    n: int = 3,
    minimum: int = 2,
) -> list[Distractor]:
    """Up to `n` tagged distractors from `(value, misconception id)` candidates, in priority order.

    A candidate is dropped when it falls outside the open interval `(lo, hi)` (an impossible length or angle), repeats
    another candidate, or lies within the number grader's tolerance of the key (the grader would accept it).
    """
    key = Fraction(key)
    pool: list[Distractor] = []
    seen_text: set[str] = set()
    for value, tag in candidates:
        value, shown_places = _on_grid(value, places)
        if (lo is not None and value <= lo) or (hi is not None and value >= hi):
            continue
        text = stated_precision_string(value, shown_places)
        if text in seen_text:
            continue
        seen_text.add(text)
        pool.append(Distractor(text, tag, value))
    key_text = num_text(key, places)
    kept = pick_distractors(key_text, pool, n, tolerance=GRADER_TOLERANCE)
    check(len(kept) >= minimum, f"only {len(kept)} usable distractors for {key_text}")
    check(all(abs(d.value - key) > GRADER_TOLERANCE for d in kept), "a distractor is within the grader's tolerance")
    return kept


def number_part(
    prompt: str,
    value: Fraction,
    places: int,
    marks: int,
    working: list[str],
    distractors: list[Distractor],
    *,
    follow_through: str | None = None,
    sensible_range: tuple | None = None,
) -> dict:
    """A `parts` entry whose answer is `value`, stated to `places` decimal places."""
    value = Fraction(value)
    check(round_half_up(value, places) == value, f"{value} is not on the {places}-place grid")
    text = num_text(value, places)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if follow_through is not None:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def redraw(build: Callable, rng, what: str, tries: int = 300):
    """The first `build(rng)` that does not raise `VerificationError`. A draw whose wrong answers collapse onto the key
    (a zero, a repeated value) is dropped and drawn again; the error is raised only if `tries` draws all fail."""
    for _ in attempts(tries, what):
        try:
            return build(rng)
        except VerificationError:
            continue
