"""Shared building blocks for the multi-part exam chains whose parts are all single numbers.

Every chain in the `exd_x_r2i_*` modules builds a part the same way: an exact `Fraction` value, stated at a
fixed number of decimal places, with tagged distractors that are checked against the part's own answer.
Keeping that here means each chain module holds only its own mathematics.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    GRADER_TOLERANCE,
    check,
    latex_answer,
    numeric_answer,
    pick_distractors,
    rational_text,
    round_half_up,
    stated_precision_string,
)

#: How far a displayed answer must sit from a rounding boundary, so a learner who rounds an earlier part and
#: carries it forward lands on the same displayed value as the exact route.
MARGIN = Fraction(1, 200)


def shown(value: Fraction, places: int | None) -> str:
    """`value` rounded half up and written at exactly `places` decimal places (a bare integer for 0). With
    `places=None` the value is exact and written as an integer or a fraction ("7/3")."""
    if places is None:
        return rational_text(Fraction(value))
    rounded = round_half_up(Fraction(value), places)
    return str(int(rounded)) if places == 0 else stated_precision_string(rounded, places)


def clear_of_boundary(value: Fraction, places: int, margin: Fraction = Fraction(1, 20)) -> bool:
    """True when `value` is not within `margin` (as a share of one unit in the last place) of a rounding tie."""
    offset = (Fraction(value) * 10**places) % 1
    return abs(offset - Fraction(1, 2)) > margin


def tagged(value: Fraction, tag: str, places: int) -> Distractor:
    """A distractor for the wrong value `value`, stated at `places` and tagged with a catalogue id."""
    rounded = Fraction(value) if places is None else round_half_up(Fraction(value), places)
    return Distractor(shown(value, places), tag, exact_value(rounded))


def pick(answer_text: str, candidates: list[tuple], places: int, n: int = 3) -> list[Distractor]:
    """Up to `n` distinct tagged distractors from `(value, tag)` candidates, none equal to the answer or within
    the grader's tolerance of it. The candidates are listed in order of how likely the error is. A candidate may
    be `(value, tag, places)` to state that one wrong value at its own precision (a time left in hours is 0.83,
    not 1)."""
    options = [tagged(c[0], c[1], c[2] if len(c) > 2 else places) for c in candidates]
    return pick_distractors(answer_text, options, n, tolerance=GRADER_TOLERANCE)


def number_part(
    prompt: str,
    value: Fraction,
    places: int | None,
    marks: int,
    working: list[str],
    candidates: list[tuple],
    *,
    follow_through: str | None = None,
    sensible_range: tuple | None = None,
    min_distractors: int = 2,
) -> dict:
    """One numeric part: the canonical `value` stated at `places`, and its tagged distractors. `places=None` makes
    an exact `fraction` part (an integer or a ratio of integers), which is what follow-through needs when an
    earlier slip leaves a non-terminating value."""
    text = shown(value, places)
    distractors = pick(text, candidates, places)
    check(len(distractors) >= min_distractors, f"part {prompt[:30]!r}: too few usable distractors")
    if places is None:
        kind, answer = "fraction", exact_value(Fraction(value))
    else:
        kind, answer = "number", numeric_answer(round_half_up(Fraction(value), places))
    part = dict(
        prompt=prompt,
        answer_kind=kind,
        answer=answer,
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part
