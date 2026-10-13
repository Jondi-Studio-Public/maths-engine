"""Part builders shared by the `exd_x_r3f_*` multi-part exam generators.

Each builder returns the dict for one part of a multi-part question (see `mathgen.core.Part`), with the answer,
the answer text at its stated precision, and distractors tagged with a misconception id. Distractors are built at
the key's own number of decimal places and are dropped when the real grader would accept them.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    GRADER_TOLERANCE,
    VerificationError,
    latex_answer,
    numeric_answer,
    pick_distractors,
    rational_text,
    round_half_up,
    stated_precision_string,
)

#: How far a displayed value must sit from a rounding boundary, so that a learner who rounds an earlier answer and
#: uses it (or does not) reaches the same last digit.
ROUNDING_MARGIN = Fraction(1, 1000)


def clear_of_boundary(value: Fraction, places: int, margin: Fraction = ROUNDING_MARGIN) -> bool:
    """True when `value` is at least `margin` away from every boundary between two `places`-decimal-place values."""
    scale = 10**places
    offset = (Fraction(value) * scale) % 1
    return abs(offset - Fraction(1, 2)) / scale > margin


def number_text(value: Fraction, places: int) -> str:
    """`value` rounded half up to `places` decimal places, spelled with exactly that many places."""
    rounded = round_half_up(Fraction(value), places)
    if places == 0:
        return str(int(rounded))
    return stated_precision_string(rounded, places)


def number_part(
    prompt: str,
    value,
    candidates,
    *,
    places: int = 0,
    marks: int = 1,
    follow_through: str | None = None,
    working=(),
    sensible_range=None,
    n: int = 3,
) -> dict:
    """A `number` part whose key is `value` to `places` decimal places.

    `candidates` is a list of `(value, misconception_id)`; a value of None is skipped. Each is rounded to the same
    number of places as the key, so the options share the key's form.
    """
    key = round_half_up(Fraction(value), places)
    text = number_text(key, places)
    low = sensible_range[0] if sensible_range else None
    options = []
    for candidate, tag in candidates:
        if candidate is None or (low is not None and candidate <= low):
            continue  # a length, an area or a count is never offered as a negative or zero distractor
        rounded = round_half_up(Fraction(candidate), places)
        options.append(Distractor(number_text(rounded, places), tag, exact_value(rounded)))
    distractors = pick_distractors(text, options, n, tolerance=GRADER_TOLERANCE if places else 0)
    if not distractors:
        raise VerificationError(f"no distractor survives for the answer {text}")
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(key),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=list(working),
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def fraction_part(
    prompt: str,
    value,
    candidates,
    *,
    marks: int = 1,
    follow_through: str | None = None,
    working=(),
    sensible_range=None,
    n: int = 3,
) -> dict:
    """A `fraction` part whose key is the exact rational `value` (graded by value, so 2/4 would match 1/2)."""
    value = Fraction(value)
    text = rational_text(value)
    options = []
    for candidate, tag in candidates:
        if candidate is None:
            continue
        candidate = Fraction(candidate)
        if candidate == value:
            continue
        options.append(Distractor(rational_text(candidate), tag, exact_value(candidate)))
    distractors = pick_distractors(text, options, n)
    if not distractors:
        raise VerificationError(f"no distractor survives for the answer {text}")
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=list(working),
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part
