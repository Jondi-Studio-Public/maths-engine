"""Shared builders for the exam-tier chains in the `exd_x_n1_*` modules.

Every part is a number or fraction part whose canonical answer is built from exact `Fraction` arithmetic and
rounded half up once, at display. Wrong answers are offered as `Distractor` objects carrying their exact value, so
a typed reply can be matched by meaning.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    check,
    latex_answer,
    numeric_answer,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)


def tolerance(dp: int) -> Fraction:
    """How close a wrong value may sit to the key before a grader could accept it."""
    return stated_precision_tolerance(dp) if dp > 0 else Fraction(1, 2)


def show(value: Fraction, dp: int) -> str:
    """`value` rounded half up and written at exactly `dp` decimal places."""
    return stated_precision_string(round_half_up(value, dp), dp)


def clear_of_boundary(value: Fraction, dp: int, margin: Fraction = Fraction(1, 10**4)) -> bool:
    """True when `value` is at least `margin` (in units of the last place) away from a rounding boundary."""
    scaled = value * 10**dp
    offset = scaled - (scaled.numerator // scaled.denominator)
    return abs(offset - Fraction(1, 2)) > margin


def money(value: Fraction, dp: int = 2) -> str:
    """A dollar amount with thousands separators, e.g. 1,234.50."""
    text = show(value, dp)
    whole, _, frac = text.partition(".")
    sign = "-" if whole.startswith("-") else ""
    whole = f"{int(whole.lstrip('-')):,}"
    return f"{sign}{whole}.{frac}" if frac else f"{sign}{whole}"


def number_part(
    prompt: str,
    value: Fraction,
    dp: int,
    marks: int,
    working: list[str],
    candidates: list,
    *,
    follow_through: str | None = None,
    sensible_range: tuple | None = None,
) -> dict:
    """One `number` part. `candidates` are `(value, misconception id or None)` pairs for the wrong answers a
    learner would reach; a candidate within the grader's tolerance of the key, or repeating another, is dropped.
    """
    key = round_half_up(value, dp)
    tol = tolerance(dp)
    distractors: list[Distractor] = []
    seen = [key]
    for cand, tag in candidates:
        cand = round_half_up(Fraction(cand), dp)
        if any(abs(cand - s) <= tol for s in seen):
            continue
        seen.append(cand)
        distractors.append(Distractor(show(cand, dp), tag, exact_value(cand)))
        if len(distractors) == 3:
            break
    check(len(distractors) >= 2, f"fewer than two distinct wrong answers for: {prompt}")
    text = stated_precision_string(key, dp)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(key),
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


def fraction_part(
    prompt: str,
    value: Fraction,
    marks: int,
    working: list[str],
    candidates: list,
    *,
    follow_through: str | None = None,
    sensible_range: tuple | None = None,
) -> dict:
    """One `fraction` part, answer written `p/q` in lowest terms."""
    value = Fraction(value)
    distractors: list[Distractor] = []
    seen = [value]
    for cand, tag in candidates:
        cand = Fraction(cand)
        if cand in seen:
            continue
        seen.append(cand)
        distractors.append(Distractor(f"{cand.numerator}/{cand.denominator}", tag, exact_value(cand)))
        if len(distractors) == 3:
            break
    check(len(distractors) >= 2, f"fewer than two distinct wrong answers for: {prompt}")
    text = f"{value.numerator}/{value.denominator}"
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
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
