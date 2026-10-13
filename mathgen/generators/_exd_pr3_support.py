"""Shared helpers for the multi-part exam sources on primary measurement, geometry and money.

Not a generator module: the package skips names that start with an underscore. It only builds the `parts`
dictionaries the generators return, so every part is assembled the same way: an exact `Fraction` key, a display at a
stated number of decimal places, tagged distractors whose `value` is the exact number the error reaches, and a check
that no distractor can be accepted by the number grader.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    check,
    latex_answer,
    numeric_answer,
    pick_distractors,
    stated_precision_string,
)

_NUMBER = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.]*\d)")


def show(value: Fraction, dp: int) -> str:
    """`value` spelled at exactly `dp` decimal places (a whole number when `dp` is 0)."""
    value = Fraction(value)
    if dp == 0:
        check(value.denominator == 1, f"{value} is not a whole number")
        return str(value.numerator)
    return stated_precision_string(value, dp)


def on_grid(value: Fraction, dp: int) -> bool:
    return (Fraction(value) * 10**dp).denominator == 1


def leaks(text: str, keys) -> bool:
    """True when any key appears as a number in `text`, so the prompt would give the answer away."""
    present = {Fraction(m.group(0)) for m in _NUMBER.finditer(text)}
    return any(Fraction(k) in present for k in keys)


def part(
    prompt: str,
    value,
    *,
    dp: int = 0,
    marks: int = 1,
    working: list[str],
    candidates: list[tuple],
    follow_through: str | None = None,
    sensible_range=(0, None),
) -> dict:
    """One number part. `candidates` is a list of `(value, misconception id or None)` in order of likelihood."""
    value = Fraction(value)
    text = show(value, dp)
    options = []
    for v, tag in candidates:
        v = Fraction(v)
        if v == value:
            continue
        # A distractor is spelled at the key's own places: a different decimal-point presence or number of
        # places gives the key away, so a value that does not sit on the key's grid is not offered.
        if not on_grid(v, dp):
            continue
        options.append(Distractor(show(v, dp), tag, v))
    picked = pick_distractors(text, options, 3, tolerance=GRADER_TOLERANCE if dp else 0)
    check(len(picked) >= 2, "a part needs at least two distractors")
    out = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=picked,
        sensible_range=sensible_range,
    )
    if follow_through is not None:
        out["follow_through"] = follow_through
    return out


def article(n: int) -> str:
    """ "a" or "an" for a number read aloud: "an 8", "an 11", "an 18", "an 80", "a 40"."""
    return "an" if str(n)[0] == "8" or n in (11, 18) else "a"
