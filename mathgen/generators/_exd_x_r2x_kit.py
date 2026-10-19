"""Part builders shared by the `exd_x_r2x_*` exam chains.

Each helper returns one entry of a multi-part question's `parts` list (see
`docs/design/multi-part-questions.md`). A decimal part is displayed at its stated number of decimal places, and every
distractor is shown at the same number of places as the key, so the options do not give the key away by their format.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)


def frac_text(value: Fraction) -> str:
    """A fraction in lowest terms as ``p/q`` (a whole number is written without a denominator)."""
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def num_part(prompt, exact, places, marks, working, wrong, follow=None, sensible=(0, None)):
    """A typed decimal (or whole number, `places=0`) part.

    `wrong` lists ``(value, misconception id or None)``; each value is rounded to `places` and dropped if it lands
    on the key, on a non-positive value when `sensible` forbids it, or on a repeat.
    """
    shown = round_half_up(Fraction(exact), places)
    text = stated_precision_string(shown, places) if places else str(int(shown))
    cands = []
    for value, tag in wrong:
        rounded = round_half_up(Fraction(value), places)
        if sensible and sensible[0] is not None and rounded <= sensible[0]:
            continue
        dtext = stated_precision_string(rounded, places) if places else str(int(rounded))
        cands.append(Distractor(dtext, tag, exact_value(rounded)) if tag else dtext)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(shown),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        sensible_range=sensible,
        working=working,
        distractors=pick_distractors(text, cands, 3, tolerance=stated_precision_tolerance(places)),
    )
    if follow:
        part["follow_through"] = follow
    return part


def frac_part(prompt, exact, marks, working, wrong, follow=None, sensible=(0, 1)):
    """A fraction part; `wrong` lists ``(Fraction, misconception id or None)``."""
    exact = Fraction(exact)
    text = frac_text(exact)
    cands = []
    for value, tag in wrong:
        value = Fraction(value)
        if sensible and sensible[0] is not None and value <= sensible[0]:
            continue
        if sensible and sensible[1] is not None and value >= sensible[1]:
            continue
        cands.append(Distractor(frac_text(value), tag, exact_value(value)) if tag else frac_text(value))
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(exact),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        sensible_range=sensible,
        working=working,
        distractors=pick_distractors(text, cands, 3),
    )
    if follow:
        part["follow_through"] = follow
    return part
