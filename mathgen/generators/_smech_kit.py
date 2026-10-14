"""Shared helpers for the senior mechanics and residual generators (`smech_*_build.py`).

Every quantity is a `Fraction` until the moment it is shown, so a displayed answer is one rounding of an exact value.
Numbers are shown at a stated number of decimal places, never fewer, because the number grader reads its tolerance
from the places the canonical answer displays (see `core.stated_precision_string`).

`G` is the acceleration due to gravity the whole batch uses, 9.8 m/s². Every prompt that needs it says so.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    exact_decimal_string,
    check,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

G = Fraction(98, 10)
G_NOTE = "Use g = 9.8 m/s²."


def shown(value, dp: int) -> Fraction:
    """`value` rounded half-up to `dp` places."""
    return round_half_up(Fraction(value), dp)


def text_of(value, dp: int) -> str:
    """`value` shown at exactly `dp` decimal places."""
    return stated_precision_string(shown(value, dp), dp)


def is_exact(value, dp: int) -> bool:
    """Whether `value` already sits on the `dp`-place grid, so showing it rounds nothing."""
    return (Fraction(value) * 10**dp).denominator == 1


def clear_of_boundary(value, dp: int, margin=Fraction(1, 20)) -> bool:
    """Whether `value` is at least `margin` of a unit in the last place away from a rounding boundary."""
    scaled = abs(Fraction(value)) * 10**dp
    return abs((scaled % 1) - Fraction(1, 2)) >= margin


def tagged(value, dp: int, misconception: str) -> Distractor:
    """A wrong numeric option: `value` shown at `dp` places and tagged with the error that gives it."""
    return Distractor(text_of(value, dp), misconception, exact_value(shown(value, dp)))


def wrong_options(answer_text: str, wrong, dp: int, *, want: int = 2) -> list[Distractor]:
    """The tagged options for `wrong`, a list of `(value, misconception)`, with every one that the grader would
    accept as the answer (within its stated precision) or that repeats another dropped."""
    tolerance = stated_precision_tolerance(dp) if dp > 0 else 0
    picked = pick_distractors(answer_text, [tagged(v, dp, tag) for v, tag in wrong], tolerance=tolerance)
    check(len(picked) >= want, f"only {len(picked)} usable distractors for {answer_text}")
    return picked


def number_fields(value, dp: int, wrong) -> dict:
    """The answer fields of a single-answer `number` question."""
    text = text_of(value, dp)
    return dict(
        answer=numeric_answer(shown(value, dp)) if dp == 0 else float(shown(value, dp)),
        answer_text=text,
        answer_latex=latex_answer(text),
        answer_kind="number",
        distractors=wrong_options(text, wrong, dp),
    )


def number_part(prompt, value, dp, marks, working, wrong, *, follow=None, sensible=None) -> dict:
    """One `number` part of an exam chain."""
    part = dict(prompt=prompt, marks=marks, working=working, **number_fields(value, dp, wrong))
    if follow:
        part["follow_through"] = follow
    if sensible is not None:
        part["sensible_range"] = sensible
    return part


def dec(value, places: int = 4) -> str:
    """`value` as the shortest exact decimal (at most `places` places) for use inside a prompt or a worked step."""
    return exact_decimal_string(Fraction(value), places)
