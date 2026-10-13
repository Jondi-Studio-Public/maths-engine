"""Shared helpers for the `exd_x_r2j_*` exam chains: building number and fraction parts whose distractors are
tagged, unique and rejected by the real grader.

Nothing here registers a generator. Every function takes exact values (ints, `Fraction`s) so an answer is rounded
once, at display, and a distractor is rounded the same way.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    check,
    latex_answer,
    numeric_answer,
    rational_text,
    round_half_up,
    stated_precision_string,
)
from mathgen.grading import typed


def shown(value, places: int) -> Fraction:
    """`value` rounded half up to `places` decimal places, as the displayed answer."""
    return round_half_up(Fraction(value), places)


def text_of(value, places: int) -> str:
    """`value` displayed at exactly `places` decimal places (an integer when `places` is 0)."""
    rounded = shown(value, places)
    return stated_precision_string(rounded, places) if places else str(int(rounded))


def clear_of_boundary(value, places: int, margin: Fraction = Fraction(1, 40)) -> bool:
    """True when `value` is at least `margin` of a unit in the last place away from a rounding boundary."""
    scaled = Fraction(value) * 10**places
    return abs((scaled % 1) - Fraction(1, 2)) > margin if places else True


def _reference(prompt: str, key: Fraction, text: str):
    """A stand-in for a number part, carrying what the number grader reads."""
    return SimpleNamespace(
        answer=numeric_answer(key),
        answer_text=text,
        text=prompt,
        answer_kind="number",
        answer_alt=None,
        answer_form=None,
    )


def number_part(
    *,
    prompt: str,
    exact,
    places: int,
    marks: int,
    working: list[str],
    wrong: list[tuple],
    follow_through: str | None = None,
    sensible_range=None,
    keep: int = 3,
) -> dict:
    """A number part dict. `wrong` lists `(exact wrong value, catalogue id)` in priority order.

    A wrong value that displays as the key, repeats an earlier distractor, or is accepted by the number grader is
    skipped. At least two tagged distractors must survive."""
    key = shown(exact, places)
    text = text_of(exact, places)
    reference = _reference(prompt, key, text)
    distractors, seen = [], {text}
    for value, tag in wrong:
        shown_text = text_of(value, places)
        if shown_text in seen or typed.grade("number", reference, shown_text).correct:
            continue
        seen.add(shown_text)
        distractors.append(Distractor(shown_text, tag, shown(value, places)))
        if len(distractors) == keep:
            break
    check(len(distractors) >= 2, f"fewer than two usable distractors for {prompt!r}")
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
    *,
    prompt: str,
    exact: Fraction,
    marks: int,
    working: list[str],
    wrong: list[tuple],
    follow_through: str | None = None,
    sensible_range=None,
    keep: int = 3,
) -> dict:
    """A fraction part dict (exact rational answer). `wrong` lists `(exact wrong Fraction, catalogue id)`."""
    exact = Fraction(exact)
    text = rational_text(exact)
    reference = SimpleNamespace(
        answer=exact_value(exact),
        answer_text=text,
        text=prompt,
        answer_kind="fraction",
        answer_alt=None,
        answer_form=None,
    )
    distractors, seen = [], {text}
    for value, tag in wrong:
        value = Fraction(value)
        shown_text = rational_text(value)
        if shown_text in seen or typed.grade("fraction", reference, shown_text).correct:
            continue
        seen.add(shown_text)
        distractors.append(Distractor(shown_text, tag, value))
        if len(distractors) == keep:
            break
    check(len(distractors) >= 2, f"fewer than two usable distractors for {prompt!r}")
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(exact),
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


def expression_part(
    *,
    prompt: str,
    key,
    key_text: str,
    marks: int,
    working: list[str],
    wrong: list[tuple],
    follow_through: str | None = None,
    answer_form: str | None = None,
    keep: int = 3,
) -> dict:
    """An `expression` part dict. `key` and each wrong expression are SymPy expressions; `wrong` lists
    `(expression, text, catalogue id)`. A wrong expression is kept only when it differs from the key as a
    function, and its text differs from the key's text."""
    import sympy

    distractors, seen = [], {key_text}
    for expr, text, tag in wrong:
        if text in seen or sympy.expand(expr - key) == 0:
            continue
        seen.add(text)
        distractors.append(Distractor(text, tag, text))
        if len(distractors) == keep:
            break
    check(len(distractors) >= 2, f"fewer than two usable distractors for {prompt!r}")
    part = dict(
        prompt=prompt,
        answer_kind="expression",
        answer=key,
        answer_text=key_text,
        answer_latex=latex_answer(key),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if answer_form:
        part["answer_form"] = answer_form
    return part
