"""Shared part builders for the `exd_x_r3j_*` multi-part exam generators.

A part is a dict for `mathgen.core.Part`. `number_part` states its answer at a fixed number of decimal places and
keeps every distractor at that same number of places; `fraction_part` states an exact rational. Both take
`slips`: a list of `(value, misconception id)` pairs in order of likelihood, from which up to three distinct
wrong options are kept. A slip is dropped when it repeats the answer, repeats an earlier slip or is accepted by
the real grader for the part (so the options a learner sees are always marked wrong).
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

from ..answers import Distractor, exact_value
from ..core import (
    check,
    latex_answer,
    numeric_answer,
    rational_text,
    round_half_up,
    stated_precision_string,
)
from ..grading import GRADERS


def number_text(value: Fraction, places: int) -> str:
    """`value` rounded half up to `places` places and written at exactly that many places."""
    rounded = round_half_up(Fraction(value), places)
    return stated_precision_string(rounded, places) if places else str(int(rounded))


def _accepted(kind: str, prompt: str, answer, answer_text: str, reply: str) -> bool:
    probe = SimpleNamespace(
        answer=answer, answer_text=answer_text, text=prompt, answer_alt=None, answer_form=None, answer_kind=kind
    )
    return bool(GRADERS[kind](probe, reply).correct)


def number_part(prompt, value, places, marks, working, slips, *, follow_through=None, sensible_range=None, keep=3):
    value = Fraction(value)
    text = number_text(value, places)
    answer = numeric_answer(round_half_up(value, places))
    options: list[Distractor] = []
    seen = {text}
    for slip, tag in slips:
        shown = number_text(Fraction(slip), places)
        if shown in seen or _accepted("number", prompt, answer, text, shown):
            continue
        seen.add(shown)
        options.append(Distractor(shown, tag, exact_value(Fraction(shown))))
        if len(options) == keep:
            break
    check(len(options) >= 2, f"fewer than two distractors for {prompt!r}")
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=answer,
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=options,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def fraction_part(prompt, value, marks, working, slips, *, follow_through=None, sensible_range=None, keep=3):
    value = Fraction(value)
    text = rational_text(value)
    answer = exact_value(value)
    options: list[Distractor] = []
    seen = {text}
    for slip, tag in slips:
        slip = Fraction(slip)
        shown = rational_text(slip)
        if shown in seen or slip == value:
            continue
        seen.add(shown)
        options.append(Distractor(shown, tag, exact_value(slip)))
        if len(options) == keep:
            break
    check(len(options) >= 2, f"fewer than two distractors for {prompt!r}")
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=answer,
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=options,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part
