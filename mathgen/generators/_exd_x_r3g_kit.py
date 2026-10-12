"""Shared part builders for the `exd_x_r3g_*` multi-part exam generators.

A part is a dict for `mathgen.core.Part`. `number_part` states its answer at a fixed number of decimal places and
keeps every distractor at that same number of places; `fraction_part` states an exact rational. Both take `slips`:
a list of `(value, misconception id)` pairs in order of likelihood, from which up to three distinct wrong options
are kept. A slip is dropped when it repeats the answer, repeats an earlier slip, or is accepted by the real grader
for the part, so every option a learner sees is marked wrong.
"""

from __future__ import annotations

import re
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

_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:/\d+)?(?![\w.])")


def number_text(value: Fraction, places: int) -> str:
    """`value` rounded half up to `places` places and written at exactly that many places."""
    rounded = round_half_up(Fraction(value), places)
    return stated_precision_string(rounded, places) if places else str(int(rounded))


def _accepted(kind: str, prompt: str, answer, answer_text: str, reply: str) -> bool:
    probe = SimpleNamespace(
        answer=answer, answer_text=answer_text, text=prompt, answer_alt=None, answer_form=None, answer_kind=kind
    )
    return bool(GRADERS[kind](probe, reply).correct)


def _finish(kind, prompt, answer, text, marks, working, options, follow_through, sensible_range):
    check(len(options) >= 2, f"fewer than two distractors for {prompt!r}")
    part = dict(
        prompt=prompt,
        answer_kind=kind,
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


def number_part(prompt, value, places, marks, working, slips, *, follow_through=None, sensible_range=None, keep=3):
    value = Fraction(value)
    text = number_text(value, places)
    answer = numeric_answer(round_half_up(value, places))
    options: list[Distractor] = []
    seen = {text}
    for item in slips:
        if item is None or item[0] is None:
            continue
        slip, tag = item
        shown = number_text(Fraction(slip), places)
        if shown in seen or _accepted("number", prompt, answer, text, shown):
            continue
        seen.add(shown)
        options.append(Distractor(shown, tag, exact_value(Fraction(shown))))
        if len(options) == keep:
            break
    return _finish("number", prompt, answer, text, marks, working, options, follow_through, sensible_range)


def fraction_part(prompt, value, marks, working, slips, *, follow_through=None, sensible_range=None, keep=3):
    value = Fraction(value)
    text = rational_text(value)
    answer = exact_value(value)
    options: list[Distractor] = []
    seen = {text}
    for item in slips:
        if item is None or item[0] is None:
            continue
        slip, tag = item
        slip = Fraction(slip)
        shown = rational_text(slip)
        if shown in seen or slip == value or _accepted("fraction", prompt, answer, text, shown):
            continue
        seen.add(shown)
        options.append(Distractor(shown, tag, exact_value(slip)))
        if len(options) == keep:
            break
    return _finish("fraction", prompt, answer, text, marks, working, options, follow_through, sensible_range)


def clashes(keys, *texts: str) -> bool:
    """True when any key of two or more characters appears as a whole number in the given texts."""
    shown = {m for t in texts for m in _NUMBER_RE.findall(t)}
    return any(len(str(k)) >= 2 and str(k) in shown for k in keys)
