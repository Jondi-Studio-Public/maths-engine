"""Shared part builders for the `exd_x_r3b_*` multi-part exam generators.

Nothing here registers a topic. Each builder turns an exact value, its stated precision and a list of
``(misconception id, wrong value)`` candidates into one part dictionary (the shape `mathgen.core.Part` takes):
the answer in the grader's own representation, and tagged distractors that are
  * spelled at exactly the key's decimal places (a number part), so no option is told apart by its format,
  * rejected by the real grader for the key, and
  * distinct from each other.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

from ..answers import Distractor, exact_value
from ..core import (
    VerificationError,
    latex_answer,
    numeric_answer,
    rational_text,
    round_half_up,
    stated_precision_string,
)
from ..grading.typed import grade


def number_text(value: Fraction, places: int) -> str:
    """`value` (already rounded to `places`) as a number part states it: exactly `places` decimal places."""
    return stated_precision_string(value, places) if places else str(int(value))


def _stand_in(kind: str, answer, text: str):
    return SimpleNamespace(
        answer=answer,
        answer_text=text,
        answer_kind=kind,
        answer_form=None,
        answer_alt=None,
        answer_latex="",
        distractors=[],
        option_notes={},
        distractor_values={},
        distractor_credit={},
    )


def _finish(kind, key_answer, key_text, offered, keep, what):
    """Keep up to `keep` of `offered` ((text, id, value) triples) that the real grader rejects for the key."""
    stand_in = _stand_in(kind, key_answer, key_text)
    seen, out = {key_text}, []
    for text, mid, value in offered:
        if text in seen or grade(kind, stand_in, text).correct:
            continue
        seen.add(text)
        out.append(Distractor(text, mid, value))
        if len(out) == keep:
            break
    if len(out) < 2:
        raise VerificationError(f"{what}: fewer than two usable distractors")
    return out


def _tail(prompt, kind, answer, text, latex, marks, working, distractors, follow_through, sensible_range):
    part = dict(
        prompt=prompt,
        answer_kind=kind,
        answer=answer,
        answer_text=text,
        answer_latex=latex,
        marks=marks,
        working=list(working),
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def number_part(
    prompt,
    value,
    places,
    marks,
    working,
    candidates,
    *,
    follow_through=None,
    sensible_range=None,
    keep=3,
):
    """A `number` part whose key is `value` rounded half up to `places` decimal places."""
    key = round_half_up(Fraction(value), places)
    text = number_text(key, places)
    offered = []
    for mid, wrong in candidates:
        if wrong is None:
            continue
        rounded = round_half_up(Fraction(wrong), places)
        offered.append((number_text(rounded, places), mid, int(rounded) if rounded.denominator == 1 else rounded))
    distractors = _finish("number", numeric_answer(key), text, offered, keep, prompt)
    return _tail(
        prompt, "number", numeric_answer(key), text, latex_answer(text), marks, working, distractors,
        follow_through, sensible_range,
    )  # fmt: skip


def fraction_part(
    prompt,
    value,
    marks,
    working,
    candidates,
    *,
    follow_through=None,
    sensible_range=None,
    keep=3,
):
    """A `fraction` part whose key is the exact rational `value`, spelled "3", "3/4" or "-1/2"."""
    key = Fraction(value)
    text = rational_text(key)
    offered = []
    for mid, wrong in candidates:
        if wrong is None:
            continue
        w = Fraction(wrong)
        offered.append((rational_text(w), mid, exact_value(w)))
    distractors = _finish("fraction", exact_value(key), text, offered, keep, prompt)
    return _tail(
        prompt, "fraction", exact_value(key), text, latex_answer(text), marks, working, distractors,
        follow_through, sensible_range,
    )  # fmt: skip


def expression_part(
    prompt,
    text,
    latex,
    marks,
    working,
    candidates,
    *,
    follow_through=None,
    keep=3,
):
    """An `expression` part. `text` is the key as typed.

    `candidates` are ``(id, typed text)`` pairs, or ``(id, typed text, differs)`` triples where the generator
    has already proved by exact algebra that the text is a different function from the key (``differs`` true);
    those skip the grader, which parses each text and is the dominant cost of making a question.
    """
    offered, proven = [], set()
    for entry in candidates:
        if entry[1] is None:
            continue
        offered.append((entry[1], entry[0], entry[1]))
        if len(entry) > 2 and entry[2]:
            proven.add(entry[1])
    stand_in = _stand_in("expression", text, text)
    seen, distractors = {text}, []
    for wrong, mid, value in offered:
        if wrong in seen or (wrong not in proven and grade("expression", stand_in, wrong).correct):
            continue
        seen.add(wrong)
        distractors.append(Distractor(wrong, mid, value))
        if len(distractors) == keep:
            break
    if len(distractors) < 2:
        raise VerificationError(f"{prompt}: fewer than two usable distractors")
    return _tail(prompt, "expression", text, text, latex, marks, working, distractors, follow_through, None)
