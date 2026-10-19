"""Shared builders for the General and Essential number, measure and data skills (`r3j_*`).

* `single_number` builds a one-answer number question; `number_part` builds one number part of a multi-part exam
  chain. Both write the key at a fixed number of decimal places and keep every distractor at that same number of
  places. A slip (a wrong value with the misconception id that produces it) is dropped when it repeats the key,
  repeats an earlier slip, lies within the grader's tolerance of the key, or is accepted by the real grader.
* `fraction_part` and `single_fraction` do the same for an exact rational answer.
* `choice_question` and `choice_part` build a multiple-choice question or part whose options are tagged wrong
  answers; `set_part` builds a part whose answer is a set of outcomes.
* `PEOPLE` and `dollars` give names and money text for a stem.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    check,
    latex_answer,
    numeric_answer,
    rational_text,
    round_half_up,
    stated_precision_string,
)
from ..grading import GRADERS
from ..grading.typed import set_answer_text

PEOPLE = [
    "Mei",
    "Tom",
    "Aisha",
    "Liam",
    "Chloe",
    "Ben",
    "Priya",
    "Jack",
    "Sophie",
    "Daniel",
    "Nina",
    "Hamid",
    "Ruby",
    "Kofi",
    "Zara",
    "Callum",
]

_TOLERANCE = Fraction(GRADER_TOLERANCE).limit_denominator(1000)


def number_text(value, places: int) -> str:
    """`value` rounded half up to `places` places and written at exactly that many places."""
    rounded = round_half_up(Fraction(value), places)
    return stated_precision_string(rounded, places) if places else str(int(rounded))


def dollars(value, places: int | None = None) -> str:
    """A dollar amount for a stem: `$12,500` when whole, `$12,500.40` otherwise (or at `places`)."""
    value = Fraction(value)
    if places is None:
        places = 0 if value.denominator == 1 else 2
    text = number_text(value, places)
    whole, _, cents = text.partition(".")
    sign = "-" if whole.startswith("-") else ""
    whole = f"{int(whole.lstrip('-')):,}"
    return f"{sign}${whole}" + (f".{cents}" if cents else "")


def plain(value) -> str:
    """A number without trailing zeros: 12, 1.5, 0.075."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(int(value))
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def sane(slips, key, factor=12):
    """`slips` without the wrong values that are more than `factor` times bigger, or smaller, than the key."""
    key = abs(Fraction(key))
    if key == 0:
        return list(slips)
    return [(v, t) for v, t in slips if v != 0 and key / factor <= abs(Fraction(v)) <= key * factor]


def _accepted(kind: str, prompt: str, answer, answer_text: str, reply: str) -> bool:
    probe = SimpleNamespace(
        answer=answer, answer_text=answer_text, text=prompt, answer_alt=None, answer_form=None, answer_kind=kind
    )
    return bool(GRADERS[kind](probe, reply).correct)


def _number_options(prompt, value, places, slips, keep, spread=15):
    value = Fraction(value)
    text = number_text(value, places)
    answer = numeric_answer(round_half_up(value, places))
    options: list[Distractor] = []
    seen = {text}
    for slip, tag in sane(slips, value, spread):
        shown = number_text(Fraction(slip), places)
        if shown in seen or abs(Fraction(shown) - Fraction(text)) <= _TOLERANCE:
            continue
        if _accepted("number", prompt, answer, text, shown):
            continue
        seen.add(shown)
        options.append(Distractor(shown, tag, exact_value(Fraction(shown))))
        if len(options) == keep:
            break
    check(len(options) >= 2, f"fewer than two distractors for {prompt!r}: {[s[1] for s in slips]}")
    return text, answer, options


def single_number(text, value, places, working, slips, *, keep=3, spread=15, **extra) -> dict:
    """A one-answer number question dict."""
    shown, answer, options = _number_options(text, value, places, slips, keep, spread)
    return dict(
        text=text,
        answer_kind="number",
        answer=answer,
        answer_text=shown,
        answer_latex=latex_answer(shown),
        working=working,
        distractors=options,
        **extra,
    )


def number_part(
    prompt, value, places, marks, working, slips, *, follow_through=None, sensible_range=None, keep=3, spread=15
):
    """One number part of an exam chain; `follow_through` is a SymPy string over `part_a`, `part_b`, ..."""
    shown, answer, options = _number_options(prompt, value, places, slips, keep, spread)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=answer,
        answer_text=shown,
        answer_latex=latex_answer(shown),
        marks=marks,
        working=working,
        distractors=options,
        sensible_range=sensible_range if sensible_range is not None else (0, None),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _fraction_options(value, slips, keep, what):
    value = Fraction(value)
    text = rational_text(value)
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
    check(len(options) >= 2, f"fewer than two distractors for {what!r}")
    return text, options


def single_fraction(text, value, working, slips, *, keep=3, **extra) -> dict:
    shown, options = _fraction_options(value, slips, keep, text)
    return dict(
        text=text,
        answer_kind="fraction",
        answer=exact_value(Fraction(value)),
        answer_text=shown,
        answer_latex=latex_answer(shown),
        working=working,
        distractors=options,
        **extra,
    )


def fraction_part(prompt, value, marks, working, slips, *, follow_through=None, sensible_range=(0, 1), keep=3):
    shown, options = _fraction_options(value, slips, keep, prompt)
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(Fraction(value)),
        answer_text=shown,
        answer_latex=latex_answer(shown),
        marks=marks,
        working=working,
        distractors=options,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def _choice_options(key, slips, keep, what):
    options: list[Distractor] = []
    seen = {key}
    for text, tag in slips:
        if text in seen:
            continue
        seen.add(text)
        options.append(Distractor(text, tag))
        if len(options) == keep:
            break
    check(len(options) >= 2, f"fewer than two distractors for {what!r}")
    return options


def choice_question(text, key, working, slips, *, keep=3, **extra) -> dict:
    """A multiple-choice question: `key` is the right option's text, `slips` are `(option text, misconception id)`."""
    return dict(
        text=text,
        answer_kind="mcq",
        answer=key,
        answer_text=key,
        answer_latex=latex_answer(key, plain=True),
        working=working,
        distractors=_choice_options(key, slips, keep, text),
        **extra,
    )


def choice_part(prompt, key, marks, working, slips, *, keep=3):
    return dict(
        prompt=prompt,
        answer_kind="mcq",
        answer=key,
        answer_text=key,
        answer_latex=latex_answer(key, plain=True),
        marks=marks,
        working=working,
        distractors=_choice_options(key, slips, keep, prompt),
    )


def _set_options(items, slips, keep, what):
    key = set_answer_text(items)
    probe = SimpleNamespace(answer=[list(i) if isinstance(i, tuple) else i for i in items], answer_text=key)
    check(GRADERS["set"](probe, key).correct, "the set grader rejects its own canonical answer")
    options: list[Distractor] = []
    seen = {key}
    for wrong, tag in slips:
        text = set_answer_text(wrong)
        if text in seen or GRADERS["set"](probe, text).correct:
            continue
        seen.add(text)
        options.append(Distractor(text, tag))
        if len(options) == keep:
            break
    check(len(options) >= 2, f"fewer than two distractors for {what!r}")
    return key, options


def single_set(text, items, working, slips, *, keep=3, **extra) -> dict:
    """A question whose answer is a set of outcomes; an outcome is a string or a tuple (an ordered pair)."""
    key, options = _set_options(items, slips, keep, text)
    return dict(
        text=text,
        answer_kind="set",
        answer=[list(i) if isinstance(i, tuple) else i for i in items],
        answer_text=key,
        working=working,
        distractors=options,
        **extra,
    )


def set_part(prompt, items, marks, working, slips, *, keep=3):
    key, options = _set_options(items, slips, keep, prompt)
    return dict(
        prompt=prompt,
        answer_kind="set",
        answer=[list(i) if isinstance(i, tuple) else i for i in items],
        answer_text=key,
        marks=marks,
        working=working,
        distractors=options,
    )
