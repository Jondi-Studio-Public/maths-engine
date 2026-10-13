"""Helpers shared by the `exd_x_r3c_*` multi-part exam generators: build one part dict from an exact value.

A part carries its own answer text, a tagged distractor for each principled wrong answer, and (for a later part)
the follow-through expression naming the earlier parts it uses. Both builders take the exact value as a
`Fraction`, round it once at display, and drop any wrong answer that the real grader would accept as the key (it
lands on the same displayed value) or that repeats another option.

This module registers no topics.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    GRADER_TOLERANCE,
    check,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)


def _distractors(answer_text, key, wrongs, shown, tolerance, n):
    """Tagged distractors for the wrong values `wrongs` ([(value, catalogue id)]), shown by `shown(value)`.

    A wrong value that cannot be shown in the answer's own format (`shown` returns None) is skipped.
    """
    candidates = []
    for value, tag in wrongs:
        text = shown(Fraction(value))
        if text is not None:
            candidates.append(Distractor(text, tag, Fraction(text)))
    picked = pick_distractors(answer_text, candidates, n, tolerance=tolerance)
    check(len(picked) >= 2, f"fewer than two usable distractors for {answer_text!r}")
    check(all(Fraction(d.text) != key for d in picked), "a distractor equals the answer")
    return picked


def number_part(
    prompt,
    value,
    *,
    dp=0,
    marks=1,
    working,
    wrongs,
    follow_through=None,
    sensible_range=None,
    n=3,
):
    """A `number` part whose answer is `value` rounded half up to `dp` places.

    `dp=0` needs a whole `value`. Every distractor is shown at the key's own number of places, so a whole-number
    answer gets whole-number distractors and a 2 d.p. answer gets 2 d.p. ones.
    """
    value = Fraction(value)
    key = round_half_up(value, dp)
    if dp == 0:
        check(value.denominator == 1, f"a whole-number part got {value}")
    answer_text = stated_precision_string(key, dp) if dp else str(int(key))

    def shown(v):
        if dp == 0 and v.denominator != 1:
            return None  # a whole-number answer is offered only whole-number wrong answers
        v = round_half_up(v, dp)
        return stated_precision_string(v, dp) if dp else str(int(v))

    picked = _distractors(answer_text, key, wrongs, shown, GRADER_TOLERANCE if dp else 0, n)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(key),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        marks=marks,
        working=working,
        distractors=picked,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def fraction_part(prompt, value, *, marks=1, working, wrongs, follow_through=None, sensible_range=None, n=3):
    """A `fraction` part whose answer is the exact rational `value` (a whole number is a fraction over 1)."""
    value = Fraction(value)
    answer_text = str(value)

    def shown(v):
        return str(Fraction(v))

    picked = _distractors(answer_text, value, wrongs, shown, 0, n)
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        marks=marks,
        working=working,
        distractors=picked,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part
