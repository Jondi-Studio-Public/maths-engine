"""Small helpers shared by the measurement exam generators (`exd_x_m6_*`).

They build one numeric part of a multi-part question and tag its distractors, so each generator states only
the arithmetic that is particular to it. Nothing here registers a topic.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    VerificationError,
    check,
    latex_answer,
    numeric_answer,
    pick_distractors,
    rounded_decimal_string,
    stated_precision_string,
)


class TooFewDistractors(VerificationError):
    """The draw left fewer than three usable distractors for a part: the generator redraws."""


def shown(value: Fraction, dp: int | None = None) -> str:
    """`value` as a plain decimal string: exactly `dp` places when given, else the shortest form to 2 places."""
    value = Fraction(value)
    return stated_precision_string(value, dp) if dp is not None else rounded_decimal_string(value, 2)


def round_to(value: Fraction, dp: int) -> Fraction:
    """`value` rounded half up to `dp` places (for a non-negative value)."""
    scale = 10**dp
    return Fraction(int(value * scale + Fraction(1, 2)), scale)


def tagged(
    value: Fraction, tag: str | None = None, dp: int | None = None, *, allow_negative: bool = False
) -> Distractor | None:
    """A distractor for `value`, tagged with a catalogue id (or untagged); None when it is not a usable amount.

    A zero, or a negative amount unless `allow_negative`, is not offered: it is not an answer a learner would give
    for a length, area, cost or count.
    """
    value = Fraction(value)
    if value < 0 and not allow_negative:
        return None
    text = shown(round_to(abs(value), dp) * (1 if value >= 0 else -1), dp) if dp is not None else shown(value)
    if Fraction(text) == 0 or (Fraction(text) < 0 and not allow_negative):
        return None
    return Distractor(text, tag, Fraction(text))


def shows_no_key(prompt_texts: list[str], values: list) -> bool:
    """True when no number printed in the prompts equals one of the answers (so the key is never given away)."""
    printed = {Fraction(t) for s in prompt_texts for t in re.findall(r"\d+(?:\.\d+)?", s)}
    return not any(Fraction(v) in printed for v in values)


def number_part(
    prompt: str,
    value: Fraction,
    candidates: list,
    *,
    marks: int,
    working: list[str],
    follow_through: str | None = None,
    dp: int = 0,
) -> dict:
    """A `number` part whose canonical answer is `value` (exactly on the `dp`-place grid)."""
    value = Fraction(value)
    text = stated_precision_string(value, dp)
    picked = pick_distractors(text, [c for c in candidates if c is not None], 3, tolerance=GRADER_TOLERANCE)
    if len(picked) < 3:
        raise TooFewDistractors("fewer than three usable distractors")
    check(all(abs(Fraction(str(d)) - value) > GRADER_TOLERANCE for d in picked), "a distractor is within the grader")
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        sensible_range=(0, None),
        working=working,
        distractors=picked,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part
