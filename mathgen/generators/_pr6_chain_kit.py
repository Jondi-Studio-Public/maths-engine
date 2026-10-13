"""Shared pieces for the primary-school exam chains: a whole-number part with tagged distractors, and the
checks every chain makes (no answer is printed in the question, no two answers coincide).

Not a generator module: the leading underscore keeps it out of the generator import loop.
"""

from __future__ import annotations

import re

from mathgen.answers import Distractor
from mathgen.core import GRADER_TOLERANCE, check, latex_answer, pick_distractors

_DIGITS = re.compile(r"\d+")


def whole_part(
    prompt: str,
    value: int,
    marks: int,
    candidates: list[tuple[int, str]],
    *,
    follow_through: str | None = None,
    working: list[str] | None = None,
) -> dict:
    """One number-graded part whose answer is the whole number `value`.

    `candidates` lists `(wrong value, misconception id)` in priority order; the first three that the real
    grader would reject (and that differ from each other) become tagged distractors.
    """
    check(isinstance(value, int) and value > 0, "a part answer must be a positive whole number")
    tagged = [Distractor(str(v), mid, v) for v, mid in candidates if isinstance(v, int) and v > 0]
    picked = pick_distractors(str(value), tagged, 3, tolerance=GRADER_TOLERANCE)
    check(len(picked) >= 2, "a part needs at least two distractors")
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        answer_latex=latex_answer(str(value)),
        marks=marks,
        sensible_range=(0, None),
        working=working or [],
        distractors=picked,
    )
    if follow_through is not None:
        part["follow_through"] = follow_through
    return part


def check_chain(text: str, answers: list[int]) -> None:
    """Fail unless no answer appears as a number in `text` and the answers are all different."""
    shown = {int(n) for n in _DIGITS.findall(text)}
    check(not shown & set(answers), "an answer is printed in the question")
    check(len(set(answers)) == len(answers), "two parts have the same answer")
