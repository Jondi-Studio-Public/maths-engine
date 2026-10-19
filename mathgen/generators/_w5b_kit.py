"""Shared builders for the `w5b_*` generators: a single-answer `number` question with tagged distractors.

Nothing here registers a topic. `number_question` turns an exact value, its decimal places and a list of
``(misconception id, wrong value)`` candidates into the dict a tier generator returns. Every distractor is spelled at
exactly the key's decimal places and is rejected by the real grader, as the multi-part builders in
`exd_x_r3b_common` do for a part.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import latex_answer, numeric_answer, round_half_up
from .exd_x_r3b_common import _finish, number_text


def number_question(text, value, places, working, candidates, *, keep=3):
    """A `number` question whose key is `value` rounded half up to `places` places; `candidates` are
    ``(misconception id, wrong value or None)`` pairs in priority order."""
    key = round_half_up(Fraction(value), places)
    key_text = number_text(key, places)
    offered = []
    for mid, wrong in candidates:
        if wrong is None:
            continue
        rounded = round_half_up(Fraction(wrong), places)
        offered.append((number_text(rounded, places), mid, int(rounded) if rounded.denominator == 1 else rounded))
    distractors = _finish("number", numeric_answer(key), key_text, offered, keep, text)
    return dict(
        text=text,
        answer_kind="number",
        answer=numeric_answer(key),
        answer_text=key_text,
        answer_latex=latex_answer(key_text),
        working=list(working),
        distractors=distractors,
    )
