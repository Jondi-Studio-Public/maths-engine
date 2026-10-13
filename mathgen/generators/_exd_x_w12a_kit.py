"""Small helpers shared by the ``exd_x_w12a_*`` exam chains (graph-reading and time skills).

Not a generator module (the leading underscore keeps ``mathgen.generators`` from importing it as one).
"""

from __future__ import annotations

import re

from mathgen.answers import Distractor
from mathgen.core import GRADER_TOLERANCE, latex_answer, pick_distractors


def number_part(prompt, key, marks, working, wrong, *, follow=None, want=3):
    """A ``number`` part whose key is a whole number.

    ``wrong`` lists ``(value, misconception id)`` in priority order; values that are not positive whole numbers,
    or that the grader would accept for ``key``, are dropped. Returns None when fewer than ``want`` survive, so
    the caller can draw again.
    """
    text = str(key)
    cands = [Distractor(str(int(v)), tag, int(v)) for v, tag in wrong if v is not None and v > 0 and v == int(v)]
    picked = pick_distractors(text, cands, want, tolerance=GRADER_TOLERANCE)
    if len(picked) < want:
        return None
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=key,
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        sensible_range=(0, None),
        working=working,
        distractors=picked,
    )
    if follow:
        part["follow_through"] = follow
    return part


def mcq_part(prompt, key, marks, working, wrong, *, want=3):
    """An ``mcq`` part. ``wrong`` lists ``(option text, misconception id)``; returns None when fewer than
    ``want`` distinct options other than ``key`` survive."""
    cands = [Distractor(text, tag) for text, tag in wrong]
    picked = pick_distractors(key, cands, want)
    if len(picked) < want:
        return None
    return dict(
        prompt=prompt,
        answer_kind="mcq",
        answer=key,
        answer_text=key,
        answer_latex=latex_answer(key, plain=True),
        marks=marks,
        working=working,
        distractors=picked,
    )


_LABEL = re.compile(r"\(([a-d])\) ")


def split_parts(text: str) -> dict[str, str]:
    """The prompt of each part of a composed multi-part question text, by label ("(a) ... (b) ...")."""
    pieces = _LABEL.split(text)
    return {pieces[i]: pieces[i + 1].strip() for i in range(1, len(pieces) - 1, 2)}
