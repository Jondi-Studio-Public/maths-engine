"""Small shared builder for the whole-number exam chains in the ``exd_x_pc1_*`` and ``exd_x_tg1_*`` modules.

Not a generator module (the leading underscore keeps ``mathgen.generators`` from importing it as one).
"""

from __future__ import annotations

from ..answers import Distractor
from ..core import GRADER_TOLERANCE, latex_answer, pick_distractors


def count_part(prompt, key, marks, working, wrong, *, follow=None, want=3, signed=False):
    """A ``number`` part whose key is a positive whole number (any non-zero whole number when ``signed``).

    ``wrong`` lists ``(value, misconception id)`` in priority order. A value that is not a whole number (or not positive, unless ``signed``), or
    that the grader would accept for ``key``, is dropped. Returns None when fewer than ``want`` survive, so the
    caller draws again.
    """
    key = int(key)
    text = str(key)
    cands = [
        Distractor(str(int(v)), tag, int(v)) for v, tag in wrong if v is not None and (v > 0 or signed) and v == int(v)
    ]
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
        working=working,
        distractors=picked,
    )
    if not signed:
        part["sensible_range"] = (0, None)
    if follow:
        part["follow_through"] = follow
    return part
