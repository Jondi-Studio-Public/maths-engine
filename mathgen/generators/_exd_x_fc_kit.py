"""Small shared builder for the multi-part exam chains whose answers are probabilities (``exd_x_fc_*`` modules).

``fraction_part`` returns a part dict for ``parts=[...]`` (see ``docs/design/multi-part-questions.md``). Fractions
are graded by value, so a wrong option is kept only when it is a different number strictly between 0 and 1 and
differs from every other option.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import check, latex_answer, pick_distractors, rational_text


def fraction_part(prompt, value, marks, working, cands, *, follow_through=None, keep=3):
    """A ``fraction`` part whose key is `value` (strictly between 0 and 1).

    `cands` is a list of ``(wrong value, misconception id)`` pairs in priority order.
    """
    value = Fraction(value)
    check(0 < value < 1, "a probability part must lie strictly between 0 and 1")
    key_text = rational_text(value)
    options = [
        Distractor(rational_text(Fraction(w)), mid, exact_value(Fraction(w)))
        for w, mid in cands
        if 0 < Fraction(w) < 1 and Fraction(w) != value
    ]
    picked = pick_distractors(key_text, options, keep)
    check(len(picked) >= 2, "fewer than two usable distractors")
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=key_text,
        answer_latex=latex_answer(key_text),
        marks=marks,
        working=working,
        distractors=picked,
        sensible_range=(0, 1),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part
