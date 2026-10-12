"""Small shared builders for the multi-part exam generators in the `exd_x_sc4_*` modules.

Each builder returns a part dict for ``parts=[...]`` (see ``docs/design/multi-part-questions.md``). A number part
is stated at a fixed number of decimal places, its distractors are tagged ``Distractor`` objects, and every
distractor is kept only when the real grader would reject it.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor
from ..core import (
    GRADER_TOLERANCE,
    VerificationError,
    check,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)


def txt(value: Fraction, dp: int) -> str:
    """`value` rounded half up to `dp` places and displayed at exactly that many places."""
    rounded = round_half_up(Fraction(value), dp)
    return str(int(rounded)) if dp == 0 else stated_precision_string(rounded, dp)


def num_part(prompt, value, dp, marks, working, cands, *, follow_through=None, keep=3, signed=False, key_places=False):
    """A ``number`` part whose key is `value` at `dp` places.

    `cands` is a list of ``(wrong value, misconception id or None)`` pairs in priority order. A candidate is dropped
    when it reads the same as the key at the grader's tolerance, is not positive (unless `signed`), or repeats
    another option. A part is given a ``sensible_range`` of positive values unless `signed`.

    By default a whole-number key may be offered a decimal wrong answer (shown to 1 or 2 places). With `key_places`
    every option is written to the key's own `dp` places instead, so the format cannot give the key away, and a
    nonzero slip that rounds to 0 is dropped.
    """
    value = Fraction(value)
    key_text = txt(value, dp)
    key = round_half_up(value, dp)
    options = []
    for wrong, mid in cands:
        wrong = Fraction(wrong)
        if wrong <= 0 and not signed:
            continue
        if key_places:
            shown_dp = dp
            shown = round_half_up(wrong, dp)
            if shown == 0 and wrong != 0:
                continue  # a nonzero slip that rounds to 0 is not a plausible answer
        else:
            shown_dp = dp if dp > 0 else (0 if wrong.denominator == 1 else (2 if abs(wrong) < 1 else 1))
            shown = round_half_up(wrong, shown_dp)
        if shown <= 0 and not signed:
            continue
        options.append(Distractor(txt(shown, shown_dp), mid, int(shown) if shown.denominator == 1 else shown))
    picked = pick_distractors(key_text, options, keep, tolerance=GRADER_TOLERANCE)
    for d in picked:
        check(abs(Fraction(d.text) - key) > GRADER_TOLERANCE, "a distractor reads as the key")
    check(len(picked) >= 2, "fewer than two usable distractors")
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(key),
        answer_text=key_text,
        answer_latex=latex_answer(key_text),
        marks=marks,
        working=working,
        distractors=picked,
    )
    if not signed:
        part["sensible_range"] = (0, None)
    if follow_through:
        part["follow_through"] = follow_through
    return part


def count_part(prompt, value, marks, working, cands, *, follow_through=None, lo=0, hi=None):
    """A whole-number ``number`` part (a count) with tagged distractors in ``[lo, hi]``."""
    value = Fraction(value)
    check(value.denominator == 1, "a count must be whole")
    options = []
    for wrong, mid in cands:
        wrong = Fraction(wrong)
        if wrong.denominator != 1 or wrong < lo or (hi is not None and wrong > hi):
            continue
        options.append(Distractor(str(int(wrong)), mid, int(wrong)))
    picked = pick_distractors(str(int(value)), options, 3)
    check(len(picked) >= 2, "fewer than two usable distractors")
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(value),
        answer_text=str(int(value)),
        answer_latex=latex_answer(str(int(value))),
        marks=marks,
        working=working,
        distractors=picked,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


__all__ = ["VerificationError", "count_part", "num_part", "txt"]
