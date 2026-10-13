"""Small helpers shared by the multi-part measurement exam sources (`exd_x_m2_*`)."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import GRADER_TOLERANCE, pick_distractors, round_half_up, stated_precision_string

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def display(value: Fraction, dp: int) -> str:
    """`value` rounded half-up and shown at exactly `dp` decimal places."""
    return stated_precision_string(round_half_up(Fraction(value), dp), dp)


def tagged_distractors(answer_text: str, dp: int, candidates, n: int = 3) -> list[Distractor]:
    """Up to `n` tagged distractors from `(value, misconception_or_None)` pairs, in priority order.

    Non-positive values are dropped (every quantity here is a length, area, volume or price), and any candidate
    the grader would accept for the answer is removed by `pick_distractors`.
    """
    options = []
    for value, tag in candidates:
        value = Fraction(value)
        if value <= 0:
            continue
        shown = round_half_up(value, dp)
        options.append(Distractor(display(value, dp), tag, shown if shown.denominator != 1 else int(shown)))
    return pick_distractors(answer_text, options, n, tolerance=GRADER_TOLERANCE)


def keys_absent(stem: str, prompts: list[str], keys: list[str]) -> bool:
    """True when no part's answer appears as a number in the stem or in any prompt."""
    present = {float(m) for m in _NUMBER_RE.findall(" ".join([stem, *prompts]))}
    return all(float(key) not in present for key in keys)
