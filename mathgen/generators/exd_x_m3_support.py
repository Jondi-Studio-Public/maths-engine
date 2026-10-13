"""Small helpers shared by the multi-part measurement exam sources (`exd_x_m3_*`).

Every source builds its parts the same way: one exact `Fraction` per answer, rounded once at display, with
tagged distractors drawn from that part's own errors. Keeping that in one place keeps the sources short.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_MARGIN = Fraction(1, 1000)  # keep a displayed value this far (in units of the last place) from a rounding tie


def display(value, dp: int) -> str:
    """`value` rounded half-up and shown at exactly `dp` decimal places."""
    return stated_precision_string(round_half_up(Fraction(value), dp), dp)


def clear_of_tie(value, dp: int) -> bool:
    """True when rounding `value` to `dp` places is unambiguous, so a learner who rounds an earlier answer first
    still lands on the same last digit as the exact route."""
    offset = (Fraction(value) * 10**dp) % 1
    return abs(offset - Fraction(1, 2)) > _MARGIN


def distractor_options(answer_text: str, dp: int, candidates, n: int = 3) -> list[Distractor]:
    """Up to `n` tagged distractors from `(value, misconception_id_or_None)` pairs, in priority order.

    Non-positive values are dropped (every quantity here is a length, area, volume, count or price). A value
    that is not a whole number is shown to at least one decimal place so it never reads as a different integer.
    Anything the grader would accept for the answer is removed by `pick_distractors`.
    """
    options = []
    for value, tag in candidates:
        value = Fraction(value)
        if value <= 0:
            continue
        places = dp if dp > 0 else (0 if value.denominator == 1 else 1)
        shown = round_half_up(value, places)
        if shown <= 0:
            continue
        options.append(Distractor(display(value, places), tag, shown if shown.denominator != 1 else int(shown)))
    return pick_distractors(answer_text, options, n, tolerance=GRADER_TOLERANCE)


def numeric_part(
    prompt: str,
    value,
    dp: int,
    marks: int,
    working: list[str],
    wrong,
    follow_through: str | None = None,
) -> dict:
    """One `number` part whose canonical answer is `value` shown to `dp` places.

    `wrong` is a list of `(value, misconception_id_or_None)` candidates for this part's own distractors.
    """
    shown = round_half_up(Fraction(value), dp)
    text = display(value, dp)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(shown),
        answer_text=text,
        answer_latex=text,
        marks=marks,
        sensible_range=(0, None),
        working=working,
        distractors=distractor_options(text, dp, wrong),
    )
    if follow_through is not None:
        part["follow_through"] = follow_through
    return part


def keys_absent(stem: str, prompts: list[str], keys: list[str]) -> bool:
    """True when no part's answer appears as a number in the stem or in any prompt."""
    present = {float(m) for m in _NUMBER_RE.findall(" ".join([stem, *prompts]))}
    return all(float(key) not in present for key in keys)


def finish(stem: str, parts: list[dict]) -> dict | None:
    """The generator's return dict, or None when a part is unusable (fewer than two distractors, or an answer
    that already appears in the wording) and the caller should redraw."""
    if any(len(p["distractors"]) < 2 for p in parts):
        return None
    if not keys_absent(stem, [p["prompt"] for p in parts], [p["answer_text"] for p in parts]):
        return None
    return dict(text=stem, parts=parts)
