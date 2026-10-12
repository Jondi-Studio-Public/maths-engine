"""Small shared helpers for the `exd_x_r2g_*` multi-part exam generators.

Not a generator itself. `build_part` turns one exact value, its worked steps and a list of tagged wrong values
into a part dict for `parts=[...]`, so every module states its parts the same way: the key is an exact `Fraction`,
a distractor is a (value, misconception id) pair, and the helper writes the answer text, drops any wrong value that
the grader would accept as the key, and returns tagged `Distractor` objects.
"""

from __future__ import annotations

import functools
import re
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    VerificationError,
    check,
    latex_answer,
    latex_prompt,
    numeric_answer,
    rational_text,
    round_half_up,
    stated_precision_string,
)


class Redraw(VerificationError):
    """A draw left a part with too few distinct wrong answers; `redraws` asks for a fresh draw."""


def redraws(build):
    """Decorate a generator function so that a `Redraw` raised while building it triggers a fresh draw (the same
    random stream carries on, so every seed stays deterministic). Bounded: 200 draws, then `VerificationError`."""

    @functools.wraps(build)
    def run(rng, *args, **kwargs):
        for _ in range(200):
            try:
                return build(rng, *args, **kwargs)
            except Redraw:
                continue
        raise VerificationError(f"{build.__name__}: no draw gave every part enough distinct wrong answers")

    return run


_SET_RE = re.compile(r"(\{[^{}]*\})")
_SHORT_WORDS_RE = re.compile(r"[A-Za-z]{2,}(?: +[A-Za-z]{2,})*")


def latex_with_sets(text: str) -> str:
    """LaTeX for prose that contains set braces such as ``{1, 2, 3}``.

    `latex_prompt` escapes a brace inside prose, which leaves the sentence as one long run of letters that cannot be
    told from hand-rolled text. Here every brace group becomes real maths (``\\{1, 2, 3\\}``) and the prose between
    the groups goes through `latex_prompt` on its own.
    """
    pieces = []
    for piece in _SET_RE.split(text):
        if not piece:
            continue
        if piece.startswith("{"):
            pieces.append("\\{" + piece[1:-1] + "\\}")
        else:
            out = latex_prompt(piece)
            if out == piece:  # no long word, so `latex_prompt` kept it as maths: set short words upright
                out = _SHORT_WORDS_RE.sub(lambda m: "\\text{" + m.group(0) + "}", piece)
            pieces.append(out)
    return "".join(pieces)


def value_text(kind: str, value: Fraction, dp: int | None) -> str:
    """The canonical spelling of `value`: a fraction as p/q, a whole number plainly, a decimal at exactly `dp` places."""
    if kind == "fraction":
        return rational_text(value)
    if dp:
        return stated_precision_string(round_half_up(value, dp), dp)
    check(value.denominator == 1, "a whole-number part was given a fractional value")
    return str(value.numerator)


def build_part(
    prompt: str,
    value,
    *,
    kind: str,
    marks: int,
    working: list[str],
    wrong: list,
    dp: int | None = None,
    follow_through: str | None = None,
    sensible_range: tuple | None = None,
    need: int = 3,
    pad_counts: bool = False,
) -> dict:
    """One part: `value` is the exact key, `wrong` is a list of (exact wrong value, misconception id) in priority
    order. `kind` is "fraction" (graded exactly) or "number" (graded at its stated precision, `dp` places or whole).
    A counting part may set `pad_counts`: when the listed errors leave fewer than three distinct wrong counts, the
    neighbouring counts (one or two out) fill the gap, tagged as off-by-one slips."""
    value = Fraction(value)
    if pad_counts:
        wrong = [*wrong, *((value + k, "off-by-one-count") for k in (1, -1, 2, -2, 3) if value + k >= 0)]
    if kind == "number" and dp:
        value = round_half_up(value, dp)
    text = value_text(kind, value, dp)
    key_answer = exact_value(value) if kind == "fraction" else numeric_answer(value)
    tolerance = GRADER_TOLERANCE if kind == "number" else Fraction(0)
    out: list[Distractor] = []
    seen = {text}
    for wrong_value, tag in wrong:
        if wrong_value is None:
            continue
        wrong_value = Fraction(wrong_value)
        if kind == "number" and dp:
            wrong_value = round_half_up(wrong_value, dp)
        if kind == "number" and not dp and wrong_value.denominator != 1:
            continue
        shown = value_text(kind, wrong_value, dp)
        if shown in seen or abs(wrong_value - value) <= tolerance:
            continue
        seen.add(shown)
        out.append(Distractor(shown, tag, wrong_value))
        if len(out) == 3:
            break
    if len(out) < need:
        raise Redraw(f"only {len(out)} usable distractors for {prompt!r}")
    part = dict(
        prompt=prompt,
        **({"prompt_latex": latex_with_sets(prompt)} if "{" in prompt else {}),
        answer_kind=kind,
        answer=key_answer,
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=out,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part
