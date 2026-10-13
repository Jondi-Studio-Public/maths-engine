"""Small shared helpers for the `exd_x_r2h_*` multi-part exam generators.

Not a generator itself. `build_part` turns one exact value, its worked steps and a list of tagged wrong values into
a part dict for `parts=[...]`: the key is an exact `Fraction`, a distractor is a (value, misconception id) pair, and
the helper writes the answer text, drops any wrong value that the grader would accept as the key, and returns tagged
`Distractor` objects. `expression_part` does the same for a part whose answer is a SymPy expression.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    VerificationError,
    latex_answer,
    numeric_answer,
    rational_text,
    round_half_up,
    stated_precision_string,
)


def value_text(kind: str, value: Fraction, dp: int | None) -> str:
    """The canonical spelling of `value`: a fraction as p/q, a whole number plainly, a decimal at exactly `dp` places."""
    if kind == "fraction":
        return rational_text(value)
    if dp:
        return stated_precision_string(round_half_up(value, dp), dp)
    if value.denominator != 1:
        raise VerificationError("a whole-number part was given a fractional value")
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
) -> dict:
    """One part. `value` is the exact key; `wrong` is a list of (exact wrong value, misconception id) in priority
    order. `kind` is "fraction" (graded exactly) or "number" (graded at its stated precision, `dp` places, or as a
    whole number when `dp` is None)."""
    value = Fraction(value)
    text = value_text(kind, value, dp)
    key_answer = exact_value(value) if kind == "fraction" else numeric_answer(value)
    # A fraction part whose key is a whole number is stored as an int, which the fraction grader reads with the
    # transcript tolerance; a key stored as {"p", "q"} is compared exactly.
    loose = kind == "number" or (kind == "fraction" and value.denominator == 1)
    tolerance = GRADER_TOLERANCE if loose else Fraction(0)
    out: list[Distractor] = []
    seen = {text}
    for wrong_value, tag in wrong:
        if wrong_value is None:
            continue
        wrong_value = Fraction(wrong_value)
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
        raise VerificationError(f"only {len(out)} usable distractors for {prompt!r}")
    part = dict(
        prompt=prompt,
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


def typed(expr) -> str:
    """A SymPy expression as a learner types it: `^` for powers, `*` kept."""
    return sympy.sstr(expr).replace("**", "^")


_SAMPLE_POINTS = (Fraction(37, 100), Fraction(-263, 100))


def _same_function(left, right) -> bool:
    """Whether two expressions agree (to 1e-9) at two generic points: a cheap stand-in for `simplify`, which is
    slow on exponentials and surds. A pair that agrees at both points is treated as equal."""
    symbols = sorted((left - right).free_symbols, key=str)
    for point in _SAMPLE_POINTS:
        sub = {sym: sympy.Rational(point.numerator, point.denominator) + i for i, sym in enumerate(symbols)}
        if abs(sympy.N((left - right).subs(sub), 15)) > 1e-9:
            return False
    return True


def expression_part(
    prompt: str,
    value,
    *,
    marks: int,
    working: list[str],
    wrong: list,
    follow_through: str | None = None,
    need: int = 3,
) -> dict:
    """A part whose answer is the SymPy expression `value`. `wrong` is a list of (wrong expression, misconception
    id); one that SymPy can show equal to the key is dropped."""
    text = typed(value)
    out: list[Distractor] = []
    seen = {text}
    for wrong_value, tag in wrong:
        if wrong_value is None:
            continue
        shown = typed(wrong_value)
        if shown in seen or _same_function(wrong_value, value):
            continue
        seen.add(shown)
        out.append(Distractor(shown, tag, shown))
        if len(out) == 3:
            break
    if len(out) < need:
        raise VerificationError(f"only {len(out)} usable distractors for {prompt!r}")
    part = dict(
        prompt=prompt,
        answer_kind="expression",
        answer=sympy.sstr(value),
        answer_text=text,
        answer_latex=latex_answer(value),
        marks=marks,
        working=working,
        distractors=out,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part
