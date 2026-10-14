"""Shared helpers for the second-year numerical-methods builds (`r4b_*`).

* Exact arithmetic is done with `Fraction`; a typed decimal answer is a rounded exact value, written at exactly its
  stated number of decimal places.
* `fraction_distractors` and `number_distractors` turn ``(misconception id, value)`` candidates into tagged options
  that the real graders cannot accept for the answer: a fraction is compared by value, a decimal differs from the
  answer by at least one unit in its last stated place.
* `poly_text`, `poly_prompt`, `poly_sympy`: a polynomial given as integer or fraction coefficients, as a student
  reads it (prompt, with superscripts), as a student types it (answer, with ``^``) and as SymPy.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, rational_text, round_half_up, stated_precision_string

X = sympy.Symbol("x")
_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def fr(v) -> Fraction:
    return v if isinstance(v, Fraction) else Fraction(v)


def sup(n: int) -> str:
    return "" if n == 1 else str(n).translate(_SUP)


def sub(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def dec(v, dp: int) -> str:
    """`v` rounded half away from zero to exactly `dp` decimal places, as text."""
    return stated_precision_string(round_half_up(fr(v), dp), dp)


def poly_terms(coeffs):
    """[(degree, coefficient)] from the highest degree down, zero terms dropped. `coeffs[k]` multiplies x^k."""
    return [(k, fr(c)) for k, c in sorted(enumerate(coeffs), reverse=True) if c != 0]


def _term(c: Fraction, k: int, caret: bool) -> tuple[bool, str]:
    mag = abs(c)
    if k == 0:
        return c < 0, rational_text(mag)
    power = "x" if k == 1 else (f"x^{k}" if caret else f"x{sup(k)}")
    if mag == 1:
        return c < 0, power
    if mag.denominator == 1:
        return c < 0, f"{mag.numerator}{power}"
    return c < 0, f"({mag.numerator}/{mag.denominator}){power}"


def _poly(coeffs, caret: bool) -> str:
    items = [_term(c, k, caret) for k, c in poly_terms(coeffs)]
    if not items:
        return "0"
    out = []
    for i, (neg, body) in enumerate(items):
        if i == 0:
            out.append(("−" if not caret else "-") + body if neg else body)
        else:
            out.append(f"{'−' if neg and not caret else '-' if neg else '+'} {body}")
    return " ".join(out)


def poly_prompt(coeffs) -> str:
    """The polynomial as printed in a question: ``x³ − 2x + 5``."""
    return _poly(coeffs, caret=False)


def poly_text(coeffs) -> str:
    """The polynomial as a student types it: ``x^3 - 2x + 5``."""
    return _poly(coeffs, caret=True)


def poly_sympy(coeffs):
    return sum((sympy.Rational(c.numerator, c.denominator) * X**k for k, c in poly_terms(coeffs)), sympy.Integer(0))


def poly_machine(coeffs) -> str:
    return str(poly_sympy(coeffs)).replace("**", "^")


def poly_eval(coeffs, x) -> Fraction:
    x = fr(x)
    out = Fraction(0)
    for c in reversed(coeffs):
        out = out * x + fr(c)
    return out


def poly_differs(left, right) -> bool:
    return any(poly_eval(left, p) != poly_eval(right, p) for p in (Fraction(1, 3), Fraction(7, 5), Fraction(-5, 2)))


def fraction_distractors(answer, cands, n: int = 4, *, floor=None):
    """Tagged options from ``(misconception id, value)`` pairs, the answer and repeats dropped."""
    answer = fr(answer)
    seen = {rational_text(answer)}
    out = []
    for tag, v in cands:
        if v is None:
            continue
        v = fr(v)
        text = rational_text(v)
        if v == answer or text in seen or abs(v) > 10**7:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, exact_value(v)))
        if len(out) == n:
            break
    return out


def number_distractors(answer, dp: int, cands, n: int = 4):
    """Tagged decimal options at exactly `dp` places, each at least one unit of the last place from the answer."""
    a = round_half_up(fr(answer), dp)
    unit = Fraction(1, 10**dp)
    seen = {stated_precision_string(a, dp)}
    out = []
    for tag, v in cands:
        if v is None:
            continue
        v = round_half_up(fr(v), dp)
        if abs(v - a) < unit or abs(v) > 10**7:
            continue
        text = stated_precision_string(v, dp)
        if text in seen:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, float(v)))
        if len(out) == n:
            break
    return out


def integer_distractors(answer: int, cands, n: int = 4):
    seen = {str(answer)}
    out = []
    for tag, v in cands:
        if v is None or int(v) != v or int(v) <= 0 or int(v) == answer or str(int(v)) in seen:
            continue
        seen.add(str(int(v)))
        out.append(Distractor(str(int(v)), tag, int(v)))
        if len(out) == n:
            break
    return out


def margin_ok(value, dp: int, margin: float = 0.1) -> bool:
    """True when rounding `value` to `dp` places is not close to a ".5" boundary."""
    scaled = abs(float(value)) * 10**dp
    return abs(scaled - int(scaled) - 0.5) > margin


class Redraw(Exception):
    """A draw gave too few distractors; the generator draws again."""


def with_redraw(draw, what: str, limit: int = 40):
    for _ in attempts(limit, what):
        try:
            return draw()
        except Redraw:
            continue
    raise AssertionError("unreachable")


def need(options, n: int = 2):
    if len(options) < n:
        raise Redraw
    return options


def fraction_part(prompt, value, cands, marks, *, ft=None, working=(), value_range=None, n=4, min_options=2):
    value = fr(value)
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=rational_text(value),
        marks=marks,
        working=list(working),
        distractors=need(fraction_distractors(value, cands, n), min_options),
    )
    if ft:
        part["follow_through"] = ft
    if value_range:
        part["sensible_range"] = value_range
    return part


def number_part(prompt, value, dp, cands, marks, *, ft=None, working=(), value_range=None, n=4, min_options=2):
    value = fr(value)
    rounded = round_half_up(value, dp)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=float(rounded) if dp else int(rounded),
        answer_text=dec(value, dp),
        marks=marks,
        working=list(working),
        distractors=need(number_distractors(value, dp, cands, n), min_options),
    )
    if ft:
        part["follow_through"] = ft
    if value_range:
        part["sensible_range"] = value_range
    return part


def integer_part(prompt, value, cands, marks, *, ft=None, working=(), value_range=None, n=4, min_options=2):
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(value),
        answer_text=str(int(value)),
        marks=marks,
        working=list(working),
        distractors=need(integer_distractors(int(value), cands, n), min_options),
    )
    if ft:
        part["follow_through"] = ft
    if value_range:
        part["sensible_range"] = value_range
    return part


def as_single(part: dict, text: str, working=None) -> dict:
    """A single-answer question dict from a part dict built by one of the `*_part` helpers."""
    out = dict(
        text=text,
        answer=part["answer"],
        answer_text=part["answer_text"],
        answer_kind=part["answer_kind"],
        distractors=part["distractors"],
        working=list(working if working is not None else part["working"]),
    )
    return out


def check_equal(a, b, msg: str):
    check(sympy.nsimplify(a) == sympy.nsimplify(b), msg)
