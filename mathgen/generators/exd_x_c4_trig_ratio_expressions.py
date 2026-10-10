"""Exam source for right-angled-triangle ratios: evaluate an expression built from two ratios.

One ratio of an acute angle is given as a fraction; the learner has to recover the triangle (or work with the
identity sin^2 + cos^2 = 1), find the other ratio, and combine them in a small expression such as
``sin(theta) + cos(theta)`` or ``(sin(theta) - cos(theta))/(sin(theta) + cos(theta))``. That is two ideas
(the right-angled-triangle ratios, then exact fraction arithmetic on a composite) rather than one ratio read off.

Built from a primitive Pythagorean triple, so every ratio is a reduced fraction and the key is exact.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Rational, Symbol, simplify, solve, sqrt

from ..core import check, check_distinct, generator, pick_distractors, rational_text

_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29)]


def _expr_value(kind: str, s: Fraction, c: Fraction) -> Fraction:
    if kind == "sum":
        return s + c
    if kind == "product":
        return s * c
    if kind == "square_diff":
        return c * c - s * s
    return (s - c) / (s + c)


_EXPR_TEXT = {
    "sum": "sin(theta) + cos(theta)",
    "product": "sin(theta) × cos(theta)",
    "square_diff": "cos(theta)^2 - sin(theta)^2",
    "ratio": "(sin(theta) - cos(theta))/(sin(theta) + cos(theta))",
}


def _wrong_values(kind: str, o: int, a: int, h: int) -> list[Fraction]:
    F = Fraction
    if kind == "sum":
        return [F(o + a, 2 * h), F(o * a, h * h), F(1)]  # added tops and bottoms; multiplied; "sin + cos = 1"
    if kind == "product":
        return [F(o * a, h), F(o + a, h), F(h * h, o * a)]  # one h only; added; reciprocal
    if kind == "square_diff":
        return [
            F(a - o, h),
            F(a * a - o * o, h),
            F(o * o - a * a, h * h),
            F(1),
        ]  # no squares; h not squared; reversed; 1
    return [
        F(o - a, h),
        F(o + a, o - a),
        F(a - o, o + a),
        F(o - a, 2 * h),
    ]  # undivided; inverted; reversed; tops/bottoms


def _build(rng: random.Random) -> dict:
    x, y, h = rng.choice(_TRIPLES)
    o, a = (x, y) if rng.random() < 0.5 else (y, x)
    given = rng.choice(["sin", "cos", "tan"])
    kind = rng.choice(list(_EXPR_TEXT))
    s, c = Fraction(o, h), Fraction(a, h)
    given_value = {"sin": s, "cos": c, "tan": Fraction(o, a)}[given]
    answer = _expr_value(kind, s, c)

    # Independent route: recover sin and cos from the given ratio with sympy alone and recompute the expression.
    sn, cs = Symbol("sn", positive=True), Symbol("cs", positive=True)
    gv = Rational(given_value.numerator, given_value.denominator)
    eqs = {
        "sin": [sn - gv, sn**2 + cs**2 - 1],
        "cos": [cs - gv, sn**2 + cs**2 - 1],
        "tan": [sn - gv * cs, sn**2 + cs**2 - 1],
    }
    sol = solve(eqs[given], [sn, cs], dict=True)
    check(len(sol) == 1, "the acute triangle must be unique")
    sv, cv = sol[0][sn], sol[0][cs]
    recomputed = {
        "sum": sv + cv,
        "product": sv * cv,
        "square_diff": cv**2 - sv**2,
        "ratio": (sv - cv) / (sv + cv),
    }[kind]
    check(simplify(recomputed - Rational(answer.numerator, answer.denominator)) == 0, "expression value disagrees")

    wrong = _wrong_values(kind, o, a, h)
    for w in wrong:
        check_distinct(Rational(answer.numerator, answer.denominator), Rational(w.numerator, w.denominator))
    answer_text = rational_text(answer)
    text = (
        f"For an acute angle theta in a right-angled triangle, {given}(theta) = {rational_text(given_value)}. "
        f"Find the exact value of {_EXPR_TEXT[kind]}, as a fraction in lowest terms."
    )
    return {
        "text": text,
        "latex": f"plain:{text}",
        "answer": {"p": answer.numerator, "q": answer.denominator},
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": pick_distractors(answer_text, [rational_text(w) for w in wrong]),
        "working": [
            f"Rebuild the triangle: opposite = {o}, adjacent = {a}, hypotenuse = {h}",
            f"sin(theta) = {rational_text(s)}, cos(theta) = {rational_text(c)}",
            f"{_EXPR_TEXT[kind]} = {answer_text}",
        ],
    }


@generator("exd_x_c4_trig_ratio_expressions", level=17, presets={"exam": dict(tier="exam")})
def exd_x_c4_trig_ratio_expressions(rng: random.Random, tier: str = "exam") -> dict:
    """Given one ratio of an acute angle, evaluate an expression combining sin and cos."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
