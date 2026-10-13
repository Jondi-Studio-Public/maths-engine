"""Exam source of `senior-examine-the-area-problem-and-use-sums-of`: a rectangle estimate, the exact area, the error.

The practice tiers evaluate one Riemann sum. An exam question compares an estimate with the truth:

(a) the area estimated by a left, right or midpoint rectangle rule with n equal strips;
(b) the exact area under the curve, by integration;
(c) the absolute error of the estimate, which follows through from (a) and (b) as |a - b|.

All three answers are exact (whole numbers or fractions). The curve is a quadratic or a cubic with whole coefficients that
stays above the axis on the interval; the strip width is chosen so the interval has whole-number ends. The independent
check sums the heights with SymPy's own substitution and integrates with SymPy.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, join_terms, latex_answer, rational_text
from mathgen.answers import exact_value

from .exd_x_r3d_common import tagged
from ._exd_distinct_parts import distinct_part_answers

_SHAPES = (
    (4, Fraction(1, 2)),
    (4, Fraction(1)),
    (4, Fraction(2)),
    (4, Fraction(3, 2)),
    (5, Fraction(1)),
    (5, Fraction(2)),
    (6, Fraction(1, 2)),
    (6, Fraction(1)),
)
_RULES = {
    "left": ("the left-endpoint rectangle rule", "left endpoints"),
    "right": ("the right-endpoint rectangle rule", "right endpoints"),
    "mid": ("the midpoint rectangle rule", "midpoints"),
}
_STEMS = (
    "The curve y = f(x), where f(x) = {f}, lies above the x-axis for {a} ≤ x ≤ {b}. The area under it between "
    "x = {a} and x = {b} is estimated with {n} rectangles of equal width.",
    "Let f(x) = {f}. The area between the graph of y = f(x), the x-axis and the lines x = {a} and x = {b} is "
    "approximated by {n} rectangles of equal width.",
    "A region is bounded by y = {f}, the x-axis, x = {a} and x = {b}. Its area is to be estimated with {n} "
    "rectangles of equal width.",
)
_PROMPT_A = (
    "Estimate the area using {rule}. Give an exact value, as a whole number or a fraction.",
    "Use {rule} to estimate the area. Give the exact value, as a whole number or a fraction.",
)
_PROMPT_B = (
    "Find the exact area under the curve. Give the value as a whole number or a fraction.",
    "Calculate the exact area of the region, as a whole number or a fraction.",
)
_PROMPT_C = (
    "Find the absolute error of your estimate in part (a), as a whole number or a fraction.",
    "By how much does your estimate in part (a) differ from the exact area? Give the size of the difference.",
)


def _poly(rng: random.Random):
    """(text, coefficient list highest degree first) of a positive quadratic or cubic."""
    if rng.random() < 0.3:
        p, q, r = rng.choice([1, 2]), rng.randint(0, 4), rng.randint(1, 6)
        terms = [(p, "x^3"), (q, "x"), (r, "")]
        coeffs = [p, 0, q, r]
    else:
        p, q, r = rng.randint(1, 3), rng.randint(-3, 4), rng.randint(1, 8)
        terms = [(p, "x^2"), (q, "x"), (r, "")]
        coeffs = [p, q, r]
    return join_terms(terms), coeffs


def _value(coeffs, x) -> Fraction:
    out = Fraction(0)
    for c in coeffs:
        out = out * x + c
    return out


def _antiderivative(coeffs, x, *, divide: bool = True) -> Fraction:
    degree = len(coeffs) - 1
    total = Fraction(0)
    for i, c in enumerate(coeffs):
        power = degree - i + 1
        total += Fraction(c) * x**power / (power if divide else 1)
    return total


def _fmt(v) -> str:
    return rational_text(Fraction(v))


@generator("exd_x_r3d_riemann_error_chain", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3d_riemann_error_chain(rng: random.Random, tier: str = "exam") -> dict:
    """A rectangle-rule estimate, the exact area, and the error between them."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(3000, "a curve and interval with usable estimates"):
        text, coeffs = _poly(rng)
        n, width = rng.choice(_SHAPES)
        a = rng.randint(0, 3)
        b = a + n * width
        rule = rng.choice(list(_RULES))
        if b.denominator != 1:
            continue
        lefts = [a + i * width for i in range(n)]
        sums = {
            "left": sum((_value(coeffs, x) for x in lefts), Fraction(0)) * width,
            "right": sum((_value(coeffs, x + width) for x in lefts), Fraction(0)) * width,
            "mid": sum((_value(coeffs, x + width / 2) for x in lefts), Fraction(0)) * width,
        }
        estimate = sums[rule]
        exact = _antiderivative(coeffs, b) - _antiderivative(coeffs, Fraction(a))
        if estimate == exact or any(_value(coeffs, a + k * width / 2) <= 0 for k in range(2 * n + 1)):
            continue
        error = abs(estimate - exact)
        other = sums["right" if rule == "left" else "left"]
        trapezium = (sums["left"] + sums["right"]) / 2
        no_width = estimate / width
        last = {"left": lefts[-1], "right": lefts[-1] + width, "mid": lefts[-1] + width / 2}[rule]
        short = estimate - _value(coeffs, last) * width
        a_cands = [
            (other, "wrong-endpoint-for-stated-rule"),
            (no_width, "strip-width-omitted"),
            (short, "off-by-one-rectangle"),
            (trapezium if rule == "mid" or trapezium != estimate else None, "trapezium-for-rectangle-rule"),
        ]
        b_cands = [
            (_antiderivative(coeffs, b), "forgot-lower-limit"),
            (_value(coeffs, b) - _value(coeffs, Fraction(a)), "differentiated-instead-of-integrated"),
            (
                _antiderivative(coeffs, b, divide=False) - _antiderivative(coeffs, Fraction(a), divide=False),
                "power-rule-integration-divisor-omitted",
            ),
            (estimate, "estimate-reported-as-exact"),
        ]
        c_cands = [
            (error / exact, "relative-error-for-absolute-error"),
            (estimate - exact if estimate < exact else None, "error-sign-kept"),
            (abs(other - exact), "error-of-other-estimate"),
            (abs(trapezium - exact), "error-of-other-estimate"),
        ]
        a_d = tagged(_fmt(estimate), [c for c in a_cands if c[0] is not None], _fmt)
        b_d = tagged(_fmt(exact), [c for c in b_cands if c[0] is not None], _fmt)
        c_d = tagged(_fmt(error), [c for c in c_cands if c[0] is not None], _fmt)
        if len(a_d) >= 3 and len(b_d) >= 3 and len(c_d) >= 2:
            break
    # Independent route: powers rather than Horner's rule for the heights, and Simpson's rule (exact for a cubic) for the area.
    degree = len(coeffs) - 1

    def height(x: Fraction) -> Fraction:
        return sum((Fraction(c) * x ** (degree - i) for i, c in enumerate(coeffs)), Fraction(0))

    offset = {"left": Fraction(0), "right": Fraction(1), "mid": Fraction(1, 2)}[rule]
    check(sum(height(a + (i + offset) * width) for i in range(n)) * width == estimate, "estimate disagrees")
    mid = Fraction(a + b, 2)
    check(
        (b - a) * (height(Fraction(a)) + 4 * height(mid) + height(b)) / 6 == exact, "exact area disagrees with Simpson"
    )

    rule_phrase, endpoints = _RULES[rule]
    stem = rng.choice(_STEMS).format(f=text, a=a, b=int(b), n=n)
    return dict(
        text=stem,
        parts=[
            _part(
                rng.choice(_PROMPT_A).format(rule=rule_phrase),
                estimate,
                3,
                a_d,
                [
                    f"Width {rational_text(width)}; sum the heights at the {endpoints} and multiply by the width: {_fmt(estimate)}."
                ],
                sensible_range=(0, None),
            ),
            _part(
                rng.choice(_PROMPT_B),
                exact,
                2,
                b_d,
                [f"The integral from {a} to {int(b)} is {_fmt(exact)}."],
                sensible_range=(0, None),
            ),
            _part(
                rng.choice(_PROMPT_C),
                error,
                1,
                c_d,
                [f"|{_fmt(estimate)} - {_fmt(exact)}| = {_fmt(error)}."],
                follow_through="Abs(part_a - part_b)",
                sensible_range=(0, None),
            ),
        ],
    )


def _part(prompt, value, marks, distractors, working, *, sensible_range=None, follow_through=None) -> dict:
    value = Fraction(value)
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=_fmt(value),
        answer_latex=latex_answer(_fmt(value)),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    if follow_through:
        part["follow_through"] = follow_through
    return part
