"""Exam-tier source for `ucv1-double-integral-rectangle`: an integrand that is a sum, then an average value.

The practice tiers integrate a single product g(x) h(y). Here the integrand is a sum of a pure-x term, a cross
term xy, a pure-y term and a constant, so the learner has to split the integral term by term: each single-variable
term picks up the width of the other interval and only the cross term factorises. Part (b) uses the result of
part (a) to find the average value of f over the rectangle (the integral divided by the rectangle's area), with
`follow_through` so a slip in (a) still earns (b).

All arithmetic is exact (`Fraction`); the answers are the exact integrals rounded once to 2 decimal places.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    join_terms,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_X, _Y = sympy.symbols("x y")
_THIRD = Fraction(1, 3)
_HALF = Fraction(1, 2)


def _dp2(value: Fraction) -> Fraction:
    return round_half_up(value, 2)


def _text(value: Fraction) -> str:
    return stated_precision_string(_dp2(value), 2)


def _clear(value: Fraction) -> bool:
    return abs((value * 100) % 1 - _HALF) > Fraction(1, 20)


@generator("exd_x_u3_double_integral_average", level=18, presets={"exam": dict(tier="exam")})
def exd_x_u3_double_integral_average(rng, tier: str = "exam") -> dict:
    """Exam tier: integrate a sum-form f over a rectangle, then find its average value."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a rectangle and integrand with a clear rounding"):
        a, b, c = (rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(3))
        e = rng.randint(-3, 3)
        x0, y0 = rng.randint(-2, 2), rng.randint(-2, 2)
        wx, wy = rng.randint(1, 4), rng.randint(1, 4)
        x1, y1 = x0 + wx, y0 + wy
        x2, x3 = Fraction(x1**2 - x0**2), Fraction(x1**3 - x0**3)
        y2, y3 = Fraction(y1**2 - y0**2), Fraction(y1**3 - y0**3)
        total = a * x3 * _THIRD * wy + b * x2 * _HALF * y2 * _HALF + c * wx * y3 * _THIRD + e * wx * wy
        area = wx * wy
        mean = total / area
        if not (_clear(total) and _clear(mean) and total != 0 and mean != 0):
            continue
        if _dp2(_dp2(total) / area) != _dp2(mean):  # the follow-through route must land on the same value
            continue
        break

    f = a * _X**2 + b * _X * _Y + c * _Y**2 + e
    exact = sympy.integrate(f, (_X, x0, x1), (_Y, y0, y1))
    check(Fraction(int(exact.p), int(exact.q)) == total, "double integral disagrees with SymPy")
    check(Fraction(int(exact.p), int(exact.q)) / area == mean, "mean value arithmetic")

    expr = join_terms([(a, "x^2"), (b, "xy"), (c, "y^2"), (e, "")])
    stem = f"Let f(x, y) = {expr} and let R be the rectangle {x0} ≤ x ≤ {x1}, {y0} ≤ y ≤ {y1}."

    def dec(value: Fraction, tag: str | None = None):
        n = _dp2(value)
        return Distractor(stated_precision_string(n, 2), tag, n)

    upper_only = a * Fraction(x1**3) * _THIRD * y1 + b * Fraction(x1**2) * _HALF * Fraction(y1**2) * _HALF
    upper_only += c * x1 * Fraction(y1**3) * _THIRD + e * x1 * y1
    corner = (a * x1**2 + b * x1 * y1 + c * y1**2 + e) * area
    no_width = a * x3 * _THIRD + b * x2 * _HALF * y2 * _HALF + c * y3 * _THIRD + e * wx * wy
    cross_added = a * x3 * _THIRD * wy + b * (x2 * _HALF + y2 * _HALF) + c * wx * y3 * _THIRD + e * wx * wy
    centre = Fraction(a * (x0 + x1) ** 2, 4) + Fraction(b * (x0 + x1) * (y0 + y1), 4)
    centre += Fraction(c * (y0 + y1) ** 2, 4) + e

    a_cand = [
        dec(upper_only, "lower-limit-forgotten"),
        dec(corner, "corner-value-times-area"),
        dec(no_width, "other-variable-width-dropped"),
        dec(cross_added, "sum-instead-of-product"),
        dec(total - e * area),  # the constant term left out
    ]
    b_cand = [
        dec(total, "average-not-divided-by-area"),
        dec(total / (wx + wy), "average-divided-by-wrong-size"),
        dec(centre),  # the integrand at the centre of R taken as the average
        dec(total / wx),
        dec(total / wy),
        dec(mean - e),
        dec(mean + e),
    ]
    a_text, b_text = _text(total), _text(mean)
    a_d = pick_distractors(a_text, a_cand, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(b_text, b_cand, 3, tolerance=GRADER_TOLERANCE)
    check(len(a_d) >= 2 and len(b_d) >= 2, f"too few distractors {len(a_d)} {len(b_d)}")
    check(a_text not in stem and b_text not in stem, "an answer appears in the stem")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Evaluate the double integral of f over R, correct to 2 decimal places.",
                answer_kind="number",
                answer=numeric_answer(_dp2(total)),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=3,
                working=[
                    "Integrate term by term: a term in x alone is multiplied by the width of the y interval, "
                    "a term in y alone by the width of the x interval, and the xy term factorises.",
                    f"∬ f dA = {a_text} (2 d.p.).",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="Using your answer to (a), find the average value of f over R, correct to 2 decimal places.",
                answer_kind="number",
                answer=numeric_answer(_dp2(mean)),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                follow_through=f"part_a/{area}",
                working=[f"Area of R = {wx} × {wy} = {area}. Average = {a_text} ÷ {area} = {b_text} (2 d.p.)."],
                distractors=b_d,
            ),
        ],
    )
