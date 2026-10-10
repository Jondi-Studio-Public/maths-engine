"""Exam-tier source for ``definite-integral``: the area enclosed between a parabola and a line.

The learner is not handed the limits of integration.  They have to see that the region is
bounded by the two intersection points, solve ``parabola = line`` for them, decide which curve
is on top, and only then evaluate a definite integral.  That is a different kind of question
from "evaluate this integral between these limits": it composes equation solving with
integration and leaves the method to the learner.

Built backwards.  The two intersection abscissae ``r1 < r2`` and the leading coefficient ``a`` of
the parabola are drawn first, together with an arbitrary line ``y = m x + n``.  The parabola is then
``y = a x^2 + b x + c`` with ``b = m - a (r1 + r2)`` and ``c = n + a r1 r2``, so that
``parabola - line = a (x - r1)(x - r2)`` exactly.  The enclosed area is ``|a| (r2 - r1)^3 / 6``
(checked independently in exact fractions: the quadratic formula for the limits, then the integral of the difference).

Misconceptions offered as distractors:

- reversing the subtraction (lower minus upper), which gives the negative of the area;
- integrating the parabola alone between the intersection points;
- integrating the line alone between the intersection points.
"""

from __future__ import annotations

import random

from fractions import Fraction
from math import isqrt

from ..core import check, check_distinct, generator, join_terms, phrase, pick_distractors

_PATTERNS = (
    "Find the area of the region enclosed by the parabola y = {parabola} and the line y = {line}.",
    "The curve y = {parabola} and the line y = {line} bound a finite region. Find its area.",
    "Calculate the area enclosed between the line y = {line} and the parabola y = {parabola}.",
)


def _value(value: Fraction):
    """Integer when whole, else the ``{p, q}`` fraction form the number grader reads exactly."""
    return int(value) if value.denominator == 1 else {"p": value.numerator, "q": value.denominator}


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _integral(coefficients, lo, hi) -> Fraction:
    """Exact definite integral of ``c0 x^2 + c1 x + c2`` (coefficients highest power first) from lo to hi."""
    c0, c1, c2 = (Fraction(v) for v in coefficients)

    def anti(t):
        return c0 * t**3 / 3 + c1 * t**2 / 2 + c2 * t

    return anti(Fraction(hi)) - anti(Fraction(lo))


def _exam(rng: random.Random) -> dict:
    while True:
        a = rng.choice([1, 2, 3, -1, -2])
        r1 = rng.randint(-4, 2)
        r2 = r1 + rng.randint(1, 5)
        m = rng.randint(-4, 4)
        n = rng.randint(-5, 5)
        if m == 0 and n == 0:
            continue
        break
    b = m - a * (r1 + r2)
    c = n + a * r1 * r2

    # Independent route, in exact Fractions: solve parabola - line = 0 with the quadratic formula, then integrate
    # (upper - lower) between the roots from the polynomial's coefficients.
    parabola = (a, b, c)
    line = (0, m, n)
    diff = [Fraction(p - q) for p, q in zip(parabola, line)]
    disc = diff[1] ** 2 - 4 * diff[0] * diff[2]
    root = Fraction(isqrt(int(disc)))
    check(root * root == disc and disc > 0, "intersection points are not rational")
    lo, hi = sorted([(-diff[1] - root) / (2 * diff[0]), (-diff[1] + root) / (2 * diff[0])])
    check([lo, hi] == [r1, r2], "intersection points do not match the construction")
    sign = 1 if a > 0 else -1  # which curve is on top between the roots
    area = _integral([sign * -d for d in diff], lo, hi)
    check(area == Fraction(abs(a) * (r2 - r1) ** 3, 6), "area disagrees with |a| (r2 - r1)^3 / 6")
    check(area > 0, "area must be positive")

    parabola_only = _integral(parabola, r1, r2)
    line_only = _integral(line, r1, r2)
    check_distinct(area, -area)
    # A coincidental match with the key (possible for particular numbers) is not offered as a wrong answer.
    candidates = [-area] + [v for v in (parabola_only, line_only) if v != area]

    parabola_text = join_terms([(a, "x^2"), (b, "x"), (c, "")])
    line_text = join_terms([(m, "x"), (n, "")])
    text = phrase(rng, _PATTERNS, parabola=parabola_text, line=line_text)
    answer_text = _fraction_text(area)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=_value(area),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Intersections: x = {r1} and x = {r2}",
            f"Area = integral from {r1} to {r2} of (upper - lower) dx = {answer_text}",
        ],
        distractors=pick_distractors(answer_text, [_fraction_text(v) for v in candidates]),
    )


@generator("exd_x_c2_definite_integral", level=15, presets={"exam": dict(tier="exam")})
def exd_x_c2_definite_integral(rng: random.Random, tier: str = "exam") -> dict:
    """Area between a parabola and a line, limits found by the learner."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
