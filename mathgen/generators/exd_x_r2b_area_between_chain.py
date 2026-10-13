"""Generates `exd_x_r2b_area_between_chain` questions: a multi-part exam chain on the area between a parabola and a line.

Skill: definite-integral (exam tier).

A parabola `y = a x^2 + b x + c` and a line `y = m x + n` meet at two points, `x = lo` and `x = hi`, and enclose a region:

(a) the x-coordinate of the left-hand intersection                fraction, 1 mark
(b) the x-coordinate of the right-hand intersection               fraction, 1 mark
(c) the exact area of the region                                  fraction, 3 marks, a definite integral between (a) and (b)
(d) the average vertical distance between the curves              fraction, 1 mark,  area / (hi - lo)

## Construction: choose the intersections first

`a`, `lo`, `hi`, `m` and `n` are drawn, and `b = m - a (lo + hi)`, `c = n + a lo hi`, so that `parabola - line = a (x -
lo)(x - hi)` exactly. The area is `|a| (hi - lo)^3 / 6`, checked by solving for the intersections with the quadratic
formula and integrating the top curve minus the bottom curve in exact fractions.

The follow-through for (c) is the definite integral of `top - bottom` written out between the learner's own (a) and
(b) (the line is on top when `a > 0`), so a wrong limit carries into the area honestly.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

from sympy import Rational

from ..core import generator, join_terms, phrase
from .exd_x_r2b_support import Redraw, draw, fraction_part

_STEMS = (
    "The parabola y = {p} and the line y = {l} intersect at two points and enclose a region R.",
    "Let R be the region enclosed between the parabola y = {p} and the line y = {l}.",
    "A region R is bounded by the curve y = {p} and the line y = {l}.",
)


def _integral(coefficients, lo, hi) -> Fraction:
    """Exact definite integral of `c0 x^2 + c1 x + c2` from lo to hi."""
    c0, c1, c2 = (Fraction(v) for v in coefficients)

    def anti(t):
        return c0 * t**3 / 3 + c1 * t**2 / 2 + c2 * t

    return anti(Fraction(hi)) - anti(Fraction(lo))


def _rational_roots(a, b, c):
    """The rational roots of a x^2 + b x + c (none when the discriminant is not a perfect square)."""
    disc = b * b - 4 * a * c
    if a == 0 or disc < 0 or isqrt(disc) ** 2 != disc:
        return []
    root = isqrt(disc)
    return sorted({Fraction(-b - root, 2 * a), Fraction(-b + root, 2 * a)})


def _build(rng):
    a = rng.choice([1, 2, 3, 4, -1, -2, -3])
    lo = rng.randint(-5, 3)
    hi = lo + rng.randint(2, 6)
    m, n = rng.randint(-4, 4), rng.randint(-5, 5)
    if m == 0 and n == 0:
        raise Redraw("the line is the x-axis")
    b, c = m - a * (lo + hi), n + a * lo * hi
    parabola, line = (a, b, c), (0, m, n)

    # Independent route: solve parabola = line with the quadratic formula, then integrate top minus bottom.
    diff = [p - q for p, q in zip(parabola, line)]
    roots = _rational_roots(*diff)
    if roots != [lo, hi]:
        raise Redraw("the intersections are not the constructed ones")
    top_minus_bottom = [-d for d in diff] if a > 0 else diff
    area = _integral(top_minus_bottom, lo, hi)
    if area != Fraction(abs(a) * (hi - lo) ** 3, 6):
        raise Redraw("area disagrees with |a| (hi - lo)^3 / 6")
    mean = area / (hi - lo)

    stem = phrase(
        rng,
        _STEMS,
        p=join_terms([(a, "x^2"), (b, "x"), (c, "")]),
        l=join_terms([(m, "x"), (n, "")]),
    )

    # (a), (b): the intersections. Wrong ones: the wrong sign, the other intersection, a sign slip on the line.
    slip = _rational_roots(a, b + m, c + n)  # the line's terms added instead of subtracted
    a_wrong = [(-lo, "sign-flip-one-root"), (hi, "wrong-root-chosen")] + [(r, "sign-error-moving-term") for r in slip]
    part_a = fraction_part(
        "Find the x-coordinate of the left-hand point where the parabola and the line meet.",
        lo,
        1,
        [f"{join_terms([(a, 'x^2'), (b - m, 'x'), (c - n, '')])} = 0, so x = {lo} or x = {hi}."],
        a_wrong,
    )
    b_wrong = [(-hi, "sign-flip-one-root"), (lo, "wrong-root-chosen")] + [
        (r, "sign-error-moving-term") for r in reversed(slip)
    ]
    part_b = fraction_part(
        "Find the x-coordinate of the right-hand point where the parabola and the line meet.",
        hi,
        1,
        [f"The intersections are x = {lo} and x = {hi}."],
        b_wrong,
    )

    # (c) the area. The follow-through is the integral of top - bottom written out between the learner's limits.
    t0, t1, t2 = top_minus_bottom
    ft_area = (
        f"Rational({t0.numerator}, {3 * t0.denominator})*(part_b**3 - part_a**3)"
        f" + Rational({t1.numerator}, {2 * t1.denominator})*(part_b**2 - part_a**2)"
        f" + Rational({t2.numerator}, {t2.denominator})*(part_b - part_a)"
    )
    c_wrong = [
        (-area, "sign-reversed-result"),
        (_integral(parabola, lo, hi), "integrated-one-curve-only"),
        (_integral(line, lo, hi), "integrated-one-curve-only"),
        (_integral(top_minus_bottom, 0, hi), "lower-limit-taken-as-zero"),
    ]
    part_c = fraction_part(
        "Find the exact area of R.",
        area,
        3,
        [
            f"Between x = {lo} and x = {hi} the {'line' if a > 0 else 'parabola'} is above the "
            f"{'parabola' if a > 0 else 'line'}.",
            f"Area = integral of ({join_terms([(t0, 'x^2'), (t1, 'x'), (t2, '')])}) from {lo} to {hi} = {area}.",
        ],
        c_wrong,
        follow_through=ft_area,
        sensible_range=(0, None),
    )

    # (d) the average vertical distance.
    d_wrong = [
        (area, "total-not-divided-by-width"),
        (area / hi if hi else None, "divided-by-endpoint-not-width"),
        (area * (hi - lo), "multiplied-instead-of-divided"),
    ]
    part_d = fraction_part(
        "Hence find the average vertical distance between the parabola and the line over the interval between "
        "the two points where they meet.",
        mean,
        1,
        [f"Average = area / (width) = {area} / {hi - lo} = {mean}."],
        [(Rational(v.numerator, v.denominator) if v is not None else None, t) for v, t in d_wrong],
        follow_through="part_c/(part_b - part_a)",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r2b_area_between_chain", level=17, presets={"exam": dict(tier="exam")})
def exd_x_r2b_area_between_chain(rng, tier: str = "exam"):
    """The area between a parabola and a line: both intersections, the area, the average vertical distance."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_build, rng)
