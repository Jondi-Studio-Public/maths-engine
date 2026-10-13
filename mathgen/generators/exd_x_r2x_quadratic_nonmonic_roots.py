"""Exam chain for `solve-quadratics-factorable`: a non-monic quadratic that models a path, read through its roots.

The practice tiers solve a quadratic equation with integer roots. The exam question gives the path of a jet of water or
a thrown ball as ``y = -s n x^2 + s (n T + m) x - s m T`` with ``n`` in 2, 4, 5 or 10 and a scale of 1 to 3, so the expression factorises as
``-s (n x - m)(x - T)``. The learner solves it, then uses the roots:

(a) the smaller x at which the height is zero (the root m / n)     2 d.p.,  2 marks
(b) the larger x at which the height is zero (the root T)           2 d.p.,  1 mark
(c) the horizontal distance between the two points                 2 d.p.,  1 mark,  follow-through (a), (b)
(d) the greatest height, at the middle of those points              2 d.p.,  2 marks, follow-through (c)

## Construction

The roots come first (``m / n`` with ``n`` in 2, 4, 5, 10 and a whole number ``T``), the coefficients are built from
them, and ``sympy`` re-solves the expanded equation as an independent route. The greatest height is
``s n (T - m / n)^2 / 4`` (the value at the midpoint of the roots) and is cross-checked against the vertex formula
``c - b^2 / (4 a)``. A draw is kept only when every answer is exact to two decimal places, so nothing is rounded, and
the span is wide enough (at least 1.5) and the height reasonable (at most 16).
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

import sympy

from mathgen.core import attempts, check, generator, phrase
from mathgen.generators._exd_x_r2x_kit import num_part

_SCENES = (
    "A jet of water from a fountain follows the path y = {eq}, where x is the horizontal distance in metres "
    "from a marker post at the edge of the pool and y is the height in metres above the water. The jet is at water "
    "level at two places.",
    "A sprinkler on a school oval sends a jet of water along the path y = {eq}, where x metres is the horizontal "
    "distance from the base of a fence post and y metres is the height above the grass. The jet is at ground level "
    "at two places.",
    "A gardener's hose throws water along the path y = {eq}, where x is the horizontal distance in metres from the "
    "tap and y is the height in metres above the garden bed. The water is at bed level at two places.",
)


def _eq(a: int, b: int, c: int) -> str:
    """``-2x^2 + 11x - 12`` written the way a person would write it."""
    out = [f"{'-' if a < 0 else ''}{abs(a) if abs(a) != 1 else ''}x^2"]
    out.append(f" {'+' if b > 0 else '-'} {abs(b) if abs(b) != 1 else ''}x")
    out.append(f" {'+' if c > 0 else '-'} {abs(c)}")
    return "".join(out)


@generator("exd_x_r2x_quadratic_nonmonic_roots", level=10, presets={"exam": {"tier": "exam"}})
def exd_x_r2x_quadratic_nonmonic_roots(rng, tier: str = "exam"):
    """Solve a non-monic quadratic, then use the roots for a span and a greatest height."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "roots with an exact greatest height"):
        n = rng.choice([2, 4, 5, 10])
        scale = rng.choice([1, 1, 2, 3])
        big = rng.randint(3, 12)
        m = rng.randint(1, n * big - 1)
        if gcd(n, m) != 1:
            continue
        lo = Fraction(m, n)
        span = big - lo
        height = scale * n * span * span / 4
        if span < Fraction(3, 2) or height > 16 or (height * 100).denominator != 1:
            continue
        if lo == big or lo.denominator not in (1, 2, 4, 5, 10, 20, 25, 50, 100):
            continue
        break
    a, b, c = -n * scale, scale * (n * big + m), -scale * m * big
    lead = n * scale
    x = sympy.symbols("x")
    roots = sorted(sympy.solve(sympy.Eq(a * x**2 + b * x + c, 0), x))
    check([Fraction(int(r.p), int(r.q)) for r in roots] == [lo, Fraction(big)], "roots disagree with sympy.solve")
    vertex_height = Fraction(c) - Fraction(b * b, 4 * a)
    check(vertex_height == height, "greatest height disagrees with the vertex formula")

    stem = phrase(rng, _SCENES, eq=_eq(a, b, c))
    scale_text = f"{scale}" if scale > 1 else ""
    hi = Fraction(big)
    mid = (lo + hi) / 2
    parts = [
        num_part(
            "Find the smaller value of x at which the height is zero. Give the answer to 2 decimal places.",
            lo,
            2,
            2,
            [
                f"Factorise: {_eq(a, b, c)} = -{scale_text}({n}x - {m})(x - {big}).",
                f"{n}x - {m} = 0 gives x = {float(lo):g}.",
            ],
            [
                (Fraction(m), "r2x-quad-ignores-leading-coefficient"),
                (Fraction(n, m), "r2x-quad-root-inverted"),
                (hi, "r2x-quad-wrong-root-chosen"),
            ],
        ),
        num_part(
            "Find the larger value of x at which the height is zero. Give the answer to 2 decimal places.",
            hi,
            2,
            1,
            [f"x - {big} = 0 gives x = {big}."],
            [
                (lo, "r2x-quad-wrong-root-chosen"),
                (mid, "r2x-quad-vertex-for-root"),
                (lo + hi, "r2x-quad-sum-of-roots"),
                (hi * n, "r2x-quad-ignores-leading-coefficient"),
            ],
        ),
        num_part(
            "Using your answers to (a) and (b), find the horizontal distance between the two places where the height "
            "is zero. Give the answer to 2 decimal places.",
            span,
            2,
            1,
            [f"{big} - {float(lo):g} = {float(span):g} m."],
            [
                (lo + hi, "r2x-quad-sum-of-roots"),
                (span / 2, "r2x-quad-half-span"),
                (lo * hi, "r2x-quad-product-of-roots"),
                (hi - m, "r2x-quad-numerator-subtracted"),
            ],
            follow="part_b - part_a",
        ),
        num_part(
            "The path is highest halfway between those two places. Using your answer to (c), find the greatest height. "
            "Give the answer to 2 decimal places.",
            height,
            2,
            2,
            [
                f"The path is -{lead}(x - {float(lo):g})(x - {big}), so at the midpoint (a distance {float(span / 2):g} "
                f"from each root) the height is {lead} × {float(span / 2):g} × {float(span / 2):g}.",
                f"Greatest height = {float(height):g}.",
            ],
            [
                (span * span / 4, "r2x-quad-ignores-leading-coefficient"),
                (lead * span / 2, "r2x-quad-height-linear-not-squared"),
                (lead * span * span / 2, "r2x-quad-half-span"),
                (mid, "r2x-quad-vertex-x-for-height"),
            ],
            follow=f"{lead}*part_c**2/4",
        ),
    ]
    return dict(text=stem, parts=parts)
