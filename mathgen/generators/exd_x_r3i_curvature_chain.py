"""Exam-tier source for `ucv1-parametric-curvature`: curvature of a cubic parametric curve at an interior parameter.

The stem gives x(t) and y(t), each a polynomial of degree up to three with no constant term, and a parameter value
t0 that is not zero. The parts build the curvature from its pieces:

(a) the speed sqrt(x'^2 + y'^2) at t0                      number, 1 mark
(b) the cross term x'y'' - y'x'' at t0                     number, 2 marks
(c) the curvature |x'y'' - y'x''| / speed^3, 4 d.p.        number, 2 marks, |part_b| / part_a^3
(d) the radius of curvature, 2 d.p.                        number, 1 mark,  part_a^3 / |part_b|

## Construction

The velocity at t0 is drawn first as a Pythagorean pair (3, 4), (5, 12), (8, 15) or a multiple, in either order and
with either signs, so the speed is a whole number. The linear coefficients r and w are then solved so that
x'(t0) and y'(t0) take those values; the quadratic and cubic coefficients are drawn freely. The cross term is
non-zero (a zero would mean zero curvature and no radius). The follow-through inputs are whole numbers, so a
learner's own (a) and (b) give an exact target for (c) and (d).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, join_terms, paren_negative, phrase, round_half_up
from ._exd_x_r3_distinct import distinct_answers
from .exd_x_r3i_support import number_part

_PAIRS = ((3, 4), (4, 3), (5, 12), (12, 5), (8, 15), (15, 8), (6, 8), (8, 6))


def _near_tie(value: Fraction, dp: int) -> bool:
    """True when `value` sits within a hundredth of a unit of a rounding boundary at `dp` places."""
    scaled = value * 10**dp
    return abs((scaled % 1) - Fraction(1, 2)) < Fraction(1, 50)


@generator("exd_x_r3i_curvature_chain", level=17, presets={"exam": dict(tier="exam")})
@distinct_answers
def exd_x_r3i_curvature_chain(rng, tier: str = "exam") -> dict:
    """Speed, cross term, curvature and radius of curvature of a parametric cubic at t = t0."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a curve with a non-zero cross term"):
        t0 = rng.choice([-3, -2, -1, 1, 2, 3])
        vx, vy = rng.choice(_PAIRS)
        vx *= rng.choice([-1, 1])
        vy *= rng.choice([-1, 1])
        p, s = rng.choice([-2, -1, 0, 1, 2]), rng.choice([-2, -1, 0, 1, 2])
        q, u = rng.randint(-4, 4), rng.randint(-4, 4)
        r = vx - 3 * p * t0**2 - 2 * q * t0
        w = vy - 3 * s * t0**2 - 2 * u * t0
        ax, ay = 6 * p * t0 + 2 * q, 6 * s * t0 + 2 * u
        cross = vx * ay - vy * ax
        if cross == 0 or (p == 0 and s == 0) or abs(r) > 40 or abs(w) > 40:
            continue
        speed = (vx * vx + vy * vy) ** 0.5
        h = round(speed)
        kappa = Fraction(abs(cross), h**3)
        radius = Fraction(h**3, abs(cross))
        if _near_tie(kappa, 4) or _near_tie(radius, 2) or kappa < Fraction(1, 40) or radius > 80:
            continue
        break
    check(h * h == vx * vx + vy * vy, "speed is not a whole number")
    # Independent route: differentiate the polynomials term by term and evaluate.
    check(3 * p * t0**2 + 2 * q * t0 + r == vx and 3 * s * t0**2 + 2 * u * t0 + w == vy, "velocity disagrees")
    check(6 * p * t0 + 2 * q == ax and 6 * s * t0 + 2 * u == ay, "acceleration disagrees")

    x_text = join_terms([(p, "t^3"), (q, "t^2"), (r, "t")])
    y_text = join_terms([(s, "t^3"), (u, "t^2"), (w, "t")])
    stem = phrase(
        rng,
        (
            "A curve is given parametrically by x(t) = {x}, y(t) = {y}. Consider the point P on the curve where t = {t0}.",
            "The path of a point is described by x(t) = {x} and y(t) = {y}. Its position when t = {t0} is the point P.",
            "A plane curve has parametric equations x(t) = {x}, y(t) = {y}. Let P be the point where t = {t0}.",
        ),
        x=x_text,
        y=y_text,
        t0=t0,
    )
    k = Fraction(abs(cross))
    part_a = number_part(
        f"Find the speed of the point on the curve at t = {t0}.",
        Fraction(h),
        0,
        1,
        [
            f"x'({t0}) = {vx} and y'({t0}) = {vy}.",
            f"Speed = sqrt({paren_negative(vx)}^2 + {paren_negative(vy)}^2) = {h}.",
        ],
        [
            ("forgot-square-root", Fraction(h * h)),
            ("speed-adds-components", Fraction(abs(vx) + abs(vy))),
            ("speed-adds-components", Fraction(abs(vx + vy))),
            (
                "speed-from-accelerations",
                Fraction(int((ax * ax + ay * ay) ** 0.5))
                if int((ax * ax + ay * ay) ** 0.5) ** 2 == ax * ax + ay * ay
                else None,
            ),
        ],
        sensible_range=(0, None),
    )
    part_b = number_part(
        f"Find the value of x'(t)y''(t) - y'(t)x''(t) at t = {t0}.",
        Fraction(cross),
        0,
        2,
        [
            f"x''({t0}) = {ax} and y''({t0}) = {ay}.",
            f"x'y'' - y'x'' = {paren_negative(vx)} × {paren_negative(ay)} - {paren_negative(vy)} × {paren_negative(ax)} = {cross}.",
        ],
        [
            ("sign-of-result-reversed", Fraction(-cross)),
            ("dot-and-cross-product-swapped", Fraction(vx * ax + vy * ay)),
            ("sign-slip-in-formula", Fraction(vx * ay + vy * ax)),
            ("cross-term-product-of-first-derivatives", Fraction(vx * vy - ax * ay)),
        ],
    )
    part_c = number_part(
        "Hence find the curvature of the curve at P, correct to 4 decimal places.",
        round_half_up(kappa, 4),
        4,
        2,
        [f"κ = |{cross}| / {h}^3 = {k}/{h**3} = {float(round_half_up(kappa, 4)):.4f}."],
        [
            ("curvature-wrong-power-of-speed", k / h**2),
            ("curvature-wrong-power-of-speed", k / h),
            ("curvature-radius-confused", radius),
            ("forgot-absolute-value", Fraction(cross, h**3) if cross < 0 else None),
        ],
        follow_through="Abs(part_b)/part_a**3",
        sensible_range=(0, None),
    )
    part_d = number_part(
        "Find the radius of curvature at P, correct to 2 decimal places.",
        round_half_up(radius, 2),
        2,
        1,
        [f"R = 1/κ = {h**3}/{k} = {float(round_half_up(radius, 2)):.2f}."],
        [
            ("curvature-radius-confused", kappa),
            ("curvature-wrong-power-of-speed", Fraction(h**2) / k),
            ("curvature-wrong-power-of-speed", Fraction(h) / k),
        ],
        follow_through="part_a**3/Abs(part_b)",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
