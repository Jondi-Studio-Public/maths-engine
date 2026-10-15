"""Generates `fy1ms_polar_integral` questions and the exam chain `exd_x_fy1ms_polar_integral`.

Skill: `ucv1-polar-double-integral` (multivariable-and-vector-calculus). Double integrals over discs, sectors and rings,
done in polar coordinates, where dA = r dr dtheta.

Tiers (topic `fy1ms_polar_integral`; every answer is a decimal to a stated number of places):

* easy   -- a radial integrand (x^2 + y^2, the distance sqrt(x^2 + y^2), or its fourth power) over a whole disc;
* medium -- a half disc, quarter disc, whole ring or quarter ring, with an integrand that may depend on the angle
            (x, y, xy, x^2, y^2) as well as the radius, so the angular integral is not just the width of the region;
* hard   -- one of: an iterated Cartesian integral over a quarter or half disc, which the learner must recognise and
            convert; the Gaussian e^(-(x^2 + y^2)) over a disc, sector or half disc; the volume under the paraboloid
            z = k - x^2 - y^2 or the cone z = h - sqrt(x^2 + y^2), where the learner must first find the radius.

Exam (topic `exd_x_fy1ms_polar_integral`): a metal plate in the shape of a quarter of a ring. (a) its area (1 mark),
(b) the integral of x over the plate (2 marks), (c) the x-coordinate of the centroid (2 marks, follow-through from
(a) and (b)), (d) the distance from the origin to the centroid, by symmetry (1 mark, follow-through from (c)).

Every key is computed from closed forms (an angular factor times a radial factor) and checked a second way by
Gauss-Legendre quadrature of the Cartesian integrand in polar coordinates (numpy), which shares no formula with the
first route.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from functools import cache


@cache
def _np():
    import numpy

    return numpy


from ..core import attempts, check, generator
from ._fy1ms_common import (
    Fm,
    Redraw,
    build_until,
    clear_of_tie,
    dp_text,
    ft_close,
    num_dist,
    number_part,
    number_question,
    say,
)

# ---------------------------------------------------------------------------
# Regions and integrands

#: Angle sets: (lower angle, upper angle) in units of pi/2, so the quarter plane is (0, 1).
_ANGLES = {"disc": (0, 4), "upper": (0, 2), "right": (-1, 1), "quarter": (0, 1)}

#: The integral over each angle set of cos^i sin^j, as (coefficient, power of pi): value = coefficient * pi**power.
_ANGULAR = {
    (0, 0): {
        "disc": (Fraction(2), 1),
        "upper": (Fraction(1), 1),
        "right": (Fraction(1), 1),
        "quarter": (Fraction(1, 2), 1),
    },
    (1, 0): {"right": (Fraction(2), 0), "quarter": (Fraction(1), 0)},
    (0, 1): {"upper": (Fraction(2), 0), "quarter": (Fraction(1), 0)},
    (1, 1): {"quarter": (Fraction(1, 2), 0)},
    (2, 0): {
        "disc": (Fraction(1), 1),
        "upper": (Fraction(1, 2), 1),
        "right": (Fraction(1, 2), 1),
        "quarter": (Fraction(1, 4), 1),
    },
    (0, 2): {
        "disc": (Fraction(1), 1),
        "upper": (Fraction(1, 2), 1),
        "right": (Fraction(1, 2), 1),
        "quarter": (Fraction(1, 4), 1),
    },
}

#: name -> (plain, tex, power of r in the integrand, (i, j) for cos^i sin^j, Cartesian function)
_INTEGRANDS = {
    "r2": ("(x² + y²)", "(x^2 + y^2)", 2, (0, 0), lambda a, b: a * a + b * b),
    "r": ("√(x² + y²)", r"\sqrt{x^2 + y^2}", 1, (0, 0), lambda a, b: _np().sqrt(a * a + b * b)),
    "r4": ("(x² + y²)²", r"(x^2 + y^2)^2", 4, (0, 0), lambda a, b: (a * a + b * b) ** 2),
    "x": ("x", "x", 1, (1, 0), lambda a, b: a),
    "y": ("y", "y", 1, (0, 1), lambda a, b: b),
    "xy": ("xy", "xy", 2, (1, 1), lambda a, b: a * b),
    "x2": ("x²", "x^2", 2, (2, 0), lambda a, b: a * a),
    "y2": ("y²", "y^2", 2, (0, 2), lambda a, b: b * b),
}


@cache
def _gl():
    return _np().polynomial.legendre.leggauss(24)


def _quadrature(f, angles: tuple[float, float], r_lo: float, r_hi: float) -> float:
    """Gauss-Legendre approximation of the integral of f(x, y) over the polar rectangle, with dA = r dr dtheta."""
    _GL_X, _GL_W = _gl()
    th = 0.5 * (angles[1] - angles[0]) * _GL_X + 0.5 * (angles[1] + angles[0])
    rr = 0.5 * (r_hi - r_lo) * _GL_X + 0.5 * (r_hi + r_lo)
    r_grid, t_grid = _np().meshgrid(rr, th)
    values = f(r_grid * _np().cos(t_grid), r_grid * _np().sin(t_grid)) * r_grid
    weights = _np().outer(_GL_W, _GL_W) * 0.25 * (angles[1] - angles[0]) * (r_hi - r_lo)
    return float((values * weights).sum())


def _closed_form(integrand: str, region: str, r_lo: int, r_hi: int) -> tuple[Fraction, int, Fraction]:
    """(angular coefficient, power of pi, radial integral): the integral is angular * pi**power * radial."""
    _plain, _tex, power, ij, _f = _INTEGRANDS[integrand]
    coefficient, pi_power = _ANGULAR[ij][region]
    radial = Fraction(r_hi ** (power + 2) - r_lo ** (power + 2), power + 2)
    return coefficient, pi_power, radial


def _value(integrand: str, region: str, r_lo: int, r_hi: int, c: int = 1) -> float:
    coefficient, pi_power, radial = _closed_form(integrand, region, r_lo, r_hi)
    value = c * float(coefficient * radial) * math.pi**pi_power
    lo, hi = _ANGLES[region]
    check(
        abs(
            _quadrature(
                lambda a, b: c * _INTEGRANDS[integrand][4](a, b), (lo * math.pi / 2, hi * math.pi / 2), r_lo, r_hi
            )
            - value
        )
        <= 1e-9 * max(1.0, abs(value)),
        "polar integral disagrees with quadrature",
    )
    return value


def _region_words(region: str, r_lo: int, r_hi: int) -> Fm:
    ring = r_lo > 0
    base = f"{r_lo * r_lo} ≤ x² + y² ≤ {r_hi * r_hi}" if ring else f"x² + y² ≤ {r_hi * r_hi}"
    base_tex = base.replace("²", "^2").replace("≤", r"\le ")
    tail = {
        "disc": ("", ""),
        "upper": (" with y ≥ 0", r"\text{ with }y \ge 0"),
        "right": (" with x ≥ 0", r"\text{ with }x \ge 0"),
        "quarter": (" with x ≥ 0 and y ≥ 0", r"\text{ with }x \ge 0\text{ and }y \ge 0"),
    }[region]
    return Fm(base + tail[0], base_tex + tail[1])


def _integral_fm(integrand: str, c: int = 1) -> Fm:
    plain, tex, *_ = _INTEGRANDS[integrand]
    lead = "" if c == 1 else str(c)
    return Fm(f"∬_D {lead}{plain} dA", rf"\iint_D {lead}{tex}\,dA")


def _radial_cands(integrand: str, region: str, r_lo: int, r_hi: int, dp: int, c: int = 1) -> list:
    """Distractors for a polar integral: each is the value a particular slip gives."""
    _plain, _tex, power, ij, _f = _INTEGRANDS[integrand]
    coefficient, pi_power, radial = _closed_form(integrand, region, r_lo, r_hi)
    lo, hi = _ANGLES[region]
    width = Fraction(hi - lo, 2)  # in units of pi
    pi = math.pi
    true = float(coefficient * radial) * pi**pi_power
    no_jacobian_radial = Fraction(r_hi ** (power + 1) - r_lo ** (power + 1), power + 1)
    out = [
        (float(coefficient) * float(no_jacobian_radial) * pi**pi_power, "polar-jacobian-omitted"),
        (float(width) * pi * float(radial), "polar-angular-factor-ignored"),
        (float(radial), "polar-angle-factor-forgotten"),
        (float(coefficient * radial) * pi**pi_power / 2, "polar-angle-halved"),
    ]
    if r_lo > 0:
        out.append(
            (
                float(coefficient) * float(Fraction(r_hi ** (power + 2), power + 2)) * pi**pi_power,
                "polar-inner-radius-ignored",
            )
        )
        out.append(
            (
                float(coefficient) * float(Fraction((r_hi - r_lo) ** (power + 2), power + 2)) * pi**pi_power,
                "polar-ring-difference-before-power",
            )
        )
    area = float(width) * pi * (r_hi * r_hi - r_lo * r_lo) / 2
    out.append((area * r_hi**power, "integrand-boundary-value-times-area"))
    if r_lo == 0:
        out.append(
            (
                float(coefficient) * float(Fraction((r_hi * r_hi) ** (power + 2), power + 2)) * pi**pi_power,
                "polar-radius-limit-squared",
            )
        )
    if ij != (0, 0):
        radial_only = float(width) * pi * float(radial)
        out.append((radial_only, "polar-angular-factor-ignored"))
    true *= c
    return [num_dist(c * v, dp, tag) for v, tag in out if abs(c * v - true) > 1e-9 and abs(c * v) < 30 * abs(true)]


# ---------------------------------------------------------------------------
# Easy and medium


def _easy(rng: random.Random) -> dict:
    integrand = rng.choice(("r2", "r2", "r", "r4"))
    c = rng.choice((1, 1, 2, 3, 4, 5))
    radius = rng.choice((1, 2, 3, 4, 5, 6)) if integrand != "r4" else rng.choice((1, 2, 3))
    for _ in attempts(200, "an easy polar integral away from a rounding tie"):
        value = _value(integrand, "disc", 0, radius, c)
        if clear_of_tie(value, 2):
            break
        radius += 1
    dp = 2
    text, latex = say(
        "Evaluate ",
        _integral_fm(integrand, c),
        ", where D is the disc ",
        _region_words("disc", 0, radius),
        ". Give your answer correct to 2 decimal places.",
    )
    return number_question(
        text,
        latex,
        value,
        dp,
        _radial_cands(integrand, "disc", 0, radius, dp, c),
        working=[
            f"In polar coordinates dA = r dr dθ, with 0 ≤ r ≤ {radius} and 0 ≤ θ ≤ 2π.",
            f"The integral is {dp_text(value, dp)}.",
        ],
    )


def _medium(rng: random.Random) -> dict:
    for _ in attempts(2000, "a medium polar integral"):
        region = rng.choice(("upper", "right", "quarter", "quarter", "disc"))
        ring = rng.random() < 0.4
        integrand = rng.choice(list(_INTEGRANDS))
        c = rng.choice((1, 1, 2, 3))
        if _INTEGRANDS[integrand][3] not in [k for k, v in _ANGULAR.items() if region in v]:
            continue
        if region == "disc" and not ring:
            continue
        r_lo = rng.choice((1, 2, 3)) if ring else 0
        r_hi = r_lo + rng.choice((1, 2, 3)) if ring else rng.choice((2, 3, 4, 5))
        if integrand in ("r4",) and r_hi > 3:
            continue
        value = _value(integrand, region, r_lo, r_hi, c)
        if value > 0 and clear_of_tie(value, 2) and value < 4000:
            break
    dp = 2
    text, latex = say(
        "Evaluate ",
        _integral_fm(integrand, c),
        ", where D is the region ",
        _region_words(region, r_lo, r_hi),
        ". Give your answer correct to 2 decimal places.",
    )
    return number_question(
        text,
        latex,
        value,
        dp,
        _radial_cands(integrand, region, r_lo, r_hi, dp, c),
        working=[
            "In polar coordinates dA = r dr dθ.",
            f"The radial limits are {r_lo} ≤ r ≤ {r_hi}, and the integral is {dp_text(value, dp)}.",
        ],
    )


# ---------------------------------------------------------------------------
# Hard


def _hard_iterated(rng: random.Random) -> dict:
    for _ in attempts(2000, "an iterated polar integral"):
        region = rng.choice(("quarter", "upper"))
        integrand = rng.choice(("r2", "r", "x", "y", "xy", "x2", "y2"))
        if _INTEGRANDS[integrand][3] not in [k for k, v in _ANGULAR.items() if region in v]:
            continue
        radius = rng.choice((1, 2, 3, 4))
        value = _value(integrand, region, 0, radius)
        if value > 0 and clear_of_tie(value, 2):
            break
    dp = 2
    plain_f, tex_f, *_ = _INTEGRANDS[integrand]
    lower = "0" if region == "quarter" else f"-{radius}"
    sq = radius * radius
    wrapped = plain_f
    text, latex = say(
        "Evaluate the iterated integral ",
        Fm(
            f"∫ from x = {lower} to {radius} of ∫ from y = 0 to √({sq} - x²) of {wrapped} dy dx",
            rf"\int_{{{lower}}}^{{{radius}}}\int_0^{{\sqrt{{{sq}-x^2}}}} {tex_f}\,dy\,dx",
        ),
        ". Give your answer correct to 2 decimal places.",
    )
    return number_question(
        text,
        latex,
        value,
        dp,
        _radial_cands(integrand, region, 0, radius, dp),
        working=[
            f"The region is the {'quarter' if region == 'quarter' else 'upper half'} of the disc of radius {radius}: 0 ≤ r ≤ {radius}, "
            f"0 ≤ θ ≤ {'π/2' if region == 'quarter' else 'π'}.",
            f"In polar coordinates the integral is {dp_text(value, dp)}.",
        ],
    )


def _hard_gaussian(rng: random.Random) -> dict:
    for _ in attempts(500, "a Gaussian polar integral"):
        region = rng.choice(("disc", "upper", "quarter", "right"))
        radius2 = rng.choice((1, 2, 3, 4, 5))  # R squared; the radius is its root
        radius = math.sqrt(radius2)
        width = {"disc": 2, "upper": 1, "right": 1, "quarter": 0.5}[region] * math.pi
        value = width * (1 - math.exp(-radius2)) / 2
        if clear_of_tie(value, 3, Fraction(15, 100)):
            break
    dp = 3
    lo, hi = _ANGLES[region]
    check(
        abs(
            _quadrature(lambda a, b: _np().exp(-(a * a + b * b)), (lo * math.pi / 2, hi * math.pi / 2), 0.0, radius)
            - value
        )
        < 1e-9,
        "Gaussian integral disagrees with quadrature",
    )
    pi = math.pi
    cands = [
        num_dist(width * (1 - math.exp(-radius2)), dp, "substitution-factor-half-dropped"),
        num_dist(width * (1 - math.exp(-radius)) / 2, dp, "polar-gaussian-limit-not-squared"),
        num_dist(width * (1 + math.exp(-radius2)) / 2, dp, "polar-gaussian-complement-sign"),
        num_dist(width * (math.exp(-radius2) - 1) / 2, dp, "polar-gaussian-complement-sign"),
        num_dist(width / 2, dp, "polar-gaussian-bound-ignored"),
    ]
    domain = {
        "disc": f"the disc x² + y² ≤ {radius2}",
        "upper": f"the half disc x² + y² ≤ {radius2}, y ≥ 0",
        "right": f"the half disc x² + y² ≤ {radius2}, x ≥ 0",
        "quarter": f"the quarter disc x² + y² ≤ {radius2}, x ≥ 0, y ≥ 0",
    }[region]
    text, latex = say(
        "Evaluate ",
        Fm("∬_D e^(-(x² + y²)) dA", r"\iint_D e^{-(x^2 + y^2)}\,dA"),
        ", where D is ",
        domain,
        ". Give your answer correct to 3 decimal places.",
    )
    return number_question(
        text,
        latex,
        value,
        dp,
        cands,
        working=[
            "In polar coordinates the integrand is e^(-r²) and dA = r dr dθ, so the radial integral is (1 - e^(-"
            + str(radius2)
            + "))/2.",
            f"The integral is {dp_text(value, dp)}.",
        ],
    )


def _hard_volume(rng: random.Random) -> dict:
    kind = rng.choice(("paraboloid", "cone"))
    pi = math.pi
    if kind == "paraboloid":
        k = rng.choice((2, 3, 4, 5, 6, 8, 9))
        value = pi * k * k / 2
        quad = _quadrature(lambda a, b: k - a * a - b * b, (0.0, 2 * pi), 0.0, math.sqrt(k))
        cands = [
            num_dist(4 * pi * k**1.5 / 3, 2, "polar-jacobian-omitted"),
            num_dist(2 * pi * (k**3 / 2 - k**4 / 4), 2, "polar-bound-radius-from-height"),
            num_dist(pi * k * k, 2, "volume-as-base-area-times-peak-height"),
            num_dist(pi * k * k / 4, 2, "polar-angle-halved"),
            num_dist(k * k / 4, 2, "polar-angle-factor-forgotten"),
        ]
        solid = f"z = {k} - x² - y²"
        solid_tex = f"z = {k} - x^2 - y^2"
        working = [
            f"The paraboloid meets the plane z = 0 where r² = {k}, so 0 ≤ r ≤ √{k}.",
            f"V = ∫₀^2π ∫₀^√{k} ({k} - r²) r dr dθ = {dp_text(value, 2)}.",
        ]
    else:
        h = rng.choice((1, 2, 3, 4, 5))
        value = pi * h**3 / 3
        quad = _quadrature(lambda a, b: h - _np().sqrt(a * a + b * b), (0.0, 2 * pi), 0.0, float(h))
        cands = [
            num_dist(pi * h * h, 2, "polar-jacobian-omitted"),
            num_dist(pi * h**3, 2, "volume-as-base-area-times-peak-height"),
            num_dist(pi * h**3 / 6, 2, "polar-angle-halved"),
            num_dist(h**3 / 6, 2, "polar-angle-factor-forgotten"),
        ]
        solid = f"z = {h} - √(x² + y²)"
        solid_tex = f"z = {h} - \\sqrt{{x^2 + y^2}}"
        working = [
            f"The cone meets the plane z = 0 where r = {h}, so 0 ≤ r ≤ {h}.",
            f"V = ∫₀^2π ∫₀^{h} ({h} - r) r dr dθ = {dp_text(value, 2)}.",
        ]
    check(abs(quad - value) < 1e-9 * max(1.0, value), "volume disagrees with quadrature")
    if not clear_of_tie(value, 2):
        raise Redraw("volume sits on a rounding tie")
    text, latex = say(
        "Find the volume of the solid that lies above the xy-plane and below the surface ",
        Fm(solid, solid_tex),
        ". Give your answer correct to 2 decimal places.",
    )
    return number_question(text, latex, value, 2, cands, working=working)


@generator(
    "fy1ms_polar_integral",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1ms_polar_integral(rng: random.Random, tier: str = "easy") -> dict:
    """Polar double integrals: a whole disc (easy), sectors and rings (medium), an iterated integral, a Gaussian or a volume (hard)."""
    if tier == "easy":
        return build_until(rng, _easy, "an easy polar integral")
    if tier == "medium":
        return build_until(rng, _medium, "a medium polar integral")
    if tier == "hard":

        def hard(r: random.Random) -> dict:
            pick = r.random()
            return _hard_iterated(r) if pick < 0.4 else _hard_gaussian(r) if pick < 0.7 else _hard_volume(r)

        return build_until(rng, hard, "a hard polar integral")
    raise AssertionError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Exam chain: the centroid of a quarter ring


@generator("exd_x_fy1ms_polar_integral", level=19, presets={"exam": dict(tier="exam")})
def exd_x_fy1ms_polar_integral(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: area, first moment, centroid, distance from the origin (6 marks)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_until(rng, _exam_chain, "a centroid exam question")


def _exam_chain(rng: random.Random) -> dict:
    r_in = rng.choice((0, 1, 2, 3, 4))
    r_out = r_in + rng.choice((2, 3, 4, 5, 6)) if r_in else rng.choice((3, 4, 5, 6, 7, 8))
    pi = math.pi
    area = pi * (r_out**2 - r_in**2) / 4
    moment = _value("x", "quarter", r_in, r_out)
    xbar = moment / area
    dist = math.sqrt(2) * xbar
    check(abs(xbar - 4 * (r_out**3 - r_in**3) / (3 * pi * (r_out**2 - r_in**2))) < 1e-12, "centroid formula disagrees")
    dp = 2
    for value in (area, moment, xbar, dist):
        if not clear_of_tie(value, dp, Fraction(1, 10)):
            raise Redraw("a value sits on a rounding tie")
    a_text, b_text, c_text = dp_text(area, dp), dp_text(moment, dp), dp_text(xbar, dp)
    ft_close(float(b_text) / float(a_text), c_text, dp)
    ft_close(math.sqrt(2) * float(c_text), dp_text(dist, dp), dp)
    a_cands = [
        num_dist(pi * (r_out - r_in) ** 2 / 4, dp, "polar-ring-difference-before-power"),
        num_dist(pi * (r_out**2 - r_in**2), dp, "polar-quadrant-angle-forgotten"),
        num_dist(pi * r_out**2 / 4, dp, "polar-inner-radius-ignored"),
        num_dist(pi * (r_out**2 - r_in**2) / 2, dp, "polar-angle-halved"),
    ]
    b_cands = [
        num_dist((r_out**2 - r_in**2) / 2, dp, "polar-jacobian-omitted"),
        num_dist(pi / 2 * (r_out**3 - r_in**3) / 3, dp, "polar-angular-factor-ignored"),
        num_dist(r_out**3 / 3, dp, "polar-inner-radius-ignored"),
        num_dist((r_out**3 - r_in**3) / 3 * pi / 4, dp, "polar-substitution-x-as-r-squared"),
    ]
    c_cands = [
        num_dist(moment, dp, "centroid-not-divided-by-area"),
        num_dist(area / moment, dp, "mean-position-ratio-inverted"),
        num_dist(moment / (pi * (r_out**2 - r_in**2)), dp, "polar-quadrant-angle-forgotten"),
        num_dist(moment / (area * pi), dp, "mean-position-divided-by-pi"),
    ]
    d_cands = [
        num_dist(xbar, dp, "centroid-distance-uses-one-coordinate"),
        num_dist(2 * xbar, dp, "centroid-distance-coordinates-added"),
        num_dist(2 * xbar * xbar, dp, "centroid-distance-root-forgotten"),
        num_dist(xbar / math.sqrt(2), dp, "centroid-distance-divided-by-root-two"),
    ]
    sheet = rng.choice(("metal sheet", "sheet of plywood", "sheet of acrylic"))
    shape_word = "ring" if r_in else "disc"
    inequality = (
        Fm(f"{r_in * r_in} ≤ x² + y² ≤ {r_out * r_out}", rf"{r_in * r_in} \le x^2 + y^2 \le {r_out * r_out}")
        if r_in
        else Fm(f"x² + y² ≤ {r_out * r_out}", rf"x^2 + y^2 \le {r_out * r_out}")
    )
    stem, stem_latex = say(
        f"A flat plate of uniform thickness is cut from a {sheet} in the shape of D, the part of the {shape_word} ",
        inequality,
        " that lies in the first quadrant (x ≥ 0 and y ≥ 0). The coordinates are in centimetres.",
    )
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            number_part(
                "Find the area of the plate, in cm², correct to 2 decimal places.",
                area,
                dp,
                a_cands,
                marks=1,
                sensible_range=(0, None),
                working=[f"A = (1/4)π({r_out}² - {r_in}²) = {a_text} cm²."],
            ),
            number_part(
                "Find the value of ∬_D x dA, correct to 2 decimal places.",
                moment,
                dp,
                b_cands,
                marks=2,
                prompt_latex=r"\text{Find the value of }\iint_D x\,dA\text{, correct to 2 decimal places.}",
                sensible_range=(0, None),
                working=[f"∬_D x dA = ∫₀^(π/2) cos θ dθ × ∫_{r_in}^{r_out} r² dr = ({r_out}³ - {r_in}³)/3 = {b_text}."],
            ),
            number_part(
                "Hence find the x-coordinate of the centroid of the plate, correct to 2 decimal places.",
                xbar,
                dp,
                c_cands,
                marks=2,
                follow_through="part_b/part_a",
                sensible_range=(0, None),
                working=[f"x̄ = ∬ x dA / A = {b_text}/{a_text} = {c_text}."],
            ),
            number_part(
                "The plate is symmetric about the line y = x. Hence find the distance from the origin to the centroid, correct to 2 decimal places.",
                dist,
                dp,
                d_cands,
                marks=1,
                follow_through="sqrt(2)*part_c",
                sensible_range=(0, None),
                working=["By symmetry ȳ = x̄, so the distance is √2 x̄ = " + dp_text(dist, dp) + " cm."],
            ),
        ],
    )
