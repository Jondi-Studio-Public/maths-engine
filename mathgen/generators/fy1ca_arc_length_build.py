"""Generates `fy1ca_arc_length` and `fy1ca_arc_length_exam` questions.

One first-year skill, ``ucv1-arc-length``: the length of a curve from the formula L = ∫ √(1 + (dy/dx)²) dx (or
∫ √((dx/dt)² + (dy/dt)²) dt for a parametric curve). Every curve is chosen so that 1 + (dy/dx)² is a perfect square, so
the integral is elementary. The key is an exact value (a fraction, or a fraction plus a logarithm) graded as an
expression.

Tiers (the reasoning escalates):

- ``easy``: the square root simplifies by a substitution or a perfect square at once: y = (2/3)x^(3/2), or
  y = (1/3)(x² + 2)^(3/2).
- ``medium``: the integrand has to be recognised as a square, 1 + y'² = (x²/2 + 1/(2x²))², before it can be integrated;
  the integral is of a sum of a power and a negative power.
- ``hard``: a parametric curve whose speed is 2t√(1 + t²), or a curve whose length contains a natural logarithm.
- ``exam`` (topic ``fy1ca_arc_length_exam``): a four-part chain on y = x³/6 + 1/(2x): the slope at the upper end,
  the integrand there, the exact length, and by how much the curve is longer than the straight line between its ends.

Every key is re-derived by a second route: Simpson's rule on the integrand, with the slope found by central differences
of the curve itself, gives the same length.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy
from sympy import Rational, log, sqrt

from ..core import check, generator, latex_answer, phrase
from ._fy1ca_kit import (
    Redraw,
    build_with_redraws,
    clear_of_rounding,
    close,
    expression_distractors,
    fraction_distractors,
    fraction_part,
    frac,
    need_distractors,
    number_distractors,
    number_part,
    simpson,
    typed,
)

x, t = sympy.symbols("x t", positive=True)


def _numeric_length(curve, lo, hi) -> float:
    """Arc length by Simpson's rule with the slope taken by central differences of the curve (no derivative formula)."""
    f = sympy.lambdify(x, curve, "math")
    h = 1e-5

    lo, hi = float(lo), float(hi)

    def speed(v: float) -> float:
        left, right = max(v - h, lo), min(v + h, hi)  # one-sided at the ends, where the curve may not extend further
        slope = (f(right) - f(left)) / (right - left)
        return math.sqrt(1 + slope * slope)

    return simpson(speed, lo, hi, 2400)


def _numeric_parametric(xt, yt, lo, hi) -> float:
    fx, fy = sympy.lambdify(t, xt, "math"), sympy.lambdify(t, yt, "math")
    h = 1e-5

    def speed(v: float) -> float:
        dx = (fx(v + h) - fx(v - h)) / (2 * h)
        dy = (fy(v + h) - fy(v - h)) / (2 * h)
        return math.hypot(dx, dy)

    return simpson(speed, float(lo), float(hi), 2400)


def _verify_integrand(curve, integrand) -> None:
    """1 + y'^2 equals the square of the integrand the working uses."""
    check(sympy.simplify(1 + sympy.diff(curve, x) ** 2 - integrand**2) == 0, "integrand is not sqrt(1 + y'^2)")


def _question(text, answer, wrong, working) -> dict:
    expr = sympy.nsimplify(answer) if not isinstance(answer, sympy.Basic) else answer
    distractors = expression_distractors(expr, _nice(wrong))
    need_distractors(distractors)
    return dict(
        text=text,
        answer=str(expr),
        answer_text=typed(expr),
        answer_kind="expression",
        answer_latex=latex_answer(expr),
        working=working,
        distractors=distractors,
    )


def _nice(candidates):
    """Drop wrong keys a learner would never produce by hand: huge denominators or long nested radicals."""
    kept = []
    for value, tag in candidates:
        if value is None:
            continue
        value = sympy.nsimplify(value) if not isinstance(value, sympy.Basic) else value
        if value.is_Rational and value.q > 400:
            continue
        if not value.is_Rational and len(typed(value)) > 22:
            continue
        kept.append((value, tag))
    return kept


def _floats(candidates):
    """Candidate (value, tag) pairs with each exact value as a float, for a typed-number answer."""
    return [(None if v is None else float(sympy.N(v)), tag) for v, tag in candidates]


def _show(value) -> str:
    """An exact value as written in a worked step (ln, no asterisks)."""
    text = typed(sympy.nsimplify(value) if not isinstance(value, sympy.Basic) else value)
    return text.replace("*", " × ")


def _common_wrong(curve, lo, hi, antiderivative, integrand):
    """Candidate wrong keys shared by the function families."""
    y = sympy.Lambda(x, curve)
    chord = sqrt((hi - lo) ** 2 + (y(hi) - y(lo)) ** 2)
    return [
        (sympy.integrate(1 + sympy.diff(curve, x) ** 2, (x, lo, hi)), "fy1ca-arc-root-dropped"),
        (antiderivative(hi), "fy1ca-arc-lower-limit-ignored"),
        (chord, "fy1ca-arc-chord-used"),
        (y(hi) - y(lo), "fy1ca-arc-height-change-used"),
        ((antiderivative(hi) - antiderivative(lo)) * 2, "fy1ca-arc-wrong-antiderivative"),
    ]


# -- easy ------------------------------------------------------------------------------------------------------------

_ROOT_PAIRS = ((0, 3), (0, 8), (3, 8), (3, 15), (8, 15), (8, 24), (0, 15), (0, 24), (15, 24), (3, 24))


def _three_halves(rng: random.Random) -> dict:
    lo, hi = rng.choice(_ROOT_PAIRS)
    curve = Rational(2, 3) * x ** Rational(3, 2)
    integrand = sqrt(1 + x)
    _verify_integrand(curve, integrand)
    G = lambda v: Rational(2, 3) * (1 + v) ** Rational(3, 2)  # noqa: E731
    answer = G(hi) - G(lo)
    check(close(_numeric_length(curve, lo, hi), float(answer), 1e-6), "numerical length disagrees")
    wrong = _common_wrong(curve, lo, hi, G, integrand)
    text = phrase(
        rng,
        (
            "Find the exact length of the curve y = (2/3)x^(3/2) from x = {lo} to x = {hi}.",
            "A cable hangs along the curve y = (2/3)x^(3/2) between the posts at x = {lo} and x = {hi} (units in metres). Find the exact length of the cable.",
        ),
        lo=lo,
        hi=hi,
    )
    working = [
        "dy/dx = x^(1/2), so 1 + (dy/dx)² = 1 + x and the length is the integral of √(1 + x).",
        f"L = [(2/3)(1 + x)^(3/2)] from {lo} to {hi} = (2/3)({1 + hi})^(3/2) - (2/3)({1 + lo})^(3/2).",
        f"L = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _offset_cube(rng: random.Random) -> dict:
    lo = rng.choice([0, 0, 1])
    hi = rng.choice([2, 3, 4]) + lo
    curve = Rational(1, 3) * (x**2 + 2) ** Rational(3, 2)
    integrand = x**2 + 1
    _verify_integrand(curve, integrand)
    G = lambda v: Rational(v) ** 3 / 3 + v  # noqa: E731
    answer = G(hi) - G(lo)
    check(close(_numeric_length(curve, lo, hi), float(answer), 1e-6), "numerical length disagrees")
    wrong = _common_wrong(curve, lo, hi, G, integrand)
    text = phrase(
        rng,
        (
            "Find the exact length of the curve y = (1/3)(x² + 2)^(3/2) from x = {lo} to x = {hi}.",
            "A walking track follows the curve y = (1/3)(x² + 2)^(3/2) for {lo} ≤ x ≤ {hi} (units in kilometres). Find its exact length.",
        ),
        lo=lo,
        hi=hi,
    )
    working = [
        "dy/dx = x(x² + 2)^(1/2), so 1 + (dy/dx)² = 1 + x²(x² + 2) = (x² + 1)².",
        f"L = the integral of (x² + 1) from {lo} to {hi} = [x³/3 + x] = {_show(G(hi))} - {_show(G(lo))}.",
        f"L = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _easy(rng: random.Random) -> dict:
    return rng.choice([_three_halves, _offset_cube])(rng)


# -- medium --------------------------------------------------------------------------------------------------------------


def _cubic_reciprocal(rng: random.Random) -> dict:
    lo, hi = rng.choice([(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4)])
    curve = x**3 / 6 + 1 / (2 * x)
    integrand = x**2 / 2 + 1 / (2 * x**2)
    _verify_integrand(curve, integrand)
    G = lambda v: Rational(v) ** 3 / 6 - 1 / (2 * Rational(v))  # noqa: E731
    answer = G(hi) - G(lo)
    check(close(_numeric_length(curve, lo, hi), float(answer), 1e-6), "numerical length disagrees")
    wrong = _common_wrong(curve, lo, hi, G, integrand)
    wrong.append((sympy.integrate(x**2 / 2 - 1 / (2 * x**2), (x, lo, hi)), "fy1ca-arc-cross-term-sign"))
    text = phrase(
        rng,
        (
            "Find the exact length of the curve y = x³/6 + 1/(2x) from x = {lo} to x = {hi}.",
            "The edge of a sail follows the curve y = x³/6 + 1/(2x) (units in metres) for {lo} ≤ x ≤ {hi}. Find its exact length.",
        ),
        lo=lo,
        hi=hi,
    )
    working = [
        "dy/dx = x²/2 - 1/(2x²), and 1 + (dy/dx)² = 1 + x⁴/4 - 1/2 + 1/(4x⁴) = (x²/2 + 1/(2x²))².",
        f"L = the integral of (x²/2 + 1/(2x²)) from {lo} to {hi} = [x³/6 - 1/(2x)] = {_show(G(hi))} - ({_show(G(lo))}).",
        f"L = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _quartic_reciprocal(rng: random.Random) -> dict:
    lo, hi = rng.choice([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])
    curve = x**4 / 8 + 1 / (4 * x**2)
    integrand = x**3 / 2 + 1 / (2 * x**3)
    _verify_integrand(curve, integrand)
    G = lambda v: Rational(v) ** 4 / 8 - 1 / (4 * Rational(v) ** 2)  # noqa: E731
    answer = G(hi) - G(lo)
    check(close(_numeric_length(curve, lo, hi), float(answer), 1e-6), "numerical length disagrees")
    wrong = _common_wrong(curve, lo, hi, G, integrand)
    wrong.append((sympy.integrate(x**3 / 2 - 1 / (2 * x**3), (x, lo, hi)), "fy1ca-arc-cross-term-sign"))
    text = phrase(
        rng,
        (
            "Find the exact length of the curve y = x⁴/8 + 1/(4x²) from x = {lo} to x = {hi}.",
            "A rock-climbing wall is cut along the curve y = x⁴/8 + 1/(4x²) (units in metres) from x = {lo} to x = {hi}. Find the exact length of the curve.",
        ),
        lo=lo,
        hi=hi,
    )
    working = [
        "dy/dx = x³/2 - 1/(2x³), and 1 + (dy/dx)² = (x³/2 + 1/(2x³))².",
        f"L = the integral of (x³/2 + 1/(2x³)) from {lo} to {hi} = [x⁴/8 - 1/(4x²)] = {_show(G(hi))} - ({_show(G(lo))}).",
        f"L = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _medium(rng: random.Random) -> dict:
    return rng.choice([_cubic_reciprocal, _quartic_reciprocal])(rng)


# -- hard ----------------------------------------------------------------------------------------------------------------

# t values for which 1 + t² is the square of a rational
_T_VALUES = (
    Rational(0),
    Rational(5, 12),
    Rational(3, 4),
    Rational(8, 15),
    Rational(4, 3),
    Rational(15, 8),
    Rational(12, 5),
)


def _parametric(rng: random.Random) -> dict:
    lo, hi = rng.choice([(a, b) for a in _T_VALUES for b in _T_VALUES if a < b])
    xt, yt = t**2, Rational(2, 3) * t**3
    speed = 2 * t * sqrt(1 + t**2)
    check(sympy.simplify(sympy.diff(xt, t) ** 2 + sympy.diff(yt, t) ** 2 - speed**2) == 0, "speed is wrong")
    G = lambda v: Rational(2, 3) * (1 + v**2) ** Rational(3, 2)  # noqa: E731
    answer = G(hi) - G(lo)
    check(close(_numeric_parametric(xt, yt, lo, hi), float(answer), 1e-6), "numerical length disagrees")
    chord = sqrt((xt.subs(t, hi) - xt.subs(t, lo)) ** 2 + (yt.subs(t, hi) - yt.subs(t, lo)) ** 2)
    wrong = [
        (sympy.integrate(sympy.diff(xt, t) + sympy.diff(yt, t), (t, lo, hi)), "fy1ca-arc-components-added"),
        (G(hi), "fy1ca-arc-lower-limit-ignored"),
        (chord, "fy1ca-arc-chord-used"),
        (sympy.integrate(sympy.diff(xt, t) ** 2 + sympy.diff(yt, t) ** 2, (t, lo, hi)), "fy1ca-arc-root-dropped"),
        ((G(hi) - G(lo)) * 2, "fy1ca-arc-wrong-antiderivative"),
        ((G(hi) - G(lo)) / 2, "fy1ca-arc-wrong-antiderivative"),
    ]
    text = phrase(
        rng,
        (
            "A drone follows the path x = t², y = (2/3)t³ (units in hundreds of metres) for {lo} ≤ t ≤ {hi}. Find the exact length of the path.",
            "Find the exact length of the curve given parametrically by x = t² and y = (2/3)t³ for {lo} ≤ t ≤ {hi}.",
        ),
        lo=_frac_text(lo),
        hi=_frac_text(hi),
    )
    working = [
        "dx/dt = 2t and dy/dt = 2t², so (dx/dt)² + (dy/dt)² = 4t² + 4t⁴ = 4t²(1 + t²).",
        "The speed is 2t√(1 + t²), and L is the integral of 2t√(1 + t²) dt = [(2/3)(1 + t²)^(3/2)].",
        f"L = (2/3)({_show(1 + hi**2)})^(3/2) - (2/3)({_show(1 + lo**2)})^(3/2) = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _frac_text(value) -> str:
    value = sympy.Rational(value)
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def _with_logarithm(rng: random.Random) -> dict:
    hi = rng.choice([2, 3, 4, 5])
    lo = 1
    curve = x**2 / 4 - log(x) / 2
    integrand = x / 2 + 1 / (2 * x)
    _verify_integrand(curve, integrand)
    G = lambda v: Rational(v) ** 2 / 4 + log(Rational(v)) / 2  # noqa: E731
    answer = sympy.simplify(G(hi) - G(lo))
    check(close(_numeric_length(curve, lo, hi), float(answer), 1e-6), "numerical length disagrees")
    wrong = [
        (Rational(hi**2 - 1, 4), "fy1ca-arc-log-term-dropped"),
        (Rational(hi**2 - 1, 4) - log(hi) / 2, "fy1ca-arc-cross-term-sign"),
        (sympy.integrate(1 + sympy.diff(curve, x) ** 2, (x, lo, hi)), "fy1ca-arc-root-dropped"),
        (G(hi), "fy1ca-arc-lower-limit-ignored"),
        (curve.subs(x, hi) - curve.subs(x, lo), "fy1ca-arc-height-change-used"),
        (Rational(hi**2 - 1, 4) + log(hi), "fy1ca-arc-wrong-antiderivative"),
    ]
    text = phrase(
        rng,
        (
            "Find the exact length of the curve y = x²/4 - (1/2)ln x from x = {lo} to x = {hi}.",
            "A garden path follows the curve y = x²/4 - (1/2)ln x (units in metres) from x = {lo} to x = {hi}. Find the exact length of the path.",
        ),
        lo=lo,
        hi=hi,
    )
    working = [
        "dy/dx = x/2 - 1/(2x), and 1 + (dy/dx)² = 1 + x²/4 - 1/2 + 1/(4x²) = (x/2 + 1/(2x))².",
        f"L = the integral of (x/2 + 1/(2x)) from {lo} to {hi} = [x²/4 + (1/2)ln x] = {_show(G(hi))} - {_show(G(lo))}.",
        f"L = {_show(answer)}.",
    ]
    return _question(text, answer, wrong, working)


def _hard(rng: random.Random) -> dict:
    return rng.choice([_parametric, _with_logarithm])(rng)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("fy1ca_arc_length", level=19, presets={t_: dict(tier=t_) for t_ in _TIERS})
def fy1ca_arc_length(rng: random.Random, tier: str = "easy") -> dict:
    """Arc length: a perfect-square integrand (easy), a sum of powers (medium), a parametric or logarithmic curve (hard)."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return build_with_redraws(rng, _TIERS[tier], f"a {tier} arc-length question")


# -- exam ----------------------------------------------------------------------------------------------------------------


def _exam_build(rng: random.Random) -> dict:
    lo, hi = rng.choice([(a, b) for a in range(1, 7) for b in range(a + 1, 9)])
    curve = x**3 / 6 + 1 / (2 * x)
    slope = sympy.diff(curve, x).subs(x, hi)
    integrand = sqrt(1 + slope**2)
    check(sympy.simplify(integrand - (Rational(hi) ** 2 / 2 + 1 / (2 * Rational(hi) ** 2))) == 0, "integrand mismatch")
    G = lambda v: Rational(v) ** 3 / 6 - 1 / (2 * Rational(v))  # noqa: E731
    length = G(hi) - G(lo)
    check(close(_numeric_length(curve, lo, hi), float(length), 1e-6), "numerical length disagrees")
    y_hi, y_lo = curve.subs(x, hi), curve.subs(x, lo)
    chord = sqrt((hi - lo) ** 2 + (y_hi - y_lo) ** 2)
    excess = float(length - chord)
    if not clear_of_rounding(excess, 2) or excess < 0.05:
        raise Redraw

    slope_f = frac(Fraction(int(slope.p), int(slope.q)))
    integrand_f = float(integrand)
    a_dist = fraction_distractors(
        slope_f,
        [
            (Rational(hi) ** 2 / 2 + 1 / (2 * Rational(hi) ** 2), "fy1ca-arc-derivative-sign"),
            (Rational(hi) ** 2 / 2 - 1 / (2 * Rational(hi)), "fy1ca-arc-power-rule-slip"),
            (y_hi, "fy1ca-arc-function-not-differentiated"),
            (Rational(hi) ** 2 / 2 - 1 / (2 * Rational(hi) ** 3), "fy1ca-arc-power-rule-slip"),
        ],
    )
    b_dist = number_distractors(
        integrand_f,
        _floats(
            [
                (1 + slope**2, "fy1ca-arc-root-dropped"),
                (1 + abs(slope), "fy1ca-arc-square-dropped"),
                (slope, "fy1ca-arc-integrand-is-derivative"),
                (sqrt(1 + slope), "fy1ca-arc-square-dropped"),
                (sqrt(slope**2 - 1) if slope**2 > 1 else None, "fy1ca-arc-sign-slip"),
            ]
        ),
        3,
    )
    c_dist = fraction_distractors(
        length,
        _nice(
            [
                (sympy.integrate(1 + sympy.diff(curve, x) ** 2, (x, lo, hi)), "fy1ca-arc-root-dropped"),
                (G(hi), "fy1ca-arc-lower-limit-ignored"),
                (y_hi - y_lo, "fy1ca-arc-height-change-used"),
                (sympy.integrate(x**2 / 2 - 1 / (2 * x**2), (x, lo, hi)), "fy1ca-arc-cross-term-sign"),
                (length * 2, "fy1ca-arc-wrong-antiderivative"),
            ]
        ),
    )
    d_dist = number_distractors(
        excess,
        _floats(
            [
                (length, "fy1ca-arc-chord-ignored"),
                (chord, "fy1ca-arc-arc-ignored"),
                (float(length + chord), "fy1ca-arc-chord-added"),
                (float(chord / length), "fy1ca-arc-ratio-for-difference"),
                (float(length - (hi - lo)), "fy1ca-arc-chord-used"),
            ]
        ),
        2,
    )
    for dist in (a_dist, b_dist, c_dist, d_dist):
        need_distractors(dist)
    dy = sympy.Rational(y_hi - y_lo)
    stem = phrase(
        rng,
        (
            "A curved retaining wall follows the curve y = x³/6 + 1/(2x), where x and y are in metres, from the point where x = {lo} to the point where x = {hi}.",
            "The top edge of a ramp has the shape y = x³/6 + 1/(2x) (x and y in metres) between x = {lo} and x = {hi}.",
            "A skate-park rail is bent to follow y = x³/6 + 1/(2x), with x and y in metres, for {lo} ≤ x ≤ {hi}.",
            "The path of a zip-line is modelled by y = x³/6 + 1/(2x) (metres) over the horizontal range from x = {lo} to x = {hi}.",
        ),
        lo=lo,
        hi=hi,
    )
    parts = [
        fraction_part(
            f"Find dy/dx when x = {hi}.",
            slope_f,
            1,
            [f"dy/dx = x²/2 - 1/(2x²), so at x = {hi} it is {_show(slope)}."],
            a_dist,
        ),
        number_part(
            f"Hence find the value of √(1 + (dy/dx)²) at x = {hi}, correct to 3 decimal places.",
            integrand_f,
            3,
            2,
            [f"√(1 + ({_show(slope)})²) = {integrand_f:.3f}."],
            b_dist,
            follow_through="sqrt(1 + part_a**2)",
            sensible_range=(0, None),
        ),
        fraction_part(
            f"Find the exact length of the curve from x = {lo} to x = {hi}.",
            length,
            3,
            [
                "1 + (dy/dx)² = (x²/2 + 1/(2x²))², so L is the integral of (x²/2 + 1/(2x²)) dx = [x³/6 - 1/(2x)].",
                f"L = {_show(G(hi))} - ({_show(G(lo))}) = {_show(length)}.",
            ],
            c_dist,
        ),
        number_part(
            "By how many metres is the curve longer than the straight line joining its two ends? Give your answer correct to 2 decimal places.",
            excess,
            2,
            2,
            [
                f"The straight line has length √({hi - lo}² + ({_show(dy)})²) = {float(chord):.2f} m.",
                f"{float(length):.2f} - {float(chord):.2f} = {excess:.2f} m.",
            ],
            d_dist,
            follow_through=f"part_c - sqrt({hi - lo}**2 + ({_show(dy)})**2)",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("fy1ca_arc_length_exam", level=19, presets={"exam": {}})
def fy1ca_arc_length_exam(rng: random.Random) -> dict:
    """A curve y = x³/6 + 1/(2x): slope, integrand, exact length, and the excess over the chord."""
    return build_with_redraws(rng, _exam_build, "an arc-length exam chain")
