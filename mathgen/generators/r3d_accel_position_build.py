"""Acceleration as a function of position, v dv/dx = f(x): `r3d_accel_position` (easy, medium, hard) and
`r3d_accel_position_exam`.

Skill: acceleration-function-of-position. When the acceleration is given in terms of the position x, the time is not
in the problem: a = v dv/dx = d(v²/2)/dx, so v² = v0² + 2 times the integral of f(x) from x0 to x.

Tiers differ in the decision they ask for, not in the size of the numbers:

* easy: a = p + qx (a constant plus a multiple of x, or just a multiple of x). Integrate from the starting position and
  find the speed at another position.
* medium: the integral is not a polynomial rule applied to a single term: an inverse-square attraction a = -k/x² (the
  antiderivative is k/x, and the particle moves towards O), or a power law a = kx². The starting speed may be zero.
* hard: the question is about the motion's turning points and extremes rather than a stated position. For
  a = c - kx (with k not a perfect square, so the motion is not the standard form) the first position of rest depends on
  which way the particle starts out; for an inverse-square attraction moving outwards the greatest distance is found from 1/x; the greatest speed is
  where the acceleration is zero.
* exam: a = p - qx from O. (a) the expression for v², (b) the speed at a stated position, (c) the greatest distance from
  O, (d) the greatest speed (where a = 0).

## Construction

Each family is searched in whole numbers so that the quantity asked for is exact: a perfect square for a speed, a
whole position for a turning point. Every v² is recomputed with SymPy's symbolic integral of f(x), a different route
from the antiderivative written into the generator.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..core import VerificationError, attempts, check, generator, join_terms
from ._r3d_kit import (
    X,
    clear_of_boundary,
    dec,
    exact,
    expression_part,
    fit_dp,
    number_fields,
    number_part,
    sqrt_of,
)

F = Fraction
_x = X


def _rat(v) -> sympy.Rational:
    v = F(v)
    return sympy.Rational(v.numerator, v.denominator)


def _perfect_square(q: Fraction) -> Fraction | None:
    """The exact rational square root of `q`, or None."""
    if q <= 0:
        return None
    n, d = q.numerator, q.denominator
    rn, rd = math.isqrt(n), math.isqrt(d)
    return F(rn, rd) if rn * rn == n and rd * rd == d else None


def _poly_text(terms) -> str:
    """'6 - 2x' from [(6, ''), (-2, 'x')] with integer or Fraction coefficients."""
    return join_terms([(c, var) for c, var in terms])


def _accel_text(p, q) -> str:
    return join_terms(
        [(int(p) if F(p).denominator == 1 else float(p), ""), (int(q) if F(q).denominator == 1 else float(q), "x")]
    )


def _v2_symbolic(f, x0, v0, x1) -> Fraction:
    """v² at x1 from the symbolic integral of f from x0 to x1."""
    if sympy.sympify(f).is_polynomial(_x):  # Poly.integrate is far quicker than the general integrator
        anti = sympy.Poly(f, _x).integrate()
        area = anti.eval(_rat(x1)) - anti.eval(_rat(x0))
    else:
        area = sympy.integrate(f, (_x, _rat(x0), _rat(x1)))
    return exact(_rat(v0) ** 2 + 2 * area)


def _v2_poly(coeffs, x0, v0, x1) -> Fraction:
    """v² at x1 for f(x) = sum of c * x^k, coeffs = {k: c}, in exact fractions (the search route; SymPy checks the winner)."""
    total = F(0)
    for k, c in coeffs.items():
        total += F(c) * (F(x1) ** (k + 1) - F(x0) ** (k + 1)) / (k + 1)
    return F(v0) ** 2 + 2 * total


def _stays_positive(coeffs, x0, v0, x1) -> bool:
    lo, hi = sorted((F(x0), F(x1)))
    steps = 24
    return all(_v2_poly(coeffs, x0, v0, lo + (hi - lo) * i / steps) > 0 for i in range(steps + 1))


# ------------------------------------------------------------------------------------------------ easy


def _easy(rng):
    for _ in attempts(5000, "a linear acceleration with a perfect-square speed"):
        p = rng.randint(-6, 6)
        q = rng.choice([c for c in range(-4, 5) if c != 0])
        x0 = rng.randint(0, 3)
        x1 = x0 + rng.choice((1, 2, 3, 4, 5))
        v0 = rng.randint(1, 8)
        coeffs = {0: p, 1: q}
        v1sq = _v2_poly(coeffs, x0, v0, x1)
        v1 = _perfect_square(v1sq)
        if v1 is None or v1 == v0 or v1 != int(v1) or not _stays_positive(coeffs, x0, v0, x1):
            continue
        # the slips below must give at least three different wrong speeds, none equal to the answer
        antider = F(p * (x1 - x0)) + F(q * (x1 * x1 - x0 * x0), 2)
        slips = {
            sqrt_of(v0 * v0 + 2 * (p + q * x0) * (x1 - x0)),
            sqrt_of(v0 * v0 + antider),
            sqrt_of(2 * antider),
            v0 + antider,
            v1sq,
        }
        if len({c for c in slips if c == c and c > 0 and c != v1}) >= 3:
            break
    f = _rat(p) + _rat(q) * _x
    check(_v2_symbolic(f, x0, v0, x1) == v1sq, "v² differs from the symbolic integral")
    a_text = _accel_text(p, q)
    antider = F(p * (x1 - x0)) + F(q * (x1 * x1 - x0 * x0), 2)
    check(v0 * v0 + 2 * antider == v1sq, "v² differs from the closed form")
    a0 = p + q * x0
    text = (
        f"A particle moves along the x-axis. When it is at position x metres its acceleration is ({a_text}) m/s². "
        f"The particle passes x = {x0} with speed {v0} m/s, moving in the positive direction. Find its speed when "
        f"x = {x1}, in metres per second."
    )
    wrong = [
        (sqrt_of(v0 * v0 + 2 * a0 * (x1 - x0)), "r3d-constant-acceleration-assumed"),
        (sqrt_of(v0 * v0 + antider), "r3d-factor-two-dropped"),
        (sqrt_of(2 * antider), "r3d-initial-speed-dropped"),
        (v0 + antider, "r3d-integrated-once-for-speed"),
        (v1sq, "r3d-speed-squared-reported"),
    ]
    working = [
        f"Since a = v dv/dx, v²/2 = ∫({a_text}) dx, so v² = {v0}² + 2[{_antiderivative_text(p, q)}] evaluated from x = {x0} to x = {x1}.",
        f"v² = {v0 * v0} + 2({dec(antider, 3)}) = {dec(v1sq, 3)}, so v = {dec(v1, 3)} m/s.",
    ]
    return dict(text=text, working=working, **number_fields(v1, fit_dp(v1, 1, 3) or 1, wrong, nonneg=True))


def _antiderivative_text(p, q) -> str:
    return join_terms([(p, "x"), (F(q, 2) if q % 2 else q // 2, "x²")])


# ------------------------------------------------------------------------------------------------ medium


def _medium(rng):
    variant = rng.choice(("inverse_square_rest", "inverse_square_moving", "power"))
    if variant.startswith("inverse"):
        rest = variant.endswith("rest")
        for _ in attempts(20000, "an inverse-square question with an exact speed"):
            k = rng.choice((2, 3, 4, 6, 8, 9, 12, 18, 24, 36))
            x0 = rng.choice((2, 3, 4, 6, 8, 9, 12))
            x1 = rng.choice([c for c in (1, 2, 3, 4, 6, 8) if c < x0])
            v0 = F(0) if rest else F(rng.randint(1, 6))
            v1sq = v0 * v0 + 2 * k * (F(1, x1) - F(1, x0))
            v1 = _perfect_square(v1sq)
            if v1 is not None and fit_dp(v1, 0, 2) is not None:
                break
        f = -_rat(k) / _x**2
        check(_v2_symbolic(f, x0, v0, x1) == v1sq, "v² differs from the symbolic integral")
        dp = fit_dp(v1, 1, 3)
        start = (
            f"It is released from rest at x = {x0}."
            if rest
            else f"It passes x = {x0} with speed {dec(v0, 2)} m/s, moving towards O."
        )
        text = (
            f"A particle moves along the positive x-axis towards the origin O under an attraction that gives it an "
            f"acceleration of magnitude {k}/x² m/s² towards O when it is x metres from O. {start} Find its speed when "
            f"x = {x1}, in metres per second."
        )
        wrong = [
            (sqrt_of(v0 * v0 - 2 * k * (F(1, x1) - F(1, x0))), "r3d-integral-sign"),
            (sqrt_of(v0 * v0 + 2 * k * F(math.log(x0 / x1)).limit_denominator(10**9)), "r3d-log-for-inverse-power"),
            (sqrt_of(v0 * v0 + k * (F(1, x1) - F(1, x0))), "r3d-factor-two-dropped"),
            (sqrt_of(v0 * v0 + 2 * F(k, x0 * x0) * (x0 - x1)), "r3d-constant-acceleration-assumed"),
            (v1sq, "r3d-speed-squared-reported"),
        ]
        working = [
            f"The acceleration is -{k}/x² (towards O, so negative). v²/2 = ∫(-{k}/x²) dx = {k}/x + C.",
            f"v² = {dec(v0 * v0, 3)} + 2({k}/{x1} - {k}/{x0}) = {dec(v1sq, 3)}, so v = {dec(v1, 3)} m/s.",
        ]
        return dict(text=text, working=working, **number_fields(v1, dp, wrong, nonneg=True))
    for _ in attempts(20000, "a power-law question with an exact speed"):
        k = rng.choice((3, 6, 9, -3, -6, -9))
        x0 = rng.randint(0, 2)
        x1 = x0 + rng.randint(1, 3)
        v0 = rng.randint(0, 6)
        v1sq = _v2_poly({2: k}, x0, v0, x1)
        v1 = _perfect_square(v1sq)
        if v1 is not None and v1 == int(v1) and _stays_positive({2: k}, x0, v0, x1):
            break
    check(_v2_symbolic(_rat(k) * _x**2, x0, v0, x1) == v1sq, "v² differs from the symbolic integral")
    text = (
        f"A particle moves along the x-axis. When it is at position x metres its acceleration is {k}x² m/s². "
        f"The particle passes x = {x0} with speed {v0} m/s moving in the positive direction. Find its speed when "
        f"x = {x1}, in metres per second."
    )
    cube = F(k, 3) * (x1**3 - x0**3)
    wrong = [
        (sqrt_of(v0 * v0 + 2 * k * (x1**3 - x0**3)), "r3d-power-rule-no-division"),
        (sqrt_of(v0 * v0 + cube), "r3d-factor-two-dropped"),
        (sqrt_of(2 * cube), "r3d-initial-speed-dropped"),
        (sqrt_of(v0 * v0 + 2 * k * x0 * x0 * (x1 - x0)), "r3d-constant-acceleration-assumed"),
        (v1sq, "r3d-speed-squared-reported"),
    ]
    working = [
        f"v²/2 = ∫{k}x² dx = {dec(F(k, 3), 3)}x³ + C.",
        f"v² = {v0 * v0} + 2({dec(F(k, 3), 3)})({x1}³ - {x0}³) = {dec(v1sq, 3)}, so v = {dec(v1, 3)} m/s.",
    ]
    return dict(text=text, working=working, **number_fields(v1, fit_dp(v1, 1, 3) or 1, wrong, nonneg=True))


# ------------------------------------------------------------------------------------------------ hard


def _linear_motion(rng, ks=(2, 3, 5, 6)):
    """a = k(xc - x): centre xc, k, amplitude R, start x0 = xc + u, start speed v0 = k m with R² - u² = k m²."""
    for _ in attempts(20000, "a linear-acceleration motion with whole numbers"):
        k = rng.choice(ks)
        xc = rng.randint(1, 6)
        R = rng.randint(3, 12)
        u = rng.choice([c for c in range(-R + 1, R) if c != 0])
        rem = R * R - u * u
        if rem % k:
            continue
        m2 = rem // k
        m = math.isqrt(m2)
        if m * m != m2 or m == 0:
            continue
        return k, xc, R, u, k * m
    raise AssertionError("unreachable")


def _hard(rng):
    variant = rng.choice(("rest", "greatest_speed", "inverse_square_turning"))
    if variant in ("rest", "greatest_speed"):
        k, xc, R, u, v0 = _linear_motion(rng)
        c = k * xc
        x0 = xc + u
        sign = rng.choice((1, -1))
        f = _rat(c) - _rat(k) * _x
        # second route: v² from the symbolic integral is zero exactly at xc ± R
        for end in (xc + R, xc - R):
            check(_v2_symbolic(f, x0, v0, end) == 0, "v² is not zero at the turning point")
        check(_v2_symbolic(f, x0, v0, xc) == k * R * R, "v² at the centre differs from kR²")
        a_text = f"{c} - {k}x"
        if variant == "rest":
            direction = "positive" if sign == 1 else "negative"
            value = xc + sign * R
            start = f"The particle passes x = {x0} with speed {v0} m/s, moving in the {direction} direction."
            text = (
                f"A particle moves along the x-axis. When it is at position x metres its acceleration is ({a_text}) m/s². "
                f"{start} Find the x-coordinate of the position at which it first comes instantaneously to rest."
            )
            wrong = [
                (xc - sign * R, "r3d-other-turning-point"),
                (xc, "r3d-equilibrium-for-rest"),
                (
                    x0 + sign * F(v0 * v0, 2 * abs(c - k * x0)) if c != k * x0 else x0 + sign,
                    "r3d-constant-acceleration-assumed",
                ),
                (x0 + sign * R, "r3d-amplitude-from-start"),
            ]
            working = [
                f"v² = {v0}² + 2∫({a_text}) dx = {v0 * v0} + 2[{c}x - {dec(F(k, 2), 3)}x²] from {x0} to x = {k * R * R - k * (x0 - xc) ** 2 + v0 * v0 and ''}"
                f"{dec(k * R * R, 3)} - {k}(x - {xc})².",
                f"v = 0 when (x - {xc})² = {R * R}, so x = {xc + R} or x = {xc - R}. Moving in the {direction} direction, it first stops at x = {value}.",
            ]
            return dict(text=text, working=working, **number_fields(F(value), 0, wrong))
        start = f"The particle passes x = {x0} with speed {v0} m/s."
        value = R * sqrt_of(k)
        text = (
            f"A particle moves along the x-axis. When it is at position x metres its acceleration is ({a_text}) m/s². "
            f"{start} Find the greatest speed the particle reaches, in metres per second, correct to 2 decimal places."
        )
        if not clear_of_boundary(value, 2):
            return _hard(rng)
        wrong = [
            (sqrt_of(F(v0 * v0) + 2 * (c - k * x0) * abs(xc - x0)), "r3d-constant-acceleration-assumed"),
            (R * k, "r3d-factor-root-dropped"),
            (sqrt_of(v0 * v0 + k * R * R), "r3d-initial-speed-added"),
            (F(v0) + 0 if v0 != value else F(v0 + 1), "r3d-start-speed-for-greatest"),
            (sqrt_of(k) * (R + abs(u)), "r3d-distance-from-start-for-amplitude"),
        ]
        working = [
            f"The speed is greatest where a = 0: {c} - {k}x = 0, so x = {xc}.",
            f"v² = {v0 * v0} + 2[{c}x - {dec(F(k, 2), 3)}x²] from {x0} to {xc} = {k * R * R}, so v = √{k * R * R} = {float(value):.2f} m/s.",
        ]
        return dict(text=text, working=working, **number_fields(value, 2, wrong, nonneg=True))
    # inverse-square attraction, moving away from O: the greatest distance reached
    for _ in attempts(50000, "an inverse-square turning point with whole numbers"):
        k = rng.randint(1, 40)
        x0 = rng.choice((1, 2, 3, 4, 5, 6, 8, 9, 10, 12))
        xt = rng.choice((2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24))
        if xt <= x0:
            continue
        v0sq = 2 * k * (F(1, x0) - F(1, xt))
        v0 = _perfect_square(v0sq)
        if v0 is None or (2 * v0).denominator != 1 or not 0 < v0 <= 12:
            continue
        wrong = [
            (x0 + v0 * v0 / (2 * F(k, x0 * x0)), "r3d-constant-acceleration-assumed"),
            (1 / (F(1, x0) + v0 * v0 / (2 * k)), "r3d-integral-sign"),
            (1 / (F(1, x0) - v0 * v0 / k) if F(1, x0) - v0 * v0 / k > 0 else x0 + 1, "r3d-factor-two-dropped"),
            (F(1, x0) - v0 * v0 / (2 * k), "r3d-reciprocal-reported"),
            (x0 * xt, "r3d-product-for-position"),
        ]
        try:
            fields = number_fields(F(xt), 1, wrong, nonneg=True)
        except VerificationError:
            continue
        break
    f = -_rat(k) / _x**2
    check(_v2_symbolic(f, x0, v0, xt) == 0, "v² is not zero at the turning point")
    text = (
        f"A particle moves along the positive x-axis under an attraction towards the origin O that gives it an "
        f"acceleration of magnitude {k}/x² m/s² when it is x metres from O. It passes x = {x0} with speed {dec(v0, 2)} m/s, "
        f"moving away from O. Find the greatest distance from O that it reaches, in metres, correct to 1 decimal place."
    )
    working = [
        f"v²/2 = ∫(-{k}/x²) dx = {k}/x + C, so v² = {dec(v0 * v0, 3)} + 2({k}/x - {k}/{x0}).",
        f"v = 0 when {k}/x = {k}/{x0} - {dec(v0 * v0 / 2, 3)} = {dec(F(k, xt), 3)}, so x = {xt} m.",
    ]
    return dict(text=text, working=working, **fields)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}
_PRESETS = {t: {"tier": t} for t in _TIERS}


@generator("r3d_accel_position", level=18, presets=_PRESETS)
def r3d_accel_position(rng, tier: str = "easy"):
    """v² = v0² + 2 ∫f(x) dx for linear, inverse-square and power-law accelerations, then turning points and extremes."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam

_EXAM_STEMS = (
    "A particle starts from the origin O and moves along the positive x-axis with initial speed {v0} m/s. When it is at "
    "position x metres its acceleration is ({p} - {q}x) m/s².",
    "A trolley leaves the origin O at {v0} m/s along a straight track in the positive x-direction. At position x "
    "metres from O its acceleration is ({p} - {q}x) m/s².",
    "A small cart is set moving along the positive x-axis from the origin O at {v0} m/s. When it is x metres from "
    "O, its acceleration is ({p} - {q}x) m/s².",
)


@generator("r3d_accel_position_exam", level=18, presets={"exam": {}})
def r3d_accel_position_exam(rng):
    """a = p - qx from O: the expression for v², the speed at a position, the greatest distance and the greatest speed."""
    for _ in attempts(20000, "a start from O with a clear speed"):
        k, xc, R, u, v0 = _linear_motion(rng, ks=(2, 4, 6))
        if u != -xc:  # start at O
            continue
        x1 = rng.randint(1, xc + R - 1)
        v1sq = F(v0 * v0) + 2 * k * xc * x1 - k * x1 * x1
        vmax = R * sqrt_of(k)
        if v1sq > 0 and v1sq != v0 * v0 and clear_of_boundary(sqrt_of(v1sq), 2) and clear_of_boundary(vmax, 2):
            break
    p, q = k * xc, k
    f = _rat(p) - _rat(q) * _x
    v2_expr = sympy.Integer(v0) ** 2 + 2 * sympy.integrate(f, (_x, 0, _x))
    check(sympy.expand(v2_expr - (v0 * v0 + 2 * p * _x - q * _x**2)) == 0, "v² differs from the integral")
    check(v2_expr.subs(_x, xc + R) == 0, "the end of the motion is not a root of v²")
    stem = _EXAM_STEMS[rng.randrange(len(_EXAM_STEMS))].format(v0=v0, p=p, q=q)
    v2_text = join_terms([(v0 * v0, ""), (2 * p, "x"), (-q, "x²")])
    half_q = F(q, 2)
    wrong_a = [
        (
            sympy.Integer(v0 * v0) + p * _x - _rat(half_q) * _x**2,
            join_terms([(v0 * v0, ""), (p, "x"), (-half_q if half_q.denominator > 1 else int(-half_q), "x²")]),
            "r3d-factor-two-dropped",
        ),
        (2 * p * _x - q * _x**2, join_terms([(2 * p, "x"), (-q, "x²")]), "r3d-initial-speed-dropped"),
        (
            sympy.Integer(v0 * v0) + 2 * p * _x - 2 * q * _x**2,
            join_terms([(v0 * v0, ""), (2 * p, "x"), (-2 * q, "x²")]),
            "r3d-power-rule-no-division",
        ),
        (
            sympy.Integer(v0 * v0) + 2 * p * _x + q * _x**2,
            join_terms([(v0 * v0, ""), (2 * p, "x"), (q, "x²")]),
            "r3d-integral-sign",
        ),
    ]
    part_a = expression_part(
        "Show that v² depends on position only through x and find an expression for v² in terms of x.",
        v2_expr,
        v2_text,
        2,
        [
            f"v dv/dx = {p} - {q}x, so v²/2 = {p}x - {dec(half_q, 2)}x² + C.",
            f"At x = 0, v = {v0}, so C = {dec(F(v0 * v0, 2), 2)} and v² = {v2_text}.",
        ],
        wrong_a,
        _x,
    )
    v1 = sqrt_of(v1sq)
    part_b = number_part(
        f"Find the speed of the particle when x = {x1}, in metres per second, correct to 2 decimal places.",
        v1,
        2,
        2,
        [f"v² = {v0 * v0} + {2 * p}({x1}) - {q}({x1})² = {dec(v1sq, 2)}.", f"v = {float(v1):.2f} m/s."],
        [
            (v1sq, "r3d-speed-squared-reported"),
            (sqrt_of(v0 * v0 + p * x1 - F(q, 2) * x1 * x1), "r3d-factor-two-dropped"),
            (sqrt_of(2 * p * x1 - q * x1 * x1), "r3d-initial-speed-dropped"),
            (v0 + p * x1 - F(q, 2) * x1 * x1, "r3d-integrated-once-for-speed"),
            (F(p - q * x1), "r3d-acceleration-for-speed"),
            (F(v0), "r3d-given-speed-for-greatest"),
        ],
        follow=f"sqrt(Subs(part_a, (x,), ({x1},)))",
        sensible=(0, None),
        nonneg=True,
    )
    part_c = number_part(
        "Find the greatest distance from O that the particle reaches, in metres.",
        F(xc + R),
        0,
        2,
        [
            f"v = 0 when {v0 * v0} + {2 * p}x - {q}x² = 0, that is {q}(x - {xc})² = {q * R * R}.",
            f"x = {xc} ± {R}; the particle reaches x = {xc + R} m.",
        ],
        [
            (F(xc), "r3d-equilibrium-for-rest"),
            (F(xc - R), "r3d-other-turning-point"),
            (F(2 * xc), "r3d-doubled-centre"),
            (F(R), "r3d-amplitude-for-distance"),
        ],
        sensible=(0, None),
    )
    part_d = number_part(
        "Find the greatest speed of the particle, in metres per second, correct to 2 decimal places.",
        vmax,
        2,
        2,
        [
            f"The speed is greatest where a = 0: {p} - {q}x = 0, so x = {xc}.",
            f"v² = {v0 * v0} + {2 * p}({xc}) - {q}({xc})² = {q * R * R}, so v = {float(vmax):.2f} m/s.",
        ],
        [
            (F(q * R * R), "r3d-speed-squared-reported"),
            (v1, "r3d-given-speed-for-greatest"),
            (sqrt_of(v0 * v0 + p * xc), "r3d-factor-two-dropped"),
            (R * F(q), "r3d-factor-root-dropped"),
            (F(p), "r3d-acceleration-for-speed"),
            (F(xc), "r3d-position-for-speed"),
        ],
        follow=f"sqrt(Subs(part_a, (x,), ({xc},)))",
        sensible=(0, None),
        nonneg=True,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
