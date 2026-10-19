"""Generates `w5a_real_integrals` and `exd_x_w5a_real_integrals` questions: real integrals evaluated by residues.

One second-year skill, ``uc2-real-integrals-by-residues`` (domain ``complex-analysis``). The answers are exact multiples
of ``pi`` (occasionally with a surd). Each integral is built from chosen poles, so the answer is fixed before the
integrand is written down:

- over the real line, ``N(x)/D(x)`` with ``D`` a product of real quadratics ``(x - p)(x - conj p)`` with ``Im p > 0``
  (a repeated factor makes a double pole in the upper half-plane), or ``1/(x^4 + a^4)`` and ``x^2/(x^4 + a^4)``, whose
  poles are eighth roots of unity scaled by ``a``;
- over ``[0, 2*pi]``, ``(c0 + c1 cos(theta) + ...)/(a + b cos(theta))^k``, with ``a, b`` chosen so that
  ``a^2 - b^2`` is a perfect square (so the pole inside the unit circle is rational).

Tiers:

- ``easy``: ``1/((x^2 + a^2)(x^2 + b^2))``; or ``1/(a + b cos(theta))`` (or with ``sin``) over ``[0, 2*pi]``.
- ``medium``: an even numerator over two quadratics, or quadratics centred away from the imaginary axis (poles with
  non-zero real part); or ``cos(n theta)`` or ``sin^2(theta)`` over ``a + b cos(theta)``.
- ``hard``: a repeated quadratic factor (a double pole), ``1/(x^4 + a^4)`` and ``x^2/(x^4 + a^4)`` (poles at the
  eighth roots), and ``1/(a + b cos(theta))^2`` or ``cos(theta)/(a + b cos(theta))^2``.
- ``exam`` (topic ``exd_x_w5a_real_integrals``): either a four-part chain on ``N(x)/((x^2 + a^2)(x^2 + b^2))``
  (the two upper-half-plane residues as multiples of ``i``, the integral, the half-line integral), or a four-part
  chain on ``1/(a + b cos(theta))`` (the square root, the pole inside the unit circle, the integral, and the integral
  of its square).

Every answer is confirmed in `make()` by a numerical quadrature of the integrand: Simpson's rule after the
substitution ``x = tan(t)`` over the real line, and the trapezoid rule (geometrically convergent) over a period.
"""

from __future__ import annotations

import cmath
import math
import random
from fractions import Fraction as Fr

import sympy

from ..core import attempts, check, generator
from ._w5a_support import (
    join_signed,
    C,
    Link,
    Rat,
    X,
    atext,
    cx_x,
    exam_dict,
    frac_x,
    msay,
    numeric_close,
    poly_x,
    qsym,
    single,
)

TOPIC = "w5a_real_integrals"
EXAM_TOPIC = "exd_x_w5a_real_integrals"

_PI = sympy.pi
_INF = X("-∞", r"-\infty"), X("∞", r"\infty")


def _simpson_line(fn, n: int = 600) -> float:
    """The integral of fn over the real line: x = tan(t), Simpson's rule on (-pi/2, pi/2)."""
    eps = 1e-9
    lo, hi = -math.pi / 2 + eps, math.pi / 2 - eps
    h = (hi - lo) / n

    def g(t):
        x = math.tan(t)
        return fn(x) / math.cos(t) ** 2

    total = g(lo) + g(hi)
    for k in range(1, n):
        total += (4 if k % 2 else 2) * g(lo + k * h)
    return total * h / 3


def _trapezoid_period(fn, n: int = 1024) -> float:
    return sum(fn(2 * math.pi * k / n) for k in range(n)) * 2 * math.pi / n


def _quad_x(u: int, v2: int) -> X:
    """x^2 - 2ux + (u^2 + v^2): the real quadratic with roots u +/- iv."""
    return poly_x([C(u * u + v2), C(-2 * u), C(1)], "x")


def _bracket(x: X) -> X:
    return X(f"({x.p})", f"({x.t})")


def _line_integral_x(num: list, den_factors: list[tuple[X, int]], lead: bool = True) -> X:
    """The integral over the real line, as an X."""
    n = poly_x(num, "x")
    d_p = "".join(
        f"({f.p})" + (f"^{m}" if m > 1 and m not in (2, 3) else "²" if m == 2 else "³" if m == 3 else "")
        for f, m in den_factors
    )
    d_t = "".join(f"({f.t})" + (f"^{{{m}}}" if m > 1 else "") for f, m in den_factors)
    frac = frac_x(n, X(d_p, d_t))
    return X(f"∫_(-∞)^(∞) {frac.p} dx", rf"\int_{{-\infty}}^{{\infty}} {frac.t}\,\mathrm{{d}}x")


def _trig_x(num: str, num_t: str, a: int, b: int, kind: str, power: int) -> X:
    den = f"{a} {'+' if b > 0 else '-'} {abs(b) if abs(b) != 1 else ''}{kind}(θ)"
    den_t = f"{a} {'+' if b > 0 else '-'} {abs(b) if abs(b) != 1 else ''}\\{kind}\\theta"
    sup = "²" if power == 2 else ""
    sup_t = "^{2}" if power == 2 else ""
    return X(
        f"∫_0^(2π) {num}/({den}){sup} dθ",
        rf"\int_{{0}}^{{2\pi}} \frac{{{num_t}}}{{({den_t}){sup_t}}}\,\mathrm{{d}}\theta",
    )


def _lead_line(rng, integral: X):
    shapes = [
        lambda: msay("Evaluate ", integral, "."),
        lambda: msay("Use the residue theorem to find the exact value of ", integral, "."),
        lambda: msay("Find ", integral, ", giving your answer in exact form."),
    ]
    return shapes[rng.randrange(3)]()


# -- the real line ------------------------------------------------------------------------------------------------------

_POLES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 1),
    (1, 2),
    (-1, 1),
    (2, 1),
    (-2, 1),
    (-1, 2),
    (1, 3),
    (2, 2),
    (-2, 2),
    (3, 1),
]


def _build_line(pole_specs, num_coeffs):
    """(rat, value) for N/(prod Q_j^m_j): poles are (u, v, m). value = 2 pi i * (sum of upper-half-plane residues)."""
    poles = []
    for u, v, m in pole_specs:
        poles.append((C(u, v), m))
        poles.append((C(u, -v), m))
    rat = Rat([C(c) for c in num_coeffs], poles)
    total = C(0)
    for j in range(0, len(poles), 2):
        total = total + rat.residue(j)
    # 2*pi*i * total must be a real multiple of pi: total is purely imaginary
    check(total.re == 0, "the upper-half-plane residues sum to a purely imaginary number")
    value = _PI * qsym(-2 * total.im)
    return rat, value, total


def _check_line(rat: Rat, value) -> None:
    numeric_close(_simpson_line(lambda x: rat.at(complex(x)).real), float(value), "real integral", 1e-6)


def _ai(a: int) -> str:
    return "i" if a == 1 else f"{a}i"


def _easy_line(rng):
    a, b = rng.sample([1, 2, 3, 4, 5], 2)
    rat, value, total = _build_line([(0, a, 1), (0, b, 1)], [1])
    check(sympy.expand(value - _PI / (a * b * (a + b))) == 0, "closed form and residues agree")
    _check_line(rat, value)
    integral = _line_integral_x(
        [1], [(poly_x([C(a * a), C(0), C(1)], "x"), 1), (poly_x([C(b * b), C(0), C(1)], "x"), 1)]
    )
    wrongs = [
        (_PI / (a * b), "cx2-real-integral-residue-slip"),
        (2 * _PI / (a * b * (a + b)), "cx2-real-integral-doubled"),
        (_PI / (2 * a * b * (a + b)), "cx2-real-integral-half-range"),
        (_PI / (a * a * b * b), "cx2-real-integral-residue-slip"),
        (sympy.Integer(0), "cx2-both-half-planes-included"),
        (_PI * (a + b) / (a * b), "calc-ratio-inverted"),
    ]
    working = [
        f"Close the contour in the upper half-plane. The poles there are z = {_ai(a)} and z = {_ai(b)}.",
        f"Res at {_ai(a)} = 1/(2i·{a}·({b * b} - {a * a})) and Res at {_ai(b)} = 1/(2i·{b}·({a * a} - {b * b})).",
        f"The integral is 2πi times their sum: {atext(value)}.",
    ]
    return single(_lead_line(rng, integral), value, working, wrongs)


def _cos_params(rng, k: int = 1):
    a, b = rng.choice(
        [
            (5, 3),
            (5, 4),
            (10, 6),
            (10, 8),
            (13, 5),
            (13, 12),
            (17, 8),
            (17, 15),
            (25, 7),
            (25, 24),
            (15, 9),
            (15, 12),
            (20, 12),
            (20, 16),
            (26, 10),
        ]
    )
    s = math.isqrt(a * a - b * b)
    check(s * s == a * a - b * b, "a^2 - b^2 is a square")
    return a, b, s


def _i_n(a: int, b: int, s: int, n: int):
    """The integral over [0, 2*pi] of cos(n theta)/(a + b cos(theta)): 2*pi*z1^n/s, z1 = (s - a)/b."""
    z1 = Fr(s - a, b)
    return 2 * _PI * qsym(z1**n) / s


def _easy_trig(rng):
    a, b, s = _cos_params(rng)
    kind = rng.choice(["cos", "cos", "sin"])
    sign = rng.choice([1, -1])
    bb = sign * b
    value = 2 * _PI / s
    if kind == "cos":
        fn = lambda t: 1 / (a + bb * math.cos(t))  # noqa: E731
    else:
        fn = lambda t: 1 / (a + bb * math.sin(t))  # noqa: E731
    numeric_close(_trapezoid_period(fn), float(value), "trig integral", 1e-7)
    integral = _trig_x("1", "1", a, bb, kind, 1)
    wrongs = [
        (2 * _PI / sympy.sqrt(a * a + b * b), "cx2-trig-denominator-sign"),
        (2 * _PI / a, "cx2-trig-b-ignored"),
        (_PI / s, "cx2-trig-jacobian-dropped"),
        (2 * _PI / (a - b), "cx2-trig-root-slip"),
        (4 * _PI / s, "cx2-real-integral-doubled"),
        (2 * _PI * s / a, "calc-ratio-inverted"),
    ]
    wrongs = [(sympy.sympify(w), t) for w, t in wrongs]
    working = [
        "Put z = e^(iθ), so cos θ = (z + 1/z)/2 and dθ = dz/(iz).",
        f"The integral becomes ∮ 2/(i({b}z² + {2 * a}z + {b})) dz round |z| = 1; the quadratic has roots (-{a} ± {s})/{b}, and only one is inside the circle.",
        f"The residue there gives 2π/√({a}² - {b}²) = {atext(value)}.",
    ]
    return single(_lead_line(rng, integral), value, working, wrongs)


# -- medium ------------------------------------------------------------------------------------------------------------


def _medium_line(rng):
    for _ in attempts(500, "a medium real-line integral"):
        mode = rng.choice(["even", "shifted", "shifted"])
        if mode == "even":
            a, b = rng.sample([1, 2, 3, 4], 2)
            specs = [(0, a, 1), (0, b, 1)]
            num = [rng.choice([0, 1, 2, 3, -1]), 0, rng.choice([1, 2, 3])]
        else:
            u1, u2 = rng.sample([-2, -1, 0, 1, 2], 2)
            v1, v2 = rng.sample([1, 2, 3], 2)
            specs = [(u1, v1, 1), (u2, v2, 1)]
            num = [rng.choice([1, 2, 3, -1]), rng.choice([0, 1, -1, 2]), 0]
        if num[0] == 0 and num[2] == 0:
            continue
        rat, value, total = _build_line(specs, num)
        if value == 0 or sympy.Rational(value / _PI).q > 400:
            continue
        break
    _check_line(rat, value)
    nfull = num if mode == "shifted" else num
    integral = _line_integral_x(nfull, [(_quad_x(u, v * v), 1) for u, v, _ in specs])
    ratio = qsym(value / _PI) if False else value / _PI
    one_pole = _PI * 2 * sympy.I * 0
    # wrong answers: only one pole; both half-planes; half range; forgotten numerator
    res_up = [rat.residue(0), rat.residue(2)]
    only_first = sympy.simplify(2 * _PI * sympy.I * res_up[0].sym())
    wrongs = [
        (sympy.nsimplify(sympy.expand(only_first)), "ignores-pole-enclosed-by-larger-contour"),
        (value / 2, "cx2-real-integral-half-range"),
        (value * 2, "cx2-real-integral-doubled"),
        (-value, "cx2-lower-half-plane-orientation"),
        (sympy.Integer(0), "cx2-both-half-planes-included"),
        (value / _PI, "forgets-i-in-residue-theorem"),
    ]
    del one_pole, ratio
    names = ", ".join(cx_x(C(u, v)).p for u, v, _ in specs)
    working = [
        f"The poles in the upper half-plane are z = {names}.",
        "Close the contour with a large semicircle above the real axis; its contribution tends to 0 because the integrand is O(1/x²).",
        f"The integral is 2πi times the sum of the two residues: {atext(value)}.",
    ]
    return single(_lead_line(rng, integral), value, working, wrongs)


def _medium_trig(rng):
    a, b, s = _cos_params(rng)
    mode = rng.choice(["cos2", "cos1", "sin2"])
    if mode == "cos1":
        value = _i_n(a, b, s, 1)
        num, num_t = "cos(θ)", r"\cos\theta"
        fn = lambda t: math.cos(t) / (a + b * math.cos(t))  # noqa: E731
        wrongs = [
            (_i_n(a, b, s, 0), "cx2-trig-numerator-ignored"),
            (-value, "cx2-trig-root-slip"),
            (2 * _PI * qsym(Fr(a - s, b)) / s * -1 * -1, "cx2-trig-root-slip"),
            (value * 2, "cx2-real-integral-doubled"),
            (_i_n(a, b, s, 2), "cx2-trig-numerator-ignored"),
        ]
    elif mode == "cos2":
        value = _i_n(a, b, s, 2)
        num, num_t = "cos(2θ)", r"\cos 2\theta"
        fn = lambda t: math.cos(2 * t) / (a + b * math.cos(t))  # noqa: E731
        wrongs = [
            (_i_n(a, b, s, 1), "cx2-trig-numerator-ignored"),
            (_i_n(a, b, s, 0), "cx2-trig-numerator-ignored"),
            (-value, "cx2-trig-root-slip"),
            (value * 2, "cx2-real-integral-doubled"),
            (2 * _PI * qsym(Fr(s - a, b)) ** 1 / s, "cx2-trig-numerator-ignored"),
        ]
    else:
        value = (_i_n(a, b, s, 0) - _i_n(a, b, s, 2)) / 2
        num, num_t = "sin²(θ)", r"\sin^{2}\theta"
        fn = lambda t: math.sin(t) ** 2 / (a + b * math.cos(t))  # noqa: E731
        wrongs = [
            ((_i_n(a, b, s, 0) + _i_n(a, b, s, 2)) / 2, "cx2-trig-numerator-ignored"),
            (_i_n(a, b, s, 0) - _i_n(a, b, s, 2), "cx2-real-integral-doubled"),
            (_i_n(a, b, s, 0) / 2, "cx2-trig-numerator-ignored"),
            (-value, "cx2-trig-root-slip"),
            (_i_n(a, b, s, 2), "cx2-trig-numerator-ignored"),
        ]
    numeric_close(_trapezoid_period(fn), float(sympy.N(value, 15)), "trig integral", 1e-7)
    integral = _trig_x(num, num_t, a, b, "cos", 1)
    working = [
        "Put z = e^(iθ); then cos(nθ) is the real part of z^n, and the sine part integrates to 0 by symmetry.",
        f"The only pole inside |z| = 1 is z1 = ({s} - {a})/{b} = {atext(qsym(Fr(s - a, b)))}, and ∫ cos(nθ)/({a} + {b}cos θ) dθ = 2π z1^n/{s}.",
        f"Combining the terms gives {atext(value)}.",
    ]
    return single(_lead_line(rng, integral), value, working, wrongs, fallback_tag="cx2-trig-root-slip")


# -- hard --------------------------------------------------------------------------------------------------------------


def _hard_double(rng):
    for _ in attempts(500, "a double-pole real integral"):
        u1 = rng.choice([0, 0, 1, -1])
        v1 = rng.choice([1, 2, 3])
        extra = rng.random() < 0.5
        specs = [(u1, v1, 2)]
        if extra:
            u2 = rng.choice([0, 1, -1, 2])
            v2 = rng.choice([1, 2, 3])
            if (u2, v2) == (u1, v1):
                continue
            specs.append((u2, v2, 1))
        num = (
            [rng.choice([1, 2, 3]), 0, rng.choice([0, 0, 1])]
            if u1 == 0 and (not extra or specs[1][0] == 0)
            else [rng.choice([1, 2, 3]), rng.choice([0, 1]), 0]
        )
        rat, value, total = _build_line(specs, num)
        if value == 0 or sympy.Rational(value / _PI).q > 600 or sympy.Rational(value / _PI).p == 0:
            continue
        break
    _check_line(rat, value)
    integral = _line_integral_x(num, [(_quad_x(u, v * v), m) for u, v, m in specs])
    simple = sympy.expand(
        2 * _PI * sympy.I * sum(rat.laurent(j, 1)[-rat.poles[j][1]].sym() for j in range(0, len(rat.poles), 2))
    )
    wrongs = [
        (sympy.nsimplify(simple), "cx2-residue-simple-formula"),
        (value / 2, "cx2-real-integral-half-range"),
        (value * 2, "cx2-real-integral-doubled"),
        (-value, "cx2-lower-half-plane-orientation"),
        (value / _PI, "forgets-i-in-residue-theorem"),
        (sympy.Integer(0), "cx2-both-half-planes-included"),
    ]
    names = ", ".join(f"{cx_x(C(u, v)).p} (order {m})" for u, v, m in specs)
    working = [
        f"The poles in the upper half-plane are z = {names}.",
        "The residue at a double pole needs the derivative of the rest of the integrand at the pole.",
        f"The integral is 2πi times the sum of the upper-half-plane residues: {atext(value)}.",
    ]
    return single(_lead_line(rng, integral), value, working, wrongs)


def _hard_quartic(rng):
    a = rng.choice([1, 2, 3])
    mode = rng.choice(["one", "x2"])
    if mode == "one":
        value = _PI * sympy.sqrt(2) / (2 * a**3)
        num_p, num = "1", [1]
        fn = lambda x: 1 / (x**4 + a**4)  # noqa: E731
    else:
        value = _PI * sympy.sqrt(2) / (2 * a)
        num_p, num = "x²", [0, 0, 1]
        fn = lambda x: x * x / (x**4 + a**4)  # noqa: E731
    numeric_close(_simpson_line(fn), float(sympy.N(value, 15)), "quartic integral", 1e-6)
    den = X(f"x⁴ + {a**4}", f"x^{{4}} + {a**4}")
    frac = frac_x(poly_x([C(c) for c in num], "x"), den)
    integral = X(f"∫_(-∞)^(∞) {frac.p} dx", rf"\int_{{-\infty}}^{{\infty}} {frac.t}\,\mathrm{{d}}x")
    wrongs = [
        (_PI / (a**3 if mode == "one" else a), "cx2-quartic-surd-dropped"),
        (
            _PI * sympy.sqrt(2) / (4 * a**3) if mode == "one" else _PI * sympy.sqrt(2) / (4 * a),
            "cx2-real-integral-half-range",
        ),
        (
            _PI * sympy.sqrt(2) / (2 * a) if mode == "one" else _PI * sympy.sqrt(2) / (2 * a**3),
            "cx2-quartic-scaling-slip",
        ),
        (_PI * sympy.sqrt(2) / a**3 if mode == "one" else _PI * sympy.sqrt(2) / a, "cx2-real-integral-doubled"),
        (_PI / (2 * a**3) if mode == "one" else _PI / (2 * a), "cx2-quartic-surd-dropped"),
        (sympy.Integer(0), "cx2-both-half-planes-included"),
    ]
    working = [
        f"The poles are the four fourth roots of -{a**4}: {a}e^(iπ/4), {a}e^(3iπ/4), {a}e^(5iπ/4), {a}e^(7iπ/4). Two lie in the upper half-plane.",
        "At a simple pole z0, the residue of N/D is N(z0)/D'(z0) = N(z0)/(4 z0³).",
        f"Adding the two residues and multiplying by 2πi gives {atext(value)}.",
    ]
    return single(_lead_line(rng, integral), value, working, wrongs)


def _hard_trig_square(rng):
    a, b, s = _cos_params(rng)
    mode = rng.choice(["one", "cos"])
    if mode == "one":
        value = 2 * _PI * qsym(Fr(a, s**3))
        fn = lambda t: 1 / (a + b * math.cos(t)) ** 2  # noqa: E731
        num, num_t = "1", "1"
        wrongs = [
            (2 * _PI * qsym(Fr(1, s**2)), "cx2-trig-double-pole-slip"),
            (2 * _PI * qsym(Fr(a, s**2)), "cx2-trig-double-pole-slip"),
            (_PI * qsym(Fr(a, s**3)), "cx2-trig-jacobian-dropped"),
            ((2 * _PI / s) ** 2 / (2 * _PI), "cx2-trig-double-pole-slip"),
            (2 * _PI * qsym(Fr(b, s**3)), "cx2-trig-double-pole-slip"),
        ]
    else:
        value = -2 * _PI * qsym(Fr(b, s**3))
        fn = lambda t: math.cos(t) / (a + b * math.cos(t)) ** 2  # noqa: E731
        num, num_t = "cos(θ)", r"\cos\theta"
        wrongs = [
            (2 * _PI * qsym(Fr(b, s**3)), "cx2-trig-root-slip"),
            (-2 * _PI * qsym(Fr(a, s**3)), "cx2-trig-double-pole-slip"),
            (-2 * _PI * qsym(Fr(b, s**2)), "cx2-trig-double-pole-slip"),
            (-_PI * qsym(Fr(b, s**3)), "cx2-trig-jacobian-dropped"),
            (2 * _PI * qsym(Fr(a, s**3)), "cx2-trig-double-pole-slip"),
        ]
    numeric_close(_trapezoid_period(fn), float(sympy.N(value, 15)), "trig integral", 1e-7)
    integral = _trig_x(num, num_t, a, b, "cos", 2)
    working = [
        "With z = e^(iθ) the integrand becomes a rational function of z with a double pole at the root of the quadratic inside |z| = 1.",
        "Its residue needs one derivative.",
        f"The integral is {atext(value)}.",
    ]
    return single(_lead_line(rng, integral), value, working, wrongs, fallback_tag="cx2-trig-double-pole-slip")


_TIERS = {
    "easy": [_easy_line, _easy_trig],
    "medium": [_medium_line, _medium_trig],
    "hard": [_hard_double, _hard_quartic, _hard_trig_square],
}


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5a_real_integrals(rng: random.Random, tier: str = "easy") -> dict:
    """A real integral over the real line or over [0, 2*pi], by residues: an exact multiple of pi."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return rng.choice(_TIERS[tier])(rng)


# -- exam ------------------------------------------------------------------------------------------------------------


def _exam_line(rng):
    for _ in attempts(500, "a real-line chain"):
        a, b = rng.sample([1, 2, 3, 4], 2)
        num = [rng.choice([1, 2, 3]), 0, rng.choice([0, 0, 1, -1])]
        rat, value, total = _build_line([(0, a, 1), (0, b, 1)], num)
        k_a, k_b = rat.residue(0).im, rat.residue(2).im
        if value == 0 or k_a == 0 or k_b == 0:
            continue
        break
    _check_line(rat, value)
    half = value / 2
    integral = _line_integral_x(
        num, [(poly_x([C(a * a), C(0), C(1)], "x"), 1), (poly_x([C(b * b), C(0), C(1)], "x"), 1)]
    )
    stem = msay("Let ", X("I = ", "I = "), integral, ".")
    w_a = Link(
        msay(
            "The integrand f(x) extends to a function of z with poles at ",
            X(f"z = {cx_x(C(0, a)).p}", f"z = {cx_x(C(0, a)).t}"),
            ". Write its residue there as ",
            X("k i", "k\\,i"),
            ". Find k.",
        ),
        qsym(k_a),
        "fraction",
        2,
        [
            f"At z0 = {_ai(a)} the denominator's derivative is 2z0(z0² + {b * b}), so Res = N(z0)/D'(z0) = {atext(qsym(k_a))} i."
        ],
        [
            (-qsym(k_a), "residue-sign-error"),
            (qsym(k_b), "residue-at-wrong-pole"),
            (qsym(k_a * a), "cx2-residue-factor-omitted"),
            (qsym(k_a * 2), "cx2-denominator-derivative-slip"),
        ],
    )
    w_b = Link(
        msay(
            "The integrand also has a pole at ",
            X(f"z = {cx_x(C(0, b)).p}", f"z = {cx_x(C(0, b)).t}"),
            ". Write its residue there as ",
            X("k i", "k\\,i"),
            ". Find k.",
        ),
        qsym(k_b),
        "fraction",
        2,
        [
            f"At z0 = {_ai(b)} the denominator's derivative is 2z0(z0² + {a * a}), so Res = N(z0)/D'(z0) = {atext(qsym(k_b))} i."
        ],
        [
            (-qsym(k_b), "residue-sign-error"),
            (qsym(k_a), "residue-at-wrong-pole"),
            (qsym(k_b * b), "cx2-residue-factor-omitted"),
            (qsym(k_b * 2), "cx2-denominator-derivative-slip"),
        ],
    )
    w_c = Link(
        msay("Hence find the exact value of ", X("I", "I"), ", using the poles in the upper half-plane."),
        value,
        "expression",
        3,
        [
            f"I = 2πi (Res at {_ai(a)} + Res at {_ai(b)}) = 2πi·({join_signed([atext(qsym(k_a)) + '·i', atext(qsym(k_b)) + '·i'])}) = -2π({atext(qsym(k_a + k_b))}) = {atext(value)}."
        ],
        [
            (-value, "cx2-lower-half-plane-orientation"),
            (value / 2, "cx2-real-integral-half-range"),
            (2 * _PI * qsym(k_a + k_b), "calc-sign-slip"),
            (-2 * _PI * qsym(k_a), "ignores-pole-enclosed-by-larger-contour"),
        ],
        ft="-2*pi*(part_a + part_b)",
    )
    w_d = Link(
        msay("Hence find ", X("∫_0^(∞) f(x) dx", r"\int_{0}^{\infty} f(x)\,\mathrm{d}x"), "."),
        half,
        "expression",
        2,
        ["The integrand is even, so the integral over [0, ∞) is half the integral over the whole line."],
        [
            (value, "cx2-real-integral-doubled"),
            (value / 4, "cx2-real-integral-half-range"),
            (-half, "cx2-lower-half-plane-orientation"),
            (2 * value, "cx2-real-integral-doubled"),
        ],
        ft="part_c/2",
    )
    return exam_dict(stem, [w_a, w_b, w_c, w_d])


def _exam_trig(rng):
    a, b, s = _cos_params(rng)
    value = 2 * _PI / s
    square = 2 * _PI * qsym(Fr(a, s**3))
    integral = _trig_x("1", "1", a, b, "cos", 1)
    stem = msay(
        "Let ",
        X("J = ", "J = "),
        integral,
        ". The substitution ",
        X("z = e^(iθ)", r"z = e^{i\theta}"),
        " turns J into a contour integral round |z| = 1 whose denominator is ",
        X(f"{b}z² + {2 * a}z + {b}", f"{b}z^{{2}} + {2 * a}z + {b}"),
        ".",
    )
    z1 = Fr(s - a, b)
    w_a = Link(
        msay(
            "Find the positive number ",
            X("s = √(a² - b²)", r"s = \sqrt{a^{2}-b^{2}}"),
            " for ",
            X(f"a = {a}, b = {b}", f"a = {a},\\ b = {b}"),
            ".",
        ),
        s,
        "number",
        1,
        [f"√({a}² - {b}²) = √{a * a - b * b} = {s}."],
        [
            (a, "cx2-trig-root-slip"),
            (b, "cx2-trig-root-slip"),
            (math.isqrt(a * a + b * b), "cx2-trig-denominator-sign"),
            (a - b, "cx2-trig-root-slip"),
        ],
        sensible=(0, 100),
    )
    w_b = Link(
        msay(
            "The roots of ",
            X(f"{b}z² + {2 * a}z + {b}", f"{b}z^{{2}} + {2 * a}z + {b}"),
            " are ",
            X("(-a ± s)/b", r"\frac{-a \pm s}{b}"),
            ", where s is your answer to part (a). Find the root that lies inside the circle |z| = 1.",
        ),
        qsym(z1),
        "fraction",
        2,
        [
            f"The roots are ({-a} + {s})/{b} = {atext(qsym(z1))} and ({-a} - {s})/{b} = {atext(qsym(Fr(-a - s, b)))}; their product is 1, so exactly one is inside."
        ],
        [
            (qsym(Fr(-a - s, b)), "cx2-trig-root-slip"),
            (-qsym(z1), "calc-sign-slip"),
            (qsym(Fr(a - s, b)), "calc-sign-slip"),
            (qsym(Fr(1, b)), "cx2-trig-root-slip"),
        ],
        ft="(part_a - " + str(a) + ")/" + str(b),
    )
    w_c = Link(
        msay("Hence find the exact value of J."),
        value,
        "expression",
        3,
        [f"J = 2π/s = 2π/{s} = {atext(value)}."],
        [
            (_PI / s, "cx2-trig-jacobian-dropped"),
            (4 * _PI / s, "cx2-real-integral-doubled"),
            (2 * _PI / a, "cx2-trig-b-ignored"),
            (2 * _PI * s, "calc-ratio-inverted"),
        ],
        ft="2*pi/part_a",
    )
    w_d = Link(
        msay(
            "Differentiating J with respect to a (with b fixed) gives ",
            X("-∫_0^(2π) dθ/(a + b cos θ)²", r"-\int_{0}^{2\pi}\frac{\mathrm{d}\theta}{(a+b\cos\theta)^{2}}"),
            ". Hence find the exact value of ",
            X(
                f"∫_0^(2π) dθ/({a} + {b}cos θ)²",
                rf"\int_{{0}}^{{2\pi}}\frac{{\mathrm{{d}}\theta}}{{({a}+{b}\cos\theta)^{{2}}}}",
            ),
            ".",
        ),
        square,
        "expression",
        3,
        [f"J(a) = 2π(a² - b²)^(-1/2), so -dJ/da = 2πa(a² - b²)^(-3/2) = 2π·{a}/{s}³ = {atext(square)}."],
        [
            (2 * _PI * qsym(Fr(1, s**2)), "cx2-trig-double-pole-slip"),
            (-square, "calc-sign-slip"),
            (_PI * qsym(Fr(a, s**3)), "cx2-trig-jacobian-dropped"),
            (value**2 / (2 * _PI), "cx2-trig-double-pole-slip"),
        ],
        ft="2*pi*" + str(a) + "/part_a**3",
    )
    del z1
    return exam_dict(stem, [w_a, w_b, w_c, w_d])


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w5a_real_integrals(rng: random.Random, tier: str = "exam") -> dict:
    """A four-part chain: residues and the integral over the real line, or the substitution z = e^(i theta)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return (_exam_line if rng.random() < 0.5 else _exam_trig)(rng)
