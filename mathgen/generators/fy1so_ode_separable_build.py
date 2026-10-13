"""Generates `fy1so_ode_separable` (easy, medium, hard) and the exam chain `fy1so_exd_ode_separable`.

Skill ``ucv1-ode-separable-product`` (domain ``ordinary-differential-equations``, answer ``expression``): genuinely separable
first-order ODEs dy/dx = g(x) h(y) with an initial condition. The existing ``solve-ode-separable-poly`` is only dy/dx = f(x).

Tiers (the reasoning escalates):

- ``easy``: h(y) = y, so dy/y = g(x) dx: dy/dx = a x y, dy/dx = a x² y with y(0) given, or x dy/dx = a y with y(1) given. The
  answer is y0 e^(...) or y0 x^a, so ln y has to be exponentiated and the initial condition used as a factor.
- ``medium``: h(y) = y² (the reciprocal -1/y), h(y) = 1/y (so y dy = a x dx and y² has to be square-rooted) or h(y) = e^(-y)
  (e^y dy = a x dx, so the solution is a logarithm).
- ``hard``: the logistic equation dy/dx = k y (M - y), which needs partial fractions, or dy/dx = a x (1 + y²) with y(0) = 0, whose
  solution is a tangent.
- ``exam`` (topic ``fy1so_exd_ode_separable``): four parts on one of dy/dx = a x y² (the solution blows up) or dy/dx = -a x / y
  (the solution stops at y = 0). (a) the constant C from y(0) (b) y at a given x, exactly (follows through from (a))
  (c) the x at which the solution blows up or stops, to 3 decimal places (follows through from (a)) (d) the x at which y reaches
  a stated value, to 3 decimal places (follows through from (a)). Every part is a single value, because a follow-through from an
  expression reply cannot be marked (`reply_value` reads the learner's `e` as a plain symbol).

Every solution is checked a second way: the solution, written out again as a python function, is substituted back into the
differential equation with a central-difference slope at several points, and into the initial condition.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, rational_text
from ._fy1so_common import Link, M, build_part, build_single, rounds_clear, say

TOPIC = "fy1so_ode_separable"
EXAM_TOPIC = "fy1so_exd_ode_separable"

x = sympy.Symbol("x")
_PROBES = (0.05, 0.1, 0.15, 0.2, 0.25)


def _R(value) -> sympy.Rational:
    value = Fraction(value)
    return sympy.Rational(value.numerator, value.denominator)


def _check_solution(y, rate, x0, y0) -> None:
    """Second route: the python function `y` satisfies y' = rate(x, y) at several points, and y(x0) = y0."""
    h = 1e-6
    for t in _PROBES:
        slope = (y(t + h) - y(t - h)) / (2 * h)
        want = rate(t, y(t))
        check(abs(slope - want) < 1e-5 * (1 + abs(want)), "the solution does not satisfy the differential equation")
    check(abs(y(float(x0)) - y0) < 1e-9, "the solution does not satisfy the initial condition")


def _num(c, body: str) -> str:
    """`c body`, with the coefficient 1 left out."""
    return body if c == 1 else f"{c}{body}"


# -- easy: h(y) = y ---------------------------------------------------------------------------------------------------


def _easy(rng: random.Random):
    kind = rng.choice(["x", "x2", "power"])
    y0 = rng.randint(1, 6)
    if kind == "x":
        a = rng.choice([2, 4, 6])
        c = a // 2
        y = y0 * sympy.exp(c * x**2)
        _check_solution(lambda t: y0 * math.exp(c * t * t), lambda t, v: a * t * v, 0, y0)
        wrongs = [
            (y0 * sympy.exp(a * x**2), "ode2-power-rule-not-divided"),
            (sympy.exp(c * x**2), "ode2-ic-constant-dropped"),
            (y0 + c * x**2, "ode2-ln-not-exponentiated"),
            (y0 + sympy.exp(c * x**2), "ode2-constant-added-not-multiplied"),
            (y0 * sympy.exp(a * x), "ode2-power-rule-misapplied"),
        ]
        ode = rf"\frac{{dy}}{{dx}} = {_num(a, 'xy')}"
        cond, point = f"y(0) = {y0}", 0
        working = [f"dy/y = {a}x dx, so ln y = {c}x² + C.", f"y = Ae^({c}x²), and y(0) = {y0} gives A = {y0}."]
        shown = f"dy/dx = {a}xy"
    elif kind == "x2":
        a = rng.choice([3, 6, 9])
        c = a // 3
        y = y0 * sympy.exp(c * x**3)
        _check_solution(lambda t: y0 * math.exp(c * t**3), lambda t, v: a * t * t * v, 0, y0)
        wrongs = [
            (y0 * sympy.exp(a * x**3), "ode2-power-rule-not-divided"),
            (sympy.exp(c * x**3), "ode2-ic-constant-dropped"),
            (y0 + c * x**3, "ode2-ln-not-exponentiated"),
            (y0 * sympy.exp(c * x**2), "ode2-power-rule-misapplied"),
            (y0 + sympy.exp(c * x**3), "ode2-constant-added-not-multiplied"),
        ]
        ode = rf"\frac{{dy}}{{dx}} = {_num(a, 'x^{2}y')}"
        cond, point = f"y(0) = {y0}", 0
        working = [f"dy/y = {a}x² dx, so ln y = {c}x³ + C.", f"y = Ae^({c}x³), and y(0) = {y0} gives A = {y0}."]
    else:
        a = rng.choice([2, 3, 4, 5])
        y = y0 * x**a
        _check_solution(lambda t: y0 * t**a, lambda t, v: a * v / t, 1, y0)
        wrongs = [
            (y0 * x ** (a + 1), "ode2-power-rule-misapplied"),
            (a * sympy.log(x) + y0, "ode2-ln-not-exponentiated"),
            (x**a, "ode2-ic-constant-dropped"),
            (y0 + x**a, "ode2-constant-added-not-multiplied"),
            (y0 * sympy.exp(a * x), "ode2-power-rule-misapplied"),
        ]
        ode = rf"x\,\frac{{dy}}{{dx}} = {_num(a, 'y')}"
        cond, point = f"y(1) = {y0}", 1
        working = [f"dy/y = {a} dx/x, so ln y = {a} ln x + C, giving y = Ax^{a}.", f"y(1) = {y0} gives A = {y0}."]
    domain = ", x > 0" if kind == "power" else ""
    prompt = say(
        "Solve the initial value problem ",
        M(ode),
        domain,
        ", ",
        M(cond),
        ", where y > 0. Give y as a function of x.",
    )
    del point
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


# -- medium: y^2, 1/y, e^(-y) -----------------------------------------------------------------------------------------


def _medium(rng: random.Random):
    return rng.choice([_medium_square, _medium_reciprocal, _medium_exponential])(rng)


def _medium_square(rng: random.Random):
    for _ in attempts(100, "an initial value for y squared"):
        c = rng.choice([1, 2, 3])
        y0 = rng.choice([-3, -2, -1, 1, 2, 3])
        if y0 * c <= 3:
            break
    a = 2 * c
    y = y0 / (1 - c * y0 * x**2)
    _check_solution(lambda t: 1 / (1 / y0 - c * t * t), lambda t, v: a * t * v * v, 0, y0)
    wrongs = [
        (y0 / (1 + c * y0 * x**2), "ode2-reciprocal-sign"),
        (y0 / (1 - a * y0 * x**2), "ode2-power-rule-not-divided"),
        (y0 * sympy.exp(c * x**2), "ode2-y-power-ignored"),
        (y0 / (1 - c * x**2), "ode2-ic-constant-misapplied"),
        (-y0 / (1 - c * y0 * x**2), "ode2-reciprocal-forgotten"),
    ]
    prompt = say(
        "Solve the initial value problem ",
        M(rf"\frac{{dy}}{{dx}} = {_num(a, 'xy^{2}')}"),
        ", ",
        M(f"y(0) = {y0}"),
        ". Give y as a function of x for x close to 0.",
    )
    working = [
        f"dy/y² = {a}x dx, so -1/y = {c}x² + C.",
        f"y(0) = {y0} gives C = {rational_text(Fraction(-1, y0))}, so y = 1/({rational_text(Fraction(1, y0))} - {c}x²).",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


def _medium_reciprocal(rng: random.Random):
    a = rng.choice([2, 4, 6, 8])
    y0 = rng.randint(1, 5)
    y = sympy.sqrt(y0**2 + a * x**2)
    _check_solution(lambda t: math.sqrt(y0**2 + a * t * t), lambda t, v: a * t / v, 0, y0)
    wrongs = [
        (sympy.sqrt(y0 + a * x**2), "ode2-ic-squared-forgotten"),
        (sympy.sqrt(y0**2 + a * x**2 / 2), "ode2-factor-two-mishandled"),
        (sympy.sqrt(y0**2 - a * x**2), "ode2-sign-error-root"),
        (y0 + a * x**2 / 2, "ode2-y-dy-integrated-as-y"),
        (sympy.sqrt(y0**2 + 2 * a * x**2), "ode2-factor-two-mishandled"),
    ]
    prompt = say(
        "Solve the initial value problem ",
        M(rf"\frac{{dy}}{{dx}} = \frac{{{_num(a, 'x')}}}{{y}}"),
        ", ",
        M(f"y(0) = {y0}"),
        ", where y > 0. Give y as a function of x.",
    )
    working = [
        f"y dy = {a}x dx, so y²/2 = {a}x²/2 + C.",
        f"y(0) = {y0} gives y² = {a}x² + {y0**2}, and y > 0 takes the positive root.",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


def _medium_exponential(rng: random.Random):
    c = rng.choice([1, 2, 3, 4])
    a = 2 * c
    y = sympy.log(1 + c * x**2)
    _check_solution(lambda t: math.log(1 + c * t * t), lambda t, v: a * t * math.exp(-v), 0, 0)
    wrongs = [
        (sympy.log(1 + a * x**2), "ode2-power-rule-not-divided"),
        (sympy.log(c * x**2), "ode2-integration-constant-dropped"),
        (sympy.log(c * x**2 + 1) + 1, "ode2-ic-constant-misapplied"),
        (c * x**2, "ode2-ln-not-exponentiated"),
        (sympy.exp(c * x**2), "ode2-exponential-inverted"),
    ]
    prompt = say(
        "Solve the initial value problem ",
        M(rf"\frac{{dy}}{{dx}} = {_num(a, 'x')}\,e^{{-y}}"),
        ", ",
        M("y(0) = 0"),
        ". Give y as a function of x.",
    )
    working = [f"e^y dy = {a}x dx, so e^y = {c}x² + C.", "y(0) = 0 gives C = 1, so y = ln(1 + " + f"{c}x²)."]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


# -- hard: partial fractions, and a tangent ---------------------------------------------------------------------------


def _hard(rng: random.Random):
    return _hard_logistic(rng) if rng.random() < 0.65 else _hard_tangent(rng)


def _hard_logistic(rng: random.Random):
    for _ in attempts(100, "a logistic setting"):
        big = rng.choice([2, 3, 4, 5, 6])
        k = rng.choice([1, 2, 3])
        y0 = rng.randint(1, big - 1)
        if y0 * 2 != big:
            break
    rate = k * big
    g = math.gcd(y0, big)

    def curve(n, p, q, e):
        return sympy.Integer(n // g) / (sympy.Integer(p // g) + sympy.Integer(q // g) * sympy.exp(e * x))

    y = curve(big * y0, y0, big - y0, -rate)
    _check_solution(lambda t: big * y0 / (y0 + (big - y0) * math.exp(-rate * t)), lambda t, v: k * v * (big - v), 0, y0)
    wrongs = [
        (curve(big * y0, y0, big - y0, -k), "ode2-partial-fraction-factor-missed"),
        (curve(big * y0, y0, -(big - y0), -rate), "ode2-partial-fraction-sign"),
        (curve(big * g, g, y0 * g, -rate), "ode2-ic-constant-misapplied"),
        (curve(big * y0, y0, big - y0, rate), "ode2-exponent-sign"),
        (curve(big * y0, big, big - y0, -rate), "ode2-ic-constant-misapplied"),
    ]
    prompt = say(
        "Solve the initial value problem ",
        M(rf"\frac{{dy}}{{dx}} = {_num(k, 'y')}({big} - y)"),
        ", ",
        M(f"y(0) = {y0}"),
        f", where 0 < y < {big}. Give y as a function of x.",
    )
    working = [
        f"dy / (y({big} - y)) = {k} dx. By partial fractions, 1/(y({big} - y)) = (1/{big})(1/y + 1/({big} - y)).",
        f"So (1/{big}) ln(y/({big} - y)) = {k}x + C, giving y/({big} - y) = Ae^({rate}x).",
        f"y(0) = {y0} gives A = {rational_text(Fraction(y0, big - y0))}, and y = {big * y0}/({y0} + {big - y0}e^(-{rate}x)).",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


def _hard_tangent(rng: random.Random):
    c = rng.choice([1, 2, 3])
    a = 2 * c
    y = sympy.tan(c * x**2)
    _check_solution(lambda t: math.tan(c * t * t), lambda t, v: a * t * (1 + v * v), 0, 0)
    wrongs = [
        (sympy.tan(a * x**2), "ode2-power-rule-not-divided"),
        (c * x**2, "ode2-arctan-not-inverted"),
        (sympy.atan(c * x**2), "ode2-arctan-inverse-confused"),
        (sympy.tan(c * x**2 + 1), "ode2-ic-constant-misapplied"),
        (sympy.tan(c * x), "ode2-power-rule-misapplied"),
    ]
    prompt = say(
        "Solve the initial value problem ",
        M(rf"\frac{{dy}}{{dx}} = {_num(a, 'x')}(1 + y^{{2}})"),
        ", ",
        M("y(0) = 0"),
        ". Give y as a function of x for x close to 0.",
    )
    working = [
        f"dy / (1 + y²) = {a}x dx, so arctan y = {c}x² + C.",
        f"y(0) = 0 gives C = 0, so y = tan({c}x²).",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1so_ode_separable(rng: random.Random, tier: str = "easy") -> dict:
    """A separable ODE with a product right-hand side: exponential, reciprocal, logarithmic, logistic or tangent solution."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    prompt, link = {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)
    return build_single(prompt, link)


# -- exam: a four-part chain -----------------------------------------------------------------------------------------


def _frac_text(f: Fraction) -> str:
    return f"({f.numerator}/{f.denominator})"


def _places_clear(value) -> bool:
    return rounds_clear(value, 3)


def _exam_blow_up(rng: random.Random):
    """dy/dx = a x y^2, y(0) = y0 > 0: -1/y = (a/2)x^2 + C, blowing up at x* = sqrt(2/(a y0))."""
    for _ in attempts(500, "an exam blow-up draw"):
        a = rng.choice([1, 2, 3, 4, 6])
        y0 = rng.choice([Fraction(1, 2), Fraction(2), Fraction(3), Fraction(1, 3), Fraction(3, 2)])
        x1 = rng.choice([Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)])
        yt = y0 * rng.choice([2, 3, 4, 5])
        c = Fraction(-1) / y0
        star2 = -2 * c / a
        if a * x1 * x1 / 2 >= -c * Fraction(9, 10):
            continue
        y1 = -1 / (a * x1 * x1 / 2 + c)
        xt2 = 2 * (-c - 1 / yt) / a
        star, xt = math.sqrt(star2), math.sqrt(xt2)
        if _places_clear(star) and _places_clear(xt) and star > 0.15 and x1 < star:
            break
    _check_solution(lambda t: 1 / (1 / float(y0) - a * t * t / 2), lambda t, v: a * t * v * v, 0, float(y0))
    check(abs(float(y1) - 1 / (1 / float(y0) - a * float(x1) ** 2 / 2)) < 1e-9, "y(x1) disagrees")
    stem = say(
        "Consider the initial value problem ",
        M(rf"\frac{{dy}}{{dx}} = {_num(a, 'xy^{2}')}"),
        ", ",
        M(f"y(0) = {rational_text(y0)}"),
        ". Separating variables and integrating gives ",
        M(rf"-\frac{{1}}{{y}} = {_coef_half(a)}x^{{2}} + C"),
        ".",
    )
    x1_text = rational_text(x1)
    p_a = say("Use the initial condition to find the value of ", M("C"), ".")
    p_b = say("Hence find ", M(f"y({x1_text})"), " exactly, as a fraction.")
    p_c = say(
        "The solution becomes unbounded as ",
        M("x"),
        " increases to a value ",
        M("X"),
        ". Find ",
        M("X"),
        ", correct to 3 decimal places.",
    )
    p_d = say(
        "Find the value of ", M("x"), f" at which y first reaches {rational_text(yt)}, correct to 3 decimal places."
    )
    ay = _R(a)
    a_wrongs = [
        (-y0, "ode2-reciprocal-forgotten"),
        (1 / y0, "ode2-reciprocal-sign"),
        (y0, "ode2-reciprocal-forgotten"),
        (-1 / (y0 * y0), "ode2-reciprocal-squared"),
    ]
    b_wrongs = [
        (_safe(lambda: 1 / (1 / y0 + a * x1 * x1 / 2)), "ode2-reciprocal-sign"),
        (_safe(lambda: 1 / (1 / y0 - a * x1 * x1)), "ode2-factor-two-mishandled"),
        (_safe(lambda: y0 / (1 - a * x1 * x1 / 2)), "ode2-ic-constant-misapplied"),
        (_safe(lambda: -1 / (a * x1 * x1 / 2 + y0)), "ode2-reciprocal-forgotten"),
        (y0 + a * x1 * x1 / 2, "ode2-ln-not-exponentiated"),
    ]
    c_wrongs = [
        (math.sqrt(1 / float(a * y0)), "ode2-factor-two-mishandled"),
        (2 / float(a * y0), "ode2-square-root-forgotten"),
        (math.sqrt(float(2 * y0 / a)), "ode2-reciprocal-forgotten"),
        (math.sqrt(2 / float(a * y0)) * 2, "ode2-factor-two-mishandled"),
        (math.sqrt(float(1 / (a * y0)) * 4), "ode2-factor-two-mishandled"),
    ]
    d_wrongs = [
        (math.sqrt(float(2 * (yt - y0) / a)), "ode2-reciprocal-forgotten"),
        (math.sqrt(float((1 / y0 - 1 / yt) / a)), "ode2-factor-two-mishandled"),
        (float(2 * (1 / y0 - 1 / yt) / a), "ode2-square-root-forgotten"),
        (math.sqrt(float(2 * (1 / y0 + 1 / yt) / a)), "ode2-sign-error-root"),
        (math.sqrt(float(2 * (1 / y0 - 1 / yt) / a) * 4), "ode2-factor-two-mishandled"),
    ]
    del ay
    n1, d1 = x1.numerator, x1.denominator
    yt_text = _frac_text(yt)
    bound = int(4 * float(y0 + y1 + abs(c)) + 10)
    links = [
        Link(
            p_a,
            "fraction",
            c,
            1,
            [f"-1/{rational_text(y0)} = C, so C = {rational_text(c)}."],
            a_wrongs,
            sensible=(-bound, bound),
        ),
        Link(
            p_b,
            "fraction",
            y1,
            2,
            [f"y = -1/({rational_text(Fraction(a, 2))}x² + C), so y({x1_text}) = {rational_text(y1)}."],
            b_wrongs,
            ft=f"-1/({a}*({n1}/{d1})**2/2 + part_a)",
            sensible=(-bound, bound),
        ),
        Link(
            p_c,
            "number",
            star,
            2,
            [f"y is unbounded when {rational_text(Fraction(a, 2))}x² + C = 0, so x* = sqrt(-2C/{a}) = {star:.5f}."],
            c_wrongs,
            ft=f"sqrt(-2*part_a/{a})",
            sensible=(0, 50),
            places=3,
        ),
        Link(
            p_d,
            "number",
            xt,
            2,
            [
                f"-1/{rational_text(yt)} = {rational_text(Fraction(a, 2))}x² + C, so x = sqrt(2(-C - 1/{rational_text(yt)})/{a}) = {xt:.5f}."
            ],
            d_wrongs,
            ft=f"sqrt(2*(-part_a - 1/{yt_text})/{a})",
            sensible=(0, 50),
            places=3,
        ),
    ]
    return stem, links


def _coef_half(a: int) -> str:
    """The coefficient a/2 written before x^2: 3x^2, x^2 or (3/2)x^2."""
    if a % 2:
        return rf"\left(\frac{{{a}}}{{2}}\right)"
    return "" if a == 2 else str(a // 2)


def _safe(fn):
    """The value of `fn()`, or None when it divides by zero (a wrong answer that cannot be formed is simply not offered)."""
    try:
        return fn()
    except ZeroDivisionError:
        return None


def _exam_stop(rng: random.Random):
    """dy/dx = -a x / y, y(0) = y0 > 0: y^2/2 = -(a/2)x^2 + C, reaching y = 0 at x* = y0/sqrt(a)."""
    for _ in attempts(2000, "an exam circle draw"):
        a = rng.choice([1, 2, 3, 4, 5])
        y0 = rng.randint(3, 13)
        x1 = rng.randint(1, 6)
        yt = rng.randint(1, y0 - 1)
        s2 = y0 * y0 - a * x1 * x1
        if s2 <= 0 or math.isqrt(s2) ** 2 != s2:
            continue
        star, xt = y0 / math.sqrt(a), math.sqrt((y0 * y0 - yt * yt) / a)
        if _places_clear(star) and _places_clear(xt) and xt > 0.2 and abs(star - round(star)) > 1e-9:
            break
    c = Fraction(y0 * y0, 2)
    s = math.isqrt(s2)
    _check_solution(lambda t: math.sqrt(y0 * y0 - a * t * t), lambda t, v: -a * t / v, 0, y0)
    stem = say(
        "Consider the initial value problem ",
        M(rf"\frac{{dy}}{{dx}} = -\frac{{{_num(a, 'x')}}}{{y}}"),
        ", ",
        M(f"y(0) = {y0}"),
        ", where y > 0. Separating variables and integrating gives ",
        M(rf"\frac{{y^{{2}}}}{{2}} = -{_coef_half(a)}x^{{2}} + C"),
        ".",
    )
    p_a = say("Use the initial condition to find the value of ", M("C"), ".")
    p_b = say("Hence find ", M(f"y({x1})"), " exactly.")
    p_c = say(
        "The solution is defined for 0 ≤ x < ",
        M("X"),
        ", where y reaches 0. Find ",
        M("X"),
        ", correct to 3 decimal places.",
    )
    p_d = say("Find the value of ", M("x"), f" at which y = {yt}, correct to 3 decimal places.")
    a_wrongs = [
        (y0 * y0, "ode2-half-forgotten"),
        (Fraction(y0, 2), "ode2-ic-squared-forgotten"),
        (-c, "ode2-sign-error-root"),
        (y0, "ode2-ic-squared-forgotten"),
    ]
    b_wrongs = [
        (s2, "ode2-square-root-forgotten"),
        (Fraction(s2, 2), "ode2-factor-two-mishandled"),
        (Fraction(y0) - Fraction(a * x1 * x1, 2), "ode2-y-dy-integrated-as-y"),
        (Fraction(y0 - a * x1), "ode2-power-rule-misapplied"),
        (Fraction(y0 + s, 2), "ode2-sign-error-root"),
    ]
    c_wrongs = [
        (math.sqrt(y0 * y0 / (2 * a)), "ode2-factor-two-mishandled"),
        (y0 * y0 / a, "ode2-square-root-forgotten"),
        (math.sqrt(y0 / a), "ode2-ic-squared-forgotten"),
        (y0 / a, "ode2-square-root-forgotten"),
        (math.sqrt(2 * y0 * y0 / a), "ode2-half-forgotten"),
    ]
    d_wrongs = [
        (math.sqrt((y0 * y0 - yt) / a), "ode2-ic-squared-forgotten"),
        (math.sqrt((y0 * y0 - yt * yt) / (2 * a)), "ode2-factor-two-mishandled"),
        ((y0 * y0 - yt * yt) / a, "ode2-square-root-forgotten"),
        (math.sqrt((y0 * y0 + yt * yt) / a), "ode2-sign-error-root"),
        (math.sqrt((y0 - yt) / a), "ode2-ic-squared-forgotten"),
    ]
    bound = int(4 * (y0 * y0 + 5))
    links = [
        Link(
            p_a,
            "fraction",
            c,
            1,
            [f"{y0}²/2 = C, so C = {rational_text(c)}."],
            a_wrongs,
            sensible=(-bound, bound),
        ),
        Link(
            p_b,
            "fraction",
            s,
            2,
            [f"y² = 2C - {a}x² = {s2}, so y({x1}) = {s} (taking y > 0)."],
            b_wrongs,
            ft=f"sqrt(2*part_a - {a}*{x1}**2)",
            sensible=(0, bound),
        ),
        Link(
            p_c,
            "number",
            star,
            2,
            [f"y = 0 when {a}x² = 2C, so x* = sqrt(2C/{a}) = {star:.5f}."],
            c_wrongs,
            ft=f"sqrt(2*part_a/{a})",
            sensible=(0, 50),
            places=3,
        ),
        Link(
            p_d,
            "number",
            xt,
            2,
            [f"{yt}² = 2C - {a}x², so x = sqrt((2C - {yt * yt})/{a}) = {xt:.5f}."],
            d_wrongs,
            ft=f"sqrt((2*part_a - {yt}**2)/{a})",
            sensible=(0, 50),
            places=3,
        ),
    ]
    return stem, links


@generator(EXAM_TOPIC, level=19, presets={"exam": {}})
def fy1so_exd_ode_separable(rng: random.Random) -> dict:
    """A separable IVP: the constant, y at a point, where the solution ends, and where it reaches a stated value."""
    stem, links = _exam_blow_up(rng) if rng.random() < 0.5 else _exam_stop(rng)
    return dict(text=stem[0], latex=stem[1], parts=[build_part(link) for link in links])
