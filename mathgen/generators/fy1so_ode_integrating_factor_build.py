"""Generates `fy1so_ode_integrating_factor` (easy, medium, hard) and the exam chain `fy1so_exd_ode_integrating_factor`.

Skill ``ucv1-first-order-linear-integrating-factor`` (domain ``ordinary-differential-equations``, answer ``expression``):
the first-order linear ODE y' + P(x) y = Q(x) solved with the integrating factor mu = e^(integral of P).

Tiers (the reasoning escalates):

- ``easy``: the integrating factor itself, for P a constant a, P = k/x (x > 0) or P = 2cx.
- ``medium``: the solution of an initial value problem y' + a y = b e^(kx) (the coefficient of e^(kx) has to be divided by
  a + k, and the constant fixed from y(0)) or y' + a y = b (a constant solution plus a decaying term).
- ``hard``: either x y' + k y = c x^m for x > 0 with y(1) given (divide by x before the integrating factor x^k is read off,
  and the right-hand side becomes c x^(m-1)), or the resonant y' + a y = b e^(-ax) with y(0) given, whose integrated right-hand
  side is a constant, so the solution carries a factor x.
- ``exam`` (topic ``fy1so_exd_ode_integrating_factor``): y' + a y = p x + r, y(0) = y0, whose general solution is
  y = Ax + B + Ce^(-ax). (a) A, (b) B (integration by parts), (c) C from y(0) (follows through from (b)), (d) y(x1) to three
  decimal places (follows through from (a), (b) and (c)). The parts are single values, because a follow-through from an
  expression reply cannot be marked: `reply_value` reads the learner's `e` as a plain symbol, not Euler's number.

Every solution is checked a second way, by substituting it back into the differential equation at several points and
into the initial condition, with SymPy's derivative.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, rational_text
from ._fy1so_common import (
    Link,
    M,
    build_part,
    build_single,
    rounds_clear,
    say,
    signed_tex,
    tex_decimal,
)

TOPIC = "fy1so_ode_integrating_factor"
EXAM_TOPIC = "fy1so_exd_ode_integrating_factor"

x = sympy.Symbol("x")
C = sympy.Symbol("C")
_PROBES = (0.7, 1.3, 1.9, 3.1)


def _R(value) -> sympy.Rational:
    value = Fraction(value)
    return sympy.Rational(value.numerator, value.denominator)


def _check_solution(y, coefficient, lead, q_of_x, x0, y0) -> None:
    """Second route: the python function `y` (written out again, with a central-difference slope) satisfies
    lead(x) y' + coefficient(x) y = q(x) at several points, and y(x0) = y0."""
    h = 1e-6
    for t in _PROBES:
        slope = (y(t + h) - y(t - h)) / (2 * h)
        residual = lead(t) * slope + coefficient(t) * y(t) - q_of_x(t)
        check(abs(residual) < 1e-5 * (1 + abs(q_of_x(t))), "the solution does not satisfy the differential equation")
    check(abs(y(float(x0)) - y0) < 1e-9, "the solution does not satisfy the initial condition")


def _exp_tex(k: int | Fraction) -> str:
    k = Fraction(k)
    if k == 1:
        return "e^{x}"
    if k == -1:
        return "e^{-x}"
    return f"e^{{{_coef(k)}x}}"


def _coef(k) -> str:
    k = Fraction(k)
    return str(k.numerator) if k.denominator == 1 else f"\\frac{{{k.numerator}}}{{{k.denominator}}}"


def _lhs_tex(a: int) -> str:
    return rf"\frac{{dy}}{{dx}} {signed_tex(a, 'y')}"


def _num(c: int, body: str) -> str:
    """`c body` with c = 1 left out."""
    return body if c == 1 else f"-{body}" if c == -1 else f"{c}{body}"


# -- easy: the integrating factor ------------------------------------------------------------------------------------


def _easy(rng: random.Random):
    kind = rng.choice(["const", "const", "power", "gauss"])
    if kind == "const":
        a = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
        q_tex = rng.choice(["x", "x^{2}", "3", r"\sin x", "2x + 1"])
        mu = sympy.exp(a * x)
        lhs = _lhs_tex(a)
        domain = ""
        wrongs = [
            (sympy.exp(-a * x), "ode1-factor-sign-flipped"),
            (sympy.exp(a), "ode1-factor-integral-skipped"),
            (a * x, "ode1-factor-integral-not-exponentiated"),
            (sympy.exp(x / a) if abs(a) != 1 else sympy.exp(2 * a * x), "ode1-factor-coefficient-inverted"),
        ]
        working = [f"P(x) = {a}, so the integral of P is {a}x.", f"The integrating factor is e^({a}x)."]
    elif kind == "power":
        k = rng.randint(2, 6)
        q_tex = rng.choice(["x", "x^{2}", "3", r"\ln x"])
        mu = x**k
        lhs = rf"\frac{{dy}}{{dx}} + \left(\frac{{{k}}}{{x}}\right)y"
        domain = " for x > 0"
        wrongs = [
            (x ** (-k), "ode1-factor-sign-flipped"),
            (sympy.exp(sympy.Rational(k) / x), "ode1-factor-integral-skipped"),
            (k * sympy.log(x), "ode1-factor-integral-not-exponentiated"),
            (x, "ode1-factor-coefficient-dropped"),
            (x ** (k + 1), "ode1-factor-power-rule-misapplied"),
        ]
        working = [
            f"P(x) = {k}/x, so the integral of P is {k} ln x (x > 0).",
            f"The integrating factor is e^({k} ln x) = x^{k}.",
        ]
    else:
        c = rng.randint(1, 4)
        q_tex = rng.choice(["x", "3", r"x^{2}", "e^{x}"])
        mu = sympy.exp(c * x**2)
        lhs = rf"\frac{{dy}}{{dx}} + {2 * c}x\,y"
        domain = ""
        wrongs = [
            (sympy.exp(2 * c * x), "ode1-factor-integral-skipped"),
            (sympy.exp(2 * c * x**2), "ode1-factor-power-rule-misapplied"),
            (sympy.exp(-c * x**2), "ode1-factor-sign-flipped"),
            (c * x**2, "ode1-factor-integral-not-exponentiated"),
            (sympy.exp(c * x), "ode1-factor-power-rule-misapplied"),
        ]
        working = [f"P(x) = {2 * c}x, so the integral of P is {c}x².", f"The integrating factor is e^({c}x²)."]
    asks = [
        "Find the integrating factor that turns the left-hand side into the derivative of a product.",
        "Find an integrating factor for this equation.",
        "What integrating factor would you use to solve it?",
    ]
    prompt = say("Consider the differential equation ", M(rf"{lhs} = {q_tex}"), domain + ". ", rng.choice(asks))
    return prompt, Link(prompt, "expression", mu, working=working, wrongs=wrongs, symbols=(x,))


# -- medium: an initial value problem with constant coefficients ------------------------------------------------------


def _medium(rng: random.Random):
    if rng.random() < 0.6:
        return _medium_exponential(rng)
    return _medium_constant(rng)


def _medium_exponential(rng: random.Random):
    for _ in attempts(200, "a non-resonant exponential right-hand side"):
        a = rng.choice([1, 2, 3, 4])
        k = rng.choice([-2, -1, 1, 2, 3])
        amp = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        y0 = rng.randint(-4, 6)
        if k == -a or k == a or y0 == amp or y0 == 0:
            continue
        break
    b = amp * (a + k)
    const = y0 - amp
    y = amp * sympy.exp(k * x) + const * sympy.exp(-a * x)
    _check_solution(
        lambda t: amp * math.exp(k * t) + const * math.exp(-a * t),
        lambda t: a,
        lambda t: 1,
        lambda t: b * math.exp(k * t),
        0,
        y0,
    )
    wrongs = [
        (b * sympy.exp(k * x) + (y0 - b) * sympy.exp(-a * x), "ode1-particular-coefficient-not-divided"),
        (amp * sympy.exp(k * x) + y0 * sympy.exp(-a * x), "ode1-ic-ignores-particular-part"),
        (amp * sympy.exp(k * x) + (amp - y0) * sympy.exp(-a * x), "ode1-ic-sign-error"),
        (amp * sympy.exp(k * x) + const * sympy.exp(a * x), "ode1-homogeneous-exponent-sign"),
        (
            _R(Fraction(b, a - k)) * sympy.exp(k * x) + (y0 - _R(Fraction(b, a - k))) * sympy.exp(-a * x),
            "ode1-particular-denominator-sign",
        ),
        (amp * sympy.exp(k * x) + const, "ode1-integrating-factor-division-skipped"),
    ]
    rhs = f"{_num(b, _exp_tex(k))}"
    prompt = say(
        "Solve the initial value problem ",
        M(rf"{_lhs_tex(a)} = {rhs}"),
        ", ",
        M(f"y(0) = {y0}"),
        ". Give y as a function of x.",
    )
    working = [
        f"The integrating factor is e^({a}x). Then (e^({a}x) y)' = {b}e^({a + k}x), so e^({a}x) y = {amp}e^({a + k}x) + C.",
        f"y = {amp}e^({k}x) + Ce^(-{a}x). From y(0) = {y0}: {amp} + C = {y0}, so C = {const}.",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


def _medium_constant(rng: random.Random):
    for _ in attempts(200, "a constant right-hand side"):
        a = rng.choice([1, 2, 3, 4, 5])
        m = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5])
        y0 = rng.randint(-5, 7)
        if y0 == m or y0 == 0:
            continue
        break
    b = a * m
    const = y0 - m
    y = m + const * sympy.exp(-a * x)
    _check_solution(lambda t: m + const * math.exp(-a * t), lambda t: a, lambda t: 1, lambda t: b, 0, y0)
    wrongs = [
        (b + (y0 - b) * sympy.exp(-a * x), "ode1-particular-coefficient-not-divided"),
        (m + y0 * sympy.exp(-a * x), "ode1-ic-ignores-particular-part"),
        (m + (m - y0) * sympy.exp(-a * x), "ode1-ic-sign-error"),
        (m + const * sympy.exp(a * x), "ode1-homogeneous-exponent-sign"),
        (m + const, "ode1-integrating-factor-division-skipped"),
    ]
    prompt = say(
        "Solve the initial value problem ",
        M(rf"{_lhs_tex(a)} = {b}"),
        ", ",
        M(f"y(0) = {y0}"),
        ". Give y as a function of x.",
    )
    working = [
        f"The integrating factor is e^({a}x). Then (e^({a}x) y)' = {b}e^({a}x), so e^({a}x) y = {m}e^({a}x) + C.",
        f"y = {m} + Ce^(-{a}x). From y(0) = {y0}: {m} + C = {y0}, so C = {const}.",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


# -- hard: a lead coefficient, or resonance --------------------------------------------------------------------------


def _hard(rng: random.Random):
    return _hard_lead(rng) if rng.random() < 0.55 else _hard_resonant(rng)


def _hard_lead(rng: random.Random):
    for _ in attempts(200, "a power right-hand side"):
        k = rng.choice([2, 3, 4])
        m = rng.choice([1, 2, 3])
        amp = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        y0 = rng.randint(-5, 7)
        if y0 == amp or y0 == 0:
            continue
        break
    c = amp * (k + m)
    const = y0 - amp
    y = amp * x**m + const * x ** (-k)
    _check_solution(lambda t: amp * t**m + const * t ** (-k), lambda t: k, lambda t: t, lambda t: c * t**m, 1, y0)
    wrongs = [
        (c * x**m + (y0 - c) * x ** (-k), "ode1-particular-coefficient-not-divided"),
        (amp * x**m + y0 * x ** (-k), "ode1-ic-ignores-particular-part"),
        (amp * x**m + (amp - y0) * x ** (-k), "ode1-ic-sign-error"),
        (amp * x**m + const * x**k, "ode1-homogeneous-exponent-sign"),
        (
            _R(Fraction(c, k + m + 1)) * x ** (m + 1) + (y0 - _R(Fraction(c, k + m + 1))) * x ** (-k),
            "ode1-lead-coefficient-not-divided",
        ),
    ]
    prompt = say(
        "Solve the initial value problem ",
        M(rf"x\,\frac{{dy}}{{dx}} + {k}y = {_num(c, 'x' if m == 1 else f'x^{{{m}}}')}"),
        ", ",
        M("x > 0"),
        ", ",
        M(f"y(1) = {y0}"),
        ". Give y as a function of x.",
    )
    working = [
        f"Divide by x: y' + ({k}/x) y = {c}x^{m - 1}. The integrating factor is x^{k}.",
        f"(x^{k} y)' = {c}x^{k + m - 1}, so x^{k} y = {amp}x^{k + m} + C, and y = {amp}x^{m} + Cx^(-{k}).",
        f"From y(1) = {y0}: {amp} + C = {y0}, so C = {const}.",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


def _hard_resonant(rng: random.Random):
    for _ in attempts(200, "a resonant forcing term"):
        a = rng.choice([1, 2, 3])
        b = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5, 6])
        y0 = rng.randint(-4, 6)
        if y0 == 0:
            continue
        break
    y = (b * x + y0) * sympy.exp(-a * x)
    _check_solution(
        lambda t: (b * t + y0) * math.exp(-a * t), lambda t: a, lambda t: 1, lambda t: b * math.exp(-a * t), 0, y0
    )
    wrongs = [
        (y0 * sympy.exp(-a * x), "ode1-resonance-factor-x-missing"),
        ((b * x + y0) * sympy.exp(a * x), "ode1-factor-sign-flipped"),
        (b * x * sympy.exp(-a * x), "ode1-ic-homogeneous-term-dropped"),
        ((sympy.Rational(b, 2) * x**2 + y0) * sympy.exp(-a * x), "ode1-integral-over-integrated"),
        ((b * x - y0) * sympy.exp(-a * x), "ode1-ic-sign-error"),
    ]
    prompt = say(
        "Solve the initial value problem ",
        M(rf"{_lhs_tex(a)} = {_num(b, f'e^{{-{a}x}}' if a != 1 else 'e^{-x}')}"),
        ", ",
        M(f"y(0) = {y0}"),
        ". Give y as a function of x.",
    )
    working = [
        f"The integrating factor is e^({a}x), so (e^({a}x) y)' = {b}, a constant.",
        f"Integrating, e^({a}x) y = {b}x + C. From y(0) = {y0}, C = {y0}, so y = ({b}x + {y0})e^(-{a}x).",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1so_ode_integrating_factor(rng: random.Random, tier: str = "easy") -> dict:
    """A first-order linear ODE: the integrating factor, an initial value problem, a lead coefficient or resonance."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    prompt, link = {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)
    return build_single(prompt, link)


# -- exam: a four-part chain -----------------------------------------------------------------------------------------

_X1 = [Fraction(1, 2), Fraction(1), Fraction(3, 2)]


def _exam(rng: random.Random):
    for _ in attempts(500, "an exam initial value problem"):
        a = rng.choice([2, 3])
        p = rng.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
        r = rng.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
        y0 = rng.randint(-4, 6)
        x1 = rng.choice(_X1)
        big_a = Fraction(p, a)
        big_b = (r - big_a) / a
        const = y0 - big_b
        if const == 0 or big_a.denominator > 3 or big_b == 0 or big_a == big_b:
            continue
        value = big_a * x1 + big_b + _R(const) * sympy.exp(-a * _R(x1))
        tail = abs(_R(const) * sympy.exp(-a * _R(x1)))
        if tail < 0.08 or abs(value) < 0.3:
            continue
        if rounds_clear(value, 3):
            break
    _check_solution(
        lambda t: float(big_a) * t + float(big_b) + float(const) * math.exp(-a * t),
        lambda t: a,
        lambda t: 1,
        lambda t: p * t + r,
        0,
        y0,
    )
    x1_text = "1" if x1 == 1 else str(float(x1))
    lhs = _lhs_tex(a)
    rhs = f"{_num(p, 'x')} {signed_tex(r, '')}".strip()
    stem = say(
        "Consider the initial value problem ",
        M(rf"{lhs} = {rhs}"),
        ", ",
        M(f"y(0) = {y0}"),
        ". Multiplying by the integrating factor ",
        M(f"e^{{{a}x}}"),
        " and integrating gives a general solution of the form ",
        M(rf"y = Ax + B + Ce^{{-{a}x}}"),
        ", where C is an arbitrary constant.",
    )
    p_a = say("Find the value of ", M("A"), ".")
    p_b = say("Find the value of ", M("B"), ".")
    p_c = say("Use the initial condition to find the value of ", M("C"), ".")
    p_d = say("Find ", M(f"y({x1_text})"), " correct to 3 decimal places.")
    a_wrongs = [
        (p, "ode1-particular-coefficient-not-divided"),
        (-big_a, "ode1-particular-coefficient-sign-flipped"),
        (Fraction(p, a * a), "ode1-particular-coefficient-divided-twice"),
        (Fraction(r, a), "ode1-linear-coefficients-swapped"),
    ]
    b_wrongs = [
        (Fraction(r, a), "ode1-particular-constant-correction-missed"),
        (r, "ode1-particular-coefficient-not-divided"),
        ((r + big_a) / a, "ode1-integration-by-parts-sign"),
        ((r - big_a) / (a * a), "ode1-particular-coefficient-divided-twice"),
    ]
    c_wrongs = [
        (y0, "ode1-ic-ignores-particular-part"),
        (big_b - y0, "ode1-ic-sign-error"),
        (y0 + big_b, "ode1-ic-sign-error"),
        (y0 - big_a - big_b, "ode1-ic-evaluated-at-wrong-point"),
    ]
    true = sympy.N(value, 30)
    ea = sympy.exp(a * _R(x1))
    d_wrongs = [
        (big_a * x1 + big_b + _R(const) * ea, "ode1-eval-exponent-sign"),
        (big_a * x1 + big_b, "ode1-eval-exponential-term-dropped"),
        (big_a * x1 + big_b + const, "ode1-eval-exponential-dropped"),
        (_R(const) * sympy.exp(-a * _R(x1)), "ode1-eval-particular-dropped"),
        (y0, "ode1-eval-initial-value-reported"),
        (big_a + big_b + _R(const) * sympy.exp(-a), "ode1-eval-point-wrong") if x1 != 1 else (None, ""),
    ]
    bound = int(2 * (abs(big_a) * x1 + abs(big_b) + abs(const)) + 1)
    n1, d1 = x1.numerator, x1.denominator
    links = [
        Link(
            p_a,
            "fraction",
            big_a,
            1,
            [
                f"(e^({a}x) y)' = e^({a}x)({_num(p, 'x')} + {r}), and the x term needs a coefficient A with {a}A = {p}.",
                f"A = {rational_text(big_a)}.",
            ],
            a_wrongs,
            sensible=(-bound, bound),
        ),
        Link(
            p_b,
            "fraction",
            big_b,
            2,
            [
                f"Integrating by parts, the integral of {_num(p, 'x')}e^({a}x) is e^({a}x)({p}x/{a} - {p}/{a}²) up to a constant, so dividing by e^({a}x) gives A = {p}/{a} and a constant term of ({r} - A)/{a}.",
                f"So B = ({r} - A)/{a} = {rational_text(big_b)}.",
            ],
            b_wrongs,
            sensible=(-bound, bound),
        ),
        Link(
            p_c,
            "fraction",
            const,
            1,
            [f"y(0) = B + C = {y0}, so C = {rational_text(const)}."],
            c_wrongs,
            ft=f"{y0} - part_b",
            sensible=(-bound, bound),
        ),
        Link(
            p_d,
            "number",
            value,
            2,
            [f"y({x1_text}) = A({x1_text}) + B + Ce^(-{a * x1}) = {float(true):.5f}."],
            d_wrongs,
            ft=f"part_a*({n1}/{d1}) + part_b + part_c*E**(-{a}*{n1}/{d1})",
            sensible=(-bound, bound),
            places=3,
        ),
    ]
    return stem, links


@generator(EXAM_TOPIC, level=19, presets={"exam": {}})
def fy1so_exd_ode_integrating_factor(rng: random.Random) -> dict:
    """A linear first-order IVP: two coefficients of the general solution, the constant, and a value of the solution."""
    stem, links = _exam(rng)
    return dict(text=stem[0], latex=stem[1], parts=[build_part(link) for link in links])
