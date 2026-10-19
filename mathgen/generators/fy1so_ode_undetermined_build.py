"""Generates `fy1so_ode_undetermined` (easy, medium, hard) and the exam chain `fy1so_exd_ode_undetermined`.

Skill ``ucv1-undetermined-coefficients`` (domain ``ordinary-differential-equations``, answer ``expression``): a particular
solution of y'' + b y' + c y = g(x) by the method of undetermined coefficients, and the initial value problem built on it. The
homogeneous part (the characteristic roots) is the separate skill ``ucv1-second-order-characteristic-roots``; here the roots are
distinct integers, named in the exam stem, and only used to fix the constants.

Tiers (the reasoning escalates):

- ``easy``: g = F e^(kx) with k not a root (y_p = A e^(kx), A = F / p(k) where p is the characteristic polynomial), or g a
  line (y_p = αx + β, which needs the y' term as well as the y term).
- ``medium``: g a quadratic (y_p = αx² + βx + γ, three coefficients in turn) or g = G cos ωx + H sin ωx (y_p = P cos ωx +
  Q sin ωx, where both terms are needed even though ``g`` is a mixture of the two).
- ``hard``: resonance: g = F e^(rx) with r a root (y_p = A x e^(rx), A = F / (r1 - r2)), g = F e^(rx) with r a double root
  (y_p = (F/2) x² e^(rx)), or y'' + ω² y = F cos ωx (y_p = (F/2ω) x sin ωx).
- ``exam`` (topic ``fy1so_exd_ode_undetermined``): y'' + b y' + c y = F e^(kx), y(0) = y0, y'(0) = v0, with the homogeneous
  solution C1 e^(r1 x) + C2 e^(r2 x) given. (a) A (b) C1 (follows through from (a)) (c) C2 (follows through from (a) and (b))
  (d) y(x1) to 3 decimal places (follows through from (a), (b) and (c)). Every part is a single value, because a follow-through
  from an expression reply cannot be marked (`reply_value` reads the learner's `e` as a plain symbol).

Every particular solution is checked a second way: written out again as a python function, it is put back into the
differential equation with a five-point finite-difference y' and y'' at several points.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, rational_text
from ._fy1so_common import Link, M, build_part, build_single, rounds_clear, say, signed_tex

TOPIC = "fy1so_ode_undetermined"
EXAM_TOPIC = "fy1so_exd_ode_undetermined"

x = sympy.Symbol("x")
_PROBES = (0.3, 0.7, 1.1)
_ROOTS = [-3, -2, -1, 1, 2, 3]


def _R(value) -> sympy.Rational:
    value = Fraction(value)
    return sympy.Rational(value.numerator, value.denominator)


def _derivatives(y, t: float, h: float = 1e-3):
    """(y, y', y'') at t by five-point finite differences."""
    f = [y(t + j * h) for j in (-2, -1, 0, 1, 2)]
    d1 = (f[0] - 8 * f[1] + 8 * f[3] - f[4]) / (12 * h)
    d2 = (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * h * h)
    return f[2], d1, d2


def _check_particular(y, b, c, g) -> None:
    """Second route: y'' + b y' + c y = g(x) for the python function `y`, at several points."""
    for t in _PROBES:
        value, d1, d2 = _derivatives(y, t)
        residual = d2 + b * d1 + c * value - g(t)
        check(abs(residual) < 1e-6 * (1 + abs(g(t))), "the particular solution does not satisfy the equation")


def _lhs_tex(b: int, c: int) -> str:
    out = "y''"
    if b:
        out += f" {signed_tex(b, "y'")}"
    if c:
        out += f" {signed_tex(c, 'y')}"
    return out


def _sum_tex(terms) -> str:
    """``coefficient body`` terms as a sum: 3x^{2} - x + 5, leaving out zero terms and unit coefficients."""
    out = ""
    for coef, body in terms:
        if coef == 0:
            continue
        magnitude = abs(coef)
        piece = ("" if magnitude == 1 and body else str(magnitude)) + body
        out += (("-" if coef < 0 else "") + piece) if not out else (" - " if coef < 0 else " + ") + piece
    return out


def _num(c, body: str) -> str:
    return body if c == 1 else f"-{body}" if c == -1 else f"{c}{body}"


def _rate(k: int) -> str:
    """The exponent k x as learner text: x, -x, 3x."""
    return "x" if k == 1 else "-x" if k == -1 else f"{k}x"


def _exp_tex(k: int) -> str:
    return "e^{x}" if k == 1 else "e^{-x}" if k == -1 else f"e^{{{k}x}}"


def _trig_tex(name: str, w: int) -> str:
    return rf"\{name}\,x" if w == 1 else rf"\{name}({w}x)"


def _coefficients(rng: random.Random, distinct: bool = True):
    """Distinct integer roots (r1 < r2) and the equation y'' + b y' + c y with those characteristic roots."""
    r1, r2 = sorted(rng.sample(_ROOTS, 2)) if distinct else (rng.choice(_ROOTS),) * 2
    return r1, r2, -(r1 + r2), r1 * r2


def _poly(b: int, c: int, k) -> object:
    return k * k + b * k + c


def _safe(fn):
    try:
        return fn()
    except ZeroDivisionError:
        return None


# -- easy ------------------------------------------------------------------------------------------------------------


def _easy(rng: random.Random):
    return _easy_exponential(rng) if rng.random() < 0.6 else _easy_line(rng)


def _easy_exponential(rng: random.Random):
    for _ in attempts(200, "a non-resonant exponential"):
        r1, r2, b, c = _coefficients(rng)
        k = rng.choice([-2, -1, 1, 2, 3])
        amp = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        if k not in (r1, r2):
            break
    p = _poly(b, c, k)
    force = amp * p
    y = amp * sympy.exp(k * x)
    _check_particular(lambda t: amp * math.exp(k * t), b, c, lambda t: force * math.exp(k * t))
    wrongs = [
        (force * sympy.exp(k * x), "ode3-coefficient-not-divided"),
        (_safe(lambda: _R(Fraction(force, k * k + b + c)) * sympy.exp(k * x)), "ode3-derivative-order-ignored"),
        (
            _safe(lambda: _R(Fraction(force, _poly(b, -c, k))) * sympy.exp(k * x)),
            "ode3-sign-in-characteristic-polynomial",
        ),
        (
            _safe(lambda: _R(Fraction(force, 2 * k + b)) * sympy.exp(k * x)),
            "ode3-characteristic-polynomial-differentiated",
        ),
        (-amp * sympy.exp(k * x), "ode3-sign-error"),
    ]
    prompt = say(
        "Find a particular solution of ",
        M(rf"{_lhs_tex(b, c)} = {_num(force, _exp_tex(k))}"),
        " of the form ",
        M(rf"y_{{p}} = Ae^{{{k}x}}".replace("{1x}", "{x}").replace("{-1x}", "{-x}")),
        ".",
    )
    working = [
        f"y_p = Ae^({k}x) gives y_p' = {k}Ae^({k}x) and y_p'' = {k * k}Ae^({k}x).",
        f"Substituting, A({k * k} + ({b})({k}) + ({c})) = {force}, so {p}A = {force} and A = {amp}.",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


def _easy_line(rng: random.Random):
    for _ in attempts(200, "a linear forcing term"):
        r1, r2, b, c = _coefficients(rng)
        alpha = rng.choice([-3, -2, -1, 1, 2, 3])
        beta = rng.choice([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        if b != 0 and not (alpha * 0 == beta and beta == 0):
            break
    m, n = c * alpha, b * alpha + c * beta
    y = alpha * x + beta
    _check_particular(lambda t: alpha * t + beta, b, c, lambda t: m * t + n)
    wrongs = [
        (_R(Fraction(m, c)) * x + _R(Fraction(n, c)), "ode3-derivative-terms-dropped"),
        (alpha * x, "ode3-constant-term-dropped"),
        (alpha * x + _R(Fraction(n + b * alpha, c)), "ode3-sign-error"),
        (alpha * x + _R(Fraction(n, c)), "ode3-derivative-terms-dropped"),
        (alpha * x + beta + 1 if beta != 0 else alpha * x + 1, "ode3-constant-term-slip"),
    ]
    rhs = _sum_tex([(m, "x"), (n, "")])
    prompt = say(
        "Find a particular solution of ",
        M(rf"{_lhs_tex(b, c)} = {rhs}"),
        " of the form ",
        M(r"y_{p} = ax + b"),
        ".",
    )
    working = [
        "y_p = ax + b gives y_p' = a and y_p'' = 0.",
        f"Substituting: {b}a + {c}(ax + b) = {rhs}. Matching x: {c}a = {m}, so a = {alpha}. Matching constants: {b}a + {c}b = {n}, so b = {beta}.",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


# -- medium ----------------------------------------------------------------------------------------------------------


def _medium(rng: random.Random):
    return _medium_quadratic(rng) if rng.random() < 0.45 else _medium_trig(rng)


def _medium_quadratic(rng: random.Random):
    for _ in attempts(200, "a quadratic forcing term"):
        r1, r2, b, c = _coefficients(rng)
        alpha = rng.choice([-2, -1, 1, 2, 3])
        beta = rng.choice([-3, -2, -1, 0, 1, 2, 3])
        gamma = rng.choice([-3, -2, -1, 1, 2, 3])
        if b != 0:
            break
    m = c * alpha
    n = c * beta + 2 * b * alpha
    q = c * gamma + b * beta + 2 * alpha
    y = alpha * x**2 + beta * x + gamma
    _check_particular(lambda t: alpha * t * t + beta * t + gamma, b, c, lambda t: m * t * t + n * t + q)
    wrongs = [
        (
            _R(Fraction(m, c)) * x**2 + _R(Fraction(n, c)) * x + _R(Fraction(q, c)),
            "ode3-derivative-terms-dropped",
        ),
        (alpha * x**2 + beta * x + _R(Fraction(q - 2 * alpha, c)), "ode3-second-derivative-dropped"),
        (alpha * x**2 + _R(Fraction(n, c)) * x + gamma, "ode3-first-derivative-dropped"),
        (alpha * x**2 + beta * x + _R(Fraction(q - 2 * alpha + b * beta, c)), "ode3-sign-error"),
        (alpha * x**2 + beta * x, "ode3-constant-term-dropped"),
    ]
    rhs = _sum_tex([(m, "x^{2}"), (n, "x"), (q, "")])
    prompt = say(
        "Find a particular solution of ",
        M(rf"{_lhs_tex(b, c)} = {rhs}"),
        ".",
    )
    working = [
        "y_p = ax² + bx + d gives y_p' = 2ax + b and y_p'' = 2a.",
        f"Matching x²: {c}a = {m}, so a = {alpha}. Matching x: {c}b + {2 * b}a = {n}, so b = {beta}.",
        f"Matching constants: {c}d + {b}b + 2a = {q}, so d = {gamma}.",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


def _medium_trig(rng: random.Random):
    for _ in attempts(200, "a trigonometric forcing term"):
        r1, r2, b, c = _coefficients(rng)
        w = rng.choice([1, 2, 3])
        p_coef = rng.choice([-3, -2, -1, 1, 2, 3])
        q_coef = rng.choice([-3, -2, -1, 1, 2, 3])
        d = c - w * w
        if b != 0 and (d != 0 or b != 0):
            break
    g_cos = d * p_coef + b * w * q_coef
    g_sin = d * q_coef - b * w * p_coef
    if g_cos == 0 and g_sin == 0:  # pragma: no cover - b != 0 makes the forcing non-trivial
        g_cos = 1
    y = p_coef * sympy.cos(w * x) + q_coef * sympy.sin(w * x)
    _check_particular(
        lambda t: p_coef * math.cos(w * t) + q_coef * math.sin(w * t),
        b,
        c,
        lambda t: g_cos * math.cos(w * t) + g_sin * math.sin(w * t),
    )
    wrongs = [
        (
            _safe(lambda: _R(Fraction(g_cos, d)) * sympy.cos(w * x) + _R(Fraction(g_sin, d)) * sympy.sin(w * x)),
            "ode3-derivative-terms-dropped",
        ),
        (p_coef * sympy.cos(w * x), "ode3-trig-partner-missed"),
        (q_coef * sympy.cos(w * x) + p_coef * sympy.sin(w * x), "ode3-trig-coefficients-swapped"),
        (p_coef * sympy.cos(w * x) - q_coef * sympy.sin(w * x), "ode3-sign-error"),
        (g_cos * sympy.cos(w * x) + g_sin * sympy.sin(w * x), "ode3-coefficient-not-divided"),
    ]
    rhs = _sum_tex([(g_cos, _trig_tex("cos", w)), (g_sin, _trig_tex("sin", w))])
    prompt = say(
        "Find a particular solution of ",
        M(rf"{_lhs_tex(b, c)} = {rhs}"),
        ".",
    )
    working = [
        f"y_p' = {w}(-P sin + Q cos) and y_p'' = -{w * w}y_p.",
        f"Collecting cos: ({c} - {w * w})P + {b * w}Q = {g_cos}. Collecting sin: ({c} - {w * w})Q - {b * w}P = {g_sin}.",
        f"Solving, P = {p_coef} and Q = {q_coef}.",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


# -- hard: resonance ------------------------------------------------------------------------------------------------


def _hard(rng: random.Random):
    return rng.choice([_hard_simple, _hard_double, _hard_trig])(rng)


def _hard_simple(rng: random.Random):
    r1, r2, b, c = _coefficients(rng)
    r, other = rng.choice([(r1, r2), (r2, r1)])
    amp = rng.choice([-3, -2, -1, 1, 2, 3, 4])
    force = amp * (r - other)
    y = amp * x * sympy.exp(r * x)
    _check_particular(lambda t: amp * t * math.exp(r * t), b, c, lambda t: force * math.exp(r * t))
    wrongs = [
        (_safe(lambda: _R(Fraction(force, 2 * r + b)) * sympy.exp(r * x)), "ode3-resonance-factor-x-missing"),
        (force * x * sympy.exp(r * x), "ode3-coefficient-not-divided"),
        (_safe(lambda: _R(Fraction(force, 2 * r)) * x * sympy.exp(r * x)), "ode3-resonance-derivative-incomplete"),
        (amp * x * sympy.exp(other * x), "ode3-wrong-root-used"),
        (-amp * x * sympy.exp(r * x), "ode3-sign-error"),
        (_safe(lambda: _R(Fraction(force, 2)) * x**2 * sympy.exp(r * x)), "ode3-resonance-order-too-high"),
    ]
    prompt = say(
        "The characteristic equation of ",
        M(rf"{_lhs_tex(b, c)} = 0"),
        f" has roots {r1} and {r2}. Find a particular solution of ",
        M(rf"{_lhs_tex(b, c)} = {_num(force, _exp_tex(r))}"),
        ".",
    )
    working = [
        f"e^({r}x) already solves the homogeneous equation, so try y_p = Axe^({r}x).",
        f"y_p' = A(1 + {r}x)e^({r}x) and y_p'' = A({2 * r} + {r * r}x)e^({r}x).",
        f"Substituting, A({2 * r} + {b}) = {force}, and {2 * r + b}A = {force}, so A = {amp}.",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


def _hard_double(rng: random.Random):
    r = rng.choice(_ROOTS)
    b, c = -2 * r, r * r
    force = rng.choice([-8, -6, -4, -2, 2, 4, 6, 8])
    amp = Fraction(force, 2)
    y = _R(amp) * x**2 * sympy.exp(r * x)
    _check_particular(lambda t: float(amp) * t * t * math.exp(r * t), b, c, lambda t: force * math.exp(r * t))
    wrongs = [
        (_R(amp) * x * sympy.exp(r * x), "ode3-resonance-order-too-low"),
        (force * x**2 * sympy.exp(r * x), "ode3-factor-two-mishandled"),
        (_R(amp / 2) * x**2 * sympy.exp(r * x), "ode3-factor-two-mishandled"),
        (_R(amp) * x**3 * sympy.exp(r * x), "ode3-resonance-order-too-high"),
        (_R(amp) * sympy.exp(r * x), "ode3-resonance-factor-x-missing"),
    ]
    prompt = say(
        "The characteristic equation of ",
        M(rf"{_lhs_tex(b, c)} = 0"),
        f" has the repeated root {r}. Find a particular solution of ",
        M(rf"{_lhs_tex(b, c)} = {_num(force, _exp_tex(r))}"),
        ".",
    )
    working = [
        f"Both e^({r}x) and xe^({r}x) solve the homogeneous equation, so try y_p = Ax²e^({r}x).",
        f"Substituting gives 2A e^({r}x) = {force}e^({r}x), so A = {rational_text(amp)}.",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


def _hard_trig(rng: random.Random):
    w = rng.choice([1, 2, 3, 4])
    force = rng.choice([-3, -2, -1, 1, 2, 3, 4, 5]) * 2 * w
    amp = Fraction(force, 2 * w)
    y = _R(amp) * x * sympy.sin(w * x)
    _check_particular(lambda t: float(amp) * t * math.sin(w * t), 0, w * w, lambda t: force * math.cos(w * t))
    wrongs = [
        (_R(amp) * sympy.sin(w * x), "ode3-resonance-factor-x-missing"),
        (_R(amp) * x * sympy.cos(w * x), "ode3-trig-partner-confused"),
        (_R(Fraction(force, w)) * x * sympy.sin(w * x), "ode3-factor-two-mishandled"),
        (_R(Fraction(force, w * w)) * x * sympy.sin(w * x), "ode3-factor-two-mishandled"),
        (-_R(amp) * x * sympy.sin(w * x), "ode3-sign-error"),
    ]
    prompt = say(
        "Find a particular solution of ",
        M(rf"y'' + {w * w}y = {_num(force, _trig_tex('cos', w))}"),
        ".",
    )
    working = [
        f"cos({w}x) and sin({w}x) already solve the homogeneous equation, so try y_p = x(P cos({w}x) + Q sin({w}x)).",
        f"Substituting, y_p'' + {w * w}y_p = {2 * w}Q cos({w}x) - {2 * w}P sin({w}x), so Q = {rational_text(amp)} and P = 0.",
    ]
    return prompt, Link(prompt, "expression", y, working=working, wrongs=wrongs, symbols=(x,))


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1so_ode_undetermined(rng: random.Random, tier: str = "easy") -> dict:
    """A particular solution of y'' + b y' + c y = g by undetermined coefficients: exponential, polynomial, trig, resonance."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    prompt, link = {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)
    return build_single(prompt, link)


# -- exam: a four-part chain -----------------------------------------------------------------------------------------

_X1 = [Fraction(1, 2), Fraction(1)]


def _exam(rng: random.Random):
    for _ in attempts(1000, "an exam initial value problem"):
        r1, r2, b, c = _coefficients(rng)
        k = rng.choice([-2, -1, 1, 2, 3])
        amp = rng.choice([-3, -2, -1, 1, 2, 3])
        y0 = rng.randint(-4, 5)
        v0 = rng.randint(-5, 6)
        x1 = rng.choice(_X1)
        if k in (r1, r2):
            continue
        c1 = Fraction(v0 - k * amp - r2 * (y0 - amp), r1 - r2)
        c2 = y0 - amp - c1
        if c1 == 0 or c2 == 0 or c1.denominator > 5 or c2.denominator > 5:
            continue
        t1 = float(x1)
        approx = amp * math.exp(k * t1) + float(c1) * math.exp(r1 * t1) + float(c2) * math.exp(r2 * t1)
        if abs(approx) < 0.4 or abs(approx) > 60 or abs((approx * 1000) % 1 - 0.5) < 0.2:
            continue
        break
    value = amp * sympy.exp(k * _R(x1)) + _R(c1) * sympy.exp(r1 * _R(x1)) + _R(c2) * sympy.exp(r2 * _R(x1))
    check(rounds_clear(value, 3), "the value is too close to a rounding boundary")
    p = _poly(b, c, k)
    force = amp * p
    full = lambda t: amp * math.exp(k * t) + float(c1) * math.exp(r1 * t) + float(c2) * math.exp(r2 * t)  # noqa: E731
    _check_particular(full, b, c, lambda t: force * math.exp(k * t))
    start, slope, _ = _derivatives(full, 0.0)
    check(abs(start - y0) < 1e-6 and abs(slope - v0) < 1e-4, "the exam solution misses an initial condition")
    x1_text = rational_text(x1) if x1 == 1 else str(float(x1))
    stem = say(
        "Consider the initial value problem ",
        M(rf"{_lhs_tex(b, c)} = {_num(force, _exp_tex(k))}"),
        ", ",
        M(f"y(0) = {y0}"),
        ", ",
        M(f"y'(0) = {v0}"),
        f". The general solution of the homogeneous equation is y = Be^({_rate(r1)}) + Ce^({_rate(r2)}). "
        "The general solution of the full equation is ",
        M(rf"y = Ae^{{{k}x}} + Be^{{{r1}x}} + Ce^{{{r2}x}}".replace("{1x}", "{x}").replace("{-1x}", "{-x}")),
        ".",
    )
    p_a = say("Find the value of ", M("A"), ".")
    p_b = say("Use the initial conditions to find the value of ", M("B"), ".")
    p_c = say("Find the value of ", M("C"), ".")
    p_d = say("Find ", M(f"y({x1_text})"), " correct to 3 decimal places.")
    a_wrongs = [
        (force, "ode3-coefficient-not-divided"),
        (_safe(lambda: Fraction(force, k * k + b + c)), "ode3-derivative-order-ignored"),
        (_safe(lambda: Fraction(force, _poly(b, -c, k))), "ode3-sign-in-characteristic-polynomial"),
        (_safe(lambda: Fraction(force, 2 * k + b)), "ode3-characteristic-polynomial-differentiated"),
        (-amp, "ode3-sign-error"),
    ]
    other_c1 = Fraction(v0 - k * amp - r1 * (y0 - amp), r2 - r1)
    b_wrongs = [
        (Fraction(v0 - r2 * y0, r1 - r2), "ode3-ic-ignores-particular"),
        (other_c1, "ode3-constants-swapped"),
        (-c1, "ode3-sign-error"),
        (Fraction(v0 - r2 * (y0 - amp), r1 - r2), "ode3-derivative-ic-particular-dropped"),
        (Fraction(v0 + k * amp - r2 * (y0 - amp), r1 - r2), "ode3-sign-error"),
    ]
    c_wrongs = [
        (y0 - c1, "ode3-ic-ignores-particular"),
        (y0 - amp + c1, "ode3-sign-error"),
        (c1, "ode3-constants-swapped"),
        (y0 + amp - c1, "ode3-sign-error"),
        (Fraction(y0), "ode3-ic-ignores-particular"),
    ]
    true = sympy.N(value, 30)
    ex = lambda e: sympy.exp(_R(e))  # noqa: E731
    d_wrongs = [
        (_R(c1) * ex(r1 * x1) + _R(c2) * ex(r2 * x1), "ode3-eval-particular-dropped"),
        (amp * ex(-k * x1) + _R(c1) * ex(r1 * x1) + _R(c2) * ex(r2 * x1), "ode3-eval-exponent-sign"),
        (amp + c1 + c2, "ode3-eval-exponential-dropped"),
        (amp * ex(k) + _R(c1) * ex(r1) + _R(c2) * ex(r2), "ode3-eval-point-wrong") if x1 != 1 else (None, ""),
        (y0, "ode3-eval-initial-value-reported"),
        (amp * ex(k * x1) + _R(c1) * ex(r2 * x1) + _R(c2) * ex(r1 * x1), "ode3-constants-swapped"),
    ]
    bound = int(3 * (abs(amp) + abs(c1) + abs(c2)) + 10)
    bound_d = max(bound, int(3 * abs(float(value))) + 10)
    n1, d1 = x1.numerator, x1.denominator
    links = [
        Link(
            p_a,
            "fraction",
            amp,
            1,
            [f"y_p = Ae^({k}x): A({k * k} + ({b})({k}) + ({c})) = {force}, so {p}A = {force} and A = {amp}."],
            a_wrongs,
            sensible=(-bound, bound),
        ),
        Link(
            p_b,
            "fraction",
            c1,
            2,
            [
                f"y(0) = A + B + C = {y0} and y'(0) = {k}A + {r1}B + {r2}C = {v0}.",
                f"Eliminating C: ({r1} - {r2})B = {v0} - {k}A - ({r2})({y0} - A), so B = {rational_text(c1)}.",
            ],
            b_wrongs,
            ft=f"({v0} - {k}*part_a - ({r2})*({y0} - part_a))/({r1} - ({r2}))",
            sensible=(-bound, bound),
        ),
        Link(
            p_c,
            "fraction",
            c2,
            1,
            [f"From y(0): C = {y0} - A - B = {rational_text(c2)}."],
            c_wrongs,
            ft=f"{y0} - part_a - part_b",
            sensible=(-bound, bound),
        ),
        Link(
            p_d,
            "number",
            value,
            2,
            [
                f"y({x1_text}) = {amp}e^({k * x1}) + ({rational_text(c1)})e^({r1 * x1}) + ({rational_text(c2)})e^({r2 * x1}) = {float(true):.5f}."
            ],
            d_wrongs,
            ft=f"part_a*E**({k}*{n1}/{d1}) + part_b*E**({r1}*{n1}/{d1}) + part_c*E**({r2}*{n1}/{d1})",
            sensible=(-bound_d, bound_d),
            places=3,
        ),
    ]
    return stem, links


@generator(EXAM_TOPIC, level=19, presets={"exam": {}})
def fy1so_exd_ode_undetermined(rng: random.Random) -> dict:
    """A second-order IVP with an exponential forcing term: the particular coefficient, two constants, a value of y."""
    stem, links = _exam(rng)
    return dict(text=stem[0], latex=stem[1], parts=[build_part(link) for link in links])
