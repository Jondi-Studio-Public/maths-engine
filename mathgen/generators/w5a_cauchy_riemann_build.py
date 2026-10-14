"""Generates `w5a_cauchy_riemann` and `exd_x_w5a_cauchy_riemann` questions: the Cauchy-Riemann equations and harmonic conjugates.

One second-year skill, ``uc2-cauchy-riemann-harmonic-conjugate`` (domain ``complex-analysis``). Every answer is typed
as an expression in x and y (or in z), or a Gaussian integer. Each question is built backwards from an analytic function
f, so u = Re f and v = Im f are known exactly; the answer is then confirmed numerically against the Cauchy-Riemann
equations u_x = v_y and u_y = -v_x (derivatives taken exactly, evaluated at points), which is a different route from the
construction. The conjugate is normalised by v(0, 0) = 0 (or, for functions singular at the origin, by v = 0 on the real
axis) so that it is unique.

Tiers:

- ``easy``: the harmonic conjugate of a polynomial; the value of a parameter that makes a polynomial harmonic; recovering
  f(z) from Re f, with f(0) = 0.
- ``medium``: the conjugate of Re of e^(cz), z e^(cz), sin(cz), cos(cz), sinh(cz) or cosh(cz); the positive parameter that
  makes e^(kx) cos(my), sin(kx) sinh(my) and the like harmonic; the single point at which a given f = u + iv is complex
  differentiable.
- ``hard``: f(z) recovered from Re f for a sum of two such functions with f(0) stated; the conjugate of Re of a rational
  function and of a sum; f'(z0) at the single point where a non-analytic f is differentiable.
- ``exam`` (topic ``exd_x_w5a_cauchy_riemann``): a four-part chain on u = p x^3 + a x y^2 + r x^2 + b y^2 + t x: the two
  parameters that make u harmonic, the conjugate, and f'(1 + i) (which follows from the first two parts).
"""

from __future__ import annotations

import cmath
import math
import random
from fractions import Fraction as Fr

import sympy

from ..core import attempts, check, generator
from ._w5a_support import (
    Link,
    X,
    astr,
    atext,
    exam_dict,
    join_signed,
    shift,
    msay,
    single,
)

TOPIC = "w5a_cauchy_riemann"
EXAM_TOPIC = "exd_x_w5a_cauchy_riemann"

x, y = sympy.symbols("x y", real=True)
z = sympy.Symbol("z")
I = sympy.I  # noqa: E741


def _tex(e) -> str:
    return sympy.latex(e, ln_notation=True)


def _ex(e) -> X:
    return X(atext(e), _tex(e))


# -- building blocks -----------------------------------------------------------------------------------------------------


def zpow(n: int):
    """(Re, Im) of (x + iy)^n as polynomials in x and y."""
    re_, im_ = sympy.Integer(0), sympy.Integer(0)
    for k in range(n + 1):
        term = sympy.binomial(n, k) * x ** (n - k) * y**k
        sign = (-1) ** (k // 2)
        if k % 2 == 0:
            re_ += sign * term
        else:
            im_ += sign * term
    return re_, im_


def poly_family(coeffs):
    """f = sum (a_k + b_k i) z^k for coeffs [(a, b), ...]: (u, v, f(z))."""
    u = v = sympy.Integer(0)
    f = sympy.Integer(0)
    for k, (a, b) in enumerate(coeffs):
        if a == 0 and b == 0:
            continue
        if k == 0:
            re_, im_ = sympy.Integer(1), sympy.Integer(0)
        else:
            re_, im_ = zpow(k)
        u += a * re_ - b * im_
        v += a * im_ + b * re_
        f += (a + b * I) * z**k
    return sympy.expand(u), sympy.expand(v), f


def trans_family(kind: str, c: int):
    """(u, v, F(z), F(0)) for e^(cz), z e^(cz), sin, cos, sinh, cosh, z sin, z cos of cz."""
    cx, cy = c * x, c * y
    if kind == "exp":
        e = sympy.exp(cx)
        return e * sympy.cos(cy), e * sympy.sin(cy), sympy.exp(c * z), 1
    if kind == "zexp":
        e = sympy.exp(cx)
        return (
            e * (x * sympy.cos(cy) - y * sympy.sin(cy)),
            e * (y * sympy.cos(cy) + x * sympy.sin(cy)),
            z * sympy.exp(c * z),
            0,
        )
    if kind == "sin":
        return sympy.sin(cx) * sympy.cosh(cy), sympy.cos(cx) * sympy.sinh(cy), sympy.sin(c * z), 0
    if kind == "cos":
        return sympy.cos(cx) * sympy.cosh(cy), -sympy.sin(cx) * sympy.sinh(cy), sympy.cos(c * z), 1
    if kind == "sinh":
        return sympy.sinh(cx) * sympy.cos(cy), sympy.cosh(cx) * sympy.sin(cy), sympy.sinh(c * z), 0
    if kind == "cosh":
        return sympy.cosh(cx) * sympy.cos(cy), sympy.sinh(cx) * sympy.sin(cy), sympy.cosh(c * z), 1
    if kind == "zsin":
        return (
            x * sympy.sin(cx) * sympy.cosh(cy) - y * sympy.cos(cx) * sympy.sinh(cy),
            x * sympy.cos(cx) * sympy.sinh(cy) + y * sympy.sin(cx) * sympy.cosh(cy),
            z * sympy.sin(c * z),
            0,
        )
    if kind == "zcos":
        return (
            x * sympy.cos(cx) * sympy.cosh(cy) + y * sympy.sin(cx) * sympy.sinh(cy),
            y * sympy.cos(cx) * sympy.cosh(cy) - x * sympy.sin(cx) * sympy.sinh(cy),
            z * sympy.cos(c * z),
            0,
        )
    raise ValueError(kind)


_TRANS = ["exp", "zexp", "sin", "cos", "sinh", "cosh", "zsin", "zcos"]
_TRANS_SIMPLE = ["exp", "sin", "cos", "sinh", "cosh"]


def rational_family(a: int, n: int, m: int):
    """f = m/(z - a)^n: (u, v, f(z)); v vanishes on the real axis."""
    d = (x - a) ** 2 + y**2
    if n == 1:
        return m * (x - a) / d, -m * y / d, m / (z - a)
    return m * ((x - a) ** 2 - y**2) / d**2, -2 * m * (x - a) * y / d**2, m / (z - a) ** 2


def _num(e, px, py) -> complex:
    return complex(sympy.N(e.subs({x: px, y: py}), 15))


_POINTS = [(0.37, 0.61), (-0.83, 0.42), (0.55, -0.9)]


def check_cr(u, v, pts=None) -> None:
    """The Cauchy-Riemann equations hold at sample points: derivatives are exact, only the final evaluation is numeric."""
    ux, uy, vx, vy = sympy.diff(u, x), sympy.diff(u, y), sympy.diff(v, x), sympy.diff(v, y)
    for px, py in pts or _POINTS:
        check(abs(_num(ux - vy, px, py)) < 1e-9, "u_x = v_y")
        check(abs(_num(uy + vx, px, py)) < 1e-9, "u_y = -v_x")


def check_laplace(u) -> None:
    lap = sympy.diff(u, x, 2) + sympy.diff(u, y, 2)
    for px, py in _POINTS:
        check(abs(_num(lap, px, py)) < 1e-9, "u is harmonic")


def _poly_int_y(e):
    """Integral of a polynomial in x and y with respect to y, no constant."""
    out = sympy.Integer(0)
    for (i, j), c in sympy.Poly(sympy.expand(e), x, y).terms():
        out += c * x**i * y ** (j + 1) / (j + 1)
    return out


def _swap(e):
    return e.xreplace({x: y, y: x})


def _dist(answer, cands, n: int = 4):
    from ._w5a_support import dist_list

    return dist_list(answer, cands, n)


def _flip_terms(v):
    """v with the sign of its first term reversed."""
    terms = sympy.Add.make_args(sympy.expand(v))
    if len(terms) < 2:
        return -v
    return sympy.Add(*terms[1:]) - terms[0]


# -- easy ----------------------------------------------------------------------------------------------------------------


def _gauss(rng, lo=-3, hi=3, nonzero=True):
    while True:
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if not (nonzero and a == 0 and b == 0):
            return a, b


def _easy_conjugate_poly(rng: random.Random) -> dict:
    deg = rng.choice([2, 2, 3])
    coeffs = [(rng.randint(-3, 3), 0)] + [_gauss(rng, -3, 3, nonzero=False) for _ in range(deg)]
    if coeffs[deg] == (0, 0):
        coeffs[deg] = _gauss(rng)
    if sum(1 for k in range(1, deg + 1) if coeffs[k] != (0, 0)) < 2:
        k = 1 if deg == 2 else 2
        coeffs[k] = _gauss(rng)
    u, v, _f = poly_family(coeffs)
    check_laplace(u)
    check_cr(u, v)
    check(v.subs({x: 0, y: 0}) == 0, "v(0,0) = 0")
    ux = sympy.diff(u, x)
    partial = _poly_int_y(ux)
    wrongs = [
        (-v, "cx2-cr-conjugate-sign"),
        (partial, "cx2-cr-integration-function-missing"),
        (_swap(v), "cx2-cr-conjugate-swapped-roles"),
        (_flip_terms(v), "cx2-cr-conjugate-sign"),
        (sympy.expand(v + 1), "cx2-cr-constant-not-fixed"),
        (sympy.expand(_poly_int_y(-sympy.diff(u, y))), "cx2-cr-integration-function-missing"),
    ]
    working = [
        f"u_x = {atext(ux)}. The equation u_x = v_y gives v = ∫ u_x dy = {atext(partial)} + h(x).",
        f"The equation u_y = -v_x then fixes h(x). With v(0,0) = 0 the harmonic conjugate is v = {atext(v)}.",
    ]
    ux_ = _ex(u)
    lead = [
        lambda: msay(
            "The function ",
            X(f"u(x, y) = {ux_.p}", f"u(x, y) = {ux_.t}"),
            " is harmonic. Find its harmonic conjugate v(x, y) with v(0, 0) = 0.",
        ),
        lambda: msay(
            "Let ",
            X(f"u(x, y) = {ux_.p}", f"u(x, y) = {ux_.t}"),
            ". Find the function v(x, y), with v(0, 0) = 0, such that u + iv is analytic.",
        ),
        lambda: msay(
            "Verify that Laplace's equation holds for ",
            X(f"u = {ux_.p}", f"u = {ux_.t}"),
            ", then find the harmonic conjugate v with v(0, 0) = 0.",
        ),
    ][rng.randrange(3)]
    return single(lead(), v, working, wrongs, fallback_tag="cx2-cr-conjugate-sign")


def _easy_parameter(rng: random.Random) -> dict:
    kind = rng.choice(["cubic", "square", "quartic", "mixed"])
    p = rng.choice([1, 2, 3, -1, -2])
    k = sympy.Symbol("k")
    t = rng.choice([0, 1, 2, -1])
    if kind == "cubic":
        u = p * x**3 + k * x * y**2
        ans = -3 * p
        shown = f"{p if p != 1 else ''}x³ + kxy²".replace("-1x", "-x").replace("+ -", "- ")
        tex = f"{p if p != 1 else ''}x^3 + kxy^2".replace("-1x", "-x")
    elif kind == "square":
        u = p * x**2 + k * y**2 + t * x
        ans = -p
        shown = f"{p if abs(p) != 1 else ('-' if p < 0 else '')}x² + ky²" + (
            f" {'+' if t > 0 else '-'} {abs(t) if abs(t) != 1 else ''}x" if t else ""
        )
        tex = f"{p if abs(p) != 1 else ('-' if p < 0 else '')}x^2 + ky^2" + (
            f" {'+' if t > 0 else '-'} {abs(t) if abs(t) != 1 else ''}x" if t else ""
        )
    elif kind == "quartic":
        u = p * x**4 + k * x**2 * y**2 + p * y**4
        ans = -6 * p
        c = "" if p == 1 else "-" if p == -1 else str(p)
        shown = f"{c}x⁴ + kx²y² + {c}y⁴".replace("+ -", "- ")
        tex = f"{c}x^4 + kx^2y^2 + {c}y^4".replace("+ -", "- ")
    else:
        u = p * x**3 * y + k * x * y**3
        ans = -p
        c = "" if p == 1 else "-" if p == -1 else str(p)
        shown = f"{c}x³y + kxy³"
        tex = f"{c}x^3y + kxy^3"
    lap = sympy.diff(u, x, 2) + sympy.diff(u, y, 2)
    check(sympy.expand(lap.subs(k, ans)) == 0, "u is harmonic at the answer")
    check(sympy.expand(lap.subs(k, ans + 1)) != 0, "k is determined")
    wrongs = [
        (-ans, "cx2-cr-harmonic-sign"),
        (sympy.Integer(ans) * 3 if abs(ans) != 3 * abs(p) else sympy.Integer(ans) * 2, "cx2-cr-parameter-factor-slip"),
        (sympy.Integer(p), "cx2-cr-parameter-factor-slip"),
        (
            sympy.Integer(ans) // 3 if ans % 3 == 0 and ans != 0 and ans // 3 != ans else sympy.Rational(ans, 2),
            "cx2-cr-parameter-factor-slip",
        ),
        (sympy.Integer(ans) * 2, "cx2-cr-derivative-slip"),
    ]
    uxx, uyy = sympy.diff(u, x, 2), sympy.diff(u, y, 2)
    working = [
        f"Harmonic means u_xx + u_yy = 0. Here u_xx = {atext(uxx)} and u_yy = {atext(uyy)}.",
        f"Their sum is {atext(sympy.expand(lap))}, which is 0 for all x and y exactly when k = {ans}.",
    ]
    lead = [
        lambda: msay(
            "Find the value of the constant k for which ", X(f"u(x, y) = {shown}", f"u(x, y) = {tex}"), " is harmonic."
        ),
        lambda: msay(
            "For which value of k is ", X(f"u = {shown}", f"u = {tex}"), " the real part of an analytic function?"
        ),
        lambda: msay(
            "The function ",
            X(f"u(x, y) = {shown}", f"u(x, y) = {tex}"),
            " satisfies Laplace's equation for exactly one value of k. Find k.",
        ),
    ][rng.randrange(3)]
    return single(lead(), sympy.Integer(ans), working, wrongs, fallback_tag="cx2-cr-parameter-factor-slip")


def _easy_f_of_z(rng: random.Random) -> dict:
    deg = rng.choice([2, 2, 3])
    coeffs = [(0, 0)] + [_gauss(rng, -3, 3, nonzero=False) for _ in range(deg)]
    if coeffs[deg] == (0, 0):
        coeffs[deg] = _gauss(rng)
    if sum(1 for k in range(1, deg + 1) if coeffs[k] != (0, 0)) < 2:
        coeffs[1 if deg == 2 else 2] = _gauss(rng)
    u, v, f = poly_family(coeffs)
    check_cr(u, v)
    # the numeric route: f(x + iy) has real part u
    for px, py in _POINTS:
        zz = complex(px, py)
        fv = complex(sympy.N(f.subs(z, zz), 15))
        check(abs(fv.real - _num(u, px, py).real) < 1e-9 and abs(fv.imag - _num(v, px, py).real) < 1e-9, "f = u + iv")
    wrongs = [
        (sympy.expand(u.subs({x: z, y: 0})), "cx2-cr-f-real-part-only"),
        (
            sympy.expand(sum((a * z**k for k, (a, _b) in enumerate(coeffs)), sympy.Integer(0))),
            "cx2-cr-f-real-part-only",
        ),
        (sympy.expand(sympy.conjugate(f).subs(sympy.conjugate(z), z)), "cx2-cr-f-conjugate"),
        (sympy.expand(-f), "cx2-cr-conjugate-sign"),
        (sympy.expand(f * I), "cx2-cr-conjugate-swapped-roles"),
    ]
    ux_ = _ex(u)
    uzz = sympy.diff(u, x)
    working = [
        f"f' = u_x + i v_x = u_x - i u_y. Here u_x = {atext(uzz)} and u_y = {atext(sympy.diff(u, y))}.",
        f"Replace x + iy by z in f'(z) = {atext(sympy.expand(sympy.diff(f, z)))} and integrate with f(0) = 0: f(z) = {atext(sympy.expand(f))}.",
    ]
    lead = [
        lambda: msay(
            "An analytic function f = u + iv has ",
            X(f"u(x, y) = {ux_.p}", f"u(x, y) = {ux_.t}"),
            " and f(0) = 0. Find f(z) as an expression in z.",
        ),
        lambda: msay(
            "Let f be analytic with ",
            X(f"Re f = {ux_.p}", f"\\operatorname{{Re}} f = {ux_.t}"),
            " and f(0) = 0. Write f(z) in terms of z.",
        ),
        lambda: msay(
            "Find the analytic function f(z) with f(0) = 0 whose real part is ",
            X(f"{ux_.p}", f"{ux_.t}"),
            ". Give your answer in terms of z.",
        ),
    ][rng.randrange(3)]
    return single(lead(), sympy.expand(f), working, wrongs, fallback_tag="cx2-cr-f-conjugate")


# -- medium --------------------------------------------------------------------------------------------------------------


def _medium_conjugate(rng: random.Random) -> dict:
    kind = rng.choice(_TRANS)
    c = rng.choice([1, 2, 3, -1])
    if kind in ("sinh", "cosh", "sin", "cos") and rng.random() < 0.4:
        c = 1
    m = rng.choice([1, 1, 2, 3, -1, -2])
    u0, v0, fz, _f0 = trans_family(kind, c)
    u, v = m * u0, m * v0
    check_cr(u, v)
    check(abs(_num(v, 0, 0)) < 1e-12, "v(0,0) = 0")
    ux = sympy.diff(u, x)
    wrongs = [
        (-v, "cx2-cr-conjugate-sign"),
        (_swap_uv(kind, c, m), "cx2-cr-conjugate-swapped-roles"),
        (
            sympy.expand(m * v0 / (c if c not in (0, 1) else 2)) if c not in (0, 1) else m * 2 * v0,
            "cx2-cr-chain-factor-dropped",
        ),
        (u, "cx2-cr-conjugate-swapped-roles"),
        (_sign_one_factor(kind, c, m), "cx2-cr-derivative-slip"),
    ]
    working = [
        f"u_x = {atext(ux)}, so v_y = u_x and v = ∫ u_x dy + h(x).",
        f"Using u_y = -v_x to fix h and v(0,0) = 0 gives v = {atext(v)} (this is the imaginary part of {atext(m * fz)}).",
    ]
    ux_ = _ex(u)
    lead = [
        lambda: msay(
            "The function ",
            X(f"u(x, y) = {ux_.p}", f"u(x, y) = {ux_.t}"),
            " is harmonic. Find the harmonic conjugate v with v(0, 0) = 0.",
        ),
        lambda: msay(
            "Find the function v(x, y) with v(0, 0) = 0 such that ",
            X(f"u = {ux_.p}", f"u = {ux_.t}"),
            " and v are the real and imaginary parts of an analytic function.",
        ),
        lambda: msay(
            "Determine the harmonic conjugate, normalised by v(0, 0) = 0, of ",
            X(f"u(x, y) = {ux_.p}", f"u(x, y) = {ux_.t}"),
            ".",
        ),
    ][rng.randrange(3)]
    return single(lead(), v, working, wrongs, fallback_tag="cx2-cr-conjugate-sign")


def _swap_uv(kind, c, m):
    """A near miss: the conjugate of the function with c replaced by -c (the sign of the y-dependence)."""
    _u, v, _fz, _f0 = trans_family(kind, -c)
    return m * v


def _sign_one_factor(kind, c, m):
    u0, v0, _fz, _f0 = trans_family(kind, c)
    # use the real part with the wrong sign on y: the conjugate of f(conj z)
    return (
        m * v0.xreplace({y: -y}) * -1
        if kind in ("exp", "zexp", "sin", "cos", "sinh", "cosh", "zsin", "zcos")
        else -m * v0
    )


_PARAM_MEDIUM = [
    # (shape, template, answer relation): the answer is the positive k with k^2 = m^2
    ("expcos", "e^(kx) cos({m}y)"),
    ("expsin", "e^({m}x) sin(ky)"),
    ("sinsinh", "sin(kx) sinh({m}y)"),
    ("coscosh", "cos({m}x) cosh(ky)"),
    ("coshcos", "cosh(kx) cos({m}y)"),
    ("sincosh", "sin({m}x) cosh(ky)"),
]


def _medium_parameter(rng: random.Random) -> dict:
    shape, _t = rng.choice(_PARAM_MEDIUM)
    m = rng.choice([2, 3, 4, 5, 6, 7])
    k = sympy.Symbol("k", positive=True)
    lib = {
        "expcos": (sympy.exp(k * x) * sympy.cos(m * y), f"e^(kx)cos({m}y)", f"e^{{kx}} \\cos {m}y", "-"),
        "expsin": (sympy.exp(m * x) * sympy.sin(k * y), f"e^({m}x)sin(ky)", f"e^{{{m}x}} \\sin ky", "-"),
        "sinsinh": (sympy.sin(k * x) * sympy.sinh(m * y), f"sin(kx)sinh({m}y)", f"\\sin kx \\sinh {m}y", "+"),
        "coscosh": (sympy.cos(m * x) * sympy.cosh(k * y), f"cos({m}x)cosh(ky)", f"\\cos {m}x \\cosh ky", "+"),
        "coshcos": (sympy.cosh(k * x) * sympy.cos(m * y), f"cosh(kx)cos({m}y)", f"\\cosh kx \\cos {m}y", "-"),
        "sincosh": (sympy.sin(m * x) * sympy.cosh(k * y), f"sin({m}x)cosh(ky)", f"\\sin {m}x \\cosh ky", "+"),
    }
    u, ptxt, ptex, _s = lib[shape]
    lap = sympy.diff(u, x, 2) + sympy.diff(u, y, 2)
    for px, py in _POINTS:
        check(abs(complex(sympy.N(lap.subs({k: m, x: px, y: py}), 15))) < 1e-9, "u harmonic at k = m")
        check(abs(complex(sympy.N(lap.subs({k: m + 1, x: px, y: py}), 15))) > 1e-6, "k determined")
    ans = sympy.Integer(m)
    wrongs = [
        (sympy.Integer(m * m), "cx2-cr-parameter-factor-slip"),
        (-ans, "cx2-cr-harmonic-sign"),
        (sympy.Rational(m, 2), "cx2-cr-parameter-factor-slip"),
        (sympy.Integer(2 * m), "cx2-cr-parameter-factor-slip"),
        (sympy.sqrt(m), "cx2-cr-parameter-factor-slip"),
        (sympy.Integer(m + 1), "cx2-cr-derivative-slip"),
    ]
    uxx, uyy = sympy.diff(u, x, 2), sympy.diff(u, y, 2)
    working = [
        f"u_xx = {atext(sympy.simplify(uxx / u))}·u and u_yy = {atext(sympy.simplify(uyy / u))}·u, using the factors of u.",
        f"u_xx + u_yy = 0 needs k² = {m * m}, so k = {m} for k > 0.",
    ]
    lead = [
        lambda: msay(
            "Find the positive value of k for which ", X(f"u(x, y) = {ptxt}", f"u(x, y) = {ptex}"), " is harmonic."
        ),
        lambda: msay(
            "The function ",
            X(f"u = {ptxt}", f"u = {ptex}"),
            " is the real part of an analytic function for exactly one positive value of k. Find it.",
        ),
        lambda: msay(
            "Determine k > 0 so that ", X(f"u(x, y) = {ptxt}", f"u(x, y) = {ptex}"), " satisfies Laplace's equation."
        ),
    ][rng.randrange(3)]
    return single(lead(), ans, working, wrongs, fallback_tag="cx2-cr-parameter-factor-slip")


# the single point of differentiability

_V_COEFFS = [1, 2, 3, -1, -2]


def _one_point_data(rng: random.Random):
    for _ in attempts(500, "a one-point example"):
        x0, y0 = rng.randint(-3, 3), rng.randint(-3, 3)
        if (x0, y0) == (0, 0):
            continue
        a_, b_ = rng.choice([1, 2, -1, 3]), rng.choice([1, 2, -1, -2])
        e_, f_ = rng.choice(_V_COEFFS), rng.choice([1, 2, -1, -2, 3])
        if (2 * a_ - e_) == 0 or (2 * b_ + e_) == 0:
            continue
        c_, g_ = rng.randint(-3, 3), rng.randint(-3, 3)
        h_ = 2 * a_ * x0 + c_ - e_ * x0 - 2 * f_ * y0
        d_ = -e_ * y0 - g_ - 2 * b_ * y0
        u = a_ * x**2 + b_ * y**2 + c_ * x + d_ * y
        v = e_ * x * y + f_ * y**2 + g_ * x + h_ * y
        return x0, y0, u, v, (a_, b_, c_, d_, e_, f_, g_, h_)
    raise AssertionError("unreachable")


def _medium_one_point(rng: random.Random) -> dict:
    x0, y0, u, v, _p = _one_point_data(rng)
    ux, uy, vx, vy = sympy.diff(u, x), sympy.diff(u, y), sympy.diff(v, x), sympy.diff(v, y)
    # the CR equations are linear in x and y and hold at (x0, y0) only
    sol = sympy.solve([ux - vy, uy + vx], [x, y], dict=True)
    check(len(sol) == 1 and sol[0][x] == x0 and sol[0][y] == y0, "unique solution")
    value = x0 + I * y0
    wrongs = [
        (y0 + I * x0, "cx2-cr-conjugate-swapped-roles"),
        (x0 - I * y0, "cx2-cr-point-conjugated"),
        (-x0 - I * y0, "calc-sign-slip"),
        (x0 + I * (-y0 + 1), "cx2-cr-point-equation-slip"),
        (x0 + 1 + I * y0, "cx2-cr-point-equation-slip"),
    ]
    working = [
        f"u_x = {atext(ux)}, u_y = {atext(uy)}, v_x = {atext(vx)}, v_y = {atext(vy)}.",
        f"The equations u_x = v_y and u_y = -v_x say {atext(ux)} = {atext(vy)} and {atext(uy)} = {atext(-vx)}.",
        f"These are linear in x and y with the single solution x = {x0}, y = {y0}, so f is differentiable only at z = {atext(value)}.",
    ]
    ue, ve = _ex(u), _ex(v)
    lead = [
        lambda: msay(
            "Let f = u + iv with ",
            X(f"u = {ue.p}", f"u = {ue.t}"),
            " and ",
            X(f"v = {ve.p}", f"v = {ve.t}"),
            ". The function f is complex differentiable at exactly one point. Find that point as a complex number.",
        ),
        lambda: msay(
            "For ",
            X(f"f(x + iy) = ({ue.p}) + i({ve.p})", f"f(x + iy) = \\left({ue.t}\\right) + i\\left({ve.t}\\right)"),
            ", find the one point z at which f is complex differentiable.",
        ),
        lambda: msay(
            "Use the Cauchy-Riemann equations to find the only point at which ",
            X(f"f = ({ue.p}) + i({ve.p})", f"f = \\left({ue.t}\\right) + i\\left({ve.t}\\right)"),
            " is differentiable, giving it as x + iy.",
        ),
    ][rng.randrange(3)]
    return single(lead(), sympy.sympify(value), working, wrongs, fallback_tag="cx2-cr-point-equation-slip")


# -- hard ----------------------------------------------------------------------------------------------------------------


def _hard_f_of_z(rng: random.Random) -> dict:
    for _ in attempts(500, "two distinct families"):
        k1, k2 = rng.sample(_TRANS_SIMPLE + ["zexp"], 2)
        c1, c2 = rng.choice([1, 2, -1]), rng.choice([1, 2, 3])
        m1, m2 = rng.choice([1, 2, -1]), rng.choice([1, 1, 2, 3, -2])
        if k1 == k2:
            continue
        break
    u1, v1, f1, z1 = trans_family(k1, c1)
    u2, v2, f2, z2 = trans_family(k2, c2)
    kr, ki = rng.randint(-3, 3), rng.choice([-3, -2, -1, 1, 2, 3])
    f = m1 * f1 + m2 * f2 + kr + ki * I
    u = m1 * u1 + m2 * u2 + kr
    v = m1 * v1 + m2 * v2 + ki
    check_cr(u, v)
    f0 = m1 * z1 + m2 * z2 + kr + ki * I
    check(
        abs(_num(u, 0, 0).real - float(sympy.re(f0))) < 1e-9 and abs(_num(v, 0, 0).real - float(sympy.im(f0))) < 1e-9,
        "f(0)",
    )
    fz = lambda w: complex(sympy.N(f.subs(z, w), 15))  # noqa: E731
    for px, py in _POINTS:
        got = fz(complex(px, py))
        check(abs(got.real - _num(u, px, py).real) < 1e-8 and abs(got.imag - _num(v, px, py).real) < 1e-8, "f = u + iv")
    wrongs = [
        (m1 * f1 + m2 * f2 + kr, "cx2-cr-constant-ignored"),
        (m1 * f1 + m2 * f2 + kr - ki * I, "cx2-cr-constant-ignored"),
        (m1 * f1 - m2 * f2 + kr + ki * I, "cx2-cr-conjugate-sign"),
        (m1 * f1 + m2 * f2 + sympy.re(f0) + I * (ki + 1), "cx2-cr-constant-ignored"),
        (m2 * f1 + m1 * f2 + kr + ki * I, "cx2-cr-conjugate-swapped-roles"),
        (m1 * f1 + m2 * f2 + I * kr + ki, "cx2-cr-conjugate-swapped-roles"),
    ]
    ue = _ex(u)
    f0c = sympy.expand(f0)
    f0txt = atext(f0c)
    working = [
        f"Recognise the parts: {atext(m1 * u1)} is the real part of {atext(m1 * f1)} and {atext(m2 * u2)} is the real part of {atext(m2 * f2)}.",
        f"So f(z) = {atext(m1 * f1 + m2 * f2)} + K with K a constant whose real part is {kr}.",
        f"The condition f(0) = {f0txt} gives the imaginary part {ki} of K, so f(z) = {atext(f)}.",
    ]
    lead = [
        lambda: msay(
            "An analytic function f = u + iv has ",
            X(f"u(x, y) = {ue.p}", f"u(x, y) = {ue.t}"),
            f" and f(0) = {f0txt}. Find f(z) as an expression in z.",
        ),
        lambda: msay(
            "Let f be analytic with ",
            X(f"Re f = {ue.p}", f"\\operatorname{{Re}} f = {ue.t}"),
            f" and f(0) = {f0txt}. Give f(z) in terms of z.",
        ),
        lambda: msay(
            "Find the analytic function f(z), as an expression in z, whose real part is ",
            X(f"{ue.p}", f"{ue.t}"),
            f" and which satisfies f(0) = {f0txt}.",
        ),
    ][rng.randrange(3)]
    return single(lead(), f, working, wrongs, fallback_tag="cx2-cr-constant-ignored")


def _hard_rational(rng: random.Random) -> dict:
    a = rng.choice([-2, -1, 0, 1, 2])
    n = rng.choice([1, 2])
    m = rng.choice([1, 2, 3, -1, -2])
    u, v, f = rational_family(a, n, m)
    check_cr(u, v)
    check(abs(_num(v, a + 2, 0)) < 1e-12, "v = 0 on the real axis")
    d = (x - a) ** 2 + y**2
    wrongs = [
        (-v, "cx2-cr-conjugate-sign"),
        (m * (x - a) / d if n == 2 else m * y / d, "cx2-cr-conjugate-swapped-roles"),
        (-m * y / d if n == 2 else -m * y / d**2, "cx2-cr-chain-factor-dropped"),
        (-2 * m * y / d**2 if n == 2 else -2 * m * y / d, "cx2-cr-chain-factor-dropped"),
        (-m * (x - a) * y / d**2 if n == 2 else -m * y / (x - a), "cx2-cr-derivative-slip"),
        (v + 1, "cx2-cr-constant-not-fixed"),
    ]
    ue = _ex(u)
    working = [
        f"u is the real part of f(z) = {atext(f)}: with {shift(a)} = {shift(a, 'x')} + iy, 1/({shift(a)})^{n} has real part {atext(u / m)}.",
        f"The imaginary part is v = {atext(v)}, and it vanishes on the real axis as required.",
    ]
    lead = [
        lambda: msay(
            "The function ",
            X(f"u(x, y) = {ue.p}", f"u(x, y) = {ue.t}"),
            " is harmonic away from the point ",
            X(f"({a}, 0)", f"({a}, 0)"),
            ". Find its harmonic conjugate v, normalised so that v(x, 0) = 0 for every real x ≠ " + str(a) + ".",
        ),
        lambda: msay(
            "Let ",
            X(f"u(x, y) = {ue.p}", f"u(x, y) = {ue.t}"),
            ". Find v(x, y) with u + iv analytic for (x, y) ≠ " + f"({a}, 0)" + ", given that v = 0 on the real axis.",
        ),
        lambda: msay(
            "Find the harmonic conjugate v of ", X(f"u = {ue.p}", f"u = {ue.t}"), " that vanishes along the real axis."
        ),
    ][rng.randrange(3)]
    return single(lead(), v, working, wrongs, fallback_tag="cx2-cr-conjugate-sign")


def _hard_sum_conjugate(rng: random.Random) -> dict:
    for _ in attempts(500, "two distinct families"):
        k1, k2 = rng.sample(_TRANS, 2)
        if k1 != k2:
            break
    c1, c2 = rng.choice([1, 2, -1]), rng.choice([1, 2])
    m1, m2 = rng.choice([1, 2, -1]), rng.choice([1, 2, -2])
    u1, v1, f1, _a = trans_family(k1, c1)
    u2, v2, f2, _b = trans_family(k2, c2)
    u, v = m1 * u1 + m2 * u2, m1 * v1 + m2 * v2
    check_cr(u, v)
    check(abs(_num(v, 0, 0)) < 1e-12, "v(0,0) = 0")
    wrongs = [
        (-v, "cx2-cr-conjugate-sign"),
        (m1 * v1 - m2 * v2, "cx2-cr-conjugate-sign"),
        (m1 * v1, "cx2-cr-integration-function-missing"),
        (m2 * v2, "cx2-cr-integration-function-missing"),
        (m2 * v1 + m1 * v2, "cx2-cr-conjugate-swapped-roles"),
    ]
    ue = _ex(u)
    working = [
        f"{atext(m1 * u1)} is the real part of {atext(m1 * f1)}, and {atext(m2 * u2)} is the real part of {atext(m2 * f2)}.",
        f"Their imaginary parts add: v = {atext(v)}, and v(0, 0) = 0.",
    ]
    lead = [
        lambda: msay(
            "The function ",
            X(f"u(x, y) = {ue.p}", f"u(x, y) = {ue.t}"),
            " is harmonic. Find its harmonic conjugate v with v(0, 0) = 0.",
        ),
        lambda: msay(
            "Find v(x, y), with v(0, 0) = 0, so that u + iv is analytic where ", X(f"u = {ue.p}", f"u = {ue.t}"), "."
        ),
        lambda: msay(
            "Determine, with v(0, 0) = 0, the harmonic conjugate of ", X(f"u(x, y) = {ue.p}", f"u(x, y) = {ue.t}"), "."
        ),
    ][rng.randrange(3)]
    return single(lead(), v, working, wrongs, fallback_tag="cx2-cr-conjugate-sign")


def _hard_derivative_at_point(rng: random.Random) -> dict:
    x0, y0, u, v, _p = _one_point_data(rng)
    ux, uy, vx, vy = sympy.diff(u, x), sympy.diff(u, y), sympy.diff(v, x), sympy.diff(v, y)
    sol = sympy.solve([ux - vy, uy + vx], [x, y], dict=True)
    check(len(sol) == 1 and sol[0][x] == x0 and sol[0][y] == y0, "unique solution")
    deriv = ux.subs({x: x0, y: y0}) + I * vx.subs({x: x0, y: y0})
    check(sympy.expand(deriv - (vy.subs({x: x0, y: y0}) - I * uy.subs({x: x0, y: y0}))) == 0, "f' two ways")
    wrongs = [
        (ux.subs({x: x0, y: y0}) - I * vx.subs({x: x0, y: y0}), "cx2-cr-derivative-formula"),
        (ux.subs({x: x0, y: y0}) + I * uy.subs({x: x0, y: y0}), "cx2-cr-derivative-formula"),
        (vx.subs({x: x0, y: y0}) + I * ux.subs({x: x0, y: y0}), "cx2-cr-conjugate-swapped-roles"),
        (ux.subs({x: 0, y: 0}) + I * vx.subs({x: 0, y: 0}), "cx2-cr-derivative-at-wrong-point"),
        (ux.subs({x: y0, y: x0}) + I * vx.subs({x: y0, y: x0}), "cx2-cr-derivative-at-wrong-point"),
        (-deriv, "calc-sign-slip"),
    ]
    working = [
        f"u_x = {atext(ux)}, u_y = {atext(uy)}, v_x = {atext(vx)}, v_y = {atext(vy)}. The Cauchy-Riemann equations hold only at x = {x0}, y = {y0}.",
        f"There f' = u_x + i v_x = {atext(sympy.expand(deriv))}, which also equals v_y - i u_y.",
    ]
    ue, ve = _ex(u), _ex(v)
    lead = [
        lambda: msay(
            "Let f = u + iv with ",
            X(f"u = {ue.p}", f"u = {ue.t}"),
            " and ",
            X(f"v = {ve.p}", f"v = {ve.t}"),
            ". The function f is complex differentiable at exactly one point z0. Find f'(z0).",
        ),
        lambda: msay(
            "The function ",
            X(f"f(x + iy) = ({ue.p}) + i({ve.p})", f"f(x + iy) = \\left({ue.t}\\right) + i\\left({ve.t}\\right)"),
            " is differentiable at one point only. Find the value of the derivative there.",
        ),
        lambda: msay(
            "Find the point where ",
            X(f"f = ({ue.p}) + i({ve.p})", f"f = \\left({ue.t}\\right) + i\\left({ve.t}\\right)"),
            " is complex differentiable and give the value of f' at that point, in the form a + bi.",
        ),
    ][rng.randrange(3)]
    return single(lead(), sympy.expand(deriv), working, wrongs, fallback_tag="cx2-cr-derivative-formula")


_TIERS = {
    "easy": [_easy_conjugate_poly, _easy_parameter, _easy_f_of_z],
    "medium": [_medium_conjugate, _medium_parameter, _medium_one_point],
    "hard": [_hard_f_of_z, _hard_rational, _hard_sum_conjugate, _hard_derivative_at_point],
}


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5a_cauchy_riemann(rng: random.Random, tier: str = "easy") -> dict:
    """A Cauchy-Riemann question: a harmonic conjugate, an analytic f(z), a harmonic parameter or a point of differentiability."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return rng.choice(_TIERS[tier])(rng)


# -- exam: a four-part chain -----------------------------------------------------------------------------------------------


def _signed(c: int, mono: str, first: bool = False) -> str:
    mag = abs(c)
    body = mono if mag == 1 and mono else f"{mag}{mono}"
    if first:
        return ("-" if c < 0 else "") + body
    return (" - " if c < 0 else " + ") + body


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w5a_cauchy_riemann(rng: random.Random, tier: str = "exam") -> dict:
    """u = p x^3 + a x y^2 + r x^2 + b y^2 + t x: the two harmonic parameters, the conjugate, and f'(1 + i)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    p = rng.choice([1, 2, 3])
    r = rng.choice([1, 2, 3, -1, -2])
    t = rng.choice([-2, -1, 1, 2, 3])
    a_, b_ = -3 * p, -r
    u = p * x**3 + a_ * x * y**2 + r * x**2 + b_ * y**2 + t * x
    v = sympy.expand(p * (3 * x**2 * y - y**3) + 2 * r * x * y + t * y)
    check_laplace(u)
    check_cr(u, v)
    check(abs(_num(v, 0, 0)) < 1e-12, "v(0, 0) = 0")
    deriv = (2 * r + t) + (6 * p + 2 * r) * I
    ux, uy = sympy.diff(u, x), sympy.diff(u, y)
    check(sympy.expand(ux.subs({x: 1, y: 1}) - I * uy.subs({x: 1, y: 1}) - deriv) == 0, "f'(1 + i)")
    # the text of u with the two unknowns
    shown = _signed(p, "x³", True) + " + axy²" + _signed(r, "x²") + " + by²" + _signed(t, "x")
    tex = _signed(p, "x^3", True) + " + axy^2" + _signed(r, "x^2") + " + by^2" + _signed(t, "x")
    stem = msay(
        "Let ",
        X(f"u(x, y) = {shown}", f"u(x, y) = {tex}"),
        ", where a and b are constants, and let u be the real part of an analytic function f = u + iv.",
    )
    w_a = Link(
        msay("Laplace's equation holds for u. Find a."),
        a_,
        "number",
        2,
        [
            f"u_xx = {atext(sympy.diff(p * x**3 + r * x**2 + t * x, x, 2))} and u_yy = 2ax + 2b.",
            f"The sum is ({6 * p} + 2a)x {'+' if r >= 0 else '-'} {abs(2 * r)} + 2b, which vanishes for all x when a = {a_}.",
        ],
        [
            (-a_, "cx2-cr-harmonic-sign"),
            (-p, "cx2-cr-parameter-factor-slip"),
            (-6 * p, "cx2-cr-parameter-factor-slip"),
            (3 * p, "cx2-cr-harmonic-sign"),
        ],
    )
    w_b = Link(
        msay("Find b."),
        b_,
        "number",
        2,
        [f"The constant term of u_xx + u_yy is {2 * r} + 2b, which must vanish, so b = {b_}."],
        [
            (-b_, "cx2-cr-harmonic-sign"),
            (r * 2, "cx2-cr-parameter-factor-slip"),
            (-2 * r, "cx2-cr-parameter-factor-slip"),
            (a_, "cx2-cr-parameter-factor-slip"),
        ],
    )
    w_c = Link(
        msay("Using these values, find the harmonic conjugate v(x, y) with v(0, 0) = 0."),
        v,
        "expression",
        4,
        [
            f"With a = {a_} and b = {b_}: u_x = {atext(sympy.expand(ux))}, so v = ∫ u_x dy + h(x).",
            f"Using u_y = -v_x to find h and v(0, 0) = 0 gives v = {atext(v)}.",
        ],
        [
            (-v, "cx2-cr-conjugate-sign"),
            (_poly_int_y(sympy.diff(u, x)), "cx2-cr-integration-function-missing"),
            (_swap(v), "cx2-cr-conjugate-swapped-roles"),
            (sympy.expand(v - t * y), "cx2-cr-integration-function-missing"),
        ],
    )
    ft = f"{3 * p} + part_a + ({2 * r}) + ({t}) - I*(2*part_a + 2*part_b)"
    w_d = Link(
        msay("Find f'(1 + i), the derivative of f at z = 1 + i, in the form p + qi."),
        sympy.expand(deriv),
        "expression",
        3,
        [
            "f'(z) = u_x - i u_y.",
            f"u_x(1, 1) = {join_signed([3 * p, 'a', 2 * r, t])} = {2 * r + t}, and u_y(1, 1) = 2a + 2b = {2 * a_ + 2 * b_}, so f'(1 + i) = {atext(sympy.expand(deriv))}.",
        ],
        [
            (sympy.expand((2 * r + t) - (6 * p + 2 * r) * I), "cx2-cr-derivative-formula"),
            (sympy.expand((2 * r + t) + (2 * a_ + 2 * b_) * I), "cx2-cr-derivative-formula"),
            (sympy.expand((6 * p + 2 * r) + (2 * r + t) * I), "cx2-cr-conjugate-swapped-roles"),
            (sympy.expand(ux.subs({x: 1, y: 0}) + I * (6 * p + 2 * r)), "cx2-cr-derivative-at-wrong-point"),
        ],
        ft=ft,
    )
    return exam_dict(stem, [w_a, w_b, w_c, w_d])
