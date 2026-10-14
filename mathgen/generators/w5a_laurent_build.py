"""Generates `w5a_laurent` and `exd_x_w5a_laurent` questions: Laurent series, coefficients and principal parts.

One second-year skill, ``uc2-laurent-series-expansion`` (domain ``complex-analysis``). The answers are exact: a rational
or Gaussian-rational coefficient, a coefficient times a power of ``e``, a radius, or (medium tier) a principal part, a
sum of terms ``c/(z - a)^k`` typed as an expression in ``z``. Every question is built from the series it asks about, and
checked against the contour formula ``a_n = (1/2*pi*i) * integral of f(z)/(z - a)^(n+1) dz``, evaluated by the
trapezoid rule on a circle inside the annulus of convergence.

Tiers:

- ``easy``: one coefficient from a standard series. ``g(cz)/z^m`` with ``g`` exp, sin, cos, sinh or cosh; ``e^(cz)`` over
  ``(z - z0)^m`` about ``z0`` (the answer carries ``e^(c z0)``); or a rational ``c/(z^m (z - a))`` on ``0 < |z| < |a|``.
- ``medium``: the principal part of a rational function at a pole of order two or three, or a coefficient of the
  regular part of that expansion.
- ``hard``: a coefficient of ``c/((z - p)(z - q))`` in the annulus between its two poles (so both a geometric series
  in ``(z - w)/beta`` and one in ``alpha/(z - w)`` are needed); a coefficient at an essential singularity,
  ``z^p e^(c/z)``, ``z^p sin(c/z)`` or ``z^p cos(c/z)``; and a coefficient of ``cot(cz)``, ``1/(e^(cz) - 1)`` or
  ``1/sin^2(cz)``, which need a series division.
- ``exam`` (topic ``exd_x_w5a_laurent``): a four-part chain on ``(pz + q)/((z - a)^2 (z - b))``: the two principal-part
  coefficients, the principal part's value one unit from the pole, and the radius of the punctured disc on which the
  series converges.
"""

from __future__ import annotations

import cmath
import math
import random
from fractions import Fraction as Fr

import sympy

from ..core import attempts, check, generator
from ._w5a_support import (
    C,
    Link,
    Rat,
    X,
    atext,
    circle_mean,
    cx_x,
    exam_dict,
    fac_x,
    frac_x,
    msay,
    numeric_close,
    poly_x,
    prod_x,
    qsym,
    rat_x,
    sdiv,
    single,
    zminus_x,
)

TOPIC = "w5a_laurent"
EXAM_TOPIC = "exd_x_w5a_laurent"

_Z = sympy.Symbol("z")


def _coeff_numeric(fn, centre: complex, radius: float, n: int) -> complex:
    """a_n = (1/2*pi*i) * integral of f(z)/(z - c)^(n+1): the mean of f(z)/(z - c)^(n+1) times (z - c)."""
    return circle_mean(lambda z: fn(z) / (z - centre) ** (n + 1), centre, radius, 128)


def _term_x(a: C, n: int) -> X:
    """(z - a)^n, with a negative exponent in brackets: z^(-1), (z - 1)^(-2)."""
    base = zminus_x(a)
    pp = base.p if a.is_zero() else f"({base.p})"
    tt = base.t if a.is_zero() else f"({base.t})"
    if n == 1:
        return X(pp, tt)
    return X(f"{pp}^({n})" if n < 0 else f"{pp}^{n}", f"{tt}^{{{n}}}")


def _fline(body: X) -> X:
    return X("f(z) = " + body.p, "f(z) = " + body.t)


def _phrase(rng, fx: X, term: X, a: C, region: X | None = None):
    where = cx_x(a)
    at = X(f"z = {where.p}", f"z = {where.t}")
    reg = [" valid ", region] if region is not None else []
    shapes = [
        lambda: msay(
            "Find the coefficient of ",
            term,
            " in the Laurent series about ",
            at,
            " of ",
            fx,
            *([", valid for ", region] if region else []),
            ".",
        ),
        lambda: msay(
            "Let ",
            fx,
            ". Determine the coefficient of ",
            term,
            " in the Laurent expansion of f about ",
            at,
            *([", valid for ", region] if region else []),
            ".",
        ),
        lambda: msay(
            "The function ",
            fx,
            " has a Laurent series about ",
            at,
            *([" valid for ", region] if region else []),
            ". What is the coefficient of ",
            term,
            "?",
        ),
    ]
    del reg
    return shapes[rng.randrange(3)]()


def _tidy(c: C) -> bool:
    return c.re.denominator <= 400 and c.im.denominator <= 400


# -- Maclaurin coefficients ------------------------------------------------------------------------------------------


def _maclaurin(kind: str, c: int, k: int) -> Fr:
    """Coefficient of z^k in g(cz)."""
    base = Fr(c) ** k / math.factorial(k)
    if kind == "exp":
        return base
    if kind == "sin":
        return Fr((-1) ** ((k - 1) // 2)) * base if k % 2 else Fr(0)
    if kind == "cos":
        return Fr((-1) ** (k // 2)) * base if k % 2 == 0 else Fr(0)
    if kind == "sinh":
        return base if k % 2 else Fr(0)
    return base if k % 2 == 0 else Fr(0)  # cosh


_FN_NAME = {"exp": "e^", "sin": "sin", "cos": "cos", "sinh": "sinh", "cosh": "cosh"}


def _g_x(kind: str, arg: str) -> X:
    if kind == "exp":
        return X(f"e^({arg})", f"e^{{{arg}}}")
    return X(f"{kind}({arg})", rf"\{kind}({arg})")


def _cz(c: int, var: str = "z") -> str:
    return var if c == 1 else f"-{var}" if c == -1 else f"{c}{var}"


# -- easy ------------------------------------------------------------------------------------------------------------


def _easy_maclaurin_over_power(rng):
    for _ in attempts(500, "an easy Laurent coefficient"):
        kind = rng.choice(["exp", "exp", "sin", "cos", "sinh", "cosh"])
        c = rng.choice([1, 2, 3, -1, -2])
        m = rng.randint(2, 5)
        k = rng.randint(0, m + 3)
        n = k - m
        value = _maclaurin(kind, c, k)
        if value == 0 or n >= 3:
            continue
        break
    fx = _fline(frac_x(_g_x(kind, _cz(c)), X(f"z{'²' if m == 2 else '³' if m == 3 else '⁴⁵'[m - 4]}", f"z^{{{m}}}")))
    term = _term_x(C(0), n) if n != 0 else X("z⁰ (the constant term)", "z^{0}")
    fn = {"exp": cmath.exp, "sin": cmath.sin, "cos": cmath.cos, "sinh": cmath.sinh, "cosh": cmath.cosh}[kind]
    numeric_close(_coeff_numeric(lambda z: fn(c * z) / z**m, 0j, 0.7, n), float(value), "Laurent coefficient")
    v = qsym(value)
    wrongs = [
        (qsym(value * math.factorial(k)), "cx2-series-factorial-dropped"),
        (
            qsym(_maclaurin(kind, c, k + 1)) if _maclaurin(kind, c, k + 1) != 0 else qsym(value) * 2,
            "cx2-laurent-index-off-by-one",
        ),
        (
            qsym(_maclaurin(kind, c, k - 1)) if k >= 1 and _maclaurin(kind, c, k - 1) != 0 else -v,
            "cx2-laurent-index-off-by-one",
        ),
        (-v, "cx2-series-alternating-sign-dropped" if kind in ("sin", "cos") else "calc-sign-slip"),
        (qsym(value / Fr(c) ** k) if abs(c) != 1 else qsym(value) * 2, "cx2-laurent-argument-scale-dropped"),
        (
            qsym(_maclaurin(kind, c, n)) if n >= 0 and _maclaurin(kind, c, n) != 0 else qsym(value) * 3,
            "cx2-laurent-power-shift-ignored",
        ),
        (v * 2, "cx2-series-coefficient-slip"),
        (v / 2, "cx2-series-coefficient-slip"),
    ]
    working = [
        f"Expand {_g_x(kind, _cz(c)).p} as a Maclaurin series and divide by z^{m}.",
        f"The term in z^{n} comes from the term in z^{k} of the series, whose coefficient is {atext(v)}.",
    ]
    return single(_phrase(rng, fx, term, C(0)), v, working, wrongs)


def _easy_exp_about_point(rng):
    for _ in attempts(500, "an easy shifted exponential"):
        c = rng.choice([1, 2, 3, -1])
        z0 = rng.choice([1, 2, -1, -2])
        m = rng.randint(1, 4)
        k = rng.randint(0, m + 2)
        n = k - m
        if n >= 3:
            continue
        break
    coeff = Fr(c) ** k / math.factorial(k)
    value = qsym(coeff) * sympy.exp(c * z0)
    den = fac_x(C(z0), m)
    fx = _fline(frac_x(_g_x("exp", _cz(c)), den))
    term = (
        _term_x(C(z0), n)
        if n != 0
        else X(f"({zminus_x(C(z0)).p})⁰ (the constant term)", f"({zminus_x(C(z0)).t})^{{0}}")
    )
    numeric_close(
        _coeff_numeric(lambda z: cmath.exp(c * z) / (z - z0) ** m, complex(z0), 0.7, n),
        complex(sympy.N(value, 15)),
        "Laurent coefficient",
    )
    wrongs = [
        (qsym(coeff), "cx2-laurent-centre-factor-dropped"),
        (qsym(coeff) * sympy.exp(c), "cx2-laurent-centre-factor-dropped"),
        (qsym(coeff * math.factorial(k)) * sympy.exp(c * z0), "cx2-series-factorial-dropped"),
        (qsym(Fr(c) ** (k + 1) / math.factorial(k + 1)) * sympy.exp(c * z0), "cx2-laurent-index-off-by-one"),
        (
            qsym(Fr(1, math.factorial(k))) * sympy.exp(c * z0) if abs(c) != 1 else -value,
            "cx2-laurent-argument-scale-dropped",
        ),
        (qsym(coeff) * sympy.exp(c * z0) * (1 if c > 0 else -1) * (-1), "calc-sign-slip"),
        (value * 2, "cx2-series-coefficient-slip"),
        (value / 2, "cx2-series-coefficient-slip"),
    ]
    working = [
        f"Write z = {z0} + u, so e^({_cz(c)}) = e^({c * z0}) e^({_cz(c, 'u')}) and f = e^({c * z0}) e^({_cz(c, 'u')})/u^{m}.",
        f"The coefficient of u^{n} is e^({c * z0}) times the coefficient of u^{k} in e^({_cz(c, 'u')}), which is {atext(qsym(coeff))}.",
    ]
    return single(_phrase(rng, fx, term, C(z0)), value, working, wrongs)


def _easy_rational(rng):
    for _ in attempts(500, "an easy rational Laurent coefficient"):
        a = rng.choice(
            [C(k) for k in (-3, -2, -1, 1, 2, 3)] + [C(0, k) for k in (-2, -1, 1, 2)] + [C(1, 1), C(1, -1), C(-1, 1)]
        )
        m = rng.randint(1, 3)
        c = rng.choice([1, 2, 3, -1, -2])
        n = rng.randint(-m, 2)
        value = -C(c) / a ** (n + m + 1)
        if not _tidy(value):
            continue
        break
    den = prod_x([fac_x(C(0), m), fac_x(a)])
    fx = _fline(frac_x(poly_x([C(c)]), den))
    absa = _dist(a)
    region = X(f"0 < |z| < {absa}", f"0 < |z| < {absa}")
    term = _term_x(C(0), n) if n != 0 else X("z⁰ (the constant term)", "z^{0}")
    fn = lambda z: c / (z**m * (z - complex(a)))  # noqa: E731
    numeric_close(_coeff_numeric(fn, 0j, 0.5 * float(math.sqrt(a.abs2())), n), complex(value), "Laurent coefficient")
    s = value.sym()
    wrongs = [
        (-s, "calc-sign-slip"),
        (
            (-C(c) / a ** (n + m)).sym() if n + m >= 1 else (-C(c) / a ** (n + m + 2)).sym(),
            "cx2-laurent-index-off-by-one",
        ),
        ((-C(c) * a ** (n + m + 1)).sym(), "cx2-laurent-geometric-ratio-inverted"),
        ((C(c) / a ** (n + m + 1) * 0 + -C(c) / a ** (n + m + 2)).sym(), "cx2-laurent-index-off-by-one"),
        ((C(c) / a ** (n + m + 1)).sym() * 2, "cx2-laurent-geometric-series-slip"),
        (s * 2, "cx2-laurent-geometric-series-slip"),
        (s / 2, "cx2-laurent-geometric-series-slip"),
    ]
    working = [
        f"For |z| < {absa}, 1/(z - a) = -(1/a) times the geometric series in z/a.",
        f"So f(z) = {c}/z^{m} × (-(1/a)) Σ (z/a)^j, and the coefficient of z^{n} is -{c}/a^{n + m + 1} = {atext(s)}.",
    ]
    return single(_phrase(rng, fx, term, C(0), region), s, working, wrongs)


def _dist(a: C) -> str:
    d2 = a.abs2()
    if d2.denominator == 1:
        r = math.isqrt(d2.numerator)
        return str(r) if r * r == d2 else f"√{d2.numerator}"
    return f"√({d2.numerator}/{d2.denominator})"


# -- medium ----------------------------------------------------------------------------------------------------------


def _medium_setup(rng, m):
    for _ in attempts(800, "a higher-order Laurent setup"):
        a = rng.choice([C(k) for k in (-2, -1, 0, 1, 2)] + [C(0, 1), C(0, -1), C(1, 1), C(-1, 1)])
        b = rng.choice(
            [
                p
                for p in [C(k) for k in (-3, -2, -1, 0, 1, 2, 3)] + [C(0, 1), C(0, -1), C(0, 2), C(1, -1), C(2, 1)]
                if p != a
            ]
        )
        num = [C(rng.randint(-4, 4)), C(rng.randint(-3, 3)), C(rng.choice([0, 0, 1, -1]))]
        if all(c.is_zero() for c in num):
            continue
        if sum((num[k] * b**k for k in range(len(num))), C(0)).is_zero():
            continue
        rat = Rat(num, [(a, m), (b, 1)])
        lau = rat.laurent(0, 5)
        if lau[-m].is_zero() or lau[-1].is_zero() and m == 2 or not all(_tidy(v) for v in lau.values()):
            continue
        return a, b, num, rat, lau
    raise AssertionError


def _medium_principal(rng):
    m = rng.choice([2, 2, 3])
    a, b, num, rat, lau = _medium_setup(rng, m)
    pp = sum((lau[-k].sym() / (_Z - a.sym()) ** k for k in range(1, m + 1)), sympy.Integer(0))
    fx = _fline(rat_x(rat))
    at = cx_x(a)
    swap = sum((lau[-(m + 1 - k)].sym() / (_Z - a.sym()) ** k for k in range(1, m + 1)), sympy.Integer(0))
    top_only = lau[-m].sym() / (_Z - a.sym()) ** m
    nval = sum((num[k] * a**k for k in range(len(num))), C(0))
    rest = C(1)
    for k in range(1):
        rest = rest * (a - b)
    nonreg = sum((nval.sym() / (_Z - a.sym()) ** k * (1 if k == m else 0) for k in range(1, m + 1)), sympy.Integer(0))
    signed = sum((((-1) ** (k + 1)) * lau[-k].sym() / (_Z - a.sym()) ** k for k in range(1, m + 1)), sympy.Integer(0))
    wrongs = [
        (swap, "cx2-laurent-coefficients-swapped"),
        (top_only, "cx2-principal-part-term-dropped"),
        (signed, "cx2-laurent-pole-sign"),
        (nonreg, "cx2-residue-numerator-only"),
        (rat_sym(rat), "cx2-principal-part-includes-regular"),
        (
            sum((lau[-k].sym() / (_Z - a.sym()) ** k * k for k in range(1, m + 1)), sympy.Integer(0)),
            "cx2-laurent-derivative-for-coefficient",
        ),
        (pp + lau[0].sym(), "cx2-principal-part-includes-regular"),
        (
            sum(
                (lau[-k].sym() * (2 if k == m else 1) / (_Z - a.sym()) ** k for k in range(1, m + 1)), sympy.Integer(0)
            ),
            "cx2-series-coefficient-slip",
        ),
        (
            sum(
                (lau[-k].sym() * (-1 if k == 1 else 1) / (_Z - a.sym()) ** k for k in range(1, m + 1)), sympy.Integer(0)
            ),
            "cx2-laurent-pole-sign",
        ),
    ]
    for j, (_pole, _) in enumerate(rat.poles[:1]):
        check(j == 0, "pole index")
    # second route: every negative-power coefficient from the contour formula
    for k in range(1, m + 1):
        numeric_close(
            _coeff_numeric(rat.at, complex(a), 0.3 * min(1.0, abs(complex(a) - complex(b))), -k),
            complex(lau[-k]),
            "principal part coefficient",
        )
    working = [
        f"Write f(z) = g(z)/{fac_x(a, m).p}, where g(z) is analytic at {at.p}.",
        "The coefficients of the negative powers are g(a), g'(a), g''(a)/2, ... in turn.",
        "The principal part is " + atext(pp) + ".",
    ]
    lead = [
        lambda: msay(
            "Find the principal part of ",
            fx,
            " at its pole ",
            X(f"z = {at.p}", f"z = {at.t}"),
            ". Give it as an expression in z.",
        ),
        lambda: msay(
            "Let ",
            fx,
            ". Write down the principal part of the Laurent series of f about ",
            X(f"z = {at.p}", f"z = {at.t}"),
            ", as an expression in z.",
        ),
        lambda: msay(
            "The function ",
            fx,
            " has a Laurent series about ",
            X(f"z = {at.p}", f"z = {at.t}"),
            ". Find the sum of its terms with negative powers of ",
            X(f"({zminus_x(a).p})" if not a.is_zero() else "z", f"({zminus_x(a).t})" if not a.is_zero() else "z"),
            ".",
        ),
    ][rng.randrange(3)]
    return single(lead(), pp, working, wrongs, fallback_tag="cx2-series-coefficient-slip")


def rat_sym(rat: Rat):
    top = sum((c.sym() * _Z**k for k, c in enumerate(rat.num)), sympy.Integer(0))
    bottom = sympy.Integer(1)
    for a, m in rat.poles:
        bottom *= (_Z - a.sym()) ** m
    return top / bottom


def _medium_regular(rng):
    m = rng.choice([2, 3])
    a, b, num, rat, lau = _medium_setup(rng, m)
    n = rng.choice([0, 1, 2])
    if lau[n].is_zero():
        n = next(k for k in (0, 1, 2, 3, 4) if not lau[k].is_zero())
    value = lau[n]
    fx = _fline(rat_x(rat))
    term = (
        _term_x(a, n)
        if n
        else X(
            f"({zminus_x(a).p})⁰ (the constant term)" if not a.is_zero() else "z⁰ (the constant term)", "constant term"
        )
    )
    numeric_close(
        _coeff_numeric(rat.at, complex(a), 0.3 * min(1.0, abs(complex(a) - complex(b))), n),
        complex(value),
        "Laurent coefficient",
    )
    v = value.sym()
    wrongs = [
        (lau[n + 1].sym(), "cx2-laurent-index-off-by-one"),
        (lau[n - 1].sym() if n >= 1 else lau[-1].sym(), "cx2-laurent-index-off-by-one"),
        (-v, "calc-sign-slip"),
        (lau[-m].sym(), "cx2-residue-simple-formula"),
        (lau[n].sym() * (n + 1), "cx2-laurent-derivative-for-coefficient"),
        (v * 2, "cx2-laurent-index-off-by-one"),
    ]
    working = [
        f"Write f(z) = g(z)/{fac_x(a, m).p} with g analytic at {cx_x(a).p}, and expand g in powers of {zminus_x(a).p}.",
        f"The coefficient of {term.p} in f is the coefficient of u^{n + m} in g, where u = {zminus_x(a).p}: {atext(v)}.",
    ]
    at = cx_x(a)
    return single(_phrase(rng, fx, term, a), v, working, wrongs, fallback_tag="cx2-laurent-index-off-by-one")


# -- hard ------------------------------------------------------------------------------------------------------------

_ALPHAS = [C(1), C(-1), C(2), C(-2), C(0, 1), C(0, -1), C(0, 2), C(0, -2)]
_BETAS = [C(3), C(-3), C(4), C(-4), C(0, 3), C(0, -3), C(5), C(-5), C(3, 4), C(3, -4), C(4, 3), C(-3, 4)]
_CENTRES = [C(0), C(0), C(1), C(-1), C(2), C(0, 1), C(1, 1)]


def _hard_annulus(rng):
    for _ in attempts(500, "an annulus coefficient"):
        w = rng.choice(_CENTRES)
        alpha = rng.choice(_ALPHAS)
        beta = rng.choice(_BETAS)
        c = rng.choice([1, 2, 3, -1, -2])
        n = rng.randint(-4, 3)
        k = c * 1
        pref = C(k) / (alpha - beta)
        value = pref * beta ** (-(n + 1)) if n >= 0 else pref * alpha ** (-n - 1)
        if not _tidy(value) or value.is_zero():
            continue
        break
    p, q = w + alpha, w + beta
    fx = _fline(frac_x(poly_x([C(c)]), prod_x([fac_x(p), fac_x(q)])))
    r1, r2 = _dist(alpha), _dist(beta)
    region = X(f"{r1} < |{zminus_x(w).p}| < {r2}", f"{r1} < |{zminus_x(w).t}| < {r2}")
    term = (
        _term_x(w, n)
        if n != 0
        else X(
            f"({zminus_x(w).p})⁰ (the constant term)" if not w.is_zero() else "z⁰ (the constant term)", "constant term"
        )
    )
    fn = lambda z: c / ((z - complex(p)) * (z - complex(q)))  # noqa: E731
    rad = (math.sqrt(float(alpha.abs2())) + math.sqrt(float(beta.abs2()))) / 2
    numeric_close(_coeff_numeric(fn, complex(w), rad, n), complex(value), "annulus coefficient", 1e-6)
    v = value.sym()
    # the two wrong regions and common slips
    inner = (
        pref * (-(alpha ** (-(n + 1))) + beta ** (-(n + 1))) if n >= 0 else C(0)
    )  # |u| < |alpha|: Taylor series only
    outer = pref * (alpha ** (-n - 1) - beta ** (-n - 1)) if n < 0 else C(0)  # |u| > |beta|
    wrongs = [
        (inner.sym() if n >= 0 else C(0).sym(), "cx2-laurent-wrong-annulus"),
        (outer.sym() if n < 0 else (pref * -(alpha ** (-(n + 1)))).sym(), "cx2-laurent-wrong-annulus"),
        ((pref * beta ** (-n)).sym() if n >= 0 else (pref * alpha ** (-n)).sym(), "cx2-laurent-index-off-by-one"),
        (
            (C(c) * beta ** (-(n + 1))).sym() if n >= 0 else (C(c) * alpha ** (-n - 1)).sym(),
            "cx2-laurent-partial-fraction-factor-dropped",
        ),
        (-v, "calc-sign-slip"),
        (
            (pref * alpha ** (-(n + 1))).sym() if n >= 0 else (pref * beta ** (-n - 1)).sym(),
            "cx2-laurent-wrong-annulus",
        ),
    ]
    working = [
        f"With u = {zminus_x(w).p}, the poles are at u = {cx_x(alpha).p} and u = {cx_x(beta).p}, and the annulus {r1} < |u| < {r2} lies between them.",
        "Partial fractions: f = c/(α - β) × (1/(u - α) - 1/(u - β)).",
        f"For |u| > |α|, 1/(u - α) = Σ α^(k-1) u^(-k); for |u| < |β|, -1/(u - β) = Σ u^k/β^(k+1).",
        f"The coefficient of u^{n} is {atext(v)}.",
    ]
    return single(_phrase(rng, fx, term, w, region), v, working, wrongs, fallback_tag="cx2-laurent-wrong-annulus")


def _hard_essential(rng):
    for _ in attempts(500, "an essential-singularity coefficient"):
        kind = rng.choice(["exp", "sin", "cos"])
        c = rng.choice([1, 2, 3, -1, -2])
        p = rng.randint(0, 3)
        j = rng.randint(0, 6)
        n = p - j
        if kind == "exp":
            value = Fr(c) ** j / math.factorial(j)
        elif kind == "sin":
            value = Fr((-1) ** ((j - 1) // 2)) * Fr(c) ** j / math.factorial(j) if j % 2 else Fr(0)
        else:
            value = Fr((-1) ** (j // 2)) * Fr(c) ** j / math.factorial(j) if j % 2 == 0 else Fr(0)
        if value == 0 or n >= 3:
            continue
        break
    arg = f"{c}/z" if c != 1 else "1/z"
    inner = (
        X(f"e^({arg})", f"e^{{{_tex_c_over_z(c)}}}")
        if kind == "exp"
        else X(f"{kind}({arg})", rf"\{kind}\left({_tex_c_over_z(c)}\right)")
    )
    pre = X("", "") if p == 0 else (X("z", "z") if p == 1 else X(f"z{'²' if p == 2 else '³'}", f"z^{{{p}}}"))
    pre_dot = X(pre.p + " " if pre.p else "", pre.t + (r"\," if pre.t else ""))
    fx = X(f"f(z) = {pre_dot.p}{inner.p}", f"f(z) = {pre_dot.t}{inner.t}")
    term = _term_x(C(0), n) if n != 0 else X("z⁰ (the constant term)", "z^{0}")
    fn = {"exp": cmath.exp, "sin": cmath.sin, "cos": cmath.cos}[kind]
    numeric_close(_coeff_numeric(lambda z: z**p * fn(c / z), 0j, 1.0, n), float(value), "essential coefficient")
    v = qsym(value)
    wrongs = [
        (qsym(value * math.factorial(j)), "cx2-series-factorial-dropped"),
        (-v, "cx2-series-alternating-sign-dropped" if kind != "exp" else "calc-sign-slip"),
        (qsym(Fr(c) ** (j + 1) / math.factorial(j + 1)) * (1 if kind == "exp" else 1), "cx2-laurent-index-off-by-one"),
        (qsym(Fr(1, math.factorial(j))), "cx2-laurent-argument-scale-dropped")
        if abs(c) != 1
        else (v * 2, "cx2-series-coefficient-slip"),
        (qsym(Fr(c) ** (j - 1) / math.factorial(j - 1)) if j >= 1 else v * 3, "cx2-laurent-index-off-by-one"),
        (qsym(Fr(c) ** (p + n if p + n >= 0 else 0) / math.factorial(abs(p + n))), "cx2-laurent-power-shift-ignored"),
    ]
    working = [
        f"Substitute w = 1/z into the {'exponential' if kind == 'exp' else kind} series: the term in z^(-{j}) of {inner.p} has coefficient {atext(v) if p == 0 else atext(qsym(value))}.",
        f"Multiplying by z^{p} moves it to z^{n}, so the coefficient of z^{n} is {atext(v)}.",
    ]
    return single(
        _phrase(rng, fx, term, C(0), X("0 < |z| < ∞", r"0 < |z| < \infty")),
        v,
        working,
        wrongs,
        fallback_tag="cx2-series-coefficient-slip",
    )


def _tex_c_over_z(c: int) -> str:
    return r"\frac{1}{z}" if c == 1 else rf"\frac{{{c}}}{{z}}"


def _series_quotient(kind: str, c: int, count: int):
    """(shift, coefficients): the Laurent series of cot(cz), 1/(e^(cz) - 1) or 1/sin^2(cz) as z^(-shift) * sum q_k z^k."""
    cc = Fr(c)
    if kind == "cot":
        num = [C(_maclaurin("cos", c, k)) for k in range(count + 2)]
        den = [C(_maclaurin("sin", c, k + 1)) for k in range(count + 2)]
        return 1, sdiv(num, den, count)
    if kind == "expm1":
        den = [C(cc**k / math.factorial(k)) for k in range(1, count + 3)]
        return 1, sdiv([C(1)], den, count)
    sin_over_z = [C(_maclaurin("sin", c, k + 1)) for k in range(count + 2)]
    sq = [C(0)] * (count + 2)
    for i in range(count + 2):
        for j in range(count + 2 - i):
            sq[i + j] = sq[i + j] + sin_over_z[i] * sin_over_z[j]
    return 2, sdiv([C(1)], sq, count)


def _hard_quotient(rng):
    for _ in attempts(500, "a quotient-series coefficient"):
        kind = rng.choice(["cot", "expm1", "sin2"])
        c = rng.choice([1, 1, 2, 3, -1]) if kind != "sin2" else rng.choice([1, 2, 3])
        shift, q = _series_quotient(kind, c, 8)
        n = rng.randint(-shift, 3)
        value = q[n + shift]
        if value.is_zero() or not _tidy(value):
            continue
        break
    v = value.sym()
    names = {"cot": f"cot({_cz(c)})", "expm1": f"1/(e^({_cz(c)}) - 1)", "sin2": f"1/sin²({_cz(c)})"}
    texes = {
        "cot": rf"\cot({_cz(c)})",
        "expm1": rf"\frac{{1}}{{e^{{{_cz(c)}}}-1}}",
        "sin2": rf"\frac{{1}}{{\sin^{{2}}({_cz(c)})}}",
    }
    fx = X(f"f(z) = {names[kind]}", f"f(z) = {texes[kind]}")
    term = _term_x(C(0), n) if n != 0 else X("z⁰ (the constant term)", "z^{0}")
    fns = {
        "cot": lambda z: cmath.cos(c * z) / cmath.sin(c * z),
        "expm1": lambda z: 1 / (cmath.exp(c * z) - 1),
        "sin2": lambda z: 1 / cmath.sin(c * z) ** 2,
    }
    numeric_close(_coeff_numeric(fns[kind], 0j, 0.5, n), complex(value), "quotient coefficient", 1e-7)
    unit_shift, unit_q = _series_quotient(kind, 1, 8)
    wrongs = [
        (
            unit_q[n + unit_shift].sym() if abs(c) != 1 else -v,
            "cx2-laurent-argument-scale-dropped" if abs(c) != 1 else "calc-sign-slip",
        ),
        (q[n + shift + 1].sym() if not q[n + shift + 1].is_zero() else -v, "cx2-laurent-index-off-by-one"),
        (
            q[n + shift - 1].sym() if n + shift - 1 >= 0 and not q[n + shift - 1].is_zero() else v * 2,
            "cx2-laurent-index-off-by-one",
        ),
        (-v, "calc-sign-slip"),
        ((value * Fr(c) ** 0 / Fr(c)).sym() if c not in (1,) else v * 3, "cx2-laurent-argument-scale-dropped"),
    ]
    working = [
        "Write the numerator and denominator as Maclaurin series, divide the denominator through by its leading power of z, and divide the series term by term.",
        f"The coefficient of {term.p} is {atext(v)}.",
    ]
    return single(
        _phrase(rng, fx, term, C(0), X("0 < |z| < 1/2", "0 < |z| < \\tfrac{1}{2}")),
        v,
        working,
        wrongs,
        fallback_tag="cx2-series-coefficient-slip",
    )


_TIERS = {
    "easy": [_easy_maclaurin_over_power, _easy_exp_about_point, _easy_rational],
    "medium": [_medium_principal, _medium_principal, _medium_regular],
    "hard": [_hard_annulus, _hard_annulus, _hard_essential, _hard_quotient],
}


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5a_laurent(rng: random.Random, tier: str = "easy") -> dict:
    """A Laurent coefficient or a principal part, exact."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return rng.choice(_TIERS[tier])(rng)


# -- exam ------------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w5a_laurent(rng: random.Random, tier: str = "exam") -> dict:
    """(pz + q)/((z - a)^2 (z - b)): the two principal-part coefficients, its value at a + 1, the radius of convergence."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a Laurent chain"):
        a = C(rng.choice([-2, -1, 1, 2]))
        b = rng.choice(
            [
                C(rng.choice([-3, -2, -1, 0, 1, 2, 3])),
                C(0, rng.choice([1, 2, 3, -1, -2])),
                C(rng.choice([-1, 1]), rng.choice([1, -1])),
            ]
        )
        if b == a:
            continue
        num = [C(rng.randint(-4, 4)), C(rng.choice([1, 2, 3, -1, -2]))]
        if sum((num[k] * b**k for k in range(2)), C(0)).is_zero():
            continue
        rat = Rat(num, [(a, 2), (b, 1)])
        lau = rat.laurent(0, 2)
        c2, c1 = lau[-2], lau[-1]
        if c2.is_zero() or c1.is_zero() or not _tidy(c2) or not _tidy(c1):
            continue
        break
    radius = sympy.sqrt(sympy.Integer(int((a - b).abs2()))) if (a - b).abs2().denominator == 1 else None
    check(radius is not None, "radius")
    value = (c2 + c1).sym()
    numeric_close(_coeff_numeric(rat.at, complex(a), 0.4, -2), complex(c2), "a_-2")
    numeric_close(_coeff_numeric(rat.at, complex(a), 0.4, -1), complex(c1), "a_-1")
    fx = _fline(rat_x(rat))
    at = cx_x(a)
    stem = msay(
        "Let ",
        fx,
        ". Write a_n for the coefficient of ",
        X(f"({zminus_x(a).p})^n", f"({zminus_x(a).t})^{{n}}"),
        " in the Laurent series of f about ",
        X(f"z = {at.p}", f"z = {at.t}"),
        ".",
    )
    na = X(f"a_(-2)", "a_{-2}")
    nb = X(f"a_(-1)", "a_{-1}")
    w_a = Link(
        msay("Find the coefficient ", na, " of ", X(f"({zminus_x(a).p})^(-2)", f"({zminus_x(a).t})^{{-2}}"), "."),
        c2.sym(),
        "expression",
        2,
        [f"a_(-2) = g(a), where g(z) = N(z)/({zminus_x(b).p}): {atext(c2.sym())}."],
        [
            (-c2.sym(), "calc-sign-slip"),
            (sum((num[k] * a**k for k in range(2)), C(0)).sym(), "cx2-residue-numerator-only"),
            (c1.sym(), "cx2-laurent-coefficients-swapped"),
            ((c2 * (a - b)).sym(), "cx2-laurent-partial-fraction-factor-dropped"),
        ],
    )
    w_b = Link(
        msay("Find the coefficient ", nb, " of ", X(f"({zminus_x(a).p})^(-1)", f"({zminus_x(a).t})^{{-1}}"), "."),
        c1.sym(),
        "expression",
        3,
        [f"a_(-1) = g'(a) = (N'(a)(a - b) - N(a))/(a - b)² = {atext(c1.sym())}."],
        [
            (-c1.sym(), "calc-sign-slip"),
            (c2.sym(), "cx2-laurent-coefficients-swapped"),
            ((num[1] / (a - b)).sym(), "cx2-residue-numerator-derivative-only"),
            ((c1 * (a - b)).sym(), "cx2-laurent-partial-fraction-factor-dropped"),
        ],
    )
    w_c = Link(
        msay(
            "Hence find the value of the principal part of f at ",
            X(f"z = {cx_x(a + C(1)).p}", f"z = {cx_x(a + C(1)).t}"),
            ".",
        ),
        value,
        "expression",
        2,
        [
            f"The principal part is a_(-2)/(z - a)² + a_(-1)/(z - a). At z = a + 1 both denominators are 1, so the value is a_(-2) + a_(-1) = {atext(value)}."
        ],
        [
            (c2.sym(), "cx2-principal-part-term-dropped"),
            (c1.sym(), "cx2-principal-part-term-dropped"),
            ((c2 - c1).sym(), "cx2-laurent-pole-sign"),
            ((c2 + c1 * 2).sym(), "cx2-laurent-index-off-by-one"),
        ],
        ft="part_a + part_b",
    )
    w_d = Link(
        msay(
            "The Laurent series about ",
            X(f"z = {at.p}", f"z = {at.t}"),
            " is valid for ",
            X(f"0 < |{zminus_x(a).p}| < R", f"0 < |{zminus_x(a).t}| < R"),
            ". Find the exact value of R.",
        ),
        radius,
        "expression",
        2,
        [
            f"The other singularity is z = {cx_x(b).p}, at distance |a - b| = {atext(radius)} from a, so R = {atext(radius)}."
        ],
        [
            (radius**2, "cx2-radius-distance-squared"),
            (sympy.Integer(1), "cx2-radius-guessed-one"),
            (
                sympy.Abs(sympy.Integer(int(abs(a.re))) if abs(a.re) > 0 else sympy.Integer(1)) + 0,
                "cx2-radius-from-origin",
            ),
            (
                sympy.sqrt(sympy.Integer(int(b.abs2()))) if b.abs2().denominator == 1 and b.abs2() != 0 else radius + 1,
                "cx2-radius-from-origin",
            ),
            (sympy.Integer(int(abs((a - b).re) + abs((a - b).im))), "cx2-radius-distance-slip"),
            (sympy.Integer(int(max(abs((a - b).re), abs((a - b).im)))), "cx2-radius-distance-slip"),
            (radius + 1, "cx2-radius-distance-slip"),
            (radius + 2, "cx2-radius-distance-slip"),
            (radius - 1 if (radius - 1).is_positive else radius + 3, "cx2-radius-distance-slip"),
            (sympy.sqrt(sympy.Integer(int(a.abs2()) + int(b.abs2()))), "cx2-radius-from-origin"),
        ],
    )
    return exam_dict(stem, [w_a, w_b, w_c, w_d])
