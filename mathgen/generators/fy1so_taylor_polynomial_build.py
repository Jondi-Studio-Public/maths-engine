"""Generates `fy1so_taylor_polynomial` (easy, medium, hard) and the exam chain `fy1so_exd_taylor_polynomial`.

Skill ``ucv1-taylor-polynomial`` (domain ``single-variable-calculus``, answer ``fraction``): the coefficients of Taylor
and Maclaurin polynomials of order 2 to 4, and using such a polynomial to approximate a value. The order-1 case is the
separate skill ``ucv1-taylor-linear-approximation``.

Tiers (the reasoning escalates; the numbers stay small):

- ``easy``: the coefficient of x^k (k = 2, 3 or 4) in the Maclaurin series of e^(ax), sin(ax), cos(ax), ln(1 + ax) or
  1/(1 - ax), where the inner coefficient a has to be raised to the k-th power. A sine has no even terms and a cosine no
  odd ones, so k is chosen where the coefficient is not zero.
- ``medium``: the coefficient of (x - a)^k in the Taylor series about a point a that is not 0, of ln x, 1/x, √x,
  1/√x or x^(1/3): f^(k)(a) / k! with the derivative taken before the point is substituted.
- ``hard``: the coefficient of x^k (k = 3 or 4) in a product of two standard series (every pair of terms whose powers add
  to k has to be collected), or in a composite such as e^(a x²), whose x⁴ term comes from the second term of the outer
  series.
- ``exam`` (topic ``fy1so_exd_taylor_polynomial``): a four-part chain for a function f of the form e^(ax), ln(1 + ax) or
  √(1 + ax). (a) the coefficient of x², (b) the coefficient of x³, (c) the exact value of the cubic Maclaurin polynomial at
  a stated small x (follows through from (a) and (b)), (d) the absolute error |f(x) - P(x)| of that approximation, to a stated
  number of decimal places (follows through from (c)).

Every coefficient is computed by a closed formula (a power of the inner coefficient over a factorial, a generalised
binomial coefficient, a collected product of truncated series) and again as f^(k)(a)/k! with SymPy's derivative. The
exam's errors are re-derived from SymPy's high-precision value of f.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from ._fy1so_common import (
    Link,
    M,
    build_part,
    build_single,
    rounds_clear,
    say,
    tex_decimal,
    tex_rational,
)

TOPIC = "fy1so_taylor_polynomial"
EXAM_TOPIC = "fy1so_exd_taylor_polynomial"

_X = sympy.Symbol("x")


def _sym(f: Fraction):
    return sympy.Rational(f.numerator, f.denominator)


def _binom(p: Fraction, k: int) -> Fraction:
    """The generalised binomial coefficient p(p-1)...(p-k+1)/k!."""
    out = Fraction(1)
    for i in range(k):
        out *= p - i
    return out / math.factorial(k)


def _scaled_x_tex(a: Fraction) -> str:
    """ax as TeX: x, -x, 2x, \\frac{x}{2}, -\\frac{3x}{2}."""
    sign = "-" if a < 0 else ""
    p, q = abs(a.numerator), a.denominator
    top = "x" if p == 1 else f"{p}x"
    return f"{sign}{top}" if q == 1 else rf"{sign}\frac{{{top}}}{{{q}}}"


def _one_plus_tex(a: Fraction) -> str:
    """1 + ax as TeX: 1 + x, 1 - 2x, 1 + \\frac{x}{2}."""
    inner = _scaled_x_tex(abs(a))
    return f"1 {'-' if a < 0 else '+'} {inner}"


# -- Maclaurin series of the standard functions ----------------------------------------------------------------------


@dataclass(frozen=True)
class Series:
    """A standard function: its TeX, its SymPy form and the closed formula for its Maclaurin coefficients."""

    name: str
    tex: str
    expr: sympy.Expr
    coeff: object  # k -> Fraction


def _exp(a: Fraction) -> Series:
    return Series("exp", rf"e^{{{_scaled_x_tex(a)}}}", sympy.exp(_sym(a) * _X), lambda k: a**k / math.factorial(k))


def _sin(a: Fraction) -> Series:
    return Series(
        "sin",
        rf"\sin({_scaled_x_tex(a)})",
        sympy.sin(_sym(a) * _X),
        lambda k: Fraction(0) if k % 2 == 0 else (-1) ** ((k - 1) // 2) * a**k / math.factorial(k),
    )


def _cos(a: Fraction) -> Series:
    return Series(
        "cos",
        rf"\cos({_scaled_x_tex(a)})",
        sympy.cos(_sym(a) * _X),
        lambda k: Fraction(0) if k % 2 else (-1) ** (k // 2) * a**k / math.factorial(k),
    )


def _ln(a: Fraction) -> Series:
    return Series(
        "ln",
        rf"\ln({_one_plus_tex(a)})",
        sympy.log(1 + _sym(a) * _X),
        lambda k: Fraction(0) if k == 0 else Fraction((-1) ** (k + 1)) * a**k / k,
    )


def _geometric(a: Fraction) -> Series:
    return Series(
        "geom",
        rf"\frac{{1}}{{{_one_plus_tex(-a)}}}",
        1 / (1 - _sym(a) * _X),
        lambda k: a**k,
    )


def _root(a: Fraction) -> Series:
    return Series(
        "root",
        rf"\sqrt{{{_one_plus_tex(a)}}}",
        sympy.sqrt(1 + _sym(a) * _X),
        lambda k: _binom(Fraction(1, 2), k) * a**k,
    )


def _derivative_coefficient(expr: sympy.Expr, k: int, point=0) -> Fraction:
    """f^(k)(point) / k! by SymPy's derivative: the second route for every coefficient."""
    value = sympy.diff(expr, _X, k).subs(_X, point) / sympy.factorial(k)
    value = sympy.nsimplify(value)
    check(value.is_Rational, f"coefficient {value} is not rational")
    return Fraction(int(value.p), int(value.q))


# -- easy ------------------------------------------------------------------------------------------------------------


_EASY_A = [Fraction(v) for v in (-3, -2, -1, 1, 2, 3)] + [Fraction(1, 2), Fraction(-1, 2)]


def _easy(rng: random.Random):
    name = rng.choice(["exp", "exp", "sin", "cos", "ln", "ln", "geom"])
    a = rng.choice(_EASY_A)
    if name == "sin":
        k = 3
        a = rng.choice(
            [Fraction(v) for v in (-3, -2, 2, 3)] + [Fraction(1, 2), Fraction(-1, 2), Fraction(-1), Fraction(1)]
        )
    elif name == "cos":
        k = rng.choice([2, 4])
    elif name == "geom":
        a = rng.choice([Fraction(v) for v in (-3, -2, -1, 2, 3)])
        k = rng.choice([2, 3, 4])
    else:
        k = rng.choice([2, 3, 4])
    series = {"exp": _exp, "sin": _sin, "cos": _cos, "ln": _ln, "geom": _geometric}[name](a)
    value = series.coeff(k)
    check(value == _derivative_coefficient(series.expr, k), f"{series.tex} coefficient of x^{k} disagrees")
    check(value != 0, "zero coefficient in the easy tier")
    fact = math.factorial(k)
    raw = value * fact  # f^(k)(0)
    if name == "ln":
        wrongs = [
            (a**k / fact, "taylor-wrong-factorial"),
            (raw, "taylor-derivative-not-divided"),
            (Fraction((-1) ** (k + 1), k), "taylor-inner-coefficient-ignored"),
            (-value, "taylor-series-sign-pattern"),
            (value / a ** (k - 1), "taylor-inner-coefficient-not-powered"),
        ]
    elif name == "geom":
        wrongs = [
            (a**k / fact, "taylor-wrong-factorial"),
            (value * a, "taylor-inner-coefficient-not-powered"),
            (raw, "taylor-derivative-not-divided"),
            (Fraction(1), "taylor-inner-coefficient-ignored"),
            (-value, "taylor-series-sign-pattern"),
        ]
    else:
        wrongs = [
            (raw, "taylor-derivative-not-divided"),
            (value / a ** (k - 1), "taylor-inner-coefficient-not-powered"),
            (
                Fraction(1, fact) * (1 if name == "exp" else value.numerator // abs(value.numerator)),
                "taylor-inner-coefficient-ignored",
            ),
            (-value, "taylor-series-sign-pattern"),
            (value * k, "taylor-wrong-factorial"),
        ]
    order = {2: "x²", 3: "x³", 4: "x⁴"}[k]
    ask = rng.choice(
        [
            ("Find the coefficient of ", " in the Maclaurin series of "),
            ("Write down the coefficient of ", " in the Maclaurin series of "),
            ("What is the coefficient of ", " in the Maclaurin expansion of "),
        ]
    )
    prompt = say(ask[0], M(f"x^{{{k}}}"), ask[1], M(rf"f(x) = {series.tex}"), ".")
    working = _easy_working(name, a, k, value)
    return prompt, Link(prompt, "fraction", value, working=working, wrongs=wrongs), order


def _easy_working(name: str, a: Fraction, k: int, value: Fraction) -> list[str]:
    from ._fy1so_common import plain

    fact = math.factorial(k)
    a_text = f"({Fraction(a)})" if a.denominator != 1 or a < 0 else str(a)
    if name == "exp":
        return [
            f"e^u = 1 + u + u²/2! + u³/3! + ..., with u = {plain(_scaled_x_tex(a))}.",
            f"The x^{k} term is {a_text}^{k}/{fact}, so the coefficient is {value}.",
        ]
    if name in ("sin", "cos"):
        return [
            f"The series for {name}(u) has terms u^n/n! with alternating signs, and u = {plain(_scaled_x_tex(a))}.",
            f"The x^{k} term has coefficient {value}.",
        ]
    if name == "ln":
        return [
            "ln(1 + u) = u - u²/2 + u³/3 - u⁴/4 + ..., with u = " + plain(_scaled_x_tex(a)) + ".",
            f"The x^{k} term has coefficient {value}.",
        ]
    return [
        f"1/(1 - u) = 1 + u + u² + ..., with u = {plain(_scaled_x_tex(a))}.",
        f"The x^{k} term is {a_text}^{k}, so the coefficient is {value}.",
    ]


# -- medium ----------------------------------------------------------------------------------------------------------


def _power_coefficient(p: Fraction, a: int, k: int) -> Fraction:
    """Coefficient of (x - a)^k in x^p about a (a a perfect power so that a^(p-k) is rational)."""
    base = Fraction(a)
    root = {Fraction(1, 2): math.isqrt(a), Fraction(-1, 2): Fraction(1, math.isqrt(a)), Fraction(-1): Fraction(1, a)}
    if p in root:
        scale = Fraction(root[p])
    elif p == Fraction(1, 3):
        scale = Fraction(round(a ** (1 / 3)))
        check(scale**3 == a, "not a perfect cube")
    else:
        scale = base**p
    return _binom(p, k) * scale / base**k


_MEDIUM = [
    # name, tex, SymPy form, coefficient formula, anchor points
    ("ln", r"\ln x", sympy.log(_X), lambda a, k: Fraction((-1) ** (k + 1), k * a**k), (1, 2, 3, 4, 5)),
    ("recip", r"\frac{1}{x}", 1 / _X, lambda a, k: Fraction((-1) ** k, a ** (k + 1)), (1, 2, 3, 4, 5)),
    ("sqrt", r"\sqrt{x}", sympy.sqrt(_X), lambda a, k: _power_coefficient(Fraction(1, 2), a, k), (1, 4, 9)),
    (
        "invsqrt",
        r"\frac{1}{\sqrt{x}}",
        1 / sympy.sqrt(_X),
        lambda a, k: _power_coefficient(Fraction(-1, 2), a, k),
        (1, 4),
    ),
    (
        "cbrt",
        r"x^{1/3}",
        sympy.Pow(_X, sympy.Rational(1, 3)),
        lambda a, k: _power_coefficient(Fraction(1, 3), a, k),
        (1, 8),
    ),
]


def _medium(rng: random.Random):
    name, tex, expr, formula, anchors = rng.choice(_MEDIUM)
    a = rng.choice(anchors)
    k = rng.choice([2, 3, 4])
    value = formula(a, k)
    check(value == _derivative_coefficient(expr, k, a), f"{tex} about {a}: coefficient of (x - {a})^{k} disagrees")
    fact = math.factorial(k)
    raw = value * fact  # f^(k)(a)
    below = formula(a, k - 1) * math.factorial(k - 1) / fact  # f^(k-1)(a)/k!
    wrongs = [
        (raw, "taylor-derivative-not-divided"),
        (below, "taylor-derivative-order-off-by-one"),
        (value * k, "taylor-wrong-factorial"),
        (-value, "taylor-series-sign-pattern"),
    ]
    if a != 1:
        other = formula(1, k)
        wrongs.insert(2, (other, "taylor-expansion-point-ignored"))
    ask = rng.choice(
        [
            ("Find the coefficient of ", " in the Taylor series of "),
            ("What is the coefficient of ", " in the Taylor expansion of "),
            ("Write down the coefficient of ", " in the Taylor series of "),
        ]
    )
    prompt = say(
        ask[0],
        M(rf"(x - {a})^{{{k}}}"),
        ask[1],
        M(rf"f(x) = {tex}"),
        " about ",
        M(rf"x = {a}"),
        "?" if ask[0].startswith("What") else ".",
    )
    working = [
        f"The coefficient is f^({k})({a})/{k}!.",
        f"f^({k})({a}) = {raw}, so the coefficient is {raw}/{fact} = {value}.",
    ]
    return prompt, Link(prompt, "fraction", value, working=working, wrongs=wrongs)


# -- hard ------------------------------------------------------------------------------------------------------------


def _product_tex(f: Series, g: Series, a: Fraction, b: Fraction) -> str:
    """f(x) as TeX: a series over (1 - ax) when one factor is geometric, else the two factors side by side."""
    if f.name == "geom":
        return rf"\frac{{{g.tex}}}{{{_one_plus_tex(-a)}}}"
    if g.name == "geom":
        return rf"\frac{{{f.tex}}}{{{_one_plus_tex(-b)}}}"
    return rf"{f.tex}\,{g.tex}"


def _hard_product(rng: random.Random):
    pairs = [
        (_exp, _sin),
        (_exp, _cos),
        (_exp, _ln),
        (_geometric, _exp),
        (_geometric, _cos),
        (_ln, _geometric),
        (_sin, _cos),
    ]
    for _ in attempts(200, "a product with a non-zero coefficient"):
        first, second = rng.choice(pairs)
        a = rng.choice([Fraction(v) for v in (-2, -1, 1, 2)])
        b = rng.choice([Fraction(v) for v in (-2, -1, 1, 2)])
        f, g = first(a), second(b)
        k = rng.choice([3, 4])
        fc = [f.coeff(i) for i in range(k + 1)]
        gc = [g.coeff(i) for i in range(k + 1)]
        value = sum((fc[i] * gc[k - i] for i in range(k + 1)), Fraction(0))
        if value != 0 and fc[k] * gc[k] != value:
            break
    check(value == _derivative_coefficient(f.expr * g.expr, k), f"({f.tex})({g.tex}) coefficient of x^{k} disagrees")
    fact = math.factorial(k)
    pair_terms = [fc[i] * gc[k - i] for i in range(k + 1)]
    missing = max(pair_terms, key=lambda t: abs(t))
    wrongs = [
        (fc[k] * gc[k], "taylor-product-coefficientwise"),
        (fc[k] * gc[0] + fc[0] * gc[k], "taylor-product-cross-terms-dropped"),
        (value - missing, "taylor-product-pair-missing"),
        (value * fact, "taylor-derivative-not-divided"),
        (-value, "taylor-series-sign-pattern"),
    ]
    prompt = say(
        "Find the coefficient of ",
        M(f"x^{{{k}}}"),
        " in the Maclaurin series of ",
        M(rf"f(x) = {_product_tex(f, g, a, b)}"),
        ".",
    )
    terms = " + ".join(f"({fc[i]})({gc[k - i]})" for i in range(k + 1) if fc[i] != 0 and gc[k - i] != 0)
    working = [
        f"Multiply the two series and collect every pair of terms whose powers of x add to {k}.",
        f"The coefficient is {terms} = {value}.",
    ]
    return prompt, Link(prompt, "fraction", value, working=working, wrongs=wrongs)


def _hard_composite(rng: random.Random):
    a = rng.choice([Fraction(v) for v in (-3, -2, -1, 1, 2, 3)])
    name = rng.choice(["exp", "cos", "geom", "ln"])
    inner = f"{'-' if a < 0 else ''}{'' if abs(a) == 1 else abs(a.numerator)}x^{{2}}"
    outer = {"exp": _exp, "cos": _cos, "geom": _geometric, "ln": _ln}[name](Fraction(1))
    # the series of the outer function in u = a x^2, so the x^4 coefficient is a^2 times the u^2 coefficient
    tex_by = {
        "exp": rf"e^{{{inner}}}",
        "cos": rf"\cos\left({inner}\right)",
        "geom": rf"\frac{{1}}{{1 {'+' if a < 0 else '-'} {'' if abs(a) == 1 else abs(a.numerator)}x^{{2}}}}",
        "ln": rf"\ln\left(1 {'-' if a < 0 else '+'} {'' if abs(a) == 1 else abs(a.numerator)}x^{{2}}\right)",
    }
    expr = {
        "exp": sympy.exp(_sym(a) * _X**2),
        "cos": sympy.cos(_sym(a) * _X**2),
        "geom": 1 / (1 - _sym(a) * _X**2),
        "ln": sympy.log(1 + _sym(a) * _X**2),
    }[name]
    k = 4
    value = a**2 * outer.coeff(2)
    check(value == _derivative_coefficient(expr, k), f"{tex_by[name]} coefficient of x^4 disagrees")
    check(value != 0, "zero coefficient in the composite")
    fact = math.factorial(k)
    wrongs = [
        (a**2 * outer.coeff(4), "taylor-composite-order-mismatch"),
        (a * outer.coeff(2), "taylor-inner-coefficient-not-powered"),
        (value * fact, "taylor-derivative-not-divided"),
        (a**4 * outer.coeff(4) if name != "geom" else a**4, "taylor-composite-order-mismatch"),
        (-value, "taylor-series-sign-pattern"),
        (a**2 * outer.coeff(2) / 2, "taylor-wrong-factorial"),
    ]
    prompt = say(
        "Find the coefficient of ",
        M("x^{4}"),
        " in the Maclaurin series of ",
        M(rf"f(x) = {tex_by[name]}"),
        ".",
    )
    working = [
        "Substitute u = " + ("" if a == 1 else "-" if a == -1 else str(a)) + "x² into the standard series.",
        f"The x⁴ term comes from the u² term, so the coefficient is {value}.",
    ]
    return prompt, Link(prompt, "fraction", value, working=working, wrongs=wrongs)


def _hard(rng: random.Random):
    return _hard_product(rng) if rng.random() < 0.6 else _hard_composite(rng)


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1so_taylor_polynomial(rng: random.Random, tier: str = "easy") -> dict:
    """A Taylor or Maclaurin coefficient of order 2 to 4: standard series, a shifted centre, or a product or composite."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    if tier == "easy":
        prompt, link, _ = _easy(rng)
    elif tier == "medium":
        prompt, link = _medium(rng)
    else:
        prompt, link = _hard(rng)
    return build_single(prompt, link)


# -- exam: a four-part chain -----------------------------------------------------------------------------------------


_EXAM_X0 = [Fraction(1, 10), Fraction(1, 5), Fraction(1, 4), Fraction(3, 10), Fraction(2, 5)]


def _exam_function(rng: random.Random):
    name = rng.choice(["exp", "ln", "root", "inv"])
    a = rng.choice([Fraction(v) for v in (1, 2, 3, -1, -2)])
    if name == "exp":
        series, f0 = _exp(a), sympy.Integer(1)

        def calc(x0):
            return sympy.exp(_sym(a) * x0)

        def ft_f(x0):
            return f"exp({a}*({x0}))"

    elif name == "ln":
        series, f0 = _ln(a), sympy.Integer(0)

        def calc(x0):
            return sympy.log(1 + _sym(a) * x0)

        def ft_f(x0):
            return f"log(1 + {a}*({x0}))"

    elif name == "root":
        series, f0 = _root(a), sympy.Integer(1)

        def calc(x0):
            return sympy.sqrt(1 + _sym(a) * x0)

        def ft_f(x0):
            return f"sqrt(1 + {a}*({x0}))"

    else:
        series, f0 = _geometric(-a), sympy.Integer(1)

        def calc(x0):
            return 1 / (1 + _sym(a) * x0)

        def ft_f(x0):
            return f"1/(1 + {a}*({x0}))"

    return name, a, series, f0, calc, ft_f


def _exam(rng: random.Random):
    for _ in range(100):
        name, a, series, f0, calc, ft_f = _exam_function(rng)
        x0 = rng.choice(_EXAM_X0)
        if 1 + a * x0 <= 0:
            continue
        c = [series.coeff(i) for i in range(4)]
        c0 = Fraction(int(f0))
        check(c0 == _derivative_coefficient(series.expr, 0), "f(0) disagrees")
        poly = sum((c[i] * x0**i for i in range(4)), Fraction(0))
        poly3 = sum((c[i] * x0**i for i in range(4)), Fraction(0))
        true = calc(_sym(x0))
        err = abs(sympy.N(true, 40) - sympy.N(_sym(poly3), 40))
        if not 0 < err < sympy.Rational(1, 10):
            continue
        # places: at least two significant figures of the error, well clear of a rounding boundary
        places = None
        for p in range(4, 9):
            if err * 10**p >= 20 and rounds_clear(err, p):
                places = p
                break
        if places is not None:
            break
    else:  # pragma: no cover - the draw space always holds a usable case
        check(False, "no usable exam draw")
    check(c[1] == _derivative_coefficient(series.expr, 1), "c1 disagrees")
    check(c[2] == _derivative_coefficient(series.expr, 2), "c2 disagrees")
    check(c[3] == _derivative_coefficient(series.expr, 3), "c3 disagrees")
    # the polynomial value again, from SymPy's own derivatives at 0
    direct = sum(sympy.diff(series.expr, _X, i).subs(_X, 0) / sympy.factorial(i) * _sym(x0) ** i for i in range(4))
    check(direct == _sym(poly), "P(x0) disagrees with the derivative route")

    fact = {2: 2, 3: 6}
    x0_text = str(float(x0))
    stem = say(
        "Let ",
        M(rf"f(x) = {series.tex}"),
        ", and let ",
        M("P(x)"),
        " be the Maclaurin polynomial of degree 3 for ",
        M("f"),
        ".",
    )
    p_a = say("Find the coefficient of ", M("x^{2}"), " in ", M("P(x)"), ".")
    p_b = say("Find the coefficient of ", M("x^{3}"), " in ", M("P(x)"), ".")
    p_c = say("Hence find ", M(f"P({x0_text})"), " exactly, as a fraction.")
    p_d = say(
        "Use your calculator to find ",
        M(f"f({x0_text})"),
        ", and hence find the absolute error ",
        M(rf"\left|f({x0_text}) - P({x0_text})\right|"),
        f" in this approximation, correct to {places} decimal places.",
    )
    raw2, raw3 = c[2] * fact[2], c[3] * fact[3]
    a_wrongs = [
        (raw2, "taylor-derivative-not-divided"),
        (raw2 / 6, "taylor-wrong-factorial"),
        (-c[2], "taylor-series-sign-pattern"),
        (c[3], "taylor-order-confused"),
        (c[1] ** 2, "taylor-order-confused"),
    ]
    b_wrongs = [
        (raw3, "taylor-derivative-not-divided"),
        (raw3 / 2, "taylor-wrong-factorial"),
        (-c[3], "taylor-series-sign-pattern"),
        (c[2], "taylor-order-confused"),
        (raw3 / 24, "taylor-wrong-factorial"),
    ]
    p2 = c[0] + c[1] * x0 + c[2] * x0**2
    p4 = poly + _series_coefficient(series, 4) * x0**4
    c_wrongs = [
        (p2, "taylor-polynomial-term-dropped"),
        (c[0] + c[1] * x0 + raw2 * x0**2 + raw3 * x0**3, "taylor-derivative-not-divided"),
        (p4, "taylor-polynomial-wrong-degree"),
        (c[0] + c[1] * x0 + c[2] * x0 + c[3] * x0, "taylor-powers-of-x-ignored"),
        (c[0] + c[1] * x0 - c[2] * x0**2 + c[3] * x0**3, "taylor-series-sign-pattern"),
    ]
    true_n, poly_n = sympy.N(true, 40), sympy.N(_sym(poly), 40)
    d_wrongs = [
        (abs(true_n - sympy.N(_sym(p2), 40)), "taylor-error-of-lower-degree"),
        (poly_n, "taylor-approximation-reported-as-error"),
        (true_n, "taylor-value-reported-as-error"),
        (true_n - poly_n if true_n < poly_n else None, "taylor-error-sign-ignored"),
        (abs(true_n - sympy.N(_sym(p4), 40)), "taylor-error-of-higher-degree"),
        (abs(true_n - poly_n) / abs(true_n) if true_n != 0 else None, "taylor-relative-error"),
    ]
    ft_c = f"{c0} + {c[1]}*({x0}) + part_a*({x0})**2 + part_b*({x0})**3".replace("Fraction(", "(")
    ft_d = f"Abs({ft_f(x0)} - part_c)"
    links = [
        Link(
            p_a,
            "fraction",
            c[2],
            1,
            [f"The coefficient of x² is f''(0)/2! = {c[2]}."],
            a_wrongs,
        ),
        Link(p_b, "fraction", c[3], 1, [f"The coefficient of x³ is f'''(0)/3! = {c[3]}."], b_wrongs),
        Link(
            p_c,
            "fraction",
            poly,
            2,
            [f"P(x) = {c0} + ({c[1]})x + ({c[2]})x² + ({c[3]})x³, so P({x0_text}) = {poly}."],
            c_wrongs,
            ft=ft_c,
        ),
        Link(
            p_d,
            "number",
            err,
            2,
            [
                f"f({x0_text}) = {float(sympy.N(true, 12)):.8f}, and P({x0_text}) = {float(poly):.8f}.",
                "The error is the positive difference.",
            ],
            d_wrongs,
            ft=ft_d,
            sensible=(0, None),
            places=places,
        ),
    ]
    return stem, links


def _series_coefficient(series: Series, k: int) -> Fraction:
    return series.coeff(k)


@generator(EXAM_TOPIC, level=19, presets={"exam": {}})
def fy1so_exd_taylor_polynomial(rng: random.Random) -> dict:
    """A cubic Maclaurin polynomial: two coefficients, its value at a small x, and the error of that approximation."""
    stem, links = _exam(rng)
    return dict(text=stem[0], latex=stem[1], parts=[build_part(link) for link in links])
