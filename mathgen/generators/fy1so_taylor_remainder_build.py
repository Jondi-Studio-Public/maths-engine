"""Generates `fy1so_taylor_remainder` (easy, medium, hard) and the exam chain `fy1so_exd_taylor_remainder`.

Skill ``ucv1-taylor-remainder-bound`` (domain ``single-variable-calculus``, answer ``fraction``): the Lagrange form of the
remainder of a Taylor polynomial and the error bound it gives. The remainder is R_n(x) = f^(n+1)(c) (x - a)^(n+1) / (n + 1)!
for some c between a and x, so |R_n(x)| <= M |x - a|^(n+1) / (n + 1)! when |f^(n+1)(c)| <= M on that interval.

Tiers (the reasoning escalates; the numbers stay small):

- ``easy``: the bound M is given by the function (sin and cos have every derivative at most 1 in size), so the work is the
  power and the factorial: |x|^(n+1)/(n+1)! at a stated x.
- ``medium``: the learner finds M. The function is ln x, 1/x, √x or x^(1/3) about a point a > 0, the interval runs from a
  to a + h, and the (n+1)-th derivative shrinks as c grows, so its largest size on the interval is its size at c = a.
- ``hard``: the smallest degree n for which the bound is below a stated tolerance, for e^x (given e < 3), sin x, ln(1 + x)
  or 1/(1 + x) on [0, x]. Each family has its own M (3, 1, n!, (n+1)!), which depends on n for the last two.
- ``exam`` (topic ``fy1so_exd_taylor_remainder``): a four-part chain for f = ln(1 + x), √(1 + x) or 1/(1 + x) on [0, x]
  with a degree n of 2, 3 or 4. (a) the least M with |f^(n+1)(c)| <= M on [0, x], (b) the Lagrange bound (follows through
  from (a)), (c) the exact value of the Taylor polynomial at x, (d) the lower end of the interval that must contain f(x)
  (follows through from (b) and (c)).

Every M is computed by a closed formula and again as |f^(n+1)(a)| with SymPy's derivative; every bound is checked against
the true error computed from SymPy's high-precision value of f (which must be no larger).
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from ._fy1so_common import Link, M, build_part, build_single, say

TOPIC = "fy1so_taylor_remainder"
EXAM_TOPIC = "fy1so_exd_taylor_remainder"

_X = sympy.Symbol("x")


def _sym(f: Fraction):
    return sympy.Rational(f.numerator, f.denominator)


def _text(f: Fraction) -> str:
    """A rational as a short learner-facing number: 3, 1/2, or a decimal when it is exactly one."""
    f = Fraction(f)
    if f.denominator == 1:
        return str(f.numerator)
    for places in (1, 2):
        scaled = f * 10**places
        if scaled.denominator == 1:
            return f"{float(f):.{places}f}"
    return f"{f.numerator}/{f.denominator}"


def _tex(f: Fraction) -> str:
    f = Fraction(f)
    if f.denominator == 1:
        return str(f.numerator)
    for places in (1, 2):
        scaled = f * 10**places
        if scaled.denominator == 1:
            return f"{float(f):.{places}f}"
    return rf"\frac{{{f.numerator}}}{{{f.denominator}}}"


def _sup(n: int) -> str:
    """The derivative order as TeX: f^{(4)}."""
    return f"f^{{({n})}}"


# -- derivatives of the families (sizes at a point) --------------------------------------------------------------------


def _power_scale(p: Fraction, t0: Fraction) -> Fraction:
    """t0^p for a rational t0 that is a perfect power for the denominator of p."""
    if p.denominator == 1:
        return t0**p.numerator
    root = round(float(t0) ** (1 / p.denominator))
    check(Fraction(root) ** p.denominator == t0, f"{t0} is not a perfect power for exponent {p}")
    return Fraction(root) ** p.numerator


def _size(name: str, t0: Fraction, m: int) -> Fraction:
    """|f^(m)(t0)| for ln t, t^p families (t0 a positive rational)."""
    if name == "ln":
        return Fraction(math.factorial(m - 1)) / t0**m
    p = {"recip": Fraction(-1), "sqrt": Fraction(1, 2), "cbrt": Fraction(1, 3)}[name]
    coefficient = Fraction(1)
    for i in range(m):
        coefficient *= p - i
    return abs(coefficient) * _power_scale(p, t0) / t0**m


_EXPR = {
    "ln": sympy.log(_X),
    "recip": 1 / _X,
    "sqrt": sympy.sqrt(_X),
    "cbrt": sympy.Pow(_X, sympy.Rational(1, 3)),
}
_TEX = {"ln": r"\ln x", "recip": r"\frac{1}{x}", "sqrt": r"\sqrt{x}", "cbrt": r"x^{1/3}"}


def _sympy_size(name: str, t0: Fraction, m: int) -> Fraction:
    value = abs(sympy.diff(_EXPR[name], _X, m).subs(_X, _sym(t0)))
    value = sympy.nsimplify(value) if not value.is_Rational else value
    check(value.is_Rational, f"|f^({m})| is not rational")
    return Fraction(int(value.p), int(value.q))


def _bound(m_bound: Fraction, h: Fraction, n: int) -> Fraction:
    return m_bound * abs(h) ** (n + 1) / math.factorial(n + 1)


# -- easy ------------------------------------------------------------------------------------------------------------

_EASY_X = [
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(1, 4),
    Fraction(1, 5),
    Fraction(2, 3),
    Fraction(3, 4),
    Fraction(1, 6),
    Fraction(1, 10),
    Fraction(3, 5),
    Fraction(2, 5),
]


def _easy(rng: random.Random):
    name = rng.choice(["sin", "cos"])
    n = rng.choice([2, 3, 4])
    x0 = rng.choice(_EASY_X)
    value = _bound(Fraction(1), x0, n)
    # second route: the true error of the Taylor polynomial is no larger than the bound
    f = sympy.sin(_X) if name == "sin" else sympy.cos(_X)
    poly = sum(sympy.diff(f, _X, k).subs(_X, 0) / sympy.factorial(k) * _X**k for k in range(n + 1))
    true_error = abs(sympy.N(f.subs(_X, _sym(x0)) - poly.subs(_X, _sym(x0)), 30))
    check(true_error <= sympy.N(_sym(value), 30), "the true error exceeds the Lagrange bound")
    check(all(abs(sympy.diff(f, _X, n + 1)).subs(_X, t) <= 1 for t in (0, 1, 2)), "derivative bound")
    fact, fact_n = math.factorial(n + 1), math.factorial(n)
    wrongs = [
        (x0**n / fact, "remainder-power-off-by-one"),
        (x0 ** (n + 1) / fact_n, "remainder-factorial-index"),
        (x0 ** (n + 1), "remainder-factorial-dropped"),
        (x0**n / fact_n, "remainder-power-off-by-one"),
        (x0 ** (n + 2) / math.factorial(n + 2), "remainder-degree-off-by-one"),
    ]
    prompt = say(
        "The Taylor polynomial of degree ",
        str(n),
        " for ",
        M(rf"f(x) = \{name}\,x"),
        " about ",
        M("x = 0"),
        " is used to approximate ",
        M(rf"f({_tex(x0)})"),
        f". Every derivative of f satisfies |f⁽ᵏ⁾(x)| ≤ 1, so the Lagrange form of the remainder gives an upper bound for ",
        M(rf"\left|R_{{{n}}}({_tex(x0)})\right|"),
        ". Find that bound.",
    )
    working = [
        f"|R_{n}(x)| ≤ M|x|^{n + 1}/{n + 1}! with M = 1.",
        f"With x = {_text(x0)}: ({_text(x0)})^{n + 1}/{n + 1}! = {value}.",
    ]
    return prompt, Link(prompt, "fraction", value, working=working, wrongs=wrongs)


# -- medium ----------------------------------------------------------------------------------------------------------

_MEDIUM_FAMILIES = {
    "ln": (1,),
    "recip": (1, 2, 4),
    "sqrt": (1, 4, 9),
    "cbrt": (1, 8),
}
_MEDIUM_H = [Fraction(1, 2), Fraction(1, 4), Fraction(1, 5), Fraction(1, 10), Fraction(1, 3), Fraction(1)]


def _medium(rng: random.Random):
    name = rng.choice(list(_MEDIUM_FAMILIES))
    a = Fraction(rng.choice(_MEDIUM_FAMILIES[name]))
    n = rng.choice([2, 3, 4])
    h = rng.choice(_MEDIUM_H)
    m_bound = _size(name, a, n + 1)
    check(m_bound == _sympy_size(name, a, n + 1), f"{name}: M disagrees with SymPy at a = {a}")
    # the (n+1)-th derivative shrinks as c grows, so its largest size on [a, a + h] is at c = a
    top = sympy.diff(_EXPR[name], _X, n + 1)
    for step in range(1, 5):
        c = _sym(a + h * Fraction(step, 4))
        check(abs(sympy.N(top.subs(_X, c), 30)) <= sympy.N(_sym(m_bound), 30), "the derivative does not shrink")
    value = _bound(m_bound, h, n)
    f = _EXPR[name]
    poly = sum(sympy.diff(f, _X, k).subs(_X, _sym(a)) / sympy.factorial(k) * (_X - _sym(a)) ** k for k in range(n + 1))
    x1 = a + h
    true_error = abs(sympy.N(f.subs(_X, _sym(x1)) - poly.subs(_X, _sym(x1)), 30))
    check(true_error <= sympy.N(_sym(value), 30), "the true error exceeds the Lagrange bound")
    fact = math.factorial(n + 1)
    wrongs = [
        (_bound(_size(name, a, n), h, n), "remainder-derivative-order-off-by-one"),
        (m_bound * h ** (n + 1) / math.factorial(n), "remainder-factorial-index"),
        (m_bound * h**n / fact, "remainder-power-off-by-one"),
        (h ** (n + 1) / fact, "remainder-derivative-bound-ignored"),
        (_bound(_size(name, a, n + 2), h, n), "remainder-derivative-order-off-by-one"),
    ]
    if name in ("ln", "recip") and a + h != a:
        wrongs.insert(1, (_bound(_size(name, a + h, n + 1), h, n), "remainder-max-at-wrong-endpoint"))
    interval_end = _text(a + h)
    prompt = say(
        "The Taylor polynomial of degree ",
        str(n),
        " for ",
        M(rf"f(x) = {_TEX[name]}"),
        " about ",
        M(f"x = {_tex(a)}"),
        " is used to approximate ",
        M(f"f({interval_end})"),
        ". Take M to be the largest value of ",
        M(rf"\left|{_sup(n + 1)}(c)\right|"),
        " for c between ",
        M(_tex(a)),
        " and ",
        M(interval_end),
        ". Use the Lagrange form of the remainder to find the bound on ",
        M(rf"\left|R_{{{n}}}({interval_end})\right|"),
        " that this M gives.",
    )
    working = [
        f"|f⁽{n + 1}⁾(c)| decreases as c increases, so M = |f⁽{n + 1}⁾({_text(a)})| = {m_bound}.",
        f"|R_{n}| ≤ M|x - a|^{n + 1}/{n + 1}! = {m_bound} × ({_text(h)})^{n + 1}/{n + 1}! = {value}.",
    ]
    return prompt, Link(prompt, "fraction", value, working=working, wrongs=wrongs)


# -- hard ------------------------------------------------------------------------------------------------------------


def _family_m(name: str, n: int) -> Fraction:
    """M for the hard families on [0, x] with 0 < x <= 1 (the interval where each constant is justified)."""
    return {
        "exp": Fraction(3),
        "sin": Fraction(1),
        "ln": Fraction(math.factorial(n)),
        "recip": Fraction(math.factorial(n + 1)),
    }[name]


_HARD_X = [Fraction(1), Fraction(1, 2), Fraction(3, 4), Fraction(1, 4), Fraction(2, 3), Fraction(1, 3)]
_HARD_EPS = [
    Fraction(1, 100),
    Fraction(1, 1000),
    Fraction(1, 10000),
    Fraction(1, 500),
    Fraction(1, 5000),
    Fraction(1, 200),
    Fraction(1, 2000),
]
_HARD_TEX = {"exp": r"e^{x}", "sin": r"\sin x", "ln": r"\ln(1 + x)", "recip": r"\frac{1}{1 + x}"}
_HARD_NOTE = {
    "exp": " Use the fact that e < 3.",
    "sin": "",
    "ln": "",
    "recip": "",
}


def _least_degree(name: str, x0: Fraction, eps: Fraction, limit: int = 12) -> int | None:
    for n in range(1, limit):
        if _bound(_family_m(name, n), x0, n) < eps:
            return n
    return None


def _hard(rng: random.Random):
    for _ in attempts(300, "a tolerance with a clear least degree"):
        name = rng.choice(["exp", "sin", "ln", "recip"])
        x0 = rng.choice(_HARD_X)
        eps = rng.choice(_HARD_EPS)
        n = _least_degree(name, x0, eps)
        if n is None:
            continue
        before = _bound(_family_m(name, n - 1), x0, n - 1) if n > 1 else None
        after = _bound(_family_m(name, n), x0, n)
        if (
            3 <= n <= 9
            and before is not None
            and before >= eps
            and after < eps * Fraction(19, 20)
            and before > eps * Fraction(21, 20)
        ):
            break
    # second route: re-derive M from SymPy's derivative for the two families whose M depends on n
    f = {"exp": sympy.exp(_X), "sin": sympy.sin(_X), "ln": sympy.log(1 + _X), "recip": 1 / (1 + _X)}[name]
    top = abs(sympy.diff(f, _X, n + 1).subs(_X, 0))
    if name in ("ln", "recip"):
        check(Fraction(int(top)) == _family_m(name, n), "M disagrees with SymPy's derivative")
    else:
        check(Fraction(int(sympy.ceiling(top))) <= _family_m(name, n), "M too small")
    wrong_m = lambda k: _family_m(name, k)  # noqa: E731
    # the degree a learner reaches with one common slip each
    by_factorial_n = next((k for k in range(1, 40) if wrong_m(k) * x0 ** (k + 1) / math.factorial(k) < eps), None)
    by_power_n = next((k for k in range(1, 40) if wrong_m(k) * x0**k / math.factorial(k + 1) < eps), None)
    wrongs = [
        (n - 1, "remainder-degree-off-by-one"),
        (n + 1, "remainder-terms-for-degree"),
        (by_factorial_n, "remainder-factorial-index"),
        (by_power_n, "remainder-power-off-by-one"),
        (n + 2, "remainder-degree-off-by-one"),
    ]
    interval = _text(x0)
    prompt = say(
        "The Taylor polynomial of degree n for ",
        M(rf"f(x) = {_HARD_TEX[name]}"),
        " about ",
        M("x = 0"),
        " is used to approximate ",
        M(f"f({interval})"),
        "." + _HARD_NOTE[name] + " Use the Lagrange form of the remainder to find the smallest n that guarantees ",
        M(rf"\left|R_{{n}}({interval})\right| < {_tex(eps)}"),
        ".",
    )
    working = [
        f"With M = {_family_m(name, n)} the bound is M({interval})^{n + 1}/{n + 1}! = {after}, which is below {eps}.",
        f"For n = {n - 1} the bound is {before}, which is not below {eps}. So the smallest n is {n}.",
    ]
    return prompt, Link(prompt, "fraction", Fraction(n), working=working, wrongs=wrongs)


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1so_taylor_remainder(rng: random.Random, tier: str = "easy") -> dict:
    """The Lagrange error bound: with M given, with M found from the derivative, or the least degree for a tolerance."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    prompt, link = {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)
    return build_single(prompt, link)


# -- exam: a four-part chain -----------------------------------------------------------------------------------------

_EXAM_FAMILIES = {
    "ln": (r"\ln(1 + x)", lambda k: Fraction((-1) ** (k + 1), k), sympy.log(1 + _X)),
    "recip": (r"\frac{1}{1 + x}", lambda k: Fraction((-1) ** k), 1 / (1 + _X)),
    "sqrt": (r"\sqrt{1 + x}", None, sympy.sqrt(1 + _X)),
}
_EXAM_X = [
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(1, 4),
    Fraction(1, 5),
    Fraction(2, 3),
    Fraction(3, 4),
    Fraction(1, 10),
]


def _binomial_half(k: int) -> Fraction:
    out = Fraction(1)
    for i in range(k):
        out *= Fraction(1, 2) - i
    return out / math.factorial(k)


def _taylor_coefficient(name: str, k: int) -> Fraction:
    if name == "ln":
        return Fraction(0) if k == 0 else Fraction((-1) ** (k + 1), k)
    if name == "recip":
        return Fraction((-1) ** k)
    return _binomial_half(k)


def _exam_m(name: str, n: int) -> Fraction:
    """The largest size of the (n+1)-th derivative on [0, x]: its size at 0, since it shrinks as c grows."""
    k = n + 1
    if name == "ln":
        return Fraction(math.factorial(n))
    if name == "recip":
        return Fraction(math.factorial(k))
    return abs(_binomial_half(k)) * math.factorial(k)


def _exam(rng: random.Random):
    name = rng.choice(list(_EXAM_FAMILIES))
    tex, _, expr = _EXAM_FAMILIES[name]
    n = rng.choice([2, 3, 4])
    x0 = rng.choice(_EXAM_X)
    m_bound = _exam_m(name, n)
    check(m_bound == abs(Fraction(str(sympy.diff(expr, _X, n + 1).subs(_X, 0)))), "M disagrees with SymPy's derivative")
    for step in range(1, 5):
        c = _sym(x0 * Fraction(step, 4))
        check(abs(sympy.diff(expr, _X, n + 1).subs(_X, c)) <= _sym(m_bound), "the derivative grows along the interval")
    bound = m_bound * x0 ** (n + 1) / math.factorial(n + 1)
    coeffs = [_taylor_coefficient(name, k) for k in range(n + 1)]
    for k in range(n + 1):
        check(
            _sym(coeffs[k]) == sympy.diff(expr, _X, k).subs(_X, 0) / sympy.factorial(k),
            f"coefficient {k} disagrees with SymPy",
        )
    poly = sum((coeffs[k] * x0**k for k in range(n + 1)), Fraction(0))
    true = sympy.N(expr.subs(_X, _sym(x0)), 40)
    check(abs(true - sympy.N(_sym(poly), 40)) <= sympy.N(_sym(bound), 40), "the true error exceeds the bound")
    lower = poly - bound
    check(sympy.N(_sym(lower), 40) <= true <= sympy.N(_sym(poly + bound), 40), "f(x) lies outside the interval")

    xs = _text(x0)
    stem = say(
        "Let ",
        M(rf"f(x) = {tex}"),
        ", and let ",
        M(rf"P_{{{n}}}(x)"),
        f" be the Taylor polynomial of degree {n} for f about ",
        M("x = 0"),
        ". Its remainder is ",
        M(rf"R_{{{n}}}(x) = f(x) - P_{{{n}}}(x)"),
        ", and we approximate ",
        M(f"f({xs})"),
        " by ",
        M(f"P_{{{n}}}({xs})"),
        ".",
    )
    p_a = say(
        "Find the smallest M for which ",
        M(rf"\left|{_sup(n + 1)}(c)\right| \le M"),
        " for every c in ",
        M(f"[0, {xs}]"),
        ".",
    )
    p_b = say(
        "Hence use the Lagrange form of the remainder to find an upper bound for ",
        M(rf"\left|R_{{{n}}}({xs})\right|"),
        ".",
    )
    p_c = say("Find ", M(f"P_{{{n}}}({xs})"), " exactly, as a fraction.")
    p_d = say(
        "Hence find the lower end of the interval that must contain ",
        M(f"f({xs})"),
        ", given by ",
        M(rf"P_{{{n}}}({xs}) - \text{{(the bound from part (b))}}"),
        ".",
    )
    k = n + 1
    a_wrongs = [
        (_exam_m(name, n - 1) if n > 1 else None, "remainder-derivative-order-off-by-one"),
        (_exam_m(name, n + 1), "remainder-derivative-order-off-by-one"),
        (m_bound / k, "remainder-factorial-index"),
        (m_bound * k, "remainder-factorial-index"),
        (Fraction(1), "remainder-derivative-bound-ignored"),
    ]
    b_wrongs = [
        (m_bound * x0**n / math.factorial(k), "remainder-power-off-by-one"),
        (m_bound * x0**k / math.factorial(n), "remainder-factorial-index"),
        (x0**k / math.factorial(k), "remainder-derivative-bound-ignored"),
        (m_bound * x0**k, "remainder-factorial-dropped"),
        (_exam_m(name, n) * x0 ** (n + 2) / math.factorial(n + 2), "remainder-degree-off-by-one"),
    ]
    p_lower = poly - coeffs[n] * x0**n
    c_wrongs = [
        (p_lower, "taylor-polynomial-term-dropped"),
        (poly + _taylor_coefficient(name, k) * x0**k, "taylor-polynomial-wrong-degree"),
        (sum((coeffs[j] * x0 for j in range(n + 1)), Fraction(0)), "taylor-powers-of-x-ignored"),
        (
            sum((coeffs[j] * math.factorial(j) * x0**j for j in range(n + 1)), Fraction(0)),
            "taylor-derivative-not-divided",
        ),
    ]
    d_wrongs = [
        (poly + bound, "remainder-wrong-end-of-interval"),
        (poly, "remainder-bound-not-applied"),
        (bound - poly, "remainder-bound-subtracted-the-wrong-way"),
        (p_lower - bound, "taylor-polynomial-term-dropped"),
        (poly - m_bound * x0**n / math.factorial(k), "remainder-power-off-by-one"),
    ]
    ft_b = f"part_a*({x0})**{k}/{math.factorial(k)}"
    links = [
        Link(
            p_a,
            "fraction",
            m_bound,
            2,
            [
                f"|f⁽{k}⁾(c)| is largest at c = 0 on [0, {xs}] because it shrinks as c grows, so M = |f⁽{k}⁾(0)| = {m_bound}."
            ],
            a_wrongs,
            sensible=(0, None),
        ),
        Link(
            p_b,
            "fraction",
            bound,
            2,
            [f"|R_{n}({xs})| ≤ M({xs})^{k}/{k}! = {bound}."],
            b_wrongs,
            ft=ft_b,
            sensible=(0, None),
        ),
        Link(
            p_c,
            "fraction",
            poly,
            2,
            [
                "P(x) = "
                + " + ".join(f"({coeffs[j]})x^{j}" for j in range(n + 1) if coeffs[j] != 0)
                + f", so P_{n}({xs}) = {poly}."
            ],
            c_wrongs,
        ),
        Link(
            p_d,
            "fraction",
            lower,
            1,
            [f"{poly} - {bound} = {lower}."],
            d_wrongs,
            ft="part_c - part_b",
        ),
    ]
    return stem, links


@generator(EXAM_TOPIC, level=19, presets={"exam": {}})
def fy1so_exd_taylor_remainder(rng: random.Random) -> dict:
    """The Lagrange bound for a Taylor polynomial of ln(1 + x), √(1 + x) or 1/(1 + x), and the interval it fixes."""
    stem, links = _exam(rng)
    return dict(text=stem[0], latex=stem[1], parts=[build_part(link) for link in links])
