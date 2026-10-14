"""Independent oracles for the ``test_s2c3_*`` files: they read a question's wording and recompute the key with SymPy."""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

X = sympy.symbols("x", real=True)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def expr(text: str, local=None):
    """Parse school-style maths: `3 - 2x`, `-π/2`, `3/2`."""
    return parse_expr(
        text.replace("π", "pi").replace("−", "-"), local_dict={"x": X, **(local or {})}, transformations=_TRANSFORMS
    )


def pieces(text: str, name: str = r"f\(x\)"):
    """[(expression in x, lo, hi)] from 'f(x) = 2 for -3 ≤ x < 0 and f(x) = 4 - x for 0 ≤ x < 3'."""
    pattern = re.compile(name + r" = (.+?) for (\S+) ≤ x [<≤] (\S+?)(?= and " + name + r"|[.,]\s|\.$|,)")
    return [(expr(body), expr(lo), expr(hi)) for body, lo, hi in pattern.findall(text)]


def half_period(text: str):
    m = re.search(r"period (2π|\d+)", text)
    return sympy.pi if m.group(1) == "2π" else sympy.Integer(int(m.group(1))) / 2


def full_range_coefficient(text: str, kind: str, n: int) -> float:
    """a_n or b_n of the pieces in `text` by exact SymPy integration over the whole period."""
    L = half_period(text)
    total = 0
    for body, lo, hi in pieces(text):
        trig = sympy.cos(n * sympy.pi * X / L) if kind == "a" else sympy.sin(n * sympy.pi * X / L)
        total += sympy.integrate(body * trig, (X, lo, hi))
    return float(sympy.N(total / L, 30))


def series_value(text: str, point):
    """The Dirichlet value of the full-period series at x = point, from one-sided limits of the pieces."""
    L = half_period(text)
    ps = pieces(text)
    x0 = point - 2 * L * sympy.floor((point + L) / (2 * L))  # into [-L, L)

    def value_from(side: str):
        for body, lo, hi in ps:
            if side == "right" and lo <= x0 < hi:
                return body.subs(X, x0)
            if side == "left" and lo < x0 <= hi:
                return body.subs(X, x0)
        # left of the first end: wrap round to the right end of the last piece
        body, lo, hi = ps[-1]
        return body.subs(X, hi)

    return float(sympy.N((value_from("left") + value_from("right")) / 2, 30))


def interval_length(text: str):
    """L from 'defined on [0, L], where L = 3'."""
    m = re.search(r"where L = (π|\d+)", text)
    return sympy.pi if m.group(1) == "π" else sympy.Integer(int(m.group(1)))


def half_range_coefficient(text: str, series: str, n: int) -> float:
    """b_n (series 'b') or a_n (series 'a') of the half-range expansion of the pieces in `text`."""
    L = interval_length(text)
    total = 0
    for body, lo, hi in pieces(text):
        trig = sympy.sin(n * sympy.pi * X / L) if series == "b" else sympy.cos(n * sympy.pi * X / L)
        total += sympy.integrate(body * trig, (X, lo, hi))
    return float(sympy.N(2 * total / L, 30))


def half_range_value(text: str, series: str, point) -> float:
    """What the half-range series converges to at x = point: the average of the extension just either side."""
    L = interval_length(text)
    ps = pieces(text)
    eps = sympy.Rational(1, 10**9)

    def g(xx):
        r = xx - 2 * L * sympy.floor((xx + L) / (2 * L))
        sign = 1
        if r < 0:
            r, sign = -r, (-1 if series == "b" else 1)
        for body, lo, hi in ps:
            if lo <= r <= hi:
                return sign * body.subs(X, r)
        raise AssertionError(r)

    return float(sympy.N((g(point - eps) + g(point + eps)) / 2, 20))
