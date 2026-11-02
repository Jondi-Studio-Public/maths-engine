"""Maths for the even/odd functions subject, with no recipe-composer imports.

Everything here is exact and independent of the recipe interface, so a change to the
interface only touches the glue module.

A term is ``(coeff, kind, p)``:

* ``"pow"``  -- coeff * x**p
* ``"sin"``  -- coeff * sin(p*x)
* ``"cos"``  -- coeff * cos(p*x)
* ``"const"`` -- coeff (p ignored)

Parity of a term: pow is even for even p and odd for odd p; sin is odd; cos and const are even.
"""

from __future__ import annotations

from sympy import Rational, Symbol, cos, pi, sin, simplify

X = Symbol("x", real=True)
K = Symbol("k", integer=True)

EVEN, ODD, NEITHER = "even", "odd", "neither"

# Phase-shift table for g(x) = trig(b*x + c), b != 0: the family of c making g even or odd.
# value is (c0, period-multiple): c = c0 + k*pi
_PHASE = {
    ("sin", EVEN): pi / 2,
    ("sin", ODD): 0,
    ("cos", EVEN): 0,
    ("cos", ODD): pi / 2,
}


def term_parity(kind: str, p: int) -> str:
    if kind == "pow":
        return EVEN if p % 2 == 0 else ODD
    if kind == "sin":
        return ODD
    return EVEN  # cos, const


def term_expr(term):
    coeff, kind, p = term
    if kind == "pow":
        return coeff * X**p
    if kind == "sin":
        return coeff * sin(p * X)
    if kind == "cos":
        return coeff * cos(p * X)
    return coeff + 0 * X


def function_parity(terms) -> str:
    """Parity of a sum of terms, from the parities of its non-zero terms."""
    parities = {term_parity(k, p) for (c, k, p) in terms if c != 0}
    if not parities:
        return EVEN  # the zero function is both; call it even
    if parities == {EVEN}:
        return EVEN
    if parities == {ODD}:
        return ODD
    return NEITHER


def combine_parity(a: str, b: str, op: str) -> str:
    """Parity of f+g or f*g from the parities of f and g (neither means 'not even and not odd')."""
    if op == "+":
        return a if a == b and a != NEITHER else NEITHER
    if NEITHER in (a, b):
        return NEITHER
    return EVEN if a == b else ODD


def phase_family(trig: str, parity: str):
    """c = c0 + k*pi making trig(b*x + c) have the given parity. Returns the sympy expression in K."""
    c0 = _PHASE[(trig, parity)]
    return c0 + K * pi


def phase_family_text(trig: str, parity: str) -> str:
    c0 = _PHASE[(trig, parity)]
    if c0 == 0:
        return "k*pi"
    return "pi/2 + k*pi"


def shifted_expr(trig: str, b, c):
    return (sin if trig == "sin" else cos)(b * X + c)


def numeric_parity(expr, points=(Rational(1, 3), Rational(2, 1), Rational(7, 5), Rational(11, 4))) -> str:
    """Independent oracle: classify by exact evaluation at several points (used by tests)."""
    even = odd = True
    for t in points:
        v, w = simplify(expr.subs(X, t)), simplify(expr.subs(X, -t))
        if simplify(v - w) != 0:
            even = False
        if simplify(v + w) != 0:
            odd = False
    if even and odd:
        return EVEN
    return EVEN if even else ODD if odd else NEITHER
