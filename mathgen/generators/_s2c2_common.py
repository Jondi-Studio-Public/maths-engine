"""Shared helpers for the second-year ordinary differential equation builds (`s2c2_*`).

* `Term` and the `fn_*` functions: a solution component such as ``3e^(2t) - t e^(-t) + 2e^(3t) cos(2t)`` as a list
  of terms c * t^p * e^(at) * (cos or sin)(bt), with its student-style text, its SymPy form and its value.
* `system_text`, `initial_text`: a linear system and its initial values written the way a student reads them.
* `rk4_state`: a numerical solution of x' = A x, used as a route that shares nothing with an eigenvalue method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import sympy

from ..core import join_terms, rational_text

T = sympy.Symbol("t")


@dataclass(frozen=True)
class Term:
    """c * t^tp * e^(a t) * trig(b t), where trig is "", "cos" or "sin"."""

    c: Fraction
    a: int = 0
    tp: int = 0
    trig: str = ""
    b: int = 0

    def key(self) -> tuple:
        return (self.a, self.tp, self.trig, self.b)

    def value(self, t: float) -> float:
        out = float(self.c) * t**self.tp * math.exp(self.a * t)
        if self.trig == "cos":
            out *= math.cos(self.b * t)
        elif self.trig == "sin":
            out *= math.sin(self.b * t)
        return out

    def sym(self):
        out = sympy.Rational(self.c.numerator, self.c.denominator) * T**self.tp * sympy.exp(self.a * T)
        if self.trig == "cos":
            out *= sympy.cos(self.b * T)
        elif self.trig == "sin":
            out *= sympy.sin(self.b * T)
        return out

    def basis(self) -> str:
        parts = []
        if self.tp:
            parts.append("t" if self.tp == 1 else f"t^{self.tp}")
        if self.a:
            parts.append("e^t" if self.a == 1 else ("e^(-t)" if self.a == -1 else f"e^({self.a}t)"))
        if self.trig:
            parts.append(f"{self.trig}({'t' if self.b == 1 else str(self.b) + 't'})")
        return " ".join(parts)


def merged(terms) -> list[Term]:
    """Like terms collected, zero terms dropped, in order of first appearance."""
    order: dict[tuple, Fraction] = {}
    for t in terms:
        order[t.key()] = order.get(t.key(), Fraction(0)) + Fraction(t.c)
    return [Term(c, *key) for key, c in order.items() if c != 0]


def _body(c: Fraction, basis: str) -> tuple[bool, str]:
    mag = abs(c)
    if not basis:
        return c < 0, rational_text(mag)
    if mag == 1:
        return c < 0, basis
    if mag.denominator == 1:
        return c < 0, f"{mag.numerator}{basis}"
    return c < 0, f"({mag.numerator}/{mag.denominator}){basis}"


def fn_text(terms) -> str:
    """The sum as a student writes it: ``3e^(2t) - t e^(-t)``; ``0`` when every term cancels."""
    items = [_body(t.c, t.basis()) for t in merged(terms)]
    if not items:
        return "0"
    out = []
    for i, (neg, body) in enumerate(items):
        if i == 0:
            out.append(f"-{body}" if neg else body)
        else:
            out.append(f"{'-' if neg else '+'} {body}")
    return " ".join(out)


def fn_sym(terms):
    return sum((t.sym() for t in merged(terms)), sympy.Integer(0))


def fn_machine(terms) -> str:
    """The parseable string kept in `answer`; the display form is `answer_text`."""
    return str(fn_sym(terms)).replace("**", "^")


def fn_value(terms, t: float) -> float:
    return sum(term.value(t) for term in terms)


def fn_differs(left, right) -> bool:
    """Whether two component functions are different functions (compared at points, not by spelling)."""
    return any(abs(fn_value(left, p) - fn_value(right, p)) > 1e-7 for p in (0.31, 0.73, 1.19))


def system_text(a, names: str = "xyz") -> str:
    """``x' = 3x + y, y' = 2x - z``: one equation per row of the matrix `a`."""
    rows = []
    for i, row in enumerate(a):
        rows.append(f"{names[i]}' = " + join_terms([(int(v), names[j]) for j, v in enumerate(row)]))
    return ", ".join(rows)


def initial_text(x0, names: str = "xyz") -> str:
    return ", ".join(f"{names[i]}(0) = {int(v)}" for i, v in enumerate(x0))


def rk4_state(a, x0, t_end: float, steps: int = 400) -> list[float]:
    """x(t_end) for x' = A x by the classical Runge-Kutta rule: a route with no eigenvalues in it."""
    n = len(x0)
    af = [[float(v) for v in row] for row in a]

    def f(x):
        return [sum(af[i][j] * x[j] for j in range(n)) for i in range(n)]

    x = [float(v) for v in x0]
    h = t_end / steps
    for _ in range(steps):
        k1 = f(x)
        k2 = f([x[i] + h / 2 * k1[i] for i in range(n)])
        k3 = f([x[i] + h / 2 * k2[i] for i in range(n)])
        k4 = f([x[i] + h * k3[i] for i in range(n)])
        x = [x[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(n)]
    return x


def matmul(p, q):
    return [[sum(p[i][k] * q[k][j] for k in range(len(q))) for j in range(len(q[0]))] for i in range(len(p))]


def matvec(p, v):
    return [sum(p[i][k] * v[k] for k in range(len(v))) for i in range(len(p))]
