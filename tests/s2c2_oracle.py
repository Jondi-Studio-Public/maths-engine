"""Helpers shared by the `s2c2` second-year differential equation tests.

They read a question's own text back into SymPy objects, so a test can re-derive the key without using the generator.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
t = sympy.Symbol("t")


def read_expr(text: str, **local):
    """The learner's notation (``3e^(2t) - t e^(-t)``, ``(1/2)e^t cos(3t)``) as a SymPy expression in t."""
    text = text.replace("²", "**2").replace("³", "**3").replace("^", "**")
    text = re.sub(r"e\*\*\(([^()]*)\)", r"exp(\1)", text)
    text = re.sub(r"(?<![A-Za-z])e\*\*t\b", "exp(t)", text)
    return parse_expr(text, transformations=_TRANSFORMS, local_dict={"t": t, **local})


def agree(left, right, var, points) -> bool:
    return all(abs(complex(left.subs(var, p).evalf()) - complex(right.subs(var, p).evalf())) < 1e-7 for p in points)


def parse_system(text: str):
    """(matrix A, initial vector, variable names) read from ``x' = ..., y' = ...`` and ``x(0) = ...`` in `text`."""
    rows = re.findall(r"([xyz])' = ([-+0-9xyz ]+)", text)
    names = [name for name, _ in rows]
    symbols = [sympy.Symbol(name) for name in names]
    exprs = [
        parse_expr(rhs.strip(), transformations=_TRANSFORMS, local_dict={n: s for n, s in zip(names, symbols)})
        for _, rhs in rows
    ]
    matrix, _ = sympy.linear_eq_to_matrix(exprs, symbols)
    initial = dict(re.findall(r"([xyz])\(0\) = (-?\d+)", text))
    return matrix, [sympy.Integer(initial[n]) if n in initial else None for n in names], names


def half_up(value, places=3) -> str:
    quantum = Decimal(1).scaleb(-places)
    return str(Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


def initial_vector(text: str, matrix, names: str = "xy"):
    """The initial vector of a 2x2 stem: ``x(0)``, ``y(0)``, or ``x(0)`` with ``x'(0)`` (y(0) recovered from row 1)."""
    stated = dict(re.findall(r"([xy])\(0\) = (-?\d+)", text))
    if "y" in stated:
        return [sympy.Integer(stated["x"]), sympy.Integer(stated["y"])]
    v0 = sympy.Integer(re.search(r"x'\(0\) = (-?\d+)", text).group(1))
    x0 = sympy.Integer(stated["x"])
    return [x0, (v0 - matrix[0, 0] * x0) / matrix[0, 1]]
