"""Shared helpers for the ucv1 first-year calculus tests (L'Hopital, improper integrals, series).

`parse_math` reads the plain-text formulas the questions print (Unicode powers, `e^(...)`, `ln`, `√`, implicit
multiplication) back into SymPy, so every oracle re-derives its key from the question text alone.
"""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

X = sympy.Symbol("x")
N = sympy.Symbol("n", integer=True, positive=True)
_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication, implicit_application)


def parse_math(text: str, **symbols):
    """The SymPy expression a printed formula stands for."""
    s = text.strip()
    s = s.replace("−", "-").replace("×", "*").replace("·", "*").replace("π", "pi").replace("∞", "oo")
    s = s.replace("²", "^2").replace("³", "^3")
    s = re.sub(r"√\(", "sqrt(", s)
    s = re.sub(r"√(\w+)", r"sqrt(\1)", s)
    s = re.sub(r"(?<![A-Za-z])e\^", "E^", s)
    s = re.sub(r"(?<![A-Za-z0-9_])e(?![A-Za-z0-9_(])", "E", s)
    s = re.sub(r"(?<![A-Za-z])ln(?![A-Za-z])", "log", s)
    local = {"x": X, "n": N, "E": sympy.E, "pi": sympy.pi, "oo": sympy.oo, **symbols}
    return parse_expr(s, local_dict=local, transformations=_TRANSFORMS)


def value_of(answer_text: str):
    """The exact value an answer text states."""
    return parse_math(answer_text)
