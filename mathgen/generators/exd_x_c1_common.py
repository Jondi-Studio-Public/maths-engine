"""Shared rendering helpers for the `exd_x_c1_*` exam-tier differentiation generators.

Nothing here registers a topic. The helpers only turn small integer parameters into the plain-text and LaTeX
spellings the exam-tier prompts use, so every module writes ``2x^2 - 3x + 1`` and ``(2x + 1)^3`` the same way.
"""

from __future__ import annotations

from sympy import Symbol, latex

from ..core import join_terms

X = Symbol("x")


def x_pow(n: int) -> str:
    """``x`` for exponent 1, otherwise ``x^n``."""
    return "x" if n == 1 else f"x^{n}"


def lin_text(a: int, b: int) -> str:
    """``ax + b`` as a student writes it (``x - 3``, ``2x + 1``)."""
    return join_terms([(a, "x"), (b, "")])


def quad_text(a: int, b: int, c: int) -> str:
    """``ax^2 + bx + c`` with zero terms dropped."""
    return join_terms([(a, "x^2"), (b, "x"), (c, "")])


def powered(inner: str, n: int) -> str:
    """``(inner)^n``, or just ``(inner)`` when n is 1."""
    return f"({inner})" if n == 1 else f"({inner})^{n}"


def tex(expr) -> str:
    """LaTeX for a sympy expression."""
    return latex(expr)
