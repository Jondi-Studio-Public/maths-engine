"""A recipe subject that is a sympy expression, with its properties computed from the expression.

`ExprSubject` holds ``expr`` in ``var`` over ``domain`` and answers `Subject.check_property` by testing the
expression itself, so a subject that declares ``has_stationary_point`` is checked on every draw against the maths
rather than trusted. Subclass it, set ``families`` and ``properties`` (names from ``mathgen.recipe_vocabulary``),
and implement ``draw`` and ``stem``. Properties with no generic test (``even_candidate``, ``odd_candidate``,
``unknown_constants_*``, ``bounded_support``, ``invertible_on_domain``, ``composite_of_translation``,
``by_parts_integrable``, ``substitution_integrable``) return ``None`` here: a subject that declares one checks it
itself by overriding ``check_property``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sympy as sp
from sympy.calculus.util import continuous_domain, periodicity

from .recipes import Subject

__all__ = ["ExprSubject", "expr_text", "arguments"]

X = sp.Symbol("x", real=True)
_OUTER = (sp.log, sp.exp, sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan)


def expr_text(expr: sp.Expr) -> str:
    """The expression as plain question text: ``x**2`` is written ``x^2``."""
    return sp.sstr(expr).replace("**", "^")


def arguments(expr: sp.Expr, var: sp.Symbol) -> list[sp.Expr]:
    """The arguments (containing ``var``) of every outer function in ``expr``: ``log(u)`` gives ``u``, ``a**u`` and
    ``exp(u)`` give ``u``, ``sqrt(u)`` and other fractional powers give the base."""
    found: list[sp.Expr] = []
    for node in sp.preorder_traversal(expr):
        if isinstance(node, _OUTER) and node.args[0].has(var):
            found.append(node.args[0])
        elif isinstance(node, sp.Pow) and node.exp.has(var) and not node.base.has(var):
            found.append(node.exp)
        elif isinstance(node, sp.Pow) and node.base.has(var) and not node.exp.has(var) and not node.exp.is_Integer:
            found.append(node.base)
    return found


def _degrees(expr: sp.Expr, var: sp.Symbol) -> list[int]:
    return [sp.degree(a, var) for a in arguments(expr, var) if a.is_polynomial(var)]


class ExprSubject(Subject):
    """A subject that is the expression ``expr`` in ``var`` on ``domain`` (all reals by default)."""

    def __init__(
        self, expr: sp.Expr, var: sp.Symbol = X, domain: sp.Set = sp.S.Reals, facts: dict[str, Any] | None = None
    ):
        self.expr = sp.sympify(expr)
        self.var = var
        self.domain = domain
        self.facts = dict(facts or {})  # precomputed exact values: roots, stationary points, constants

    # ----- the computable properties -------------------------------------------------------------------------
    def _roots(self) -> list[sp.Expr] | None:
        solution = sp.solveset(self.expr, self.var, self.domain)
        return list(solution) if isinstance(solution, sp.FiniteSet) else None

    def _quadratic_in_exponential(self) -> bool:
        exps = {
            n for n in sp.preorder_traversal(self.expr) if isinstance(n, (sp.exp, sp.Pow)) and self._is_exponential(n)
        }
        bases = {sp.E if isinstance(n, sp.exp) else n.base for n in exps}
        if len(bases) != 1:
            return False
        t = sp.Dummy("t", positive=True)
        swapped = self.expr.replace(
            lambda n: self._is_exponential(n),
            lambda n: t ** sp.simplify((n.args[0] if isinstance(n, sp.exp) else n.exp) / self.var),
        )
        # a**(2x) becomes t**2 with t = a**x; an exponent that is not a multiple of x leaves a non-polynomial
        return swapped.is_polynomial(t) and sp.degree(swapped, t) == 2

    def _is_exponential(self, n: sp.Basic) -> bool:
        if isinstance(n, sp.exp):
            return n.args[0].has(self.var)
        return isinstance(n, sp.Pow) and n.exp.has(self.var) and not n.base.has(self.var)

    def check_property(self, name: str) -> bool | None:
        var, expr = self.var, self.expr
        tests: dict[str, Callable[[], bool | None]] = {
            "linear_argument": lambda: 1 in _degrees(expr, var),
            "quadratic_argument": lambda: 2 in _degrees(expr, var),
            "polynomial_argument": lambda: any(d >= 3 for d in _degrees(expr, var)),
            "quadratic_in_exponential": self._quadratic_in_exponential,
            "periodic": lambda: periodicity(expr, var) is not None,
            "restricted_domain": lambda: continuous_domain(expr, var, sp.S.Reals) != sp.S.Reals,
            "differentiable_everywhere": lambda: (
                continuous_domain(expr, var, self.domain) == self.domain
                and continuous_domain(sp.diff(expr, var), var, self.domain) == self.domain
            ),
            "has_stationary_point": lambda: _nonempty(sp.solveset(sp.diff(expr, var), var, self.domain)),
            "has_inflection": lambda: _has_inflection(expr, var, self.domain),
            "has_asymptote": lambda: _has_asymptote(expr, var),
            "has_closed_form_antiderivative": lambda: not sp.integrate(expr, var).has(sp.Integral),
            "has_surd_roots": lambda: _any_root(self, lambda r: _is_surd(r)),
            "has_log_root": lambda: _any_root(self, lambda r: r.has(sp.log)),
            "has_pi_roots": lambda: _any_root(self, lambda r: r.has(sp.pi)),
            "rational_roots_only": lambda: _all_roots(self, lambda r: r.is_Rational),
        }
        test = tests.get(name)
        return None if test is None else test()


def _nonempty(s: sp.Set) -> bool | None:
    if isinstance(s, sp.ConditionSet):
        return None
    return s != sp.S.EmptySet


def _any_root(subject: ExprSubject, pred: Callable[[sp.Expr], bool]) -> bool | None:
    roots = subject._roots()
    return None if roots is None else any(pred(r) for r in roots)


def _all_roots(subject: ExprSubject, pred: Callable[[sp.Expr], bool]) -> bool | None:
    roots = subject._roots()
    return None if roots is None else bool(roots) and all(pred(r) for r in roots)


def _is_surd(r: sp.Expr) -> bool:
    return any(isinstance(n, sp.Pow) and n.exp.is_Rational and not n.exp.is_Integer for n in sp.preorder_traversal(r))


def _has_inflection(expr: sp.Expr, var: sp.Symbol, domain: sp.Set) -> bool | None:
    points = sp.solveset(sp.diff(expr, var, 2), var, domain)
    if not isinstance(points, sp.FiniteSet):
        return None
    third = sp.diff(expr, var, 3)
    return any(sp.simplify(third.subs(var, p)) != 0 for p in points)


def _has_asymptote(expr: sp.Expr, var: sp.Symbol) -> bool:
    if sp.calculus.util.singularities(expr, var):
        return True
    for direction in (sp.oo, -sp.oo):
        lim = sp.limit(expr, var, direction)
        if lim.is_finite:
            return True
    return False
