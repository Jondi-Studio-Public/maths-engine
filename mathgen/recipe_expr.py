"""A recipe subject that is a sympy expression, with its declared properties checked against the maths.

`ExprSubject` holds ``expr`` in ``var`` over ``domain``. A subject that declares a property is checked on every
draw (`Subject.check_property`, called by `compose`), so a declaration cannot be false. The checks must be cheap,
because they run on every ``make()`` and a generator has a 250 ms ceiling, so nothing here calls ``simplify``,
``solveset``, ``integrate`` or ``limit``. Instead:

* **Structural properties** are read off the expression: the argument shapes, ``quadratic_in_exponential``,
  ``periodic``, and ``differentiable_everywhere`` (provably smooth, or containing a kink).
* **Witness properties** need the subject to say where: ``facts`` holds exact witnesses and each is verified
  numerically at 40 digits, in the domain. ``roots`` (for ``has_pi_roots``, ``has_log_root``, ``has_surd_roots``),
  ``stationary_points``, ``inflection_points``, ``vertical_asymptotes``, ``horizontal_asymptotes`` (``(L, +1 | -1)``
  pairs: the limit L at plus or minus infinity), ``excluded_points`` (for ``restricted_domain``) and
  ``antiderivative`` (an elementary F). A witness that fails its check fails the draw.

A check returns ``None`` (unchecked, never a pass or a failure) when it cannot decide: no witness was given, or the
property has no generic test (``even_candidate``, ``odd_candidate``, ``unknown_constants_*``, ``bounded_support``,
``invertible_on_domain``, ``composite_of_translation``, ``by_parts_integrable``, ``substitution_integrable``).
A subject that declares one of those overrides `check_property` itself. ``tests/test_recipe_expr.py`` holds the
adversarial expressions each check must get right.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sympy as sp
from sympy.calculus.util import periodicity

from .recipes import Subject

__all__ = ["ExprSubject", "expr_text", "arguments"]

X = sp.Symbol("x", real=True)
_OUTER = (sp.log, sp.exp, sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan)
_PRECISION = 40
_TOL = sp.Float("1e-20")
_ELEMENTARY = (sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan, sp.sinh, sp.cosh, sp.tanh, sp.Abs)
_KINKS = (sp.Abs, sp.sign, sp.Piecewise, sp.floor, sp.ceiling, sp.Min, sp.Max, sp.Heaviside)
_SMOOTH_FUNCTIONS = (sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh)


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


def _is_exponential(n: sp.Basic, var: sp.Symbol) -> bool:
    if isinstance(n, sp.exp):
        return n.args[0].has(var)
    return isinstance(n, sp.Pow) and n.exp.has(var) and not n.base.has(var)


def _quadratic_in_exponential(expr: sp.Expr, var: sp.Symbol) -> bool:
    """True when ``expr`` is a degree-2 polynomial in t = a**x for one base a. ``3**(2x+1) - 3**x`` is, because
    the constant in the exponent is a factor; ``x*3**(2x) + 3**x`` is not, because x survives."""
    expanded = sp.expand_power_exp(sp.expand(expr))
    exps = [n for n in sp.preorder_traversal(expanded) if _is_exponential(n, var)]
    bases = {sp.E if isinstance(n, sp.exp) else n.base for n in exps}
    if len(bases) != 1:
        return False
    t = sp.Dummy("t", positive=True)

    def to_t(n):
        e = n.args[0] if isinstance(n, sp.exp) else n.exp
        coeff = sp.simplify(e / var)
        return t**coeff

    swapped = expanded.replace(lambda n: _is_exponential(n, var), to_t)
    return not swapped.has(var) and swapped.is_polynomial(t) and sp.degree(swapped, t) == 2


def _value(expr: sp.Expr, var: sp.Symbol, p: sp.Expr) -> sp.Expr | None:
    """``expr`` at ``p`` to 40 digits, or None when it is undefined, infinite or not real there."""
    try:
        v = sp.N(expr.subs(var, p), _PRECISION)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not v.is_number or v.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        return None
    re, im = v.as_real_imag()
    return re if abs(im) < _TOL else None


def _in_domain(domain: sp.Set, p: sp.Expr) -> bool:
    return domain.contains(p) == sp.true


def _vanishes(expr: sp.Expr, var: sp.Symbol, domain: sp.Set, p: sp.Expr) -> bool:
    """``expr`` is defined and zero at ``p``, which lies in ``domain``."""
    v = _value(expr, var, p)
    return _in_domain(domain, p) and v is not None and abs(v) < _TOL


def _grows(expr: sp.Expr, var: sp.Symbol, a: sp.Expr, side: int) -> bool:
    """|expr| rises without bound towards ``a`` from one side: it keeps increasing at distances 1e-6, 1e-12, 1e-18
    and ends above 10. A removable gap stays level; log(x) at 0 passes."""
    values = []
    for k in (6, 12, 18):
        v = _value(expr, var, a + side * sp.Rational(1, 10**k))
        if v is None:
            return False
        values.append(abs(v))
    return values[0] + sp.Rational(1, 2) < values[1] and values[1] + sp.Rational(1, 2) < values[2] and values[2] > 10


def _tends_to(expr: sp.Expr, var: sp.Symbol, limit: sp.Expr, direction: int) -> bool:
    """expr(direction * N) is within 1e-9 of ``limit`` at several huge N (so cos(x) + 2 never passes)."""
    for n in (10**20, 3 * 10**20 + 7, 10**21 + 11):
        v = _value(expr, var, sp.Float(direction * n, _PRECISION))  # a Float: 2**(-10**20) as an Integer power hangs
        if v is None or abs(v - limit) > sp.Float("1e-9"):
            return False
    return True


def _check_witnesses(facts_key: str, check: Callable[[Any], bool], facts: dict[str, Any]) -> bool | None:
    witnesses = [sp.sympify(w) for w in facts.get(facts_key, [])]
    if not witnesses:
        return None
    return all(check(w) for w in witnesses)


class ExprSubject(Subject):
    """A subject that is the expression ``expr`` in ``var`` on ``domain`` (all reals by default).

    ``facts`` carries the exact witnesses listed in the module docstring; the properties a subject declares are
    checked on every draw by `check_property`.
    """

    def __init__(
        self, expr: sp.Expr, var: sp.Symbol = X, domain: sp.Set = sp.S.Reals, facts: dict[str, Any] | None = None
    ):
        self.expr = sp.sympify(expr)
        self.var = var
        self.domain = domain
        self.facts = dict(facts or {})

    def _smooth(self) -> bool | None:
        nodes = list(sp.preorder_traversal(self.expr))
        if any(isinstance(n, _KINKS) for n in nodes):
            return False
        funcs_ok = all(isinstance(n, _SMOOTH_FUNCTIONS) for n in nodes if isinstance(n, sp.Function))
        pow_ok = all(
            (n.exp.is_Integer and n.exp >= 0) or not n.base.has(self.var) for n in nodes if isinstance(n, sp.Pow)
        )
        return True if funcs_ok and pow_ok else None

    def _roots_verified(self) -> list[sp.Expr] | None:
        roots = [sp.sympify(r) for r in self.facts.get("roots", [])]
        if not roots:
            return None
        if not all(_vanishes(self.expr, self.var, self.domain, r) for r in roots):
            return []  # a witness that is not a root: every root property fails
        return list(roots)

    def check_property(self, name: str) -> bool | None:
        var, expr, domain, facts = self.var, self.expr, self.domain, self.facts
        d1 = lambda: sp.diff(expr, var)  # noqa: E731 -- built only by the checks that need it
        d2 = lambda: sp.diff(expr, var, 2)  # noqa: E731

        def stationary(p):
            # f' is zero at p and continuous there (|x| has f'(0) = sign(0) = 0 but jumps either side)
            h = sp.Rational(1, 10**9)
            left, right = _value(d1(), var, p - h), _value(d1(), var, p + h)
            smooth = left is not None and right is not None and abs(left - right) < sp.Float("1e-6")
            return smooth and _vanishes(d1(), var, domain, p) and _value(expr, var, p) is not None

        def inflection(p):
            h = sp.Rational(1, 1000)
            left, right = _value(d2(), var, p - h), _value(d2(), var, p + h)
            return _vanishes(d2(), var, domain, p) and left is not None and right is not None and left * right < 0

        def antiderivative_ok(f):
            if any(isinstance(n, sp.Function) and not isinstance(n, _ELEMENTARY) for n in sp.preorder_traversal(f)):
                return False
            hits = 0
            for p in (sp.Rational(1, 3), sp.Integer(2), sp.Rational(7, 5)):
                want, got = _value(expr, var, p), _value(sp.diff(f, var), var, p)
                if want is None or got is None:
                    continue
                if abs(want - got) > _TOL:
                    return False
                hits += 1
            return hits >= 2

        def asymptote_ok(w):
            limit, direction = w
            return _tends_to(expr, var, sp.sympify(limit), direction)

        roots = None
        if name in ("has_pi_roots", "has_log_root", "has_surd_roots", "rational_roots_only"):
            roots = self._roots_verified()
            if roots is not None and not roots and name != "rational_roots_only":
                return False
        tests: dict[str, Callable[[], bool | None]] = {
            "linear_argument": lambda: 1 in _degrees(expr, var),
            "quadratic_argument": lambda: 2 in _degrees(expr, var),
            "polynomial_argument": lambda: any(d >= 3 for d in _degrees(expr, var)),
            "quadratic_in_exponential": lambda: _quadratic_in_exponential(expr, var),
            "periodic": lambda: periodicity(expr, var) is not None,
            "differentiable_everywhere": self._smooth,
            "restricted_domain": lambda: _check_witnesses(
                "excluded_points", lambda p: _value(expr, var, p) is None, facts
            ),
            "has_stationary_point": lambda: _check_witnesses("stationary_points", stationary, facts),
            "has_inflection": lambda: _check_witnesses("inflection_points", inflection, facts),
            "has_asymptote": lambda: self._asymptote(asymptote_ok),
            "has_closed_form_antiderivative": lambda: (
                None if "antiderivative" not in facts else antiderivative_ok(facts["antiderivative"])
            ),
            "has_pi_roots": lambda: None if roots is None else any((r / sp.pi).is_Rational and r != 0 for r in roots),
            "has_log_root": lambda: None if roots is None else any(r.has(sp.log) for r in roots),
            "has_surd_roots": lambda: None if roots is None else any(_is_surd(r) for r in roots),
            "rational_roots_only": self._rational_roots_only,
        }
        test = tests.get(name)
        return None if test is None else test()

    def _asymptote(self, horizontal_ok: Callable[[Any], bool]) -> bool | None:
        vertical = self.facts.get("vertical_asymptotes", [])
        horizontal = self.facts.get("horizontal_asymptotes", [])
        if not vertical and not horizontal:
            return None
        grows = all(any(_grows(self.expr, self.var, sp.sympify(a), side) for side in (1, -1)) for a in vertical)
        return grows and all(horizontal_ok(w) for w in horizontal)

    def _rational_roots_only(self) -> bool | None:
        if not self.expr.is_polynomial(self.var):
            return None
        poly = sp.Poly(self.expr, self.var)
        if poly.degree() > 6:
            return None
        return all(r.is_Rational for r in poly.real_roots())


def _is_surd(r: sp.Expr) -> bool:
    return any(isinstance(n, sp.Pow) and n.exp.is_Rational and not n.exp.is_Integer for n in sp.preorder_traversal(r))
