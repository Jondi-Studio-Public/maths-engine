"""Recipe subject ``log_quadratic``: f(x) = ln(q(x)) for a quadratic q with two whole-number roots and leading
coefficient -1. A thin grid cell: it states its facts and its likely errors, and the generic asks in
``recipe_asks_generic.py`` do the asking and the checking.

The shape is VCE Methods 2023 Exam 1 Q7: ``f(x) = ln(3 + 2x - x^2)``, with (a) the implied domain, (b) the x of the
local maximum and (c) the solution of ``f(x) = 0``. See ``docs/design/recipes.md``.

## Construction

The roots ``r1 < r2`` are drawn so that the vertex ``h = (r1 + r2) / 2`` is a whole number, ``m = (r2 - r1) / 2``
is the half-width, and ``q(x) = -(x - r1)(x - r2) = m^2 - (x - h)^2``. Every fact follows from that form:

- the implied domain is where ``q > 0``, the open interval ``r1 < x < r2``;
- ln is increasing, so the stationary point of f is the maximum of q, at ``x = h``;
- ``f(x) = ln k`` is ``q(x) = k``, so ``(x - h)^2 = m^2 - k = d`` and ``x = h ± sqrt(d)``. ``k`` is drawn so that
  ``d`` is a square-free whole number above 1, so the roots are surds already in simplest form and the answer
  ``x = h ± √d`` has one spelling. ``k = 1`` (``f(x) = 0``, the exam's case) is the common draw.

The subject is exact only: ``technology_active=True`` is refused, since a calculator version would be a different
question (decimals, not a hand solution).
"""

from __future__ import annotations

import math

import sympy as sp

from ..core import attempts, join_terms
from ..recipes import RecipeError, Subject, subject

X = sp.Symbol("x", real=True)


def _squarefree(n: int) -> bool:
    return n > 1 and all(e == 1 for e in sp.factorint(n).values())


@subject("log_quadratic")
class LogQuadratic(Subject):
    """f(x) = ln(q(x)), q(x) = -(x - r1)(x - r2) = m^2 - (x - h)^2."""

    properties = frozenset({"sympy_expression", "domain_interval", "unique_stationary_x", "level_set_solutions"})

    def __init__(self, r1: int, m: int, k: int, ascending: bool):
        self.r1, self.m, self.k, self.ascending = r1, m, k, ascending
        self.r2 = r1 + 2 * m
        self.h = r1 + m
        self.d = m * m - k
        self.b = 2 * self.h
        self.c = -self.r1 * self.r2
        self.expr = sp.log(-(X - self.r1) * (X - self.r2))

    @classmethod
    def draw(cls, rng, technology_active):
        if technology_active:
            raise RecipeError("log_quadratic is exact only; technology_active=True is not built")
        for _ in attempts(500, "a log-of-a-quadratic subject"):
            m = rng.randint(2, 7)
            r1 = rng.randint(-9, 3) - m
            options = [k for k in range(1, m * m) if _squarefree(m * m - k)]
            if not options:
                continue
            k = 1 if 1 in options and rng.random() < 0.5 else rng.choice(options)
            return cls(r1, m, k, rng.random() < 0.5)
        raise AssertionError("unreachable")  # pragma: no cover

    # -- text -----------------------------------------------------------------------------------------------------

    def q_text(self) -> str:
        terms = [(self.c, ""), (self.b, "x"), (-1, "x^2")]
        return join_terms(terms if self.ascending else terms[::-1])

    def stem(self) -> str:
        return f"Let f(x) = ln({self.q_text()})."

    def fn_text(self) -> str:
        return "f"

    # -- implied domain -------------------------------------------------------------------------------------------

    def domain(self):
        return sp.Interval.open(self.r1, self.r2)

    def domain_working(self) -> list[str]:
        return [
            f"ln needs {self.q_text()} > 0, that is -(x - {self.r1})(x - {self.r2}) > 0 (the parabola opens downwards).",
            f"q is positive between its roots {self.r1} and {self.r2}.",
        ]

    def domain_wrong(self) -> list[tuple[str, str]]:
        r1, r2 = self.r1, self.r2
        return [
            (f"x < {r1} or x > {r2}", "log-domain-outside-the-roots"),
            (f"{r1} ≤ x ≤ {r2}", "log-domain-endpoints-included"),
            (f"x > {r1}", "log-domain-one-factor-only"),
            (f"{-r2} < x < {-r1}", "sign-flip-both-roots"),
        ]

    # -- stationary point -----------------------------------------------------------------------------------------

    def stationary_x(self) -> int:
        return self.h

    def stationary_working(self) -> list[str]:
        return [
            "f'(x) = q'(x) / q(x), so f'(x) = 0 where q'(x) = 0, the vertex of the parabola q.",
            f"The vertex is halfway between the roots {self.r1} and {self.r2}: x = {self.h}. "
            "q opens downwards, so this is a maximum.",
        ]

    def stationary_wrong(self) -> list[tuple[int, str]]:
        return [
            (self.m * self.m, "turning-point-gives-the-value"),
            (2 * self.h, "turning-point-sum-of-roots"),
            (self.m, "turning-point-half-width"),
            (-self.h, "calc-sign-slip"),
        ]

    # -- f(x) = c -------------------------------------------------------------------------------------------------

    def level_text(self) -> str:
        return "0" if self.k == 1 else f"ln({self.k})"

    def level_value(self):
        return sp.log(self.k)

    def solutions(self):
        root = sp.sqrt(self.d)
        return [self.h - root, self.h + root]

    def solution_text(self) -> str:
        return self._text(self.d)

    def solution_value(self) -> str:
        return f"{self.h}±sqrt({self.d})"

    def _text(self, radicand: int) -> str:
        return f"x = ± √{radicand}" if self.h == 0 else f"x = {self.h} ± √{radicand}"

    def level_working(self) -> list[str]:
        exponent = "e^0 = 1" if self.k == 1 else f"e^(ln {self.k}) = {self.k}"
        completed = (
            f"{join_terms([(1, 'x^2'), (-self.b, 'x'), (self.k - self.c, '')])} = 0, so (x - {self.h})^2 = {self.d}."
            if self.h
            else f"x^2 = {self.d}."
        )
        return [
            f"ln({self.q_text()}) = {self.level_text()}, so {self.q_text()} = {exponent}.",
            completed,
            f"{self.solution_text()}. Both values lie between {self.r1} and {self.r2}, so both are in the domain.",
        ]

    def solution_wrong(self) -> list[tuple[str, str]]:
        out = [(f"x = {self.r1} or x = {self.r2}", "log-equation-argument-set-to-zero")]
        shifted = self.d - 1  # q(x) = k + 1 instead of q(x) = k
        if self.k > 1 and shifted > 1 and math.isqrt(shifted) ** 2 != shifted:
            out.append((self._text(shifted), "log-equation-exponentiated-wrongly"))
        out.append((self._text(self.m * self.m + self.k), "sign-error-moving-term"))
        out.append(
            (f"x = {self.h} ± {self.d}" if self.h else f"x = ± {self.d}", "completing-square-root-not-taken"),
        )
        out.append((f"x = {self.h} + √{self.d}" if self.h else f"x = √{self.d}", "missed-negative-root"))
        return out
