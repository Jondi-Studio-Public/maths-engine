"""Exam-tier source for `stationary-points`.

The hard tier gives a cubic written out term by term, whose derivative is a quadratic that factorises. Here the
function is given in *factorised form*, ``f(x) = (x - p)^m (x - q)^n``, and the learner must choose to use the
product rule and factorise rather than expand (the expansion has degree m + n, up to 6). The derivative is

    f'(x) = (x - p)^(m-1) (x - q)^(n-1) [ m(x - q) + n(x - p) ],

so there are THREE stationary points: x = p, x = q, and the root of the bracket, ``(mq + np)/(m + n)``. The
third is the easy one to miss, and the two outer ones are zeros of f, which is why "solve f(x) = 0" is a
tempting wrong answer.

Built backwards from the answer: pick p, a step t and exponents m, n, then set ``q = p + (m + n)t`` so the third
stationary point ``p + m*t`` is an integer strictly between the other two.
"""

from __future__ import annotations

import random

from sympy import Poly, diff, solve, symbols

from ..core import check, generator, join_terms, phrase, pick_distractors
from .exd_x_c1_common import x_pow

_PATTERNS = (
    "Find the x-coordinates of all the stationary points of f(x) = {f}.",
    "Find every value of x at which the graph of y = {f} has a stationary point.",
    "Determine all stationary points of f(x) = {f}, giving the x-coordinate of each.",
)

x = symbols("x")


def _factor_text(p: int, power: int) -> str:
    base = "x" if p == 0 else f"({join_terms([(1, 'x'), (-p, '')])})"
    return base if power == 1 else f"{base}^{power}"


def _answer_text(values: list[int]) -> str:
    parts = [f"x = {v}" for v in values]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _build(rng: random.Random) -> dict:
    m, n = rng.choice([(2, 3), (3, 2), (2, 2), (3, 3)])
    t = rng.choice([-2, -1, 1, 2])
    p = rng.randint(-4, 3)
    q = p + (m + n) * t
    third = p + m * t
    roots = sorted({p, q, third})
    check(len(roots) == 3, "stationary exam: the three stationary points must be distinct")

    f = (x - p) ** m * (x - q) ** n
    # Independent route: solve f'(x) = 0 exactly, and cross-check the factorised derivative.
    check(sorted(solve(diff(f, x), x)) == roots, "stationary exam: sympy disagrees with the constructed roots")
    check(
        Poly(diff(f, x), x).all_coeffs()
        == Poly((x - p) ** (m - 1) * (x - q) ** (n - 1) * (m * (x - q) + n * (x - p)), x).all_coeffs(),
        "stationary exam: factorised derivative disagrees with sympy",
    )

    # Named wrong turns (all as value lists, rendered in the same text shape as the answer).
    zeros_only = sorted({p, q})  # solved f(x) = 0 instead of f'(x) = 0
    sign_flipped = sorted(-r for r in roots)  # wrong sign when reading the factors
    swapped_weights = sorted({p, q, p + n * t})  # used the exponents the wrong way round in the bracket
    candidates = []
    for values in (zeros_only, sign_flipped, swapped_weights):
        if len(values) >= 2 and values != roots:
            candidates.append(_answer_text(values) if len(values) > 1 else f"x = {values[0]}")
    candidates.append(_answer_text(roots[:2]))  # lost the bracket root
    answer_text = _answer_text(roots)
    f_text = _factor_text(p, m) + _factor_text(q, n)
    return dict(
        text=phrase(rng, _PATTERNS, f=f_text),
        latex=rf"\text{{Find the stationary points of }} f(x) = {_factor_latex(p, m)}{_factor_latex(q, n)}",
        answer=roots,
        answer_text=answer_text,
        answer_latex=r" \text{ and } ".join(f"x = {v}" for v in roots),
        working=[
            f"Product rule: f'(x) = {_deriv_text(p, q, m, n)}",
            f"Set f'(x) = 0: x = {p}, x = {q}, or the bracket is zero, x = {third}",
        ],
        distractors=pick_distractors(answer_text, candidates, unordered=True),
    )


def _factor_latex(p: int, power: int) -> str:
    base = "x" if p == 0 else rf"\left({join_terms([(1, 'x'), (-p, '')])}\right)"
    return base if power == 1 else f"{base}^{{{power}}}"


def _deriv_text(p: int, q: int, m: int, n: int) -> str:
    left = _factor_text(p, m - 1) if m > 1 else ""
    right = _factor_text(q, n - 1) if n > 1 else ""
    bracket = f"[{m}({join_terms([(1, 'x'), (-q, '')])}) + {n}({join_terms([(1, 'x'), (-p, '')])})]"
    return f"{left}{right}{bracket}"


@generator("exd_x_c1_stationary", level=14, presets={"exam": dict(tier="exam")})
def exd_x_c1_stationary(rng: random.Random, tier: str = "exam") -> dict:
    """Stationary points of (x - p)^m (x - q)^n, which has three of them."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
