"""Exam-tier source for `zero-product-property`: equations that are NOT already a product equal to zero.

Two shapes, chosen per seed. In both the learner has to choose the method, and the tempting shortcut
loses a solution:

* ``common_factor`` -- ``(mx + n)(px + q) = (mx + n)(rx + s)``. Cancelling the shared bracket is the
  tempting move and silently drops the root ``x = -n/m``. The right move is to bring everything to one
  side and take the shared bracket out as a factor.
* ``squares`` -- ``(px + q)^2 = (rx + s)^2``. Taking square roots without the plus-or-minus loses a
  solution; the equation is really ``(px + q - rx - s)(px + q + rx + s) = 0``.

Built backwards from the two solutions' defining brackets and re-solved with ``sympy.solve`` on the
expanded equation (an independent route).
"""

from __future__ import annotations

import random

import sympy
from sympy import Rational, symbols

from ..core import GRADER_TOLERANCE, check, check_equivalent, fmt_signed, generator, pick_distractors
from ..core import rational_text

x = symbols("x")
_TOL = float(GRADER_TOLERANCE)


def _bracket(a: int, b: int) -> str:
    return f"({fmt_signed(a, 'x')} {'+' if b >= 0 else '-'} {abs(b)})"


def _roots_text(roots) -> str:
    parts = [f"x = {rational_text(r)}" for r in roots]
    return parts[0] if len(parts) == 1 else f"{parts[0]} or {parts[1]}"


def _separated(roots) -> bool:
    vals = sorted(float(r) for r in roots)
    return all(b - a > 3 * _TOL for a, b in zip(vals, vals[1:]))


def _nonzero(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi) * rng.choice([1, -1])


def _common_factor(rng: random.Random):
    while True:
        m, n = rng.choice([1, 2, 3]), _nonzero(rng, 1, 7)
        p, q, r, s = rng.choice([1, 2, 3, 4]), _nonzero(rng, 1, 7), rng.choice([1, 2, 3, 4]), _nonzero(rng, 1, 7)
        if p == r:
            continue
        root_common = Rational(-n, m)
        root_other = Rational(s - q, p - r)
        roots = sorted([root_common, root_other], key=float)
        if root_common == root_other or root_common == -root_other or not _separated(roots):
            continue
        break
    swap_left, swap_right = rng.random() < 0.5, rng.random() < 0.5
    left = [_bracket(m, n), _bracket(p, q)][:: -1 if swap_left else 1]
    right = [_bracket(m, n), _bracket(r, s)][:: -1 if swap_right else 1]
    lhs = (m * x + n) * (p * x + q)
    rhs = (m * x + n) * (r * x + s)
    solved = sorted(sympy.solve(sympy.Eq(lhs, rhs), x), key=float)
    check(len(solved) == 2, "expected two solutions")
    for got, want in zip(solved, roots):
        check_equivalent(got, want, "solution disagrees with sympy.solve")
    # cancel-the-common-factor keeps only the other root; the others are the usual sign slips.
    cands = [
        [root_other],
        [-root_common, -root_other],
        sorted([-root_common, root_other], key=float),
        sorted([root_common, -root_other], key=float),
    ]
    return "".join(left), "".join(right), roots, cands


def _squares(rng: random.Random):
    while True:
        p, q, r, s = _nonzero(rng, 1, 4), _nonzero(rng, 1, 8), _nonzero(rng, 1, 4), _nonzero(rng, 1, 8)
        p, r = abs(p), abs(r)
        if p == r or p * s == q * r:
            continue
        plus = Rational(s - q, p - r)  # px + q = rx + s
        minus = Rational(-s - q, p + r)  # px + q = -(rx + s)
        roots = sorted([plus, minus], key=float)
        if plus == -minus or not _separated(roots):
            continue
        break
    lhs = (p * x + q) ** 2
    rhs = (r * x + s) ** 2
    solved = sorted(sympy.solve(sympy.Eq(lhs, rhs), x), key=float)
    check(len(solved) == 2, "expected two solutions")
    for got, want in zip(solved, roots):
        check_equivalent(got, want, "solution disagrees with sympy.solve")
    cands = [
        [plus],  # took the square root without the plus-or-minus
        [minus],
        [-plus, -minus],
        sorted([plus, Rational(s + q, p + r)], key=float),  # sign slip on the negative branch
    ]
    return f"{_bracket(p, q)}^2", f"{_bracket(r, s)}^2", roots, cands


@generator("exd_x_a1_zero_product_property", level=9, presets={"exam": dict(tier="exam")})
def exd_x_a1_zero_product_property(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: an equation that must be rearranged or factorised before the null factor law applies."""
    check(tier == "exam", f"unknown tier {tier!r}")
    left, right, roots, cand_lists = (_common_factor if rng.random() < 0.5 else _squares)(rng)
    answer_text = _roots_text(roots)
    texts = []
    for cand in cand_lists:
        cand = sorted(cand, key=float)
        if cand != roots:
            texts.append(_roots_text(cand))
    text = f"Solve {left} = {right}"
    parts = [rf"x = {sympy.latex(r)}" for r in roots]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=list(roots),
        answer_text=answer_text,
        answer_latex=rf"{parts[0]} \text{{ or }} {parts[1]}",
        working=[
            "Bring everything to one side and factorise; the null factor law then gives each solution.",
            f"{text} -> {answer_text}",
        ],
        distractors=pick_distractors(answer_text, texts, unordered=True, tolerance=GRADER_TOLERANCE),
    )
