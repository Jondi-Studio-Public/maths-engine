"""Exam-tier source for expanding products of binomials into a trinomial.

The practice tiers expand ONE product of monic binomials. The exam question is a different kind of task:
the learner must expand TWO products (the second sometimes a perfect square) and then combine them, so a
minus sign has to be distributed over a whole expanded product and the leading coefficients are not monic.
Typical shape:

    Expand and simplify (3x + 2)(x - 4) - (x - 1)(2x + 5).

The answer is built backwards: every bracket is drawn from small non-zero coefficients, the result is
accepted only if it is a genuine trinomial (all three coefficients non-zero), and the key is assembled from
hand-computed coefficient formulas and then confirmed against ``sympy.expand`` as an independent second
route.

Distractors (each a named, plausible slip):

- ``minus-on-first-term-only``: the subtraction is applied to the first term of the second product only.
- ``wrong-operation``: the two products are combined with the opposite sign to the one in the question.
- ``no-cross-terms``: each product is expanded without its middle terms (first-times-first plus
  last-times-last).
- ``square-without-middle``: a squared bracket is expanded as the sum of two squares.
"""

from __future__ import annotations

import random

from sympy import Symbol, expand

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    latex_answer,
    phrase,
    pick_distractors,
)

_VARS = ("x", "y", "n", "t")

_PATTERNS = (
    "Expand and simplify {expr}.",
    "Expand and simplify {expr}, giving your answer as a trinomial in standard form.",
    "Write {expr} as a trinomial in standard form.",
    "Simplify {expr} by expanding both products.",
)


def _lin(a: int, b: int, v: str) -> str:
    return join_terms([(a, v), (b, "")])


def _nz(rng: random.Random, lo: int, hi: int) -> int:
    return rng.choice([n for n in range(lo, hi + 1) if n != 0])


def _product(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int, int]:
    """Coefficients (x^2, x, 1) of (p0 x + p1)(q0 x + q1), by FOIL."""
    return p[0] * q[0], p[0] * q[1] + p[1] * q[0], p[1] * q[1]


def _tri(c2: int, c1: int, c0: int, v: str) -> str:
    return join_terms([(c2, f"{v}^2"), (c1, v), (c0, "")])


def _exam(rng: random.Random) -> dict:
    v = rng.choice(_VARS)
    sym = Symbol(v)
    while True:
        f1 = (_nz(rng, -3, 3), _nz(rng, -6, 6))
        f2 = (_nz(rng, -3, 3), _nz(rng, -6, 6))
        squared = rng.random() < 0.35
        if squared:
            g1 = g2 = (_nz(rng, -3, 3), _nz(rng, -6, 6))
        else:
            g1 = (_nz(rng, -3, 3), _nz(rng, -6, 6))
            g2 = (_nz(rng, -3, 3), _nz(rng, -6, 6))
        sign = rng.choice([1, -1])
        first = _product(f1, f2)
        second = _product(g1, g2)
        res = tuple(a + sign * b for a, b in zip(first, second))
        if all(res):
            break

    first_text = f"({_lin(*f1, v)})({_lin(*f2, v)})"
    second_text = f"({_lin(*g1, v)})^2" if squared else f"({_lin(*g1, v)})({_lin(*g2, v)})"
    op = "+" if sign == 1 else "-"
    expr_text = f"{first_text} {op} {second_text}"
    text = phrase(rng, _PATTERNS, expr=expr_text)

    answer_text = _tri(*res, v)
    answer_expr = expand(res[0] * sym**2 + res[1] * sym + res[2])

    # Independent second route: let sympy expand the displayed expression itself.
    lin = lambda f: f[0] * sym + f[1]  # noqa: E731
    shown = lin(f1) * lin(f2) + sign * (lin(g1) ** 2 if squared else lin(g1) * lin(g2))
    check_equivalent(answer_expr, expand(shown), "hand-built trinomial disagrees with sympy.expand")

    candidates = []
    # minus applied to the first term of the second product only
    cand_a = (first[0] + sign * second[0], first[1] + second[1], first[2] + second[2])
    # the opposite operation applied to the two products
    cand_b = tuple(a - sign * b for a, b in zip(first, second))
    # no cross terms in either product
    cand_c = (first[0] + sign * second[0], 0, first[2] + sign * second[2])
    # a squared bracket expanded as a sum of two squares
    cand_d = (first[0] + sign * g1[0] ** 2, first[1], first[2] + sign * g1[1] ** 2)
    cands = [cand_a, cand_b, cand_c] + ([cand_d] if squared else [])
    for c in cands:
        if c != res:
            candidates.append(c)
    check(bool(candidates), "no principled distractor available")
    texts = []
    for c in candidates:
        wrong = c[0] * sym**2 + c[1] * sym + c[2]
        check_distinct(answer_expr, wrong, msg="distractor equals the key")
        texts.append(_tri(*c, v))

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_expr,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_expr),
        distractors=pick_distractors(answer_text, texts),
        answer_form="expanded",
    )


@generator("exd_x_a2_expand_binomial_sums", level=9, presets={"exam": dict(tier="exam")})
def exd_x_a2_expand_binomial_sums(rng: random.Random, tier: str = "exam") -> dict:
    """Expand and combine two binomial products (one possibly a square) into a trinomial."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
