"""Exam-tier source for the recursive definition of a geometric sequence, when the terms contain an unknown.

The practice tiers show the first few terms as plain numbers, so the common ratio is one division away. Here the
first three terms are written as linear expressions in an unknown ``x``. The learner must use the geometric
property ``t2^2 = t1 * t3`` to find ``x`` before the first term and the ratio can be read off.

Two question shapes, each built backwards from a chosen ``x``, first term and whole-number ratio:

- ``unit_coefficients``: the terms are ``x + a``, ``x + b``, ``x + c``. Expanding ``t2^2 = t1 * t3`` cancels the
  ``x^2`` terms and leaves a linear equation, ``x (2b - a - c) = ac - b^2``.
- ``scaled_coefficients``: the terms are ``x + a``, ``sx + b``, ``s^2 x + c`` with ``s`` equal to 2 or 3 and the
  ratio different from ``s``. Again the ``x^2`` terms cancel and the equation is linear.

In both shapes the linear equation has exactly one solution (its coefficient is ``-t1 (r - s)^2``, non-zero
because ``r != s``), so the sequence is unique. The answer is the labelled pair ``first_term = a, r = r``.

Every key is cross-checked from the printed terms alone: the printed expressions are parsed with sympy, the
equation ``t2^2 = t1 t3`` is solved for ``x``, and the terms are evaluated and divided to recover the ratio.
Distractors, each a named slip:

- reporting the value of ``x`` as the first term;
- reporting the second term as the first term (off by one);
- reporting the common difference ``t2 - t1`` in place of the ratio;
- dropping the sign of a negative ratio;
- reporting the ratio over two steps, ``t3 / t1``.
"""

from __future__ import annotations

import random

from sympy import Symbol, solve
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from ..core import VerificationError, check, generator, join_terms, paren_negative, phrase, pick_distractors

x = Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_RESAMPLE = 300

_TEMPLATES = (
    "The first three terms of a geometric sequence are {t1}, {t2} and {t3}. Find the value of x, then state "
    "the sequence's recursive definition: the first term, and the value of r in the rule t(n+1) = r × t(n).",
    "A geometric sequence begins {t1}, {t2}, {t3}, where x is a constant. State its recursive definition: the "
    "first term, and the value of r in the rule t(n+1) = r × t(n).",
)


def _parse(text: str):
    return parse_expr(text, local_dict={"x": x}, transformations=_TRANSFORMS)


def _build(rng: random.Random, scale: int) -> dict:
    for _ in range(_RESAMPLE):
        x0 = rng.choice([v for v in range(-6, 7) if v != 0])
        first = rng.choice([v for v in range(-9, 10) if v != 0])
        r = rng.choice([2, 3, 4, -2, -3])
        if r == scale:
            continue
        t = [first, first * r, first * r * r]
        coeffs = [1, scale, scale * scale]
        consts = [t[i] - coeffs[i] * x0 for i in range(3)]
        if any(abs(c) > 60 for c in consts):
            continue
        texts = [join_terms([(coeffs[i], "x"), (consts[i], "")]) for i in range(3)]
        sum_ratio = t[1] - t[0]
        candidates = [
            (x0, r),  # reports x as the first term
            (t[1], r),  # reports the second term as the first
            (first, sum_ratio),  # common difference in place of the ratio
            (first, abs(r)) if r < 0 else (first, r * r),  # sign dropped, or the two-step ratio
            (first, r * r),
        ]
        key = (first, r)
        wrong_texts = []
        seen = {key}
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            wrong_texts.append(f"first_term = {cand[0]}, r = {cand[1]}")
        if len(wrong_texts) < 3 or first == x0:
            continue
        break
    else:
        raise VerificationError(f"geometric_recursion scale={scale}: no collision-free draw")

    # Independent check from the printed terms alone: solve t2^2 = t1 * t3 for x, then read the ratio.
    exprs = [_parse(tt) for tt in texts]
    solutions = solve(exprs[1] ** 2 - exprs[0] * exprs[2], x)
    check(solutions == [x0], f"t2^2 = t1 t3 should have exactly the solution x = {x0}, got {solutions}")
    values = [e.subs(x, solutions[0]) for e in exprs]
    check(
        values[0] != 0 and values[1] / values[0] == values[2] / values[1] == r, "terms are not geometric with ratio r"
    )
    check(values[0] == first, "first term disagrees with the printed expression at the solution")

    text = phrase(rng, _TEMPLATES, t1=texts[0], t2=texts[1], t3=texts[2])
    answer_text = f"first_term = {first}, r = {r}"
    working = [
        f"In a geometric sequence t2^2 = t1 × t3, so ({texts[1]})^2 = ({texts[0]})({texts[2]}).",
        f"The x^2 terms cancel, leaving a linear equation with solution x = {x0}.",
        f"The terms are {t[0]}, {t[1]}, {t[2]}, so r = {t[1]} / {paren_negative(t[0])} = {r} and the first term is {first}.",
    ]
    return dict(
        text=text,
        answer={"first_term": first, "r": r},
        answer_text=answer_text,
        answer_latex=rf"\text{{first term}} = {first},\; r = {r}",
        working=working,
        distractors=pick_distractors(answer_text, wrong_texts, n=3),
    )


@generator(
    "exd_x_f2_geometric_recursion",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f2_geometric_recursion(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier geometric recursion: three terms containing an unknown that the geometric property pins down."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}; this source only has 'exam'")
    return _build(rng, rng.choice([1, 2, 3]))
