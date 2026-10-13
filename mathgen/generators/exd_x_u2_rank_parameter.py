"""Exam tier for the rank of a small matrix: the parameter that makes a 3x3 matrix lose a dimension.

The practice tiers ask for the rank of a numerical matrix. Here one entry of a 3x3 matrix is an unknown k, and
the learner must find the single value of k at which the matrix has rank 2 instead of 3. Two methods work:
set the determinant to zero (it is linear in k), or find the multiples of two rows that make the third and
read k off. The prompt is worded three ways (rank 2, dependent rows, a non-zero solution of Ax = 0) so the
learner has to recognise each as the same condition.

## Construction

Two independent rows r1 and r2 are drawn, and the third row is a r1 + b r2 with both multiples non-zero. One
entry of that row is replaced by k, so the matrix is singular exactly when k equals the entry that was
replaced. The determinant is linear in k with slope equal to a signed 2x2 minor of r1 and r2, which is
required to be non-zero, so the value of k is unique and the rank is exactly 2 there and 3 everywhere else.
SymPy confirms both. The three rows are then shuffled so the dependent row is not always last.
"""

from __future__ import annotations

import random

import sympy

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, pick_distractors

_ENTRIES = [v for v in range(-9, 10) if v]
_MULTIPLES = (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)
_K_VALUES = [v for v in range(-60, 61) if v]
_K = sympy.Symbol("k")

_SHAPES = (
    "For what value of k does the matrix A = {matrix} have rank 2?",
    "Find the value of k for which the rows of the matrix A = {matrix} are linearly dependent.",
    "The matrix A = {matrix} is such that Ax = 0 has a non-zero solution. Find the value of k.",
    "Find the value of k for which the matrix A = {matrix} has rank less than 3.",
)


def _tagged(key: int, candidates: list[tuple[str, int]]) -> list:
    return pick_distractors(str(key), [Distractor(str(value), ident, value) for ident, value in candidates], 3)


@generator("exd_x_u2_rank_parameter", level=13, presets={"exam": dict(tier="exam")})
def exd_x_u2_rank_parameter(rng: random.Random, tier: str = "exam") -> dict:
    """The value of k that drops a 3x3 matrix to rank 2. See the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a 3x3 matrix with a unique rank-dropping parameter"):
        # Draw the value of k first, then a row-1 entry, two multiples and the matching row-2 entry, so that the
        # answers spread evenly over their range instead of bunching near zero.
        k0 = rng.choice(_K_VALUES)
        a, b = rng.choice(_MULTIPLES), rng.choice(_MULTIPLES)
        j = rng.randrange(3)
        r1j = rng.choice(_ENTRIES)
        if (k0 - a * r1j) % b:
            continue
        r2j = (k0 - a * r1j) // b
        if r2j == 0 or abs(r2j) > 9:
            continue
        r1 = [rng.choice(_ENTRIES) for _ in range(3)]
        r2 = [rng.choice(_ENTRIES) for _ in range(3)]
        r1[j], r2[j] = r1j, r2j
        r3 = [a * x + b * y for x, y in zip(r1, r2)]
        check(r3[j] == k0, "the entry replaced by k is not the drawn value")
        if max(abs(v) for v in r3) > 80:
            continue
        keep = [i for i in range(3) if i != j]
        if r1[keep[0]] * r2[keep[1]] - r1[keep[1]] * r2[keep[0]] == 0:
            continue

        symbolic = [list(r1), list(r2), [_K if i == j else v for i, v in enumerate(r3)]]
        order = rng.sample(range(3), 3)
        shown = [symbolic[i] for i in order]
        matrix = sympy.Matrix(shown)
        det = sympy.expand(matrix.det())
        roots = sympy.solve(det, _K)
        check(roots == [k0], "the determinant does not vanish at exactly the drawn k")
        check(matrix.subs(_K, k0).rank() == 2, "rank at the drawn k is not 2")
        check(matrix.subs(_K, k0 + 1).rank() == 3, "rank away from the drawn k is not 3")

        distractors = _tagged(
            k0,
            [
                ("sign-error-moving-term", -k0),
                ("assumed-rows-sum", r1[j] + r2[j]),
                ("dropped-one-multiple", a * r1[j]),
                ("dropped-one-multiple", b * r2[j]),
            ],
        )
        if len(distractors) < 2:
            continue

        text_matrix = "[" + ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in shown) + "]"
        return dict(
            text=rng.choice(_SHAPES).format(matrix=text_matrix),
            answer=k0,
            answer_kind="number",
            answer_text=str(k0),
            answer_latex=str(k0),
            working=[
                f"The rows are dependent when det A = 0; det A = {sympy.factor(det)}.",
                f"This is zero only for k = {k0}, and the rank is then 2.",
            ],
            distractors=distractors,
        )
