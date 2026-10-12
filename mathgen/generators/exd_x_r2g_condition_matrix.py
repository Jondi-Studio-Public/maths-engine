"""Exam question: condition number of a 2 x 2 system in the infinity norm (`ucv1-condition-number-sensitivity`).

A system A x = b has an integer 2 x 2 matrix A and a right-hand side known to a stated relative error. The parts
chain, all in the infinity norm (the largest absolute row sum):

* (a) ||A|| (a whole number);
* (b) ||A^-1||, from the inverse formula (a fraction);
* (c) the condition number kappa = ||A|| * ||A^-1|| (follow-through from (a) and (b));
* (d) the bound kappa * rho on the relative error of the solution when the relative error of b is at most rho
  (follow-through from (c)).

Every value is an exact rational, so the keys are exact.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, paren_negative
from .exd_x_r2g_support import build_part, redraws

_RHO_PERCENT = [Fraction(1, 10), Fraction(1, 5), Fraction(1, 4), Fraction(2, 5), Fraction(1, 2), 1, 2]

_STEMS = [
    "The linear system A x = b has coefficient matrix A with first row ({p}, {q}) and second row ({r}, {s}). "
    "The right-hand side b is known only to a relative error of at most {rho}% (in the infinity norm).",
    "A calibration problem reduces to solving A x = b, where A has rows ({p}, {q}) and ({r}, {s}). Measurement "
    "error means the vector b has a relative error of at most {rho}% in the infinity norm.",
    "Consider the system A x = b in which the matrix A has first row ({p}, {q}) and second row ({r}, {s}). The "
    "entries of b are rounded, giving a relative error in b of at most {rho}% in the infinity norm.",
]


def _rho_text(percent) -> str:
    return f"{float(percent):g}"


def _draw(rng: random.Random) -> dict:
    for _ in range(1000):
        p, q, r, s = (rng.choice([-1, 1]) * rng.randint(1, 7) for _ in range(4))
        det = p * s - q * r
        if not 2 <= abs(det) <= 14:
            continue
        row_sums = [abs(p) + abs(q), abs(r) + abs(s)]
        col_sums = [abs(p) + abs(r), abs(q) + abs(s)]
        inv_rows = [abs(s) + abs(q), abs(r) + abs(p)]
        if row_sums[0] == row_sums[1] or inv_rows[0] == inv_rows[1]:
            continue
        if max(row_sums) == max(col_sums):
            continue
        norm_a = max(row_sums)
        norm_inv = Fraction(max(inv_rows), abs(det))
        if norm_a * norm_inv < 3 or norm_inv == 1:
            continue
        return dict(p=p, q=q, r=r, s=s, det=det, norm_a=norm_a, norm_inv=norm_inv, col=max(col_sums))
    check(False, "no acceptable matrix draw")


@generator("exd_x_r2g_condition_matrix", level=14, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_r2g_condition_matrix(rng: random.Random, tier: str = "exam") -> dict:
    """Infinity norms, the condition number and the resulting relative-error bound for a 2 x 2 matrix."""
    check(tier == "exam", "exd_x_r2g_condition_matrix: only the exam tier exists")
    d = _draw(rng)
    p, q, r, s, det = d["p"], d["q"], d["r"], d["s"], d["det"]
    norm_a, norm_inv = d["norm_a"], d["norm_inv"]
    kappa = norm_a * norm_inv
    percent = rng.choice(_RHO_PERCENT)
    rho = Fraction(percent) / 100
    bound = kappa * rho
    stem = rng.choice(_STEMS).format(p=p, q=q, r=r, s=s, rho=_rho_text(percent))
    adj_norm = max(abs(s) + abs(q), abs(r) + abs(p))
    parts = [
        build_part(
            "Find the infinity norm of A, the largest absolute row sum.",
            norm_a,
            kind="number",
            marks=1,
            sensible_range=(0, None),
            working=[f"Row sums of absolute values: {abs(p) + abs(q)} and {abs(r) + abs(s)}.", f"||A|| = {norm_a}."],
            wrong=[
                (d["col"], "column-sum-for-row-sum"),
                (max(abs(p), abs(q), abs(r), abs(s)), "wrong-matrix-norm-formula"),
                (abs(p) + abs(q) + abs(r) + abs(s), "wrong-matrix-norm-formula"),
                (min(abs(p) + abs(q), abs(r) + abs(s)), "wrong-matrix-norm-formula"),
            ],
        ),
        build_part(
            "Find the infinity norm of the inverse matrix A^-1, as a fraction in lowest terms.",
            norm_inv,
            kind="fraction",
            marks=2,
            sensible_range=(0, None),
            working=[
                f"det A = {paren_negative(p)} x {paren_negative(s)} - {paren_negative(q)} x {paren_negative(r)} = {det}.",
                f"A^-1 = (1/{det}) x (rows ({s}, {-q}) and ({-r}, {p})).",
                f"Largest absolute row sum = {adj_norm}/{abs(det)} = {norm_inv}.",
            ],
            wrong=[
                (Fraction(adj_norm), "forgot-to-divide-by-determinant"),
                (Fraction(norm_a, abs(det)), "inverse-without-swapping-entries"),
                (Fraction(1, norm_a), "reciprocal-of-norm-for-inverse"),
                (Fraction(max(abs(s) + abs(r), abs(q) + abs(p)), abs(det)), "column-sum-for-row-sum"),
            ],
        ),
        build_part(
            "Hence find the condition number kappa = ||A|| x ||A^-1||, as a fraction in lowest terms.",
            kappa,
            kind="fraction",
            marks=1,
            sensible_range=(0, None),
            follow_through="part_a*part_b",
            working=[f"kappa = {norm_a} x {norm_inv} = {kappa}."],
            wrong=[
                (Fraction(norm_a) / norm_inv, "divides-instead-of-multiplies"),
                (norm_a + norm_inv, "adds-instead-of-multiplies"),
                (norm_inv, "norm-of-inverse-reported-as-condition-number"),
                (kappa * abs(det), "forgot-to-divide-by-determinant"),
            ],
        ),
        build_part(
            "Hence find the upper bound on the relative error in the solution x, as a fraction in lowest terms "
            "(not a percentage).",
            bound,
            kind="fraction",
            marks=2,
            sensible_range=(0, None),
            follow_through=f"part_c*({rho.numerator}/{rho.denominator})",
            working=[f"Relative error in x <= kappa x (relative error in b) = {kappa} x {rho} = {bound}."],
            wrong=[
                (kappa * percent, "percent-not-converted"),
                (rho / kappa, "inverts-the-error-amplification"),
                (kappa / rho, "divides-instead-of-multiplies"),
                (rho, "no-error-amplification"),
            ],
        ),
    ]
    return dict(text=stem, parts=parts)
