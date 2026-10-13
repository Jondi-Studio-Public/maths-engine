"""Exam-tier source for `jacobi-iteration-2x2`: two Jacobi sweeps for a system whose equations are listed out of order.

The practice tiers give a system already arranged with its large coefficients on the diagonal and ask for one sweep.
The exam question lists the two equations in the order that puts each large coefficient on the OTHER unknown, so a
Jacobi iteration taken in the listed order is not diagonally dominant. The learner must

1. choose the arrangement (each equation solved for the unknown with the larger coefficient in it) and take the first
   sweep, giving `x1` and `y1`, and
2. take a second sweep from their own first-sweep values, giving `x2` and `y2`.

All four parts are single values, so the second sweep carries follow-through from the first. The key is built from the
rearranged equations; the independent route checks each sweep against the original equations by residual.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import check, generator, join_terms, rational_text

from ._u4_kit import fraction_distractors, fraction_payload, survives_nsimplify

_INTROS = [
    "A circuit analysis gives two equations in the node voltages x and y:",
    "Balancing two flows in a pipe network gives these equations for the unknown flows x and y:",
    "A mass balance for two tanks gives these equations for the unknown concentrations x and y:",
]


def _row(a: int, b: int, rhs: int) -> str:
    return f"{join_terms([(a, 'x'), (b, 'y')])} = {rhs}"


@generator("exd_x_u4_jacobi_reordered", level=17, presets={"exam": dict(tier="exam")})
def exd_x_u4_jacobi_reordered(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: two Jacobi sweeps for a 2x2 system listed with its dominant coefficients off the diagonal."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(1000):
        q = rng.choice([-1, 1]) * rng.randint(3, 9)  # equation 1 is dominant in y
        s = rng.choice([-1, 1]) * rng.randint(3, 9)  # equation 2 is dominant in x
        p = rng.choice([-1, 1]) * rng.randint(1, abs(q) - 1)
        t = rng.choice([-1, 1]) * rng.randint(1, abs(s) - 1)
        r1, r2 = rng.randint(-20, 20), rng.randint(-20, 20)
        x0, y0 = rng.choice([-1, 1]) * rng.randint(1, 4), rng.choice([-1, 1]) * rng.randint(1, 4)
        if r1 == 0 or r2 == 0:
            continue
        # sweep 1: x from equation 2, y from equation 1, both from the starting values
        x1 = Fraction(r2 - t * y0, s)
        y1 = Fraction(r1 - p * x0, q)
        x2 = Fraction(r2 - t * y1, s)
        y2 = Fraction(r1 - p * x1, q)
        keys = (x1, y1, x2, y2)
        if len(set(keys)) < 4 or {r1, r2, x0, y0, p, q, s, t} & set(keys):
            continue
        if all(k.denominator == 1 for k in keys) or not survives_nsimplify(*keys):
            continue
        break
    else:
        check(False, "no usable system in 1000 tries")

    # Independent route: the new values satisfy the equations they were solved from, with the OLD other unknown.
    check(s * x1 + t * y0 == r2 and q * y1 + p * x0 == r1, "first sweep does not satisfy its rows")
    check(s * x2 + t * y1 == r2 and q * y2 + p * x1 == r1, "second sweep does not satisfy its rows")
    check(abs(p) < abs(q) and abs(t) < abs(s), "the rearranged system is not diagonally dominant")

    stem = (
        f"{rng.choice(_INTROS)} (1) {_row(p, q, r1)}; (2) {_row(s, t, r2)}. "
        "Jacobi iteration is to be used with the equations taken in an order that makes the coefficient matrix "
        f"diagonally dominant, starting from (x₀, y₀) = ({x0}, {y0}). Give exact values as fractions or integers."
    )
    a_candidates = [
        (Fraction(r1 - q * y0, p), "jacobi-equations-not-reordered"),
        (Fraction(r2, s), "jacobi-ignores-off-diagonal"),
        (Fraction(r2 + t * y0, s), "substitution-sign-error"),
        (Fraction(x0), None),
    ]
    b_candidates = [
        (Fraction(r2 - s * x0, t), "jacobi-equations-not-reordered"),
        (Fraction(r1 - p * x1, q), "jacobi-uses-new-value"),
        (Fraction(r1, q), "jacobi-ignores-off-diagonal"),
        (Fraction(y0), None),
    ]
    c_candidates = [
        (x1, "jacobi-reuses-starting-values"),
        (Fraction(r1 - q * y1, p), "jacobi-equations-not-reordered"),
        (Fraction(r2 + t * y1, s), "substitution-sign-error"),
    ]
    d_candidates = [
        (y1, "jacobi-reuses-starting-values"),
        (Fraction(r1 - p * x2, q), "jacobi-uses-new-value"),
        (Fraction(r2 - s * x1, t), "jacobi-equations-not-reordered"),
    ]
    texts = [rational_text(k) for k in keys]
    x1t, y1t, x2t, y2t = texts
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find x₁.",
                answer_kind="fraction",
                answer=fraction_payload(x1)[0],
                answer_text=x1t,
                marks=1,
                working=[
                    f"Equation (2) has the larger coefficient on x and equation (1) on y, so x₁ = ({r2} − ({t})y₀)/{s}.",
                    f"x₁ = {x1t}.",
                ],
                distractors=fraction_distractors(x1, a_candidates),
            ),
            dict(
                prompt="Find y₁.",
                answer_kind="fraction",
                answer=fraction_payload(y1)[0],
                answer_text=y1t,
                marks=1,
                working=[f"y₁ = ({r1} − ({p})x₀)/{q} = {y1t}."],
                distractors=fraction_distractors(y1, b_candidates),
            ),
            dict(
                prompt="Using your y₁, find x₂.",
                answer_kind="fraction",
                answer=fraction_payload(x2)[0],
                answer_text=x2t,
                marks=2,
                follow_through=f"({r2} - ({t})*part_b)/({s})",
                working=[f"x₂ = ({r2} − ({t})y₁)/{s} = {x2t}."],
                distractors=fraction_distractors(x2, c_candidates),
            ),
            dict(
                prompt="Using your x₁, find y₂.",
                answer_kind="fraction",
                answer=fraction_payload(y2)[0],
                answer_text=y2t,
                marks=2,
                follow_through=f"({r1} - ({p})*part_a)/({q})",
                working=[f"y₂ = ({r1} − ({p})x₁)/{q} = {y2t}."],
                distractors=fraction_distractors(y2, d_candidates),
            ),
        ],
    )
