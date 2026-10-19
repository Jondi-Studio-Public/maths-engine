"""Exam tier for the three cases for solutions of a system: a three-part chain on a system with two parameters.

The practice tiers classify a given system. Here the system has a parameter k in the third equation's z coefficient
and in the second equation's right-hand side, and a constant m on the third right-hand side:

    a1 . (x, y, z) = b1        a2 . (x, y, z) = b2 + p k        (l1 a1 + l2 a2 + (k - k*) e) . (x, y, z) = m

with e = (0, 0, 1). The determinant is (k - k*) times a nonzero number, so the system has a unique solution unless
k = k*. At k = k* the third equation is l1 (eq 1) + l2 (eq 2) on the left, so the system is consistent exactly when
m = l1 b1 + l2 (b2 + p k*).

(a) the critical value k*;
(b) the value m* for which the system then has infinitely many solutions, which follows through from k*;
(c) the y-coordinate of the point of the solution line with a stated x, which also follows through from k*.

Built backwards from a point P0 on the solution line, so every key is a whole number.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3h_kit import distinct_slips, number_part

_COEFS = (-3, -2, -1, 1, 2, 3)
_LAMBDAS = ((1, 1), (1, -1), (2, 1), (1, 2), (-1, 1), (2, -1), (-1, 2))


def _dot(row, point) -> int:
    return sum(r * v for r, v in zip(row, point))


@generator("exd_x_r3h_system_parameters", level=18, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3h_system_parameters(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(3000, "a system with a clean solution line"):
        a1 = [rng.choice(_COEFS) for _ in range(3)]
        a2 = [rng.choice(_COEFS) for _ in range(3)]
        d2 = a1[0] * a2[1] - a1[1] * a2[0]
        delta = a1[1] * a2[2] - a1[2] * a2[1]
        if d2 == 0 or delta == 0:
            continue
        l1, l2 = rng.choice(_LAMBDAS)
        k_star = rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
        p = rng.choice((-3, -2, -1, 1, 2, 3))
        point = [rng.randint(-4, 4) for _ in range(3)]
        if point[1] == 0 or point[0] == 0:
            continue
        c0 = l1 * a1[2] + l2 * a2[2] - k_star
        row3 = [l1 * a1[0] + l2 * a2[0], l1 * a1[1] + l2 * a2[1]]
        if row3[0] == 0 and row3[1] == 0:
            continue
        slips_k = [
            (-k_star, "sign-of-result-reversed"),
            (-c0, "set-coefficient-zero-for-determinant"),
            (c0, "set-coefficient-zero-for-determinant"),
            (k_star + l1 * a1[2] + l2 * a2[2], "dropped-the-parameter-offset-from-the-z-coefficient"),
        ]
        if distinct_slips(k_star, 0, slips_k) >= 3:
            break
    b1 = _dot(a1, point)
    rhs2 = _dot(a2, point)
    b2 = rhs2 - p * k_star
    m_star = l1 * b1 + l2 * rhs2
    x0, y0, z0 = point

    # Independent check with sympy: unique unless k = k*, and a line of solutions at (k*, m*).
    k, m = sympy.symbols("k m")
    matrix = sympy.Matrix([a1, a2, [row3[0], row3[1], l1 * a1[2] + l2 * a2[2] + k - k_star]])
    check(sympy.factor(matrix.det()) == sympy.factor(d2 * (k - k_star)), "determinant is d2 (k - k*)")
    at_star = matrix.subs(k, k_star)
    rhs = sympy.Matrix([b1, b2 + p * k_star, m_star])
    check(at_star.rank() == 2 and at_star.row_join(rhs).rank() == 2, "consistent with a line of solutions at m*")
    check(at_star.row_join(sympy.Matrix([b1, b2 + p * k_star, m_star + 1])).rank() == 3, "inconsistent off m*")
    check(at_star * sympy.Matrix(point) == rhs, "P0 lies on the solution line")

    def y_at(kv, rhs_k=None) -> Fraction:
        """y at x = x0 on the line from equations 1 and 2 when k = kv."""
        r2 = b2 + p * kv
        return Fraction((b1 - a1[0] * x0) * a2[2] - a1[2] * (r2 - a2[0] * x0), delta)

    check(y_at(k_star) == y0, "y at x0 from the first two equations")

    z3 = "k" if c0 == 0 else f"(k {'+' if c0 > 0 else '-'} {abs(c0)})"
    eq1 = join_terms([(a1[0], "x"), (a1[1], "y"), (a1[2], "z")]) + f" = {b1}"
    eq2 = join_terms([(a2[0], "x"), (a2[1], "y"), (a2[2], "z")]) + " = " + join_terms([(p, "k"), (b2, "")])
    eq3 = join_terms([(row3[0], "x"), (row3[1], "y"), (1, f"{z3}z")]) + " = m"
    stem = f"Consider the system of equations in x, y and z, where k and m are constants: {eq1}; {eq2}; {eq3}."
    kf = Fraction(k_star)
    ft_m = f"({l1})*({b1}) + ({l2})*(({b2}) + ({p})*part_a)"
    ft_y = f"((({b1}) - ({a1[0]})*({x0}))*({a2[2]}) - ({a1[2]})*(({b2}) + ({p})*part_a - ({a2[0]})*({x0})))/({delta})"
    return dict(
        text=stem,
        parts=[
            dict(
                **number_part(
                    "Find the value of k for which the system does not have a unique solution.",
                    kf,
                    0,
                    3,
                    [
                        f"The determinant of the coefficient matrix is {'' if d2 == 1 else d2}(k {'-' if k_star >= 0 else '+'} {abs(k_star)}).",
                        f"It is zero when k = {k_star}.",
                    ],
                    slips_k,
                )
            ),
            dict(
                **number_part(
                    "For this value of k, find the value of m for which the system has infinitely many solutions.",
                    Fraction(m_star),
                    0,
                    3,
                    [
                        f"At k = {k_star} the left side of the third equation is ({l1}) x equation 1 + ({l2}) x equation 2.",
                        f"So m = ({l1})({b1}) + ({l2})({b2 + p * k_star}) = {m_star}.",
                    ],
                    [
                        (l1 * b1 + l2 * b2, "left-parameter-out"),
                        (l1 * b1 + l2 * (b2 - p * k_star), "substitution-sign-error"),
                        (b1 + b2 + p * k_star, "added-equations-without-multipliers"),
                        (-m_star, "sign-of-result-reversed"),
                    ],
                    follow_through=ft_m,
                )
            ),
            dict(
                **number_part(
                    f"For these values of k and m the solutions lie on a line. Find the value of y at the point of "
                    f"the line where x = {x0}, correct to 2 decimal places.",
                    Fraction(y0),
                    2,
                    4,
                    [
                        f"Put x = {x0} and k = {k_star} in the first two equations and solve for y and z.",
                        f"y = {y0} (and z = {z0}).",
                    ],
                    [
                        (y_at(0), "left-parameter-out"),
                        (y_at(-k_star), "substitution-sign-error"),
                        (-y0, "sign-of-result-reversed"),
                        (z0, "gave-other-variable-of-solution-point"),
                        (x0, "gave-other-variable-of-solution-point"),
                    ],
                    follow_through=ft_y,
                )
            ),
        ],
    )
