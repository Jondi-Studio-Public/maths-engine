"""Generates `exd_x_r3b_gauss_seidel`: a multi-part exam question on one Gauss-Seidel step for a 2 x 2 system.

The system ``a11 x + a12 y = b1``, ``a21 x + a22 y = b2`` is strictly diagonally dominant once the equations are in
the right order, but is printed in a random order, so the learner must pick the equation in which x has the larger
coefficient to update x. The chain is the one a numerical-methods paper sets:

(a) x1, from the x-dominant equation and the starting y0              fraction, 2 marks
(b) y1, from the other equation and the new x1                        fraction, 2 marks, (b2 - a21 part_a) / a22
(c) the residual b1 - a11 x1 - a12 y1 of the equation used for x1     fraction, 1 mark,  b1 - a11 part_a - a12 part_b
(d) the error |x1 - x*| against the exact solution                    fraction, 2 marks, Abs(part_a - x*)

The exact solution (x*, y*) is drawn first, as integers, and the right-hand sides are built from it. Every key is
checked against `sympy.Matrix.solve` for x* and against the residual of the second equation, which a Gauss-Seidel
step always leaves at exactly zero.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms, phrase
from .exd_x_r3b_common import fraction_part
from ._exd_distinct_parts import distinct_part_answers

_STEMS = (
    "Consider the system {first} and {second}. Starting from (x0, y0) = ({x0}, {y0}), perform one Gauss-Seidel "
    "iteration: update x first, from the equation in which x has the larger coefficient in magnitude, then update y "
    "from the other equation using the new value of x.",
    "The system {first}, {second} is to be solved by the Gauss-Seidel method from (x0, y0) = ({x0}, {y0}). In one "
    "iteration x is updated first, using the equation in which x has the larger coefficient in magnitude, and y is then "
    "updated from the other equation using the new x.",
    "A Gauss-Seidel iteration starts at (x0, y0) = ({x0}, {y0}) for {first} and {second}. It updates x from the "
    "equation in which x has the larger coefficient in magnitude, then y from the other equation using the new x.",
)


def _pn(value) -> str:
    """A number in brackets when negative or a fraction, for use after an operator."""
    return f"({value})" if value < 0 or "/" in str(value) else str(value)


def _eq(a: int, b: int, rhs: int) -> str:
    return f"{join_terms([(a, 'x'), (b, 'y')])} = {rhs}"


def _build(rng: random.Random) -> dict:
    for _ in attempts(2000, "a dominant system whose iteration leaves a non-zero residual"):
        a11 = rng.choice([-1, 1]) * rng.randint(3, 7)
        a22 = rng.choice([-1, 1]) * rng.randint(3, 7)
        a12, a21 = rng.choice([-2, -1, 1, 2]), rng.choice([-2, -1, 1, 2])
        xs, ys = rng.randint(-4, 4), rng.randint(-4, 4)
        x0, y0 = rng.randint(-5, 5), rng.randint(-5, 5)
        if (x0, y0) == (xs, ys) or 0 in (xs, ys):
            continue
        b1, b2 = a11 * xs + a12 * ys, a21 * xs + a22 * ys
        x1 = Fraction(b1 - a12 * y0, a11)
        y1 = Fraction(b2) - a21 * x1
        y1 /= a22
        r1 = b1 - a11 * x1 - a12 * y1
        if r1 == 0 or x1 == xs or x1 == x0 or y1 == y0:
            continue
        break
    # Independent routes: the exact solution from sympy, and the second equation's residual being exactly zero.
    solution = sympy.Matrix([[a11, a12], [a21, a22]]).LUsolve(sympy.Matrix([b1, b2]))
    check((int(solution[0]), int(solution[1])) == (xs, ys), "the exact solution disagrees")
    check(b2 - a21 * x1 - a22 * y1 == 0, "the second equation must be satisfied exactly after the update")
    check(r1 == -a12 * (y1 - y0), "the residual of the first equation disagrees with -a12 (y1 - y0)")
    error = abs(x1 - xs)

    eqs = [_eq(a11, a12, b1), _eq(a21, a22, b2)]
    if rng.random() < 0.5:
        eqs.reverse()
    stem = phrase(rng, _STEMS, first=eqs[0], second=eqs[1], x0=x0, y0=y0)

    def jacobi_y():
        return Fraction(b2 - a21 * x0, a22)

    part_a = fraction_part(
        "Find x1.",
        x1,
        2,
        [
            f"x has the larger coefficient in {_eq(a11, a12, b1)}.",
            f"x1 = ({b1} - {_pn(a12)}·{_pn(y0)}) / {_pn(a11)} = {x1}.",
        ],
        [
            ("sign-error-on-off-diagonal", Fraction(b1 + a12 * y0, a11)),
            ("forgets-diagonal-divide", Fraction(b1 - a12 * y0)),
            ("ignores-dominance-order", Fraction(b2 - a22 * y0, a21)),
            ("wrong-variable-order", Fraction(b1 - a12 * jacobi_y(), a11)),
        ],
    )
    part_b = fraction_part(
        "Hence find y1.",
        y1,
        2,
        [f"y1 = ({b2} - {_pn(a21)}·{_pn(x1)}) / {_pn(a22)} = {y1}."],
        [
            ("uses-jacobi-method", jacobi_y()),
            ("sign-error-on-off-diagonal", Fraction(b2 + a21 * x1, a22)),
            ("forgets-diagonal-divide", Fraction(b2) - a21 * x1),
            ("ignores-dominance-order", Fraction(b1 - a11 * x1, a12)),
            ("coefficients-swapped-in-update", Fraction(b2 - a22 * x1, a21)),
        ],
        follow_through=f"({b2} - ({a21})*part_a)/{a22}",
    )
    part_c = fraction_part(
        "Find the residual b - (ax + by) of the equation used to find x1, evaluated at (x1, y1).",
        r1,
        1,
        [f"{b1} - {_pn(a11)}·{_pn(x1)} - {_pn(a12)}·{_pn(y1)} = {r1}."],
        [
            ("residual-of-the-wrong-equation", Fraction(0)),
            ("residual-sign-reversed", -r1),
            ("residual-at-mixed-iterates", Fraction(b1) - a11 * x0 - a12 * y1),
        ],
        follow_through=f"{b1} - ({a11})*part_a - ({a12})*part_b",
    )
    part_d = fraction_part(
        f"The exact solution of the system has x = {xs}. Find the error in x1, that is, the distance of x1 from the exact value.",
        error,
        2,
        [f"Error = |{x1} - {_pn(xs)}| = {error}."],
        [
            ("error-measured-from-start", Fraction(abs(x0 - xs))),
            ("error-in-wrong-variable", abs(y1 - ys)),
            ("signed-error-not-magnitude", x1 - xs if x1 < xs else None),
            ("step-size-for-error", abs(x1 - x0)),
        ],
        follow_through=f"Abs(part_a - ({xs}))",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r3b_gauss_seidel", level=15, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3b_gauss_seidel(rng: random.Random, tier: str = "exam") -> dict:
    """One Gauss-Seidel step for a 2 x 2 system, then its residual and its error."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
