"""Exam source of `integer-polynomial-quotient-remainder`: a remainder condition, then the quotient and a second remainder.

The practice tiers divide a given integer polynomial by a monic divisor. Here the cubic carries an unknown integer
coefficient `a` and the stem states only the remainder on division by `x - c`, so the learner has to run the
division backwards, then use the value:

    (a) find a from the stated remainder                  (2 marks)
    (b) hence the constant term of the quotient           (2 marks, follow-through)
    (c) hence the remainder on division by `x - d`        (2 marks, follow-through)

The coefficient is drawn first, so the stated remainder is exact. Part (b) is read off the synthetic-division row
and part (c) off the remainder theorem; both are re-derived with `sympy.div` before the question is returned.
"""

from __future__ import annotations

from fractions import Fraction

import sympy
from sympy import Poly, Symbol

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    pick_distractors,
    rational_text,
)

x = Symbol("x")
_ROOTS = [2, 3, 4, -1, -2, -3]
_SECOND = [1, 2, 3, -1, -2, -3]


def _divisor(c: int) -> str:
    return join_terms([(1, "x"), (-c, "")])


def _poly_text(a_symbol: bool, a_value: int, m: int, n: int) -> str:
    return join_terms([(1, "x^3"), (1, "ax^2") if a_symbol else (a_value, "x^2"), (m, "x"), (n, "")])


def _small(value: Fraction) -> bool:
    """A wrong answer worth offering: a whole number, or a fraction with a small denominator."""
    return value.denominator <= 4 and abs(value) <= 200


def _dist(value: Fraction, tag: str):
    return Distractor(rational_text(value), tag, value)


def _part(prompt, value, marks, working, candidates, follow_through=None):
    value = Fraction(value)
    text = rational_text(value)
    kept = [c for c in candidates if _small(c[0])]
    dists = pick_distractors(
        text,
        [_dist(v, tag) for v, tag in kept],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(value) if value.denominator == 1 else float(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=dists,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


@generator("exd_x_sc1_poly_remainder_unknown", level=12, presets={"exam": dict(tier="exam")})
def exd_x_sc1_poly_remainder_unknown(rng, tier="exam"):
    for _ in attempts(1000, "a cubic with an unknown coefficient"):
        c = rng.choice(_ROOTS)
        d = rng.choice([v for v in _SECOND if v != c])
        a = rng.choice([v for v in range(-12, 13) if v != 0])
        m = rng.choice([v for v in range(-9, 10) if v != 0])
        n = rng.choice([v for v in range(-9, 10) if v != 0])
        P = x**3 + a * x**2 + m * x + n
        quo, rem = sympy.div(Poly(P, x), Poly(x - c, x))
        q0 = int(quo.all_coeffs()[-1])
        r2 = int(P.subs(x, d))
        R = int(rem.as_expr())
        # Keep every figure modest and every key distinct from the numbers the stem gives.
        stem_numbers = {c, d, m, n, R}
        if abs(R) > 150 or abs(q0) > 60 or abs(r2) > 150:
            continue
        if q0 in stem_numbers or r2 in stem_numbers or R == 0 or len({a, q0, r2}) < 3:
            continue
        break

    # Independent route to (a): the remainder theorem P(c) = R is linear in a.
    solved = sympy.solve(sympy.Eq(_p_of_a(c, m, n), R), Symbol("a"))
    check(solved == [a], "the remainder condition does not give the drawn coefficient")
    check(q0 == m + c * (a + c), "synthetic-division route disagrees with sympy.div")
    check(r2 == d**3 + a * d**2 + m * d + n, "remainder theorem route disagrees")

    c2 = c * c
    lead = R - n - c * m - c**3  # c^2 * a
    a_cands = [
        (Fraction(R - n + c * m + c**3, c2), "remainder-theorem-wrong-sign"),
        (Fraction(-n - c * m - c**3, c2), "remainder-ignored-given"),
        (Fraction(lead, c), "lost-a-power-when-collecting"),
        (Fraction(R + n - c * m - c**3, c2), "sign-error-moving-term"),
        (Fraction(R - n - c * m, c2), "dropped-a-term"),
        (Fraction(R - n - c**3, c2), "dropped-a-term"),
    ]
    q1 = a + c
    b_cands = [
        (Fraction(m + a + c), "synthetic-division-no-multiply"),
        (Fraction(q1), "wrong-quotient-coefficient"),
        (Fraction(m + c * (a - c)), "substitution-sign-error"),
        (Fraction(R), "quotient-confused-with-remainder"),
    ]
    c_cands = [
        (Fraction(int(P.subs(x, -d))), "remainder-theorem-wrong-sign"),
        (Fraction(r2 - n), "dropped-constant-term"),
        (Fraction(d**2 + q1 * d + q0), "quotient-evaluated-for-remainder"),
    ]

    a_part = _part(
        "Find the value of a.",
        a,
        2,
        [
            f"Remainder theorem: P({c}) = {R}.",
            f"{c**3} + {paren_negative(c2)}a + {paren_negative(c * m)} + {paren_negative(n)} = {R}, so a = {a}.",
        ],
        a_cands,
    )
    b_part = _part(
        f"Hence find the constant term of the quotient when P(x) is divided by {_divisor(c)}.",
        q0,
        2,
        [
            f"Synthetic division by {c}: the quotient is x^2 + {paren_negative(q1)}x + {paren_negative(q0)}.",
            f"Constant term = {m} + {paren_negative(c)} × {paren_negative(q1)} = {q0}.",
        ],
        b_cands,
        follow_through=f"{m} + {paren_negative(c)}*(part_a + {paren_negative(c)})",
    )
    c_part = _part(
        f"Hence find the remainder when P(x) is divided by {_divisor(d)}.",
        r2,
        2,
        [f"P({d}) = {d**3} + {paren_negative(a * d**2)} + {paren_negative(m * d)} + {paren_negative(n)} = {r2}."],
        c_cands,
        follow_through=f"{d**3} + part_a*{d**2} + {paren_negative(m * d)} + {paren_negative(n)}",
    )

    verb = "leaves a remainder of"
    stem = (
        f"The polynomial P(x) = {_poly_text(True, a, m, n)}, where a is an integer, "
        f"{verb} {R} when it is divided by {_divisor(c)}."
    )
    return dict(text=stem, parts=[a_part, b_part, c_part])


def _p_of_a(c: int, m: int, n: int):
    """P(c) as an expression in the symbol a."""
    return c**3 + Symbol("a") * c**2 + m * c + n
