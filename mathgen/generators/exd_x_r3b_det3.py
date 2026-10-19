"""Generates `exd_x_r3b_det3`: a multi-part exam question on a 3x3 matrix with one unknown entry.

The stem gives a 3x3 matrix A whose entries are integers except one, ``k``. Expanding along the row or column that
holds ``k`` makes det(A) linear in k, ``alpha k + beta``, with ``alpha`` the cofactor of the k entry.

(a) det(A) in terms of k                            expression, 2 marks
(b) det(A) for a stated k                           number,     1 mark,  Subs(part_a, (k,), (k0,))
(c) det(c A^-1) for the same k                      fraction,   2 marks, c^3 / part_b
(d) the k for which A has no inverse                fraction,   2 marks, from part_a evaluated at k = 0 and k = 1

Every key is checked against sympy's determinant of the matrix with k left symbolic.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms, phrase
from .exd_x_r3b_common import expression_part, fraction_part, number_part
from ._exd_distinct_parts import distinct_part_answers

K = sympy.Symbol("k")

_STEMS = (
    "The matrix A = {m} has an unknown entry k.",
    "Let A = {m}, where k is a constant.",
    "A is the matrix {m}, in which k is a real number.",
)


def _minor(rows, i, j):
    keep = [[v for cj, v in enumerate(row) if cj != j] for ci, row in enumerate(rows) if ci != i]
    return keep[0][0] * keep[1][1] - keep[0][1] * keep[1][0]


def _det_with(rows, k_value):
    """det of the matrix with the symbol replaced by `k_value` (entries are ints or None for k)."""
    m = [[k_value if v is None else v for v in row] for row in rows]
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _matrix_text(rows) -> str:
    return "[" + ", ".join("[" + ", ".join("k" if v is None else str(v) for v in row) + "]" for row in rows) + "]"


def _build(rng: random.Random) -> dict:
    for _ in attempts(2000, "a matrix with usable determinants"):
        rows = [[rng.randint(-4, 4) for _ in range(3)] for _ in range(3)]
        i, j = rng.randrange(3), rng.randrange(3)
        rows[i][j] = None
        beta = _det_with(rows, 0)
        alpha = _det_with(rows, 1) - beta
        if alpha == 0 or abs(alpha) > 24 or abs(beta) > 80 or alpha == 1 and beta == 0:
            continue
        k0 = rng.choice([-3, -2, -1, 1, 2, 3, 4])
        det0 = alpha * k0 + beta
        if det0 == 0 or abs(det0) > 90 or (alpha * beta == 0):
            continue
        if len({-alpha * k0 + beta, alpha * k0, -alpha * k0 + beta, beta} - {det0}) < 3:
            continue
        root_ = Fraction(-beta, alpha)
        if (
            len(
                {-root_, Fraction(alpha, -beta), Fraction(-beta), Fraction(beta, alpha), Fraction(-alpha, beta)}
                - {root_}
            )
            < 3
        ):
            continue
        break
    sym = sympy.Matrix([[K if v is None else v for v in row] for row in rows])
    check(sympy.expand(sym.det() - (alpha * K + beta)) == 0, "linear form disagrees with sympy's determinant")
    check(sym.det().subs(K, k0) == det0, "value at k0 disagrees")
    root = Fraction(-beta, alpha)
    check(sym.det().subs(K, sympy.Rational(root.numerator, root.denominator)) == 0, "singular k is not a root")

    sign = (-1) ** (i + j)
    check(sign * _minor(rows, i, j) == alpha, "cofactor disagrees with the linear coefficient")
    # all-plus expansion along the first row, without the alternating signs
    first_row = [(c, v) for c, v in enumerate(rows[0])]
    all_plus = {0: 0, 1: 0}
    for k_value in (0, 1):
        total = 0
        for c, v in first_row:
            total += (k_value if v is None else v) * _minor(
                [[k_value if w is None else w for w in r] for r in rows], 0, c
            )
        all_plus[k_value] = total
    plus_alpha, plus_beta = all_plus[1] - all_plus[0], all_plus[0]

    def lin(a, b):
        return join_terms([(a, "k"), (b, "")])

    def at(a, b):
        return Fraction(a * k0 + b)

    a_text = lin(alpha, beta)
    c_scale = rng.choice([2, 3])
    key_c = Fraction(c_scale**3, det0)
    stem = phrase(rng, _STEMS, m=_matrix_text(rows))
    part_a = expression_part(
        "Find det(A) in terms of k.",
        a_text,
        sympy.latex(alpha * K + beta),
        2,
        [
            f"Expand along the row or column containing k. The cofactor of k is {alpha}.",
            f"det(A) = {a_text}.",
        ],
        [
            ("all-plus-expansion", lin(plus_alpha, plus_beta), (plus_alpha, plus_beta) != (alpha, beta)),
            ("cofactor-sign-error", lin(-alpha, beta), True),
            ("minor-used-for-cofactor", lin(_minor(rows, i, j), beta), _minor(rows, i, j) != alpha),
            ("dropped-constant-term", lin(alpha, 0), beta != 0),
        ],
    )
    part_b = number_part(
        f"Hence find the value of det(A) when k = {k0}.",
        Fraction(det0),
        0,
        1,
        [f"det(A) = {alpha}({k0}) + ({beta}) = {det0}."],
        [
            ("all-plus-expansion", at(plus_alpha, plus_beta)),
            ("cofactor-sign-error", at(-alpha, beta)),
            ("minor-used-for-cofactor", at(_minor(rows, i, j), beta)),
            ("dropped-constant-term", at(alpha, 0)),
            ("substitution-sign-error", Fraction(-alpha * k0 + beta)),
        ],
        follow_through=f"Subs(part_a, (k,), ({k0},))",
    )
    part_c = fraction_part(
        f"For k = {k0}, find det({c_scale}A^(-1)).",
        key_c,
        2,
        [f"det({c_scale}A^(-1)) = {c_scale}^3 × (1/det(A)) = {c_scale**3}/{det0}."],
        [
            ("scalar-not-cubed", Fraction(c_scale, det0)),
            ("scalar-squared-as-for-2x2", Fraction(c_scale**2, det0)),
            ("inverse-determinant-not-reciprocal", Fraction(c_scale**3 * det0)),
        ],
        follow_through=f"{c_scale**3}/part_b",
    )
    part_d = fraction_part(
        "Find the value of k for which A has no inverse.",
        root,
        2,
        [
            f"A has no inverse when det(A) = 0: {a_text} = 0.",
            f"k = {root.numerator}/{root.denominator}." if root.denominator != 1 else f"k = {root.numerator}.",
        ],
        [
            ("sign-error-moving-term", -root),
            ("inverted-ratio", Fraction(alpha, -beta) if beta else None),
            ("forgot-to-divide-by-coefficient", Fraction(-beta)),
            ("sign-error-moving-term", Fraction(beta, alpha) if root != Fraction(beta, alpha) else None),
            ("inverted-ratio", Fraction(-alpha, beta) if beta else None),
        ],
        follow_through="-Subs(part_a, (k,), (0,))/(Subs(part_a, (k,), (1,)) - Subs(part_a, (k,), (0,)))",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r3b_det3", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3b_det3(rng: random.Random, tier: str = "exam") -> dict:
    """A 3x3 determinant with an unknown entry: det in k, its value, det of a scaled inverse, the singular k."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
