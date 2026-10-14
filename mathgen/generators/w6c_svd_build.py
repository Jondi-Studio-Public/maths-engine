"""Generates `w6c_svd` questions (easy, medium, hard) and the exam chain `exd_x_w6c_svd`.

``uc2-svd-small-matrices`` (``linear-algebra``): the singular values of a real matrix are the square roots of the
eigenvalues of AᵀA (equivalently of AAᵀ, which is the smaller matrix when A is 2×3). Every matrix here has two
distinct positive integer singular values σ₁ > σ₂, found once by searching the integer 2×2 and 2×3 matrices of small
entries, so every key is exact.

* easy: the largest singular value (the spectral norm) of a 2×2 matrix that is neither symmetric nor diagonal, so its
  singular values are not read off its eigenvalues.
* medium: the smaller non-zero singular value of a 2×3 matrix. AᵀA is 3×3 with eigenvalues σ₁², σ₂² and 0; the work is
  done on AAᵀ.
* hard (Eckart-Young): one of the Frobenius distance from a 2×3 matrix to its best rank-1 approximation (σ₂), the
  condition number σ₁/σ₂ of a 2×2 matrix, or the share σ₁²/(σ₁² + σ₂²) of the squared Frobenius norm that the best
  rank-1 approximation keeps.
* exam (``exd_x_w6c_svd``): a 2×3 matrix. (a) σ₁, (b) σ₂, (c) the share of ‖A‖_F² kept by the best rank-1
  approximation (follows through from (a) and (b)), (d) a right singular vector for σ₁ (any non-zero multiple).

Each key is re-derived from SymPy's eigenvalues of AAᵀ, and the Frobenius share is checked against the sum of the
squared entries. Right singular vectors are the null space of AᵀA − σ₁²I in exact fractions, and checked by
A v = σ₁ u with |u| = σ₁|v|.
"""

from __future__ import annotations

import functools
from fractions import Fraction
from math import gcd, isqrt

import sympy

from ..core import check, generator, phrase
from ._ucv1_linalg import fm, mat_str, mmul, nullspace, primitive, transpose
from ._w6c_kit import redrawn, value_fields, value_part, vector_part


def _is_square(n: int) -> bool:
    return n >= 0 and isqrt(n) ** 2 == n


#: Directions (a, b) of the first eigenvector of a Gram matrix AAᵀ. With c = a/√n and s = b/√n for n = a² + b², the
#: Gram matrix c²σ₁² + s²σ₂² ... has rational entries, so every integer matrix with a rational singular direction is
#: reached.
_DIRECTIONS = [(a, b) for a in range(0, 7) for b in range(0, 7) if (a, b) != (0, 0) and gcd(a, b) == 1]


def _vectors(length, bound):
    out = [()]
    for _ in range(length):
        out = [v + (x,) for v in out for x in range(-bound, bound + 1)]
    return out


def _search(columns, bound, top):
    """``{(σ₁, σ₂): [rows]}``: integer 2×columns matrices with entries within ``bound`` and distinct positive integer
    singular values up to ``top``. The Gram matrix AAᵀ = [[p, q], [q, r]] of such a matrix is Q diag(σ₁², σ₂²) Qᵀ
    for a rotation Q with rational c², s² and cs, so the candidate Gram matrices are listed first and then matched to integer rows of that norm."""
    by_norm: dict[int, list] = {}
    for v in _vectors(columns, bound):
        by_norm.setdefault(sum(x * x for x in v), []).append(v)
    found: dict[tuple[int, int], list] = {}
    for s1 in range(2, top + 1):
        for s2 in range(1, s1):
            big, small = Fraction(s1 * s1), Fraction(s2 * s2)
            for a, b in _DIRECTIONS:
                n = a * a + b * b
                for sign in (1, -1):
                    p = (big * a * a + small * b * b) / n
                    r = (big * b * b + small * a * a) / n
                    q = sign * (big - small) * a * b / n
                    if p.denominator != 1 or r.denominator != 1 or q.denominator != 1:
                        continue
                    for r1 in by_norm.get(int(p), ()):
                        for r2 in by_norm.get(int(r), ()):
                            if sum(x * y for x, y in zip(r1, r2)) == q:
                                found.setdefault((s1, s2), []).append((r1, r2))
    return found


@functools.cache
def _pool(shape):
    """``{(σ₁, σ₂): [rows, ...]}`` for ``"2x2"`` or ``"2x3"`` integer matrices."""
    if shape == "2x2":
        pool: dict[tuple[int, int], list] = {}
        for key, matrices in _search(2, 7, 15).items():
            for rows in matrices:
                (a, b), (c, d) = rows
                if b == c or (a, b, c, d).count(0) > 1 or (b == 0 and c == 0) or (a == 0 and d == 0):
                    continue  # symmetric (singular values are eigenvalues), diagonal, or too sparse
                pool.setdefault(key, []).append(rows)
        return pool
    pool = {}
    for key, matrices in _search(3, 6, 15).items():
        for rows in matrices:
            if (rows[0] + rows[1]).count(0) <= 2 and all(any(col) for col in zip(*rows)):
                pool.setdefault(key, []).append(rows)
    return pool


def _draw(rng, shape):
    pool = _pool(shape)
    s1, s2 = rng.choice(sorted(pool))
    return [list(r) for r in rng.choice(pool[(s1, s2)])], s1, s2


def _gram(a, side):
    """AᵀA (``side="right"``) or AAᵀ (``side="left"``) as a Fraction matrix."""
    m = fm(a)
    return mmul(transpose(m), m) if side == "right" else mmul(m, transpose(m))


def _frobenius_squared(a):
    return sum(v * v for row in a for v in row)


def _verify(a, s1, s2):
    """SymPy's eigenvalues of AAᵀ (the non-zero eigenvalues of AᵀA), and the Frobenius norm identity."""
    m = sympy.Matrix(a)
    check(dict((m * m.T).eigenvals()) == {s1 * s1: 1, s2 * s2: 1}, "SymPy's eigenvalues of AAᵀ differ")
    check(_frobenius_squared(a) == s1 * s1 + s2 * s2, "the squared Frobenius norm is not the sum of the squared values")


def _right_vector(a, sigma):
    """A right singular vector for ``sigma``: the primitive null vector of AᵀA - sigma²I."""
    gram = _gram(a, "right")
    n = len(gram)
    shifted = [[gram[i][j] - (sigma * sigma if i == j else 0) for j in range(n)] for i in range(n)]
    basis = nullspace(shifted)
    check(len(basis) == 1, "the singular value is repeated")
    v = primitive(basis[0])
    image = [sum(a[i][j] * v[j] for j in range(n)) for i in range(len(a))]
    check(sum(x * x for x in image) == sigma * sigma * sum(x * x for x in v), "A v does not have length sigma |v|")
    return v


def _other_vectors(a, s2):
    """Right singular vectors for σ₂ and for the zero eigenvalue of AᵀA (a 2×3 matrix has a null vector)."""
    v2 = _right_vector(a, s2)
    null = nullspace(fm(a))
    return v2, (primitive(null[0]) if len(null) == 1 else None)


_EASY_STEMS = (
    "Find the largest singular value of A = {m}.",
    "The matrix A = {m} has singular values σ₁ > σ₂ > 0. Find σ₁.",
    "Find the spectral norm ‖A‖₂ of A = {m}.",
)
_MEDIUM_STEMS = (
    "Find the smaller non-zero singular value of the 2×3 matrix A = {m}.",
    "The 2×3 matrix A = {m} has rank 2. Find its smallest non-zero singular value.",
    "In a singular value decomposition A = UΣVᵀ of A = {m}, find the diagonal entry σ₂ of Σ.",
)
_ERROR_STEMS = (
    "Let A₁ be the rank-1 matrix closest to A = {m} in the Frobenius norm. Find ‖A − A₁‖_F.",
    "Keeping only the largest singular value of A = {m} gives a rank-1 approximation A₁. Find ‖A − A₁‖_F.",
)
_CONDITION_STEMS = (
    "Find the 2-norm condition number σ₁/σ₂ of A = {m}, as a fraction in lowest terms.",
    "The matrix A = {m} is invertible. Find ‖A‖₂ · ‖A⁻¹‖₂ as a fraction in lowest terms.",
)
_SHARE_STEMS = (
    "What fraction of ‖A‖_F² is kept by the best rank-1 approximation of A = {m}? Give it in lowest terms.",
    "The rank-1 matrix A₁ closest to A = {m} has ‖A₁‖_F² = k‖A‖_F². Find k in lowest terms.",
)


@redrawn
def _easy(rng):
    a, s1, s2 = _draw(rng, "2x2")
    _verify(a, s1, s2)
    wrong = [
        ("svd-forgot-square-root", s1 * s1),
        ("svd-wrong-ordering", s2),
        ("svd-determinant-as-value", s1 * s2),
        ("svd-frobenius-squared", s1 * s1 + s2 * s2),
        ("svd-sum-of-values", s1 + s2),
    ]
    return dict(
        text=phrase(rng, _EASY_STEMS, m=mat_str(fm(a))),
        working=[
            f"AᵀA = {mat_str(_gram(a, 'right'))}, with eigenvalues {s1 * s1} and {s2 * s2}.",
            f"The singular values are the square roots, so σ₁ = {s1}.",
        ],
        **value_fields(s1, wrong),
    )


@redrawn
def _medium(rng):
    a, s1, s2 = _draw(rng, "2x3")
    _verify(a, s1, s2)
    wrong = [
        ("svd-null-eigenvalue-used", 0),
        ("svd-forgot-square-root", s2 * s2),
        ("svd-wrong-ordering", s1),
        ("svd-determinant-as-value", s1 * s2),
        ("svd-frobenius-squared", s1 * s1 + s2 * s2),
    ]
    return dict(
        text=phrase(rng, _MEDIUM_STEMS, m=mat_str(fm(a))),
        working=[
            f"AAᵀ = {mat_str(_gram(a, 'left'))} has eigenvalues {s1 * s1} and {s2 * s2}.",
            f"AᵀA has the same non-zero eigenvalues and one more eigenvalue, 0, so σ₂ = {s2}.",
        ],
        **value_fields(s2, wrong),
    )


@redrawn
def _hard_error(rng):
    a, s1, s2 = _draw(rng, "2x3")
    _verify(a, s1, s2)
    wrong = [
        ("svd-error-uses-largest", s1),
        ("svd-forgot-square-root", s2 * s2),
        ("svd-frobenius-squared", s1 * s1 + s2 * s2),
        ("svd-determinant-as-value", s1 * s2),
    ]
    return dict(
        text=phrase(rng, _ERROR_STEMS, m=mat_str(fm(a))),
        working=[
            f"The singular values are {s1} and {s2}.",
            f"The best rank-1 approximation discards σ₂, so ‖A − A₁‖_F = σ₂ = {s2}.",
        ],
        **value_fields(s2, wrong),
    )


@redrawn
def _hard_condition(rng):
    for _ in range(40):
        a, s1, s2 = _draw(rng, "2x2")
        if s1 % s2:
            break
    _verify(a, s1, s2)
    ratio = Fraction(s1, s2)
    wrong = [
        ("svd-condition-inverted", Fraction(s2, s1)),
        ("svd-condition-squared", ratio * ratio),
        ("svd-ratio-of-sum", Fraction(s1, s1 + s2)),
        ("svd-ratio-of-eigenvalue-gap", Fraction(s1 - s2, s2)),
    ]
    return dict(
        text=phrase(rng, _CONDITION_STEMS, m=mat_str(fm(a))),
        working=[
            f"The singular values are {s1} and {s2}.",
            f"The condition number is σ₁/σ₂ = {s1}/{s2} = {ratio.numerator}/{ratio.denominator}.",
        ],
        **value_fields(ratio, wrong),
    )


@redrawn
def _hard_share(rng):
    a, s1, s2 = _draw(rng, rng.choice(["2x2", "2x3"]))
    _verify(a, s1, s2)
    share = Fraction(s1 * s1, s1 * s1 + s2 * s2)
    check(share.denominator != 1, "the share is 1")
    wrong = [
        ("svd-forgot-squares-in-share", Fraction(s1, s1 + s2)),
        ("svd-discarded-share", Fraction(s2 * s2, s1 * s1 + s2 * s2)),
        ("svd-ratio-instead-of-share", Fraction(s1 * s1, s2 * s2)),
        ("svd-squares-dropped-in-denominator", Fraction(s1 * s1, s1 * s1 + s2)),
        ("svd-forgot-squares-in-share", Fraction(s2, s1 + s2)),
    ]
    return dict(
        text=phrase(rng, _SHARE_STEMS, m=mat_str(fm(a))),
        working=[
            f"‖A‖_F² is the sum of the squared entries: {_frobenius_squared(a)} = {s1 * s1} + {s2 * s2}.",
            f"The rank-1 approximation keeps σ₁² = {s1 * s1}, a share of {share.numerator}/{share.denominator}.",
        ],
        **value_fields(share, wrong),
    )


@generator(
    "w6c_svd",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w6c_svd(rng, tier="easy"):
    """Singular values of a small integer matrix; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_error, _hard_condition, _hard_share])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_w6c_svd", level=19, presets={"exam": dict(tier="exam")})
@redrawn
def exd_x_w6c_svd(rng, tier="exam"):
    """Exam chain on a 2×3 matrix: σ₁, σ₂, the share of ‖A‖_F² kept by the rank-1 approximation, a right singular
    vector for σ₁."""
    check(tier == "exam", f"unknown tier {tier!r}")
    a, s1, s2 = _draw(rng, "2x3")
    _verify(a, s1, s2)
    v1 = _right_vector(a, s1)
    v2, v3 = _other_vectors(a, s2)
    share = Fraction(s1 * s1, s1 * s1 + s2 * s2)
    check(share.denominator != 1, "the share is 1")
    # A left singular vector padded to R³ is a plausible wrong vector, as is a row of A.
    left = nullspace([[x - (s1 * s1 if i == j else 0) for j, x in enumerate(row)] for i, row in enumerate(_gram(a, "left"))])
    padded = primitive(left[0]) + [0] if len(left) == 1 else None
    rows = [primitive(r) for r in a]
    return dict(
        text=f"Let A = {mat_str(fm(a))}.",
        parts=[
            value_part(
                "Find the largest singular value σ₁ of A.",
                s1,
                2,
                [f"AAᵀ = {mat_str(_gram(a, 'left'))} has eigenvalues {s1 * s1} and {s2 * s2}, so σ₁ = {s1}."],
                [
                    ("svd-forgot-square-root", s1 * s1),
                    ("svd-wrong-ordering", s2),
                    ("svd-determinant-as-value", s1 * s2),
                    ("svd-frobenius-squared", s1 * s1 + s2 * s2),
                ],
                sensible_range=(0, None),
            ),
            value_part(
                "Find the smaller non-zero singular value σ₂ of A.",
                s2,
                2,
                [f"The other eigenvalue of AAᵀ is {s2 * s2}, so σ₂ = {s2}."],
                [
                    ("svd-null-eigenvalue-used", 0),
                    ("svd-forgot-square-root", s2 * s2),
                    ("svd-wrong-ordering", s1),
                    ("svd-determinant-as-value", s1 * s2),
                ],
                sensible_range=(0, None),
            ),
            value_part(
                "The best rank-1 approximation A₁ of A keeps σ₁ and discards σ₂. "
                "What fraction of ‖A‖_F² is ‖A₁‖_F²? Give it in lowest terms.",
                share,
                3,
                [f"‖A₁‖_F²/‖A‖_F² = σ₁²/(σ₁² + σ₂²) = {s1 * s1}/{s1 * s1 + s2 * s2} = {share.numerator}/{share.denominator}."],
                [
                    ("svd-forgot-squares-in-share", Fraction(s1, s1 + s2)),
                    ("svd-discarded-share", Fraction(s2 * s2, s1 * s1 + s2 * s2)),
                    ("svd-ratio-instead-of-share", Fraction(s1 * s1, s2 * s2)),
                ],
                follow_through="part_a**2/(part_a**2 + part_b**2)",
                sensible_range=(0, 1),
            ),
            vector_part(
                "Find a right singular vector of A for σ₁: a non-zero vector v with AᵀAv = σ₁²v. Any non-zero multiple is accepted.",
                v1,
                3,
                [f"Solving (AᵀA − {s1 * s1}I)v = 0 gives v = {tuple(v1)}."],
                [
                    (v2, "svd-vector-for-other-value"),
                    (v3, "svd-null-direction-chosen"),
                    (padded, "svd-left-vector-used"),
                    (rows[0], "svd-row-as-singular-vector"),
                    (rows[1], "svd-row-as-singular-vector"),
                ],
            ),
        ],
    )
