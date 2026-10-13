"""Generates `fy1la_column_space` questions (easy, medium, hard) and the exam chain `exd_x_fy1la_column_space`.

``ucv1-column-space-basis-dimension`` (``linear-algebra``, answer ``number``): the dimension of a column space is the
rank, a basis is the set of pivot columns of the ORIGINAL matrix, and rank-nullity says rank + nullity = number of
columns.

* easy: rank-nullity with no matrix to reduce: a linear map ℝⁿ → ℝᵐ with a stated rank (or nullity); the other one.
* medium: how many of four listed vectors in ℝ³ or ℝ⁴ are redundant (can be deleted without changing the span), which
  is the number of vectors minus the dimension of the span, so the vectors have to be reduced and the pivots counted.
* hard: the dimension of the null space of the TRANSPOSE of a given m×n matrix, which is m minus the rank (not n minus
  the rank).
* exam (``exd_x_fy1la_column_space``): a 3×4 matrix of rank 2 whose first column is a pivot column. (a) its rank,
  (b) its nullity (follows through from (a)), (c) which columns of A form a basis of its column space (a list of
  column numbers), (d) the coefficients that write the first non-pivot column in terms of the basis columns.

Every matrix is built from a known rank (as `B C`, or as `E R` for a target RREF `R`) and re-derived by Gauss-Jordan
elimination in exact fractions and by SymPy's `rank()`, `columnspace()` and `rref()`.
"""

from __future__ import annotations

from ..core import VerificationError, attempts, check, generator, phrase
from ._fy1la_kit import (
    column,
    fm,
    mat_str,
    matrix_from_rref,
    matrix_part,
    max_abs,
    mmul,
    num_distractors,
    number_part,
    rank,
    rref,
    sym,
    text_distractors,
    transpose,
    vec_str,
    vector_distractors,
)

_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _sup(n: int) -> str:
    return str(n).translate(_SUP)


def _product(rng, m, n, r):
    """An m×n integer matrix of rank exactly r with small entries, no zero row or column, no repeated column."""
    for _ in attempts(3000, f"a {m}×{n} matrix of rank {r}"):
        b = [[rng.randint(-2, 2) for _ in range(r)] for _ in range(m)]
        c = [[rng.randint(-2, 2) for _ in range(n)] for _ in range(r)]
        mat = mmul(fm(b), fm(c))
        if max_abs(mat) > 8 or max_abs(mat) < 2 or rank(mat) != r:
            continue
        if any(all(v == 0 for v in row) for row in mat) or any(all(v == 0 for v in col) for col in zip(*mat)):
            continue
        cols = [tuple(col) for col in zip(*mat)]
        if len(set(cols)) < n or any(tuple(-v for v in col) in cols for col in cols):
            continue
        if len({tuple(row) for row in mat}) < m:
            continue
        if r > 1 and any(rank([list(cols[i]), list(cols[j])]) < 2 for i in range(n) for j in range(i + 1, n)):
            continue
        if sum(1 for row in mat for v in row if v == 0) > 3:
            continue
        return mat
    raise VerificationError("unreachable")  # pragma: no cover - attempts() raises first


def _verify_rank(mat, r):
    check(sym(mat).rank() == r and rank(mat) == r, "the rank differs between routes")
    check(len(sym(mat).columnspace()) == r, "SymPy's column space has a different dimension")


# -- easy ----------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(500, "a rank-nullity question"):
        n = rng.choice([4, 5, 6, 7])
        m = rng.choice([x for x in (3, 4, 5, 6) if x != n])
        if rng.random() < 0.5:
            r = rng.randint(2, min(m, n) - 1)
            answer, given = n - r, r
            stems = (
                "A linear map T: ℝ{sn} → ℝ{sm} has an image of dimension {r}. What is the dimension of its kernel?",
                "The {m}×{n} matrix A has rank {r}. Find the dimension of the null space of A.",
            )
            dist = num_distractors(
                answer,
                [
                    (r, "rank-nullity-swapped"),
                    (m - r, "rank-nullity-uses-rows"),
                    (n, "rank-nullity-sum-reported"),
                    (n - r + 1, "rank-nullity-off-by-one"),
                ],
                lowest=0,
            )
            working = [
                f"Rank-nullity: rank + nullity = {n}, the number of columns.",
                f"So the nullity is {n} - {r} = {answer}.",
            ]
        else:
            d = rng.randint(1, n - 2)
            r = n - d
            if r > m:
                continue
            answer, given = r, d
            stems = (
                "A linear map T: ℝ{sn} → ℝ{sm} has a kernel of dimension {r}. What is the dimension of its image?",
                "The {m}×{n} matrix A has a null space of dimension {r}. Find the dimension of the column space of A.",
            )
            dist = num_distractors(
                answer,
                [
                    (given, "rank-nullity-swapped"),
                    (m - given, "rank-nullity-uses-rows"),
                    (n, "rank-nullity-sum-reported"),
                    (answer + 1, "rank-nullity-off-by-one"),
                ],
                lowest=0,
            )
            working = [
                f"Rank-nullity: rank + nullity = {n}, the number of columns.",
                f"So the rank is {n} - {given} = {answer}.",
            ]
        if dist is None:
            continue
        return dict(
            text=phrase(rng, stems, n=n, m=m, r=given, sn=_sup(n), sm=_sup(m)),
            answer_kind="number",
            answer=answer,
            answer_text=str(answer),
            answer_latex=str(answer),
            working=working,
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


# -- medium ---------------------------------------------------------------------------------------------------------


def _medium(rng):
    for _ in attempts(500, "a span-dimension question"):
        m, n, r = rng.choice([(3, 4, 2), (4, 4, 2), (4, 4, 3)])
        mat = _product(rng, m, n, r)
        _verify_rank(mat, r)
        vectors = [column(mat, c) for c in range(n)]
        answer = n - r
        dist = num_distractors(
            answer,
            [
                (r, "rank-nullity-swapped"),
                (n, "dimension-counts-vectors"),
                (answer + 1, "rank-misses-a-pivot"),
                (answer - 1, "dimension-counts-dependent-vector"),
            ],
            lowest=0,
        )
        if dist is None:
            continue
        names = ", ".join(f"v{i + 1} = {vec_str(v)}" for i, v in enumerate(vectors))
        stems = (
            "The vectors {vs} span a subspace W of ℝ{sup}. What is the largest number of these vectors that can be "
            "deleted without changing W?",
            "Let W be the subspace of ℝ{sup} spanned by {vs}. How many of these vectors are redundant, in the sense "
            "that deleting them leaves W unchanged?",
        )
        return dict(
            text=phrase(rng, stems, sup=_sup(m), vs=names),
            answer_kind="number",
            answer=answer,
            answer_text=str(answer),
            answer_latex=str(answer),
            working=[
                f"A basis of W is a set of independent vectors, and dim W is the rank of the matrix with these columns, {mat_str(mat)}.",
                f"Row-reducing leaves {r} non-zero rows, so dim W = {r} and {n} - {r} = {answer} of the {n} vectors can be deleted.",
            ],
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


# -- hard -----------------------------------------------------------------------------------------------------------


def _hard(rng):
    for _ in attempts(500, "a left-null-space question"):
        m, n, r = rng.choice([(4, 3, 2), (4, 3, 3), (4, 5, 3), (3, 5, 2), (5, 4, 3), (4, 5, 2), (5, 3, 2)])
        mat = _product(rng, m, n, r)
        _verify_rank(mat, r)
        answer = m - r
        check(len(sym(transpose(mat)).nullspace()) == answer, "SymPy's null space of the transpose differs")
        dist = num_distractors(
            answer,
            [
                (n - r, "left-null-space-confused-with-null-space"),
                (r, "rank-nullity-swapped"),
                (m, "rank-nullity-sum-reported"),
                (answer + 1, "rank-nullity-off-by-one"),
                (n, "dimension-counts-vectors"),
            ],
            lowest=0,
        )
        if dist is None:
            continue
        stems = (
            "Let A = {a}. Find the dimension of the null space of Aᵀ.",
            "The matrix A = {a} is {m}×{n}. What is the dimension of the null space of its transpose Aᵀ?",
        )
        return dict(
            text=phrase(rng, stems, a=mat_str(mat), m=m, n=n),
            answer_kind="number",
            answer=answer,
            answer_text=str(answer),
            answer_latex=str(answer),
            working=[
                f"Row-reducing A gives rank A = {r}. Aᵀ is {n}×{m} and has the same rank {r}.",
                f"Aᵀ has {m} columns, so by rank-nullity its null space has dimension {m} - {r} = {answer}.",
            ],
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


@generator(
    "fy1la_column_space",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1la_column_space(rng, tier="easy"):
    """Dimension of a column space and rank-nullity; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -----------------------------------------------------------------------------------------------------------


@generator("exd_x_fy1la_column_space", level=17, presets={"exam": dict(tier="exam")})
def exd_x_fy1la_column_space(rng, tier="exam"):
    """Exam chain: rank, nullity, a basis of the column space by column numbers, and a dependent column's coefficients."""
    check(tier == "exam", f"unknown tier {tier!r}")
    nz = [v for v in range(-3, 4) if v]
    for _ in attempts(500, "an exam column-space chain"):
        pivots = rng.choice([(0, 2), (0, 2), (0, 2), (0, 1)])
        if pivots == (0, 2):
            target = [[1, rng.choice(nz), 0, rng.choice(nz)], [0, 0, 1, rng.choice(nz)], [0, 0, 0, 0]]
        else:
            target = [[1, 0, rng.choice(nz), rng.choice(nz)], [0, 1, rng.choice(nz), rng.choice(nz)], [0, 0, 0, 0]]
        mat = matrix_from_rref(rng, target)
        if mat is None:
            continue
        reduced, found = rref(mat)
        check(reduced == fm(target) and tuple(found) == pivots, "unexpected reduction")
        ref, ref_pivots = sym(mat).rref()
        check(tuple(ref_pivots) == pivots, "SymPy's pivot columns differ")
        free = [c for c in range(4) if c not in pivots]
        f = free[-1]  # the last column, whose two reduced entries are both non-zero in either pattern
        other_free = free[0]
        a, b = int(reduced[0][f]), int(reduced[1][f])
        p1, p2 = pivots
        check(
            all(mat[r][f] == a * mat[r][p1] + b * mat[r][p2] for r in range(3)),
            "the free column is not the stated combination",
        )

        def pair_text(i, j):
            lo, hi = sorted((i, j))
            return f"{lo + 1}, {hi + 1}"

        basis_text = pair_text(p1, p2)
        cands = [
            (pair_text(*free), "free-columns-as-basis"),
            (pair_text(0, 1), "first-columns-assumed-independent"),
            (pair_text(p1, free[0]), "basis-mixes-pivot-and-free-columns"),
            (pair_text(p2, free[-1]), "basis-mixes-pivot-and-free-columns"),
            (pair_text(1, 2), "first-columns-assumed-independent"),
        ]
        dist_c = text_distractors(basis_text, cands)
        dist_a = num_distractors(
            2, [(3, "rank-counts-all-rows"), (4, "rank-counts-all-columns"), (1, "rank-misses-a-pivot")]
        )
        dist_b = num_distractors(
            2, [(1, "free-variables-from-rows"), (3, "free-variables-miscounted"), (4, "free-variables-all-columns")]
        )
        dist_d = vector_distractors(
            [
                ([b, a], "coefficients-order-swapped"),
                ([-a, -b], "span-coefficient-sign-slip"),
                ([a, -b], "span-coefficient-sign-slip"),
                ([int(reduced[0][other_free]), int(reduced[1][other_free])], "coefficients-read-from-wrong-column"),
            ],
            [a, b],
        )
        if None in (dist_a, dist_b, dist_c, dist_d):
            continue
        # The list answer is the pivot columns; the grader reads it as a list of numbers.
        part_c = dict(
            prompt="Which columns of A (by column number) form a basis for the column space of A?",
            answer_kind="list",
            answer=[p1 + 1, p2 + 1],
            answer_text=basis_text,
            answer_latex="plain:" + basis_text,
            marks=2,
            working=[
                f"The pivots of the reduced matrix are in columns {p1 + 1} and {p2 + 1}; the matching columns of A (not of the reduced matrix) are a basis."
            ],
            distractors=dist_c,
        )
        return dict(
            text=phrase(
                rng,
                ("Let A = {a}.", "Consider the matrix A = {a}.", "The matrix A = {a} has three rows and four columns."),
                a=mat_str(mat),
            ),
            parts=[
                number_part(
                    "Find the rank of A.",
                    2,
                    dist_a,
                    marks=1,
                    working=[f"Row-reducing A leaves two non-zero rows ({mat_str(reduced)}), so the rank is 2."],
                    sensible_range=(0, 4),
                ),
                number_part(
                    "Find the dimension of the null space of A.",
                    2,
                    dist_b,
                    marks=1,
                    follow_through="4 - part_a",
                    working=["A has 4 columns, so by rank-nullity the nullity is 4 - 2 = 2."],
                    sensible_range=(0, 5),
                ),
                part_c,
                matrix_part(
                    f"Write column {f + 1} of A as a·(column {p1 + 1}) + b·(column {p2 + 1}). Give your answer as (a, b).",
                    [a, b],
                    dist_d,
                    marks=3,
                    vector=True,
                    working=[
                        f"Column {f + 1} of the reduced matrix is ({a}, {b}, 0), and the same combination of the basis columns of A gives column {f + 1} of A."
                    ],
                ),
            ],
        )
    raise VerificationError("unreachable")  # pragma: no cover
