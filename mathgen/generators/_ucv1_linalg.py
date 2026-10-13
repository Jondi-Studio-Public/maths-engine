"""Exact small-matrix helpers shared by the first-year linear algebra generators (`ucv1_eigenvectors_build`,
`ucv1_diagonalisation_build`, `ucv1_null_space_build`).

Everything here works on lists of rows of `Fraction`, so a question's numbers never pass through a float, and none of
it uses SymPy: the generators keep SymPy for their independent re-derivation, and this module is the hand route
(Gauss-Jordan elimination, adjugate-free inverse, cofactor-free determinant).

* `unimodular` draws an integer matrix of determinant 1 or -1 with small entries, so that `P D P^-1` is an integer
  matrix for any diagonal `D`;
* `nullspace`, `rank`, `inverse`, `det`, `mmul`, `mpow` are the usual operations;
* `primitive` scales a vector to coprime integers whose first non-zero entry is positive, the spelling every
  eigenvector and null-space answer is stored in;
* `vector_distractors` / `matrix_distractors` turn candidate wrong answers into tagged `Distractor`s, dropping any
  the matrix grader would accept (a multiple of the right vector, say), any repeat, and the zero vector.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from ..answers import Distractor, exact_value
from ..core import VerificationError, attempts
from ..grading.matrix import matrices_match, matrix_text, vector_text

Matrix = list[list[Fraction]]


def fm(rows) -> Matrix:
    """`rows` (lists of ints or Fractions) as a matrix of Fractions."""
    return [[Fraction(v) for v in row] for row in rows]


def identity(n: int) -> Matrix:
    return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]


def diagonal(entries) -> Matrix:
    n = len(entries)
    return [[Fraction(entries[i]) if i == j else Fraction(0) for j in range(n)] for i in range(n)]


def transpose(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def mmul(a: Matrix, b: Matrix) -> Matrix:
    cols = list(zip(*b))
    return [[sum((x * y for x, y in zip(row, col)), Fraction(0)) for col in cols] for row in a]


def mpow(a: Matrix, n: int) -> Matrix:
    result = identity(len(a))
    for _ in range(n):
        result = mmul(result, a)
    return result


def madd(a: Matrix, b: Matrix, scale=1) -> Matrix:
    return [[x + scale * y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def mscale(a: Matrix, s) -> Matrix:
    return [[Fraction(s) * x for x in row] for row in a]


def rref(a: Matrix) -> tuple[Matrix, list[int]]:
    """Reduced row-echelon form and the pivot columns."""
    m = [row[:] for row in a]
    rows, cols = len(m), len(m[0])
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        pivot = next((i for i in range(r, rows) if m[i][c] != 0), None)
        if pivot is None:
            continue
        m[r], m[pivot] = m[pivot], m[r]
        lead = m[r][c]
        m[r] = [v / lead for v in m[r]]
        for i in range(rows):
            if i != r and m[i][c] != 0:
                factor = m[i][c]
                m[i] = [v - factor * w for v, w in zip(m[i], m[r])]
        pivots.append(c)
        r += 1
    return m, pivots


def rank(a: Matrix) -> int:
    return len(rref(a)[1])


def nullspace(a: Matrix) -> list[list[Fraction]]:
    """A basis of {x : a x = 0}, one vector per free column."""
    reduced, pivots = rref(a)
    cols = len(a[0])
    basis = []
    for free in (c for c in range(cols) if c not in pivots):
        vec = [Fraction(0)] * cols
        vec[free] = Fraction(1)
        for row, pc in zip(reduced, pivots):
            vec[pc] = -row[free]
        basis.append(vec)
    return basis


def det(a: Matrix) -> Fraction:
    m = [row[:] for row in a]
    n = len(m)
    sign = 1
    value = Fraction(1)
    for c in range(n):
        pivot = next((i for i in range(c, n) if m[i][c] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != c:
            m[c], m[pivot] = m[pivot], m[c]
            sign = -sign
        value *= m[c][c]
        for i in range(c + 1, n):
            factor = m[i][c] / m[c][c]
            if factor:
                m[i] = [v - factor * w for v, w in zip(m[i], m[c])]
    return sign * value


def inverse(a: Matrix) -> Matrix:
    n = len(a)
    aug = [row[:] + identity(n)[i] for i, row in enumerate(a)]
    reduced, pivots = rref(aug)
    if pivots[:n] != list(range(n)):
        raise VerificationError("matrix is singular")
    return [row[n:] for row in reduced]


def apply(a: Matrix, v) -> list[Fraction]:
    return [sum((x * y for x, y in zip(row, v)), Fraction(0)) for row in a]


def primitive(vec) -> list[int]:
    """`vec` scaled to coprime integers with the first non-zero entry positive."""
    vec = [Fraction(v) for v in vec]
    denominator = 1
    for v in vec:
        denominator = denominator * v.denominator // gcd(denominator, v.denominator)
    ints = [int(v * denominator) for v in vec]
    divisor = 0
    for v in ints:
        divisor = gcd(divisor, abs(v))
    if divisor == 0:
        raise VerificationError("the zero vector has no primitive form")
    ints = [v // divisor for v in ints]
    first = next(v for v in ints if v != 0)
    return [-v for v in ints] if first < 0 else ints


def is_integer_matrix(a: Matrix) -> bool:
    return all(v.denominator == 1 for row in a for v in row)


def max_abs(a: Matrix) -> Fraction:
    return max(abs(v) for row in a for v in row)


def unimodular(rng, n: int, bound: int = 3) -> Matrix:
    """An integer matrix of determinant 1 or -1 with every entry within `bound`, built from random row additions."""
    for _ in attempts(500, "a small unimodular matrix"):
        m = identity(n)
        for _step in range(rng.randint(2, 3) if n == 2 else rng.randint(3, 5)):
            i, j = rng.sample(range(n), 2)
            k = rng.choice([-2, -1, 1, 2])
            m[i] = [v + k * w for v, w in zip(m[i], m[j])]
        order = list(range(n))
        rng.shuffle(order)
        signs = [rng.choice([-1, 1]) for _ in range(n)]
        m = [[m[i][order[c]] * signs[c] for c in range(n)] for i in range(n)]
        if max_abs(m) <= bound and abs(det(m)) == 1:
            return m
    raise VerificationError("unreachable")  # pragma: no cover - attempts() raises first


def draw_2x2(rng, lo: int = -4, hi: int = 5, cap: int = 30, full: bool = False, bound: int = 4):
    """`(P, [l1, l2], A)` with A = P diag(l1, l2) P^-1 an integer 2×2 whose eigenvalues are distinct integers in
    [lo, hi]. A is never diagonal; with `full` every entry of A is non-zero as well, so it is not triangular."""
    for _ in attempts(500, "a 2×2 matrix with integer eigenvalues"):
        p = unimodular(rng, 2, bound)
        l1, l2 = rng.sample(range(lo, hi + 1), 2)
        a = mmul(mmul(p, diagonal([l1, l2])), inverse(p))
        if max_abs(a) > cap:
            continue
        if (a[0][1] == 0 and a[1][0] == 0) or (full and (a[0][1] == 0 or a[1][0] == 0)):
            continue
        return p, [l1, l2], a
    raise VerificationError("unreachable")  # pragma: no cover


def column(m: Matrix, c: int) -> list[Fraction]:
    return [row[c] for row in m]


def with_columns(vectors) -> Matrix:
    return transpose([[Fraction(x) for x in v] for v in vectors])


# -- spelling and answers ---------------------------------------------------------------------------------------


def entry(value):
    """One matrix entry as the JSON the matrix grader reads: an int, or {"p": .., "q": ..}."""
    return exact_value(Fraction(value))


def rows_value(a) -> list[list]:
    """A matrix (list of rows) as the answer value: rows of exact entries."""
    return [[entry(v) for v in row] for row in a]


def column_value(vec) -> list[list]:
    """A vector as a one-column answer value."""
    return [[entry(v)] for v in vec]


def mat_str(a) -> str:
    """The stem and answer spelling of a matrix: [[1, 2], [3, 4]]."""
    return matrix_text(rows_value(a))


def vec_str(vec) -> str:
    """The spelling of a vector answer: (1, -2, 3)."""
    return vector_text([entry(v) for v in vec])


def vector_distractors(candidates, answer_vec, form="scalar_multiple") -> list[Distractor]:
    """Tagged distractors from `(vector, misconception)` pairs, in priority order.

    A candidate is kept only when the matrix grader would reject it against the answer (`form` applies), it is not the
    zero vector, and no earlier candidate spells it the same way (vectors are compared in primitive form, because a
    multiple of a wrong vector is the same wrong answer).
    """
    answer_value = column_value(answer_vec)
    out, seen = [], set()
    for vec, tag in candidates:
        if vec is None or all(v == 0 for v in vec):
            continue
        vec = primitive(vec)
        key = tuple(vec)
        if key in seen:
            continue
        value = column_value(vec)
        if matrices_match(value, answer_value, form):
            continue
        seen.add(key)
        out.append(Distractor(vec_str(vec), tag, value))
    return out


def matrix_distractors(candidates, answer_rows, form=None) -> list[Distractor]:
    """Tagged distractors from `(matrix, misconception)` pairs: rejected by the matrix grader, no repeats."""
    out, seen = [], set()
    for mat, tag in candidates:
        if mat is None:
            continue
        text = mat_str(mat)
        if text in seen:
            continue
        value = rows_value(mat)
        if matrices_match(value, answer_rows, form):
            continue
        seen.add(text)
        out.append(Distractor(text, tag, value))
    return out
