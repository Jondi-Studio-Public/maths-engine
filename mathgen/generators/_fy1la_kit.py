"""Exact small-matrix helpers shared by the first-year linear algebra generators (`fy1la_*_build`).

Everything works on lists of rows of `Fraction`, so a question's numbers never pass through a float. This is the
hand route (Gauss-Jordan elimination, determinant by elimination); each generator re-derives its answer a second
way with SymPy before it returns.

* `unimodular` draws an integer matrix of determinant 1 or -1, so `E R` is an integer matrix with the row space of `R`;
* `rref`, `echelon`, `rank`, `nullspace`, `det`, `solve_unique` are the usual operations;
* `num_distractors`, `matrix_distractors` and `vector_distractors` turn `(candidate, misconception)` pairs into
  tagged `Distractor`s, dropping repeats and anything the real grader would accept as the answer;
* `number_part`, `matrix_part` and `expression_part` build the dict for one part of an exam chain.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

import sympy

from ..answers import Distractor, exact_value
from ..core import VerificationError, attempts
from ..grading.matrix import matrices_match, matrix_text, vector_text

Matrix = list[list[Fraction]]


def fm(rows) -> Matrix:
    """`rows` (lists of ints or Fractions) as a matrix of Fractions."""
    return [[Fraction(v) for v in row] for row in rows]


def identity(n: int) -> Matrix:
    return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]


def transpose(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def mmul(a: Matrix, b: Matrix) -> Matrix:
    cols = list(zip(*b))
    return [[sum((x * y for x, y in zip(row, col)), Fraction(0)) for col in cols] for row in a]


def apply(a: Matrix, v) -> list[Fraction]:
    return [sum((Fraction(x) * Fraction(y) for x, y in zip(row, v)), Fraction(0)) for row in a]


def column(a: Matrix, c: int) -> list[Fraction]:
    return [row[c] for row in a]


def with_columns(vectors) -> Matrix:
    """The matrix whose columns are `vectors`."""
    return transpose(fm(vectors))


def max_abs(a: Matrix) -> Fraction:
    return max(abs(v) for row in a for v in row)


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


def echelon(a: Matrix) -> Matrix:
    """Row-echelon form with leading ones, eliminating below each pivot only (the form a learner reaches when they
    stop before clearing the entries above the pivots)."""
    m = [row[:] for row in a]
    rows, cols = len(m), len(m[0])
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
        for i in range(r + 1, rows):
            if m[i][c] != 0:
                factor = m[i][c]
                m[i] = [v - factor * w for v, w in zip(m[i], m[r])]
        r += 1
    return m


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


def solve_unique(a: Matrix, b) -> list[Fraction] | None:
    """The one solution of `a x = b` for square `a` of full rank, else None."""
    n = len(a)
    aug = [row[:] + [Fraction(b[i])] for i, row in enumerate(a)]
    reduced, pivots = rref(aug)
    if pivots[:n] != list(range(n)):
        return None
    return [reduced[i][n] for i in range(n)]


def consistent(a: Matrix, b) -> bool:
    """Whether `a x = b` has a solution."""
    return rank(a) == rank([row[:] + [Fraction(b[i])] for i, row in enumerate(a)])


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


def sym(a) -> sympy.Matrix:
    """A matrix of ints or Fractions as a SymPy matrix, for the second route."""
    return sympy.Matrix([[sympy.Rational(Fraction(v).numerator, Fraction(v).denominator) for v in row] for row in a])


def unimodular(rng, n: int, bound: int = 2) -> Matrix:
    """An integer matrix of determinant 1 or -1 with every entry within `bound`, built from random row additions and a
    random row order, so a product `E R` is an integer matrix with the same row space as `R`."""
    for _ in attempts(500, "a small unimodular matrix"):
        m = identity(n)
        for _step in range(rng.randint(2, 4)):
            i, j = rng.sample(range(n), 2)
            k = rng.choice([-2, -1, 1, 2])
            m[i] = [v + k * w for v, w in zip(m[i], m[j])]
        rng.shuffle(m)
        if max_abs(m) <= bound and abs(det(m)) == 1 and m != identity(n):
            return m
    raise VerificationError("unreachable")  # pragma: no cover - attempts() raises first


# -- spelling and answers ---------------------------------------------------------------------------------------


def entry(value):
    """One matrix entry as the JSON the matrix grader reads: an int, or {"p": .., "q": ..}."""
    return exact_value(Fraction(value))


def rows_value(a) -> list[list]:
    return [[entry(v) for v in row] for row in a]


def column_value(vec) -> list[list]:
    """A vector as a one-column answer value."""
    return [[entry(v)] for v in vec]


def mat_str(a) -> str:
    """The stem and answer spelling of a matrix: [[1, 2], [3, 4]]."""
    return matrix_text(rows_value(a))


def vec_str(vec) -> str:
    """The spelling of a vector: (1, -2, 3)."""
    return vector_text([entry(v) for v in vec])


# -- distractors ------------------------------------------------------------------------------------------------


def num_distractors(answer: int, candidates, n: int = 3, lowest: int | None = None) -> list[Distractor] | None:
    """Tagged whole-number distractors from `(value, misconception)` pairs in priority order.

    A value is dropped when it is the answer, a repeat, or not a whole number (the answer has no decimal places, so
    neither may an option) or below `lowest` (a dimension cannot be negative). Returns None when fewer than two
    survive, so the caller redraws.
    """
    out, seen = [], {int(answer)}
    for value, tag in candidates:
        value = Fraction(value)
        if value.denominator != 1 or int(value) in seen or (lowest is not None and value < lowest):
            continue
        seen.add(int(value))
        out.append(Distractor(str(int(value)), tag, int(value)))
        if len(out) == n:
            break
    return out if len(out) >= 2 else None


def matrix_distractors(candidates, answer_rows, form=None, n: int = 3) -> list[Distractor] | None:
    """Tagged distractors from `(matrix, misconception)` pairs: refused by the matrix grader, no repeats."""
    out, seen = [], set()
    for mat, tag in candidates:
        if mat is None:
            continue
        text = mat_str(mat)
        value = rows_value(mat)
        if text in seen or matrices_match(value, answer_rows, form):
            continue
        seen.add(text)
        out.append(Distractor(text, tag, value))
        if len(out) == n:
            break
    return out if len(out) >= 2 else None


def vector_distractors(candidates, answer_vec, form=None, n: int = 3) -> list[Distractor] | None:
    """Tagged distractors from `(vector, misconception)` pairs. With a `scalar_multiple` form, vectors are compared
    in primitive form (a multiple of a wrong vector is the same wrong answer); the zero vector is never offered."""
    answer_value = column_value(answer_vec)
    out, seen = [], set()
    for vec, tag in candidates:
        if vec is None or all(v == 0 for v in vec):
            continue
        if form == "scalar_multiple":
            vec = primitive(vec)
        text = vec_str(vec)
        value = column_value(vec)
        if text in seen or matrices_match(value, answer_value, form):
            continue
        seen.add(text)
        out.append(Distractor(text, tag, value))
        if len(out) == n:
            break
    return out if len(out) >= 2 else None


def text_distractors(answer_text: str, candidates, n: int = 3) -> list[Distractor] | None:
    """Tagged distractors for a multiple-choice or list answer from `(text, misconception)` pairs."""
    out, seen = [], {answer_text}
    for text, tag in candidates:
        if text in seen:
            continue
        seen.add(text)
        out.append(Distractor(text, tag))
        if len(out) == n:
            break
    return out if len(out) >= 2 else None


# -- exam parts -------------------------------------------------------------------------------------------------


def number_part(prompt, value, distractors, *, marks, working, follow_through=None, sensible_range=None) -> dict:
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(value),
        answer_text=str(int(value)),
        answer_latex=str(int(value)),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def matrix_part(prompt, rows, distractors, *, marks, working, form=None, vector=False) -> dict:
    """A part whose answer is a matrix (`rows` a list of rows) or, with `vector`, a column vector (a flat list)."""
    value = column_value(rows) if vector else rows_value(rows)
    text = vec_str(rows) if vector else mat_str(rows)
    part = dict(
        prompt=prompt,
        answer_kind="matrix",
        answer=value,
        answer_text=text,
        answer_latex="plain:" + text,
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if form:
        part["answer_form"] = form
    return part


def expression_part(prompt, expr, text, distractors, *, marks, working) -> dict:
    return dict(
        prompt=prompt,
        answer_kind="expression",
        answer=str(expr),
        answer_text=text,
        answer_latex="plain:" + text,
        marks=marks,
        working=working,
        distractors=distractors,
    )


def expression_distractors(answer: str, candidates, prompt: str, n: int = 3) -> list[Distractor] | None:
    """Tagged expression distractors from `(text, misconception)` pairs, refused by the real expression grader."""
    from types import SimpleNamespace

    from ..grading import GRADERS

    question = SimpleNamespace(answer=answer, answer_text=answer, answer_form=None, text=prompt, parts=None)
    out, seen = [], {answer}
    for text, tag in candidates:
        if text in seen or GRADERS["expression"](question, text).correct:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, text))
        if len(out) == n:
            break
    return out if len(out) >= 2 else None


# -- stems ------------------------------------------------------------------------------------------------------


def linear_text(coefs, names) -> str:
    """The left side of an equation: coefficients are ints, or the string "k" for a coefficient of the constant k."""
    pieces = []
    for c, v in zip(coefs, names):
        if c == 0:
            continue
        if c == "k":
            sign, mag = "+", "k" + v
        else:
            sign, mag = ("+" if c > 0 else "-"), (str(abs(c)) if abs(c) != 1 else "") + v
        pieces.append((sign, mag))
    first_sign, first = pieces[0]
    return ("-" if first_sign == "-" else "") + first + "".join(f" {s} {m}" for s, m in pieces[1:])


def system_text(rows, rhs, names) -> str:
    """Equations separated by semicolons: "2x + y = 3; x - z = 0"."""
    return "; ".join(f"{linear_text(row, names)} = {value}" for row, value in zip(rows, rhs))


def mcq_part(prompt, answer, distractors, *, marks, working) -> dict:
    """A multiple-choice part: `answer` is the text of the right option, `distractors` the other options."""
    return dict(
        prompt=prompt,
        answer_kind="mcq",
        answer=answer,
        answer_text=answer,
        marks=marks,
        working=working,
        distractors=distractors,
    )


# -- parameters -------------------------------------------------------------------------------------------------


def quadratic_in_k(build) -> tuple[Fraction, Fraction, Fraction]:
    """(c2, c1, c0) with det(build(k)) = c2 k² + c1 k + c0, by interpolation at k = -1, 0, 1."""
    d_m, d_0, d_p = (det(build(v)) for v in (-1, 0, 1))
    return (d_p + d_m) / 2 - d_0, (d_p - d_m) / 2, d_0


def int_roots(c2, c1, c0) -> list[int] | None:
    """The two distinct integer roots of c2 k² + c1 k + c0 in increasing order, or None."""
    from math import isqrt

    disc = c1 * c1 - 4 * c2 * c0
    if c2 == 0 or disc <= 0 or disc.denominator != 1 or isqrt(int(disc)) ** 2 != disc:
        return None
    s = isqrt(int(disc))
    roots = [(-c1 + s) / (2 * c2), (-c1 - s) / (2 * c2)]
    if any(r.denominator != 1 for r in roots):
        return None
    return sorted(int(r) for r in roots)


def vector_with_k(entries) -> str:
    """A vector written for a stem, with "k" wherever an entry is the string "k": (1, k, -2)."""
    return "(" + ", ".join(str(v) for v in entries) + ")"


def matrix_from_rref(rng, target, cap: int = 9, tries: int = 150) -> Matrix | None:
    """`E target` for a unimodular `E`: an integer matrix whose reduced row echelon form is `target`, with a small
    spread of entries, no zero row and no two equal or opposite rows. Returns None when `tries` unimodular matrices give no
    such product (some targets have none), so the caller draws a different target."""
    base = fm(target)
    rows = len(base)
    for _ in range(tries):
        a = mmul(unimodular(rng, rows, 2), base)
        if max_abs(a) > cap or max_abs(a) < 2 or a == base:
            continue
        if any(all(v == 0 for v in row) for row in a):
            continue
        if sum(1 for row in a for v in row if v == 0) > 3:
            continue
        keys = {tuple(row) for row in a}
        if len(keys) < rows or any(tuple(-v for v in row) in keys for row in a):
            continue
        if any(rank([r1, r2]) < 2 for i, r1 in enumerate(a) for r2 in a[i + 1 :]):
            continue
        return a
    return None


def inverse(a: Matrix) -> Matrix:
    n = len(a)
    aug = [row[:] + identity(n)[i] for i, row in enumerate(a)]
    reduced, pivots = rref(aug)
    if pivots[:n] != list(range(n)):
        raise VerificationError("matrix is singular")
    return [row[n:] for row in reduced]


def list_distractors(answer: list, candidates, n: int = 3) -> list[Distractor] | None:
    """Tagged distractors for an unordered list answer from `(list, misconception)` pairs: texts are the sorted values
    joined by ", "; a candidate that is the same set of values as the answer, or a repeat, is dropped."""
    key = sorted(answer)
    out, seen = [], {tuple(key)}
    for values, tag in candidates:
        if any(Fraction(v).denominator != 1 for v in values):
            continue
        values = sorted(int(v) for v in values)
        if tuple(values) in seen:
            continue
        seen.add(tuple(values))
        out.append(Distractor(", ".join(str(v) for v in values), tag, values))
        if len(out) == n:
            break
    return out if len(out) >= 2 else None
