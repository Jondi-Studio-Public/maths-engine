"""Shared helpers for the second-year linear algebra generators (`s2b1_*`).

Vectors of a vector space are always handled as lists of `Fraction` coordinates in the standard basis (a polynomial
as its coefficients from the constant term up, a 2×2 matrix as its entries row by row), so every question is solved
and checked with exact arithmetic. This module adds what the first-year helpers (`_ucv1_linalg`) lack:

* `poly_text`, `power_text`: polynomials and monomials written the way a student writes them;
* `sym`: a `Fraction` matrix as a SymPy matrix, for the independent re-derivation each generator performs;
* `vec_answer`, `mat_answer`: the answer attributes of a vector or matrix answer;
* `vec_distractors`, `mat_distractors`: tagged wrong options from `(value, misconception)` pairs, dropping any the
  matrix grader would accept, any repeat, and any that spell the same as an earlier option;
* `plain_stem`: a decorator that ships a stem as plain text, for stems with set braces or `[v]_B` coordinates that the
  automatic LaTeX conversion cannot express;
* `det_root_distractors`: the wrong roots a learner reaches by slipping a sign in the Leibniz expansion of a 3×3
  determinant that is linear in a parameter.
"""

from __future__ import annotations

from fractions import Fraction
from functools import wraps
from itertools import permutations

import sympy
from sympy.combinatorics import Permutation

from ..answers import Distractor
from ..core import attempts, join_terms
from ..grading.matrix import matrices_match
from ._ucv1_linalg import column_value, det, fm, mat_str, rows_value, vec_str


def plain_stem(make):
    """Wrap a generator so that its question stem is shipped as plain text (`latex` = ``plain:`` + `text`).

    The automatic conversion of a stem to LaTeX cannot express set braces `{(1, 2), (3, 4)}` or coordinate vectors
    such as `[v]_B` inside running prose, so these stems are given to the renderer as plain text instead.
    """

    @wraps(make)
    def wrapper(*args, **kwargs):
        question = make(*args, **kwargs)
        question.setdefault("latex", "plain:" + question["text"])
        return question

    return wrapper


_SUPERSCRIPT = {2: "²", 3: "³", 4: "⁴", 5: "⁵"}


def power_text(var: str, k: int) -> str:
    """`x`^k as text: "" for k = 0, "x" for 1, "x²" for 2."""
    if k == 0:
        return ""
    return var if k == 1 else var + _SUPERSCRIPT[k]


def poly_text(coeffs, var: str = "x") -> str:
    """The polynomial with ascending `coeffs`, written in descending powers: [2, -1, 3] -> "3x² - x + 2"."""
    return join_terms([(c, power_text(var, k)) for k, c in reversed(list(enumerate(coeffs)))])


def sym(m) -> sympy.Matrix:
    """A list of rows of Fractions (or ints) as an exact SymPy matrix."""
    return sympy.Matrix([[sympy.Rational(Fraction(v).numerator, Fraction(v).denominator) for v in row] for row in m])


def sym_value(value) -> Fraction:
    """A SymPy rational (or int) as a Fraction."""
    value = sympy.Rational(value)
    return Fraction(int(value.p), int(value.q))


def vec_answer(vec) -> dict:
    """The answer attributes of a coordinate vector, answered as a column."""
    text = vec_str(vec)
    return dict(answer_kind="matrix", answer=column_value(vec), answer_text=text, answer_latex="plain:" + text)


def mat_answer(mat) -> dict:
    """The answer attributes of a matrix."""
    text = mat_str(mat)
    return dict(answer_kind="matrix", answer=rows_value(mat), answer_text=text, answer_latex="plain:" + text)


def vec_distractors(candidates, answer_vec) -> list[Distractor]:
    """Tagged distractors from `(vector, misconception)` pairs, best first.

    A candidate is kept only if the matrix grader rejects it against `answer_vec` (compared exactly, not up to a
    multiple) and no earlier candidate spells it the same way. The zero vector is kept: for coordinates it is a real
    wrong answer.
    """
    answer_value = column_value(answer_vec)
    out, seen = [], set()
    for vec, tag in candidates:
        if vec is None or len(vec) != len(answer_vec):
            continue
        vec = [Fraction(v) for v in vec]
        text = vec_str(vec)
        value = column_value(vec)
        if text in seen or matrices_match(value, answer_value, None):
            continue
        seen.add(text)
        out.append(Distractor(text, tag, value))
    return out


def mat_distractors(candidates, answer_mat) -> list[Distractor]:
    """Tagged distractors from `(matrix, misconception)` pairs, compared exactly with `answer_mat`."""
    answer_value = rows_value(answer_mat)
    out, seen = [], set()
    for mat, tag in candidates:
        if mat is None or len(mat) != len(answer_mat) or len(mat[0]) != len(answer_mat[0]):
            continue
        text = mat_str(mat)
        value = rows_value(mat)
        if text in seen or matrices_match(value, answer_value, None):
            continue
        seen.add(text)
        out.append(Distractor(text, tag, value))
    return out


def int_distractors(candidates, answer) -> list[Distractor]:
    """Tagged whole-number distractors from `(value, misconception)` pairs. Values equal to the answer, repeats and
    anything that is not a whole number are dropped, so every option is spelled like the answer."""
    out, seen = [], {str(answer)}
    for value, tag in candidates:
        if value is None:
            continue
        value = Fraction(value)
        if value.denominator != 1 or value == Fraction(answer) or str(value.numerator) in seen:
            continue
        seen.add(str(value.numerator))
        out.append(Distractor(str(value.numerator), tag, value.numerator))
    return out


def leibniz_terms(entries) -> list:
    """The six signed products of a 3×3 determinant (SymPy expressions), in permutation order."""
    terms = []
    for perm in permutations(range(3)):
        sign = Permutation(list(perm)).signature()
        product = sympy.Integer(sign)
        for row, col in enumerate(perm):
            product = product * entries[row][col]
        terms.append(product)
    return terms


def det_root_distractors(entries, symbol, true_root) -> list[Distractor]:
    """Wrong roots of det = 0 for a 3×3 determinant of SymPy `entries` that is linear in `symbol`.

    One sign slip in the Leibniz expansion (`det-sarrus-sign-slip`) or one product left out (`det-sarrus-term-dropped`)
    each give a different linear equation; its root is offered when it exists and is not `true_root`.
    """
    terms = leibniz_terms(entries)
    total = sum(terms)
    order = [n for n, t in enumerate(terms) if t.has(symbol)] + [n for n, t in enumerate(terms) if not t.has(symbol)]
    out: list[tuple[Fraction, str]] = []
    for tag, build in (
        ("det-sarrus-sign-slip", lambda n: total - 2 * terms[n]),
        ("det-sarrus-term-dropped", lambda n: total - terms[n]),
    ):
        for n in order:
            expr = sympy.expand(build(n))
            poly = sympy.Poly(expr, symbol)
            if poly.degree() != 1:
                continue
            root = sym_value(-poly.coeff_monomial(1) / poly.coeff_monomial(symbol))
            if root != Fraction(true_root):
                out.append((root, tag))
    return out


def draw_cols(rng, n, lo, hi, dmin, dmax):
    """`n` integer vectors of length `n` (entries in [lo, hi], none zero) whose matrix has |det| in [dmin, dmax]."""
    for _ in attempts(3000, f"a basis of {n} vectors with determinant in [{dmin}, {dmax}]"):
        cols = [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)]
        if any(all(v == 0 for v in c) for c in cols):
            continue
        if dmin <= abs(det(fm(cols))) <= dmax:
            return cols
    raise AssertionError  # pragma: no cover


def vector_text(vec) -> str:
    """An integer vector spelled (1, -2, 3)."""
    return "(" + ", ".join(str(int(v)) for v in vec) + ")"


X = sympy.symbols("x")


def poly_expr(vec):
    """The polynomial with ascending coefficients `vec` as a SymPy expression in `X`."""
    return sum(sympy.Rational(Fraction(c).numerator, Fraction(c).denominator) * X**k for k, c in enumerate(vec))


def poly_coeffs(expr, degree: int) -> list[Fraction]:
    """The ascending coefficients (as Fractions) of `expr` up to `degree`."""
    poly = sympy.Poly(sympy.expand(expr), X)
    return [sym_value(poly.coeff_monomial(X**d)) for d in range(degree + 1)]


def std_matrix(apply, n_in: int, out_degree: int):
    """The matrix of a map on polynomials in the monomial bases: column j holds the coefficients of apply(x^j)."""
    cols = [poly_coeffs(apply(X**j), out_degree) for j in range(n_in)]
    return [list(row) for row in zip(*cols)]
