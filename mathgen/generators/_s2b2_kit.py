"""Shared helpers for the second-year linear algebra generators (`s2b2_*_build.py`).

Everything is exact: matrices are lists of rows of `Fraction`, built from `_ucv1_linalg`, and each generator
re-derives its key a second way with SymPy. This module adds only what several of those generators share:

* `sym` turns a `Fraction` matrix into a SymPy matrix for the independent check;
* `jordan_matrix` and `conjugate` build `P J P^-1` for a Jordan matrix J with a small unimodular P, so every key is
  known by construction;
* `permute` is a permutation similarity, which keeps a parameter k in a single entry of the matrix shown;
* `number_distractor`, `ints_distractor` and `matrix_candidate` turn candidate wrong values into tagged `Distractor`s;
* `combine` assembles the exam parts' working into the dict `make()` expects.
"""

from __future__ import annotations

import functools
import re
from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import VerificationError, attempts, rational_text
from ._ucv1_linalg import (  # noqa: F401  (re-exported for the s2b2 generators)
    apply,
    column,
    column_value,
    det,
    diagonal,
    entry,
    fm,
    identity,
    inverse,
    is_integer_matrix,
    madd,
    mat_str,
    matrix_distractors,
    max_abs,
    mmul,
    mpow,
    mscale,
    nullspace,
    primitive,
    rank,
    rows_value,
    transpose,
    unimodular,
    vec_str,
    vector_distractors,
    with_columns,
)


def sym(m) -> sympy.Matrix:
    """A matrix of `Fraction`s (or ints) as an exact SymPy matrix."""
    return sympy.Matrix([[sympy.Rational(Fraction(v).numerator, Fraction(v).denominator) for v in row] for row in m])


def jordan_matrix(blocks) -> list[list[Fraction]]:
    """The Jordan matrix with the given `(eigenvalue, size)` blocks in order, 1s on the superdiagonal."""
    n = sum(size for _lam, size in blocks)
    out = [[Fraction(0)] * n for _ in range(n)]
    pos = 0
    for lam, size in blocks:
        for i in range(size):
            out[pos + i][pos + i] = Fraction(lam)
            if i + 1 < size:
                out[pos + i][pos + i + 1] = Fraction(1)
        pos += size
    return out


def conjugate(rng, j, bound=2, cap=14, dense=True):
    """`(A, P)` with A = P J P^-1 an integer matrix, P unimodular with entries within `bound`, A within `cap`.

    With `dense`, A has at most one zero entry, so it is not read off as triangular or block diagonal.
    """
    n = len(j)
    for _ in attempts(2000, "an integer matrix similar to the Jordan matrix"):
        p = unimodular(rng, n, bound)
        a = mmul(mmul(p, j), inverse(p))
        if not is_integer_matrix(a) or max_abs(a) > cap:
            continue
        if dense and sum(1 for row in a for v in row if v == 0) > 1:
            continue
        return a, p
    raise VerificationError("unreachable")  # pragma: no cover


def permute(m, order):
    """The permutation similarity `m'[i][j] = m[order[i]][order[j]]`: the same matrix with its rows and columns
    renumbered, so an entry given by a letter stays a single entry."""
    return [[m[order[i]][order[j]] for j in range(len(m))] for i in range(len(m))]


def number_text(value) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else rational_text(value)


def number_distractor(value, tag) -> Distractor:
    """A wrong number (int or `Fraction`) as a tagged option that the `number` or `fraction` grader reads."""
    value = Fraction(value)
    return Distractor(number_text(value), tag, exact_value(value))


def ints_distractor(values, tag) -> Distractor:
    """A wrong ordered list of integers, spelled "(2, 1)"."""
    values = [int(v) for v in values]
    return Distractor("(" + ", ".join(map(str, values)) + ")", tag, values)


def ints_text(values) -> str:
    return "(" + ", ".join(str(int(v)) for v in values) + ")"


def expression_distractor(expr, tag, text=None) -> Distractor:
    """A wrong SymPy expression as a tagged option; `text` is its spelling if it is not `str(expr)`."""
    shown = text if text is not None else str(expr)
    return Distractor(shown, tag, shown)


# -- worked-step wording ------------------------------------------------------------------------------------------

_SIGNED_AFTER_OPERATOR = re.compile(r"([+\u2212-]) +-\s*(?=\d)")


def tidy_line(text: str) -> str:
    """A worked step with no Python notation: "- -3" read as "+ 3", "+ -3" as "- 3", `**` as `^`, `*` as a dot."""
    text = _SIGNED_AFTER_OPERATOR.sub(lambda m: "+ " if m.group(1) in "-\u2212" else "- ", text)
    return text.replace("**", "^").replace("*", "\u00b7")


def tidy_question(question: dict) -> dict:
    """`question` with every worked step (and every part's worked steps) passed through `tidy_line`."""
    if "working" in question:
        question["working"] = [tidy_line(w) for w in question["working"]]
    for part in question.get("parts", ()):
        if "working" in part:
            part["working"] = [tidy_line(w) for w in part["working"]]
    return question


def tidied(build):
    """Decorator for a generator function: its question's worked steps are passed through `tidy_question`.

    A rare draw that cannot be verified or given enough distractors is redrawn from the same random stream (at most
    six times), so a seed still gives one deterministic question."""

    @functools.wraps(build)
    def inner(rng, *args, **kwargs):
        failure = None
        for _ in range(6):
            try:
                return tidy_question(build(rng, *args, **kwargs))
            except VerificationError as exc:
                failure = exc
        raise failure

    return inner
