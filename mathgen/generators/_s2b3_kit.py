"""Exact helpers shared by the second-year inner-product-space generators (`s2b3_*`).

The questions are about vectors and matrices with rational entries (and, for QR, square roots of rationals), so the
hand route here is plain `Fraction` arithmetic; every generator then re-derives its answer with SymPy as a second
route. Nothing here draws random numbers.

* vector and matrix arithmetic on lists of `Fraction`: `dot`, `vadd`, `vsub`, `vscale`, `mat_vec`, `gram_schmidt`;
* answers: `vec_answer` and `mat_answer` build the `matrix`-kind answer fields (entries may be rationals or SymPy
  closed forms such as `sqrt(2)/2`), `fraction_answer` the `fraction`-kind ones;
* distractors: `matrix_distractors` turns `(vector or matrix, misconception)` pairs into tagged `Distractor`s the
  matrix grader rejects, with no repeats; `fraction_distractors` does the same for exact rational values.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import check, latex_answer, rational_text


def frac(x) -> Fraction:
    """`x` (an int, a Fraction or a SymPy Rational) as a Fraction."""
    if isinstance(x, (int, Fraction)):
        return Fraction(x)
    return Fraction(int(x.p), int(x.q))


def dot(u, v) -> Fraction:
    return sum((Fraction(a) * Fraction(b) for a, b in zip(u, v)), Fraction(0))


def vadd(u, v) -> list[Fraction]:
    return [Fraction(a) + Fraction(b) for a, b in zip(u, v)]


def vsub(u, v) -> list[Fraction]:
    return [Fraction(a) - Fraction(b) for a, b in zip(u, v)]


def vscale(c, u) -> list[Fraction]:
    return [Fraction(c) * Fraction(a) for a in u]


def mat_vec(a, x) -> list[Fraction]:
    return [dot(row, x) for row in a]


def gram_schmidt(vectors) -> list[list[Fraction]]:
    """The orthogonal (not normalised) vectors from `vectors`, taken in the order given."""
    out: list[list[Fraction]] = []
    for v in vectors:
        u = [Fraction(a) for a in v]
        for w in out:
            u = vsub(u, vscale(dot(v, w) / dot(w, w), w))
        out.append(u)
    return out


def to_sym(a) -> sympy.Matrix:
    """A list of rows (or a flat list, as a column) of Fractions as an exact SymPy matrix."""
    if a and not isinstance(a[0], (list, tuple)):
        return sympy.Matrix([sympy.Rational(v.numerator, v.denominator) for v in map(Fraction, a)])
    return sympy.Matrix([[sympy.Rational(v.numerator, v.denominator) for v in map(Fraction, row)] for row in a])


def sym_fractions(m: sympy.Matrix) -> list[list[Fraction]]:
    return [[frac(v) for v in m.row(i)] for i in range(m.rows)]


def sym_vector(m: sympy.Matrix) -> list[Fraction]:
    return [frac(v) for v in m]


def sizes_ok(values, bound_num: int = 40, bound_den: int = 12) -> bool:
    """Whether every entry of `values` (Fractions, nested or flat) is a small, readable rational."""
    flat = [v for row in values for v in row] if values and isinstance(values[0], (list, tuple)) else values
    return all(abs(v.numerator) <= bound_num and v.denominator <= bound_den for v in map(Fraction, flat))


# -- answers --------------------------------------------------------------------------------------------------


def entry(v):
    """One matrix entry as the JSON the matrix grader reads: an int, `{"p", "q"}`, or text for a closed form."""
    if isinstance(v, (int, Fraction)):
        return exact_value(Fraction(v))
    v = sympy.sympify(v)
    if v.is_Rational:
        return exact_value(Fraction(int(v.p), int(v.q)))
    return str(v)


def rows_json(rows) -> list[list]:
    return [[entry(v) for v in row] for row in rows]


def column_json(vec) -> list[list]:
    return [[entry(v)] for v in vec]


def sy(v) -> sympy.Expr:
    """`v` (an int, a Fraction or a SymPy expression) as an exact SymPy number."""
    if isinstance(v, (int, Fraction)):
        v = Fraction(v)
        return sympy.Rational(v.numerator, v.denominator)
    return sympy.sympify(v)


def etext(v) -> str:
    """One entry as the matrix grader spells it back: `3/2`, `-1`, `3√(2)/2`."""
    v = sy(v)
    if v.is_Rational:
        return str(v)
    return str(v).replace("sqrt", "√").replace("**", "^").replace("*", "")


def vec_text(vec) -> str:
    return "(" + ", ".join(etext(v) for v in vec) + ")"


def mat_text(rows) -> str:
    return "[" + ", ".join("[" + ", ".join(etext(v) for v in row) + "]" for row in rows) + "]"


def vec_answer(vec) -> dict:
    """The answer fields of a vector answer (`matrix` kind, graded entry by entry)."""
    text = vec_text(vec)
    return dict(answer_kind="matrix", answer=column_json(vec), answer_text=text, answer_latex="plain:" + text)


def mat_answer(rows) -> dict:
    text = mat_text(rows)
    return dict(answer_kind="matrix", answer=rows_json(rows), answer_text=text, answer_latex="plain:" + text)


def fraction_answer(value) -> dict:
    """The answer fields of an exact rational (`fraction` kind)."""
    value = sympy.Rational(value.numerator, value.denominator) if isinstance(value, Fraction) else sympy.Rational(value)
    text = rational_text(value)
    return dict(
        answer_kind="fraction",
        answer=exact_value(Fraction(int(value.p), int(value.q))),
        answer_text=text,
        answer_latex=latex_answer(text),
    )


def number_answer(value: int) -> dict:
    return dict(answer_kind="number", answer=int(value), answer_text=str(int(value)), answer_latex=str(int(value)))


def expression_answer(expr, *, decimals: int = 2) -> dict:
    """An exact closed form such as `sqrt(6)/3` (`expression` kind), with its decimal value as the numeric alternative."""
    text = str(expr)
    return dict(
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex_answer(text),
        answer_alt=round(float(expr), decimals),
    )


# -- distractors ----------------------------------------------------------------------------------------------


def _same_matrix(left, right) -> bool:
    """Entry-by-entry equality of two nested lists of exact numbers of the same shape."""
    if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right)):
        return False
    return all(sympy.expand(sy(x) - sy(y)) == 0 for a, b in zip(left, right) for x, y in zip(a, b))


def matrix_distractors(candidates, true, *, vector: bool = False, limit: int = 4) -> list[Distractor]:
    """Tagged distractors from `(vector or rows, misconception)` pairs, best first.

    `true` is the correct answer in the same shape. A candidate is kept only when the matrix grader would reject it
    against `true` (compared here entry by entry, as the grader does), no earlier candidate spells it the same way,
    and its entries are finite real numbers.
    """
    shape = column_json if vector else rows_json
    to_text = vec_text if vector else mat_text
    as_rows = (lambda v: [[x] for x in v]) if vector else (lambda m: m)
    expected = as_rows(true)
    out, seen = [], {to_text(true)}
    for cand, tag in candidates:
        if cand is None:
            continue
        flat = list(cand) if vector else [v for row in cand for v in row]
        if not flat or any(sy(v).is_real is not True or sy(v).is_finite is not True for v in flat):
            continue
        text = to_text(cand)
        if text in seen or _same_matrix(as_rows(cand), expected):
            continue
        seen.add(text)
        out.append(Distractor(text, tag, shape(cand)))
        if len(out) == limit:
            break
    return out


def fraction_distractors(true, candidates, *, limit: int = 3, positive: bool = False) -> list[Distractor]:
    """Tagged exact-rational distractors from `(value, misconception)` pairs; none equals `true`."""
    true = Fraction(true)
    out, seen = [], {true}
    for value, tag in candidates:
        if value is None:
            continue
        value = Fraction(value)
        if value in seen or (positive and value <= 0):
            continue
        seen.add(value)
        out.append(
            Distractor(rational_text(sympy.Rational(value.numerator, value.denominator)), tag, exact_value(value))
        )
        if len(out) == limit:
            break
    return out


def expression_distractors(true, candidates, *, limit: int = 3, gap: float = 0.03) -> list[Distractor]:
    """Tagged closed-form distractors from `(SymPy expression, misconception)` pairs.

    A candidate is dropped when it is within `gap` of `true`, or of an earlier candidate: the expression grader also
    accepts a decimal, so a wrong answer must sit well clear of the right one.
    """
    kept: list[float] = [float(true)]
    out = []
    for expr, tag in candidates:
        if expr is None:
            continue
        expr = sympy.radsimp(sympy.sympify(expr))
        if not expr.is_real or expr.is_finite is not True:
            continue
        value = float(expr)
        if any(abs(value - k) < gap for k in kept):
            continue
        kept.append(value)
        out.append(Distractor(str(expr), tag, str(expr)))
        if len(out) == limit:
            break
    return out


def need(distractors, count: int = 2):
    """Fail the draw loudly when fewer than `count` distractors survived."""
    check(len(distractors) >= count, f"only {len(distractors)} distractors")
    return distractors


def unreduced(vectors) -> bool:
    """Whether the vectors have a zero vector among them."""
    return any(all(v == 0 for v in vec) for vec in vectors)
