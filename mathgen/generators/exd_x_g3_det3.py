"""Exam-tier source for the skill "3x3 determinant".

The practice tiers expand a 3x3 determinant, or take ``det(AB)`` of two explicit
matrices.  This tier uses the properties of the determinant so the learner must
choose a method:

* ``scaled``  -- ``det(k A^T B^-1)`` or ``det(k adj(A))``.  Pulling a scalar out of
  a 3x3 gives ``k^3`` (not ``k``), ``det(adj A) = det(A)^2`` for 3x3, and
  ``det(A^T) = det A``.
* ``unknown`` -- one entry ``x`` of a 3x3 matrix is unknown and ``det`` is stated;
  cofactor expansion along the row or column holding ``x`` gives a linear equation.

Matrices are drawn first and the stated value is derived from them, rejecting
draws where the result would not be a whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Matrix, Symbol, latex, solve

from ..core import check, check_equivalent, fmt_signed, generator, pick_distractors


def _mat_text(rows) -> str:
    return "[" + "; ".join(" ".join(str(v) for v in row) for row in rows) + "]"


def _rand_mat(rng: random.Random, hi: int = 3):
    return [[rng.randint(-hi, hi) for _ in range(3)] for _ in range(3)]


def _scaled(rng: random.Random) -> dict:
    for _ in range(2000):
        A, B = _rand_mat(rng), _rand_mat(rng)
        dA, dB = int(Matrix(A).det()), int(Matrix(B).det())
        if dA == 0 or dB == 0 or abs(dA) > 20 or abs(dB) > 20:
            continue
        k = rng.choice([2, 3])
        if rng.random() < 0.5:
            expr, tex = f"{k}A^T B^(-1)", rf"\det\left({k}A^{{T}}B^{{-1}}\right)"
            value = Fraction(k**3 * dA, dB)
            sym = (Matrix(A).T * k) * Matrix(B).inv()
            cands = [
                Fraction(k * dA, dB),  # scalar not cubed
                Fraction(k * k * dA, dB),  # scalar squared, as for a 2x2
                Fraction(k**3 * dA * dB),  # treats B^-1 as B
                Fraction(k**3 * (dA - dB)),  # subtracts instead of dividing
            ]
        else:
            expr, tex = f"{k} adj(A)", rf"\det\left({k}\,\operatorname{{adj}}(A)\right)"
            value = Fraction(k**3 * dA**2)
            sym = Matrix(A).adjugate() * k
            cands = [
                Fraction(k * dA**2),  # scalar not cubed
                Fraction(k * k * dA**2),  # scalar squared, as for a 2x2
                Fraction(k**3 * dA),  # treats det(adj A) as det A
                Fraction(k**3 * dA**3),  # uses det(adj A) = det(A)^3
            ]
        if value.denominator != 1 or abs(value) > 20000:
            continue
        check_equivalent(sym.det(), int(value))
        answer = int(value)
        ds = pick_distractors(str(answer), [str(int(c)) for c in cands if c.denominator == 1])
        if len(ds) >= 3:
            break
    else:  # pragma: no cover
        raise ValueError("could not draw a question")
    if "adj" in expr:
        text = f"A = {_mat_text(A)}. Find det({expr})."
        latex_ = rf"A = {latex(Matrix(A))}.\ \text{{Find }}{tex}\text{{.}}"
    else:
        text = f"A = {_mat_text(A)} and B = {_mat_text(B)}. Find det({expr})."
        latex_ = rf"A = {latex(Matrix(A))},\ B = {latex(Matrix(B))}.\ \text{{Find }}{tex}\text{{.}}"
    return dict(
        text=text,
        latex=latex_,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[f"det(A) = {dA}", f"result = {answer}"],
        distractors=ds[:3],
    )


def _unknown(rng: random.Random) -> dict:
    x = Symbol("x")
    for _ in range(2000):
        A = _rand_mat(rng, 4)
        pos = (rng.randrange(3), rng.randrange(3))
        xv = rng.choice([v for v in range(-5, 6) if v != 0])
        A[pos[0]][pos[1]] = xv
        full = Matrix(A)
        D = int(full.det())
        symm = Matrix(3, 3, lambda i, j: x if (i, j) == pos else A[i][j])
        cof = int(symm.cofactor(pos[0], pos[1]))
        if cof == 0 or D == 0 or abs(D) > 120:
            continue
        check(solve(symm.det() - D, x) == [xv], "x is determined")
        rest = D - cof * xv  # the part of det that does not involve x
        cands = [
            Fraction(D, cof),  # forgets the x-free part
            Fraction(D + rest, cof),  # adds the x-free part instead of subtracting it
            Fraction(D - rest, -cof),  # sign slip on the cofactor
            Fraction(D - rest, abs(cof)),  # drops the checkerboard sign of the cofactor
        ]
        ds = pick_distractors(str(xv), [str(int(c)) for c in cands if c.denominator == 1])
        if len(ds) >= 3:
            break
    else:  # pragma: no cover
        raise ValueError("could not draw a question")
    shown = [[("x" if (i, j) == pos else A[i][j]) for j in range(3)] for i in range(3)]
    text = f"The matrix A = {_mat_text(shown)} has determinant {D}. Find x."
    latex_ = rf"\text{{The matrix }}A = {latex(symm)}\text{{ has determinant {D}. Find }}x\text{{.}}"
    return dict(
        text=text,
        latex=latex_,
        answer=xv,
        answer_text=str(xv),
        answer_latex=str(xv),
        working=[
            f"expand along the row or column containing x: det = {fmt_signed(cof, 'x')} + ({rest}) = {D}",
            f"x = {xv}",
        ],
        distractors=ds[:3],
    )


@generator(
    "exd_x_g3_det3",
    level=13,
    presets={"exam": dict(tier="exam")},
)
def exd_x_g3_det3(rng: random.Random, tier: str = "exam") -> dict:
    """3x3 determinants through the properties of det, or an unknown entry from the stated det."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _scaled(rng) if rng.random() < 0.5 else _unknown(rng)
