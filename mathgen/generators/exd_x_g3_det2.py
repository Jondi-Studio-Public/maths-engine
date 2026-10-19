"""Exam-tier source for the skill "2x2 determinant".

The practice tiers evaluate ``det A`` or ``det(AB)`` for explicit matrices.  This
tier asks for determinants that are best found by *choosing a method* from the
properties of the determinant rather than by multiplying matrices out:

* ``scaled`` -- ``det(k A^-1 B^n)`` or ``det(k A^T B^-1)``: pull out the scalar
  (``k^2`` for a 2x2), use ``det(A^-1) = 1/det A``, ``det(A^T) = det A`` and
  ``det(B^n) = det(B)^n``.
* ``unknown`` -- an entry ``x`` of ``A`` is unknown and ``det(AB)`` is stated;
  use ``det(AB) = det A det B`` to get a linear equation for ``x``.

Matrices are drawn first and the stated value is derived from them, rejecting
draws where the result would not be a whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Matrix, Symbol, latex, solve

from ..core import check, check_equivalent, generator, pick_distractors


def _mat_text(rows) -> str:
    return "[" + ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in rows) + "]"


def _rand_mat(rng: random.Random, hi: int = 5):
    return [[rng.randint(-hi, hi) for _ in range(2)] for _ in range(2)]


def _det(m) -> int:
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def _scaled(rng: random.Random) -> dict:
    for _ in range(500):
        A, B = _rand_mat(rng), _rand_mat(rng)
        dA, dB = _det(A), _det(B)
        if dA == 0 or dB == 0 or abs(dA) > 12 or abs(dB) > 12:
            continue
        k = rng.choice([2, 3, 4])
        if rng.random() < 0.5:
            n = rng.choice([2, 3])
            expr, tex = f"{k}A^(-1)B^{n}", rf"\det\left({k}A^{{-1}}B^{{{n}}}\right)"
            value = Fraction(k * k * dB**n, dA)
            sym = (Matrix(A).inv() * k) * Matrix(B) ** n
            cands = [
                Fraction(k * dB**n, dA),  # scalar not squared
                Fraction(k * k * dB**n * dA),  # treats A^-1 as A
                Fraction(k * k * dB, dA),  # ignores the power
                Fraction(k * k * (dB**n - dA)),  # subtracts instead of dividing
            ]
        else:
            expr, tex = f"{k}A^T B^(-1)", rf"\det\left({k}A^{{T}}B^{{-1}}\right)"
            value = Fraction(k * k * dA, dB)
            sym = (Matrix(A).T * k) * Matrix(B).inv()
            cands = [
                Fraction(k * dA, dB),  # scalar not squared
                Fraction(k * k * dA * dB),  # treats B^-1 as B
                Fraction(k * k * dB, dA),  # inverts the ratio
                Fraction(k * k * (dA - dB)),  # subtracts instead of dividing
            ]
        if value.denominator != 1 or abs(value) > 4000:
            continue
        check_equivalent(sym.det(), int(value))
        answer = int(value)
        ds = pick_distractors(str(answer), [str(int(c)) for c in cands if c.denominator == 1])
        if len(ds) >= 3:
            break
    else:  # pragma: no cover
        raise ValueError("could not draw a question")
    text = f"A = {_mat_text(A)} and B = {_mat_text(B)}. Find det({expr})."
    latex_ = rf"A = {latex(Matrix(A))},\ B = {latex(Matrix(B))}.\ \text{{Find }}{tex}\text{{.}}"
    return dict(
        text=text,
        latex=latex_,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[f"det(A) = {dA}, det(B) = {dB}", f"result = {answer}"],
        distractors=ds[:3],
    )


def _unknown(rng: random.Random) -> dict:
    x = Symbol("x")
    for _ in range(500):
        A, B = _rand_mat(rng), _rand_mat(rng)
        dB = _det(B)
        pos = rng.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
        xv = rng.choice([v for v in range(-6, 7) if v != 0])
        A[pos[0]][pos[1]] = xv
        dA = _det(A)
        cof = A[1 - pos[0]][1 - pos[1]] * (1 if pos[0] == pos[1] else -1)
        if dB == 0 or cof == 0 or dA == 0:
            continue
        D = dA * dB
        if abs(D) > 300:
            continue
        shown = [[("x" if (i, j) == pos else A[i][j]) for j in range(2)] for i in range(2)]
        xm = Matrix(2, 2, lambda i, j: x if (i, j) == pos else A[i][j])
        check(solve(xm.det() * dB - D, x) == [xv], "x is determined")
        rest = dA - cof * xv  # the part of det A that does not involve x
        cands = [
            Fraction(D - rest, cof),  # forgets to divide the stated value by det B
            Fraction(dA + rest, cof),  # adds the x-free part instead of subtracting it
            Fraction(dA - rest, -cof),  # sign slip on the cofactor
            Fraction(D * dB - rest, cof),  # multiplies by det B instead of dividing
        ]
        ds = pick_distractors(str(xv), [str(int(c)) for c in cands if c.denominator == 1])
        if len(ds) >= 3:
            break
    else:  # pragma: no cover
        raise ValueError("could not draw a question")
    sx = latex(Matrix(2, 2, lambda i, j: x if (i, j) == pos else A[i][j]))
    text = f"A = {_mat_text(shown)} and B = {_mat_text(B)}. Given that det(AB) = {D}, find x."
    latex_ = rf"A = {sx},\ B = {latex(Matrix(B))}.\ \text{{Given that }}\det(AB) = {D}\text{{, find }}x\text{{.}}"
    return dict(
        text=text,
        latex=latex_,
        answer=xv,
        answer_text=str(xv),
        answer_latex=str(xv),
        working=[f"det(B) = {dB}", f"det(A) = {D // dB}", f"x = {xv}"],
        distractors=ds[:3],
    )


@generator(
    "exd_x_g3_det2",
    level=13,
    presets={"exam": dict(tier="exam")},
)
def exd_x_g3_det2(rng: random.Random, tier: str = "exam") -> dict:
    """2x2 determinants through the properties of det, or an unknown entry from det(AB)."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _scaled(rng) if rng.random() < 0.5 else _unknown(rng)
