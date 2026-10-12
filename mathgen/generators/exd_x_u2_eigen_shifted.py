"""Exam tier for 2x2 eigenvalues: a full (not triangular) matrix, scaled and shifted.

The practice tiers read eigenvalues from a triangular matrix through its characteristic equation. Here the
matrix A is full, so the characteristic polynomial has to be built from the trace and the determinant, and the
question asks for the eigenvalues of M = sA + tI. A learner can form M and solve its characteristic equation
directly, or find the eigenvalues of A and transform them (each becomes s lambda + t); either route is
acceptable, and both are checked here.

## Construction: draw the eigenvalues first

Pick distinct integer eigenvalues p and q and an integer a that is neither. The matrix [[a, b], [c, d]] with
d = p + q - a has trace p + q, and has determinant p q exactly when b c = -(a - p)(a - q), so b and c are a
factorisation of that (non-zero) product. The entries then have the intended eigenvalues by construction, and
SymPy confirms it.
"""

from __future__ import annotations

import random

import sympy

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, pick_distractors

_SCALES = (2, 3, 4, -1, -2, -3)
_SHIFTS = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)


def _list_text(values: list[int]) -> str:
    return ", ".join(map(str, sorted(values)))


def _tagged(key: list[int], candidates: list[tuple[str, list[int]]]) -> list:
    out = []
    for ident, values in candidates:
        out.append(Distractor(_list_text(values), ident, sorted(values)))
    return pick_distractors(_list_text(key), out, 3, unordered=True)


def _signed(coefficient: int, name: str) -> str:
    return f"{'-' if coefficient < 0 else ''}{abs(coefficient) if abs(coefficient) != 1 else ''}{name}"


_SHAPES = (
    "Find the eigenvalues of M = {scale} {shift}, where A = {matrix}.",
    "The matrix A is {matrix}. Find the eigenvalues of M = {scale} {shift}.",
    "Let A = {matrix}. Determine both eigenvalues of the matrix M = {scale} {shift}.",
)


@generator("exd_x_u2_eigen_shifted", level=13, presets={"exam": dict(tier="exam")})
def exd_x_u2_eigen_shifted(rng: random.Random, tier: str = "exam") -> dict:
    """Eigenvalues of sA + tI for a full 2x2 matrix A. See the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a full 2x2 matrix with the intended eigenvalues"):
        p, q = sorted(rng.sample(range(-6, 7), 2))
        a = rng.randint(-6, 6)
        if a in (p, q):
            continue
        product = -(a - p) * (a - q)
        b = rng.choice([x for x in range(-9, 10) if x and product % x == 0])
        c = product // b
        if abs(c) > 12 or c == 0:
            continue
        d = p + q - a
        s, t = rng.choice(_SCALES), rng.choice(_SHIFTS)

        matrix = sympy.Matrix([[a, b], [c, d]])
        m_matrix = s * matrix + t * sympy.eye(2)
        # Two routes: eigenvalues of M directly, and the eigenvalues of A transformed.
        direct = sorted(int(v) for v in m_matrix.eigenvals())
        check(sorted(int(v) for v in matrix.eigenvals()) == [p, q], "eigenvalues of A are not the drawn ones")
        transformed = sorted([s * p + t, s * q + t])
        check(direct == transformed, "eigenvalues of sA + tI disagree with the transformed eigenvalues of A")
        check(len(set(direct)) == 2, "eigenvalues of M must be distinct")

        diagonal = [s * a + t, s * d + t]
        distractors = _tagged(
            direct,
            [
                ("ignored-shift-in-eigenvalues", [s * p, s * q]),
                ("ignored-scaling-in-eigenvalues", [p + t, q + t]),
                ("diagonal-read-as-eigenvalues", diagonal),
            ],
        )
        if len(distractors) < 2:
            continue

        a_text = f"[[{a}, {b}], [{c}, {d}]]"
        scale_text = _signed(s, "A")
        shift_text = f"{'+' if t > 0 else '-'} {abs(t) if abs(t) != 1 else ''}I"
        stem = rng.choice(_SHAPES).format(matrix=a_text, scale=scale_text, shift=shift_text)
        stem = stem.replace("  ", " ")
        answer_text = _list_text(direct)
        trace, det = p + q, p * q
        return dict(
            text=stem,
            latex=stem,
            answer=direct,
            answer_kind="list",
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"For A: trace = {a} + {d} = {trace} and determinant = {a * d - b * c}, so "
                f"λ² - {trace}λ + {det} = 0 gives λ = {p} and λ = {q}.",
                f"The eigenvalues of {scale_text} {shift_text} are {s}λ + {t}: {answer_text}.",
            ],
            distractors=distractors,
        )
