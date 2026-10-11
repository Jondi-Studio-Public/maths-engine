"""Exam-tier source for "solve Ax = b" (answer: the pair x, y).

The practice tiers hand the learner one matrix equation ``A x = b`` (or a
purchase scenario that translates straight into one). The exam tier makes
the learner first decide how to reduce the problem to a single solvable
system. Three shapes, each built backwards from a chosen integer solution so
the key is exact:

* ``product``  -- ``(A B) x = b``. The matrices are shown separately, so
  the learner must either multiply them first or peel the equation apart in
  the right order (``B x = A^-1 b``, then ``x = B^-1 (A^-1 b)``).
* ``rhs_product`` -- ``A x = B c``. The right-hand side is itself a matrix
  times a vector, which has to be evaluated before the system can be solved.
* ``both_sides`` -- ``A x = x + b``. The unknown appears on both sides, so
  the learner must rearrange to ``(A - I) x = b``; solving ``A x = b``
  directly is the natural wrong move.

Every distractor is the exact answer of a named wrong method, and each is
dropped if the pair grader would accept it (tolerance included).
"""

from __future__ import annotations

import random

from sympy import Matrix, eye, latex

from ..core import GRADER_TOLERANCE, check, generator, pick_distractors, rational_text, xy_solution

_RANGE = 3


def _mat_text(M):
    return "[" + "; ".join(" ".join(str(M[i, j]) for j in range(M.cols)) for i in range(M.rows)) + "]"


def _invertible(rng: random.Random, *, lo: int = -_RANGE, hi: int = _RANGE) -> Matrix:
    while True:
        A = Matrix(2, 2, [rng.randint(lo, hi) for _ in range(4)])
        if A.det() != 0:
            return A


def _solution(rng: random.Random) -> tuple[int, int]:
    """A small solution pair other than (0, 0), where every wrong method would also return (0, 0)."""
    while True:
        x0, y0 = rng.randint(-5, 5), rng.randint(-5, 5)
        if (x0, y0) != (0, 0):
            return x0, y0


def _pt(v) -> str:
    return f"x = {rational_text(v[0])}, y = {rational_text(v[1])}"


def _finish(text: str, latex_text: str, x0: int, y0: int, wrong: list, working: list[str]) -> dict:
    solved = xy_solution(x0, y0)
    key = Matrix([x0, y0])
    cands = []
    for w in wrong:
        if w is None or w == key:
            continue
        cands.append(_pt(w))
    solved["distractors"] = pick_distractors(solved["answer_text"], cands, tolerance=GRADER_TOLERANCE)
    return dict(text=text, latex=latex_text, working=working, **solved)


def _product(rng: random.Random) -> dict:
    x0, y0 = _solution(rng)
    while True:
        A, B = _invertible(rng), _invertible(rng)
        v = Matrix([x0, y0])
        b = A * B * v
        wrong_order = (B * A).solve(b)  # multiplied the factors the wrong way round
        only_a = A.solve(b)
        only_b = B.solve(b)
        if len({tuple(w) for w in (wrong_order, only_a, only_b, v)}) == 4:
            break
    check(B.solve(A.solve(b)) == v)
    check((A * B).solve(b) == v)
    text = f"Solve (A B) x = b where A = {_mat_text(A)}, B = {_mat_text(B)} and b = ({b[0]}, {b[1]})"
    ltx = rf"{latex(A)}{latex(B)} \mathbf{{x}} = {latex(b)}"
    working = [
        f"B x = A^-1 b = ({A.solve(b)[0]}, {A.solve(b)[1]}).",
        f"x = B^-1 (A^-1 b) = ({x0}, {y0}).",
    ]
    return _finish(text, ltx, x0, y0, [wrong_order, only_a, only_b, Matrix([y0, x0])], working)


def _rhs_product(rng: random.Random) -> dict:
    # Draw c first, then keep only draws whose solution A^-1 (B c) is a small integer pair.
    while True:
        A, B = _invertible(rng), _invertible(rng)
        c = Matrix([rng.randint(-4, 4), rng.randint(-4, 4)])
        v = A.solve(B * c)
        if all(e.q == 1 for e in v) and any(v) and all(abs(e) <= 12 for e in v):
            break
    x0, y0 = int(v[0]), int(v[1])
    rhs = B * c
    check(A.solve(rhs) == v)
    wrong_t = A.solve(c)  # forgot to multiply by B
    wrong_b = A.solve(B.T * c)  # transposed B when evaluating the right-hand side
    text = f"Solve A x = B c where A = {_mat_text(A)}, B = {_mat_text(B)} and c = ({c[0]}, {c[1]})"
    ltx = rf"{latex(A)} \mathbf{{x}} = {latex(B)}{latex(c)}"
    working = [f"B c = ({rhs[0]}, {rhs[1]}).", f"x = A^-1 (B c) = ({x0}, {y0})."]
    return _finish(text, ltx, x0, y0, [wrong_t, wrong_b, A.T.solve(rhs), Matrix([y0, x0])], working)


def _both_sides(rng: random.Random) -> dict:
    x0, y0 = _solution(rng)
    while True:
        A = _invertible(rng, lo=-4, hi=4)
        if (A - eye(2)).det() != 0 and (A + eye(2)).det() != 0:
            break
    v = Matrix([x0, y0])
    b = (A - eye(2)) * v
    check(A * v == v + b)
    ignore_x = A.solve(b)  # treated the equation as A x = b
    plus = (A + eye(2)).solve(b)  # moved x across with the wrong sign
    neg_b = (A - eye(2)).solve(-b)
    text = f"Find the vector x = (x, y) such that A x = x + b, where A = {_mat_text(A)} and b = ({b[0]}, {b[1]})"
    ltx = rf"{latex(A)} \mathbf{{x}} = \mathbf{{x}} + {latex(b)}"
    working = ["Rearrange to (A - I) x = b.", f"x = (A - I)^-1 b = ({x0}, {y0})."]
    return _finish(text, ltx, x0, y0, [ignore_x, plus, neg_b, Matrix([y0, x0])], working)


@generator(
    "exd_x_g4_matrix_solve_system",
    level=13,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_g4_matrix_solve_system(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier matrix system; see the module docstring for the three shapes."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return rng.choice((_product, _rhs_product, _both_sides))(rng)
