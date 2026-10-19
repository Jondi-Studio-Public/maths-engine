"""Exam question on ridge regression with two coefficients: the normal-equation matrix, the solution, a prediction.

The stem gives a small design matrix A (m rows, two columns), responses b and a penalty lambda, and says the ridge
solution solves (A^T A + lambda I) x = A^T b. Four parts:

(a) det(A^T A + lambda I)                                 number (whole), 2 marks
(b) x1                                                    fraction, 2 marks, follow_through (m22 c1 - m12 c2) / part_a
(c) x2                                                    fraction, 2 marks, follow_through (m11 c2 - m12 c1) / part_a
(d) the prediction u1 x1 + u2 x2 for a new input (u1, u2) fraction, 1 mark,  follow_through u1 part_b + u2 part_c

M = A^T A + lambda I and c = A^T b are computed here from A and b; the follow-through expressions carry their
entries as numbers, so a wrong determinant in (a) is carried into (b) and (c), and wrong coefficients into (d).
Every key is also re-derived by solving the 2 x 2 system with SymPy.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, paren_negative as pn, phrase
from .exd_x_r2h_support import build_part


def _inv_text(value: Fraction) -> str:
    return str(value)


_STEM = (
    "A model y = x1 u1 + x2 u2 is fitted by ridge regression to {m} observations. The design matrix A has rows {rows}, "
    "the observed responses are b = [{b}], and the penalty is lambda = {lam}. The coefficients x = (x1, x2) solve "
    "the regularised normal equations (A^T A + lambda I)x = A^T b. Let M = A^T A + lambda I.",
    "Ridge regression with penalty lambda = {lam} is used to fit {m} observations. The matrix A has rows {rows} and "
    "the response vector is b = [{b}]. The fitted coefficients x = (x1, x2) satisfy (A^T A + lambda I)x = A^T b. "
    "Write M = A^T A + lambda I.",
    "Two inputs are used to predict a response from {m} observations. The design matrix A has rows {rows}, with "
    "responses b = [{b}]. With ridge penalty lambda = {lam}, the coefficients x = (x1, x2) solve "
    "(A^T A + lambda I)x = A^T b; call the matrix on the left M.",
)
_PROMPT_A = (
    "Find the determinant of M.",
    "Calculate det(M).",
)
_PROMPT_B = (
    "Hence find x1, as a fraction in simplest form.",
    "Use det(M) to find x1, as a fraction in simplest form.",
)
_PROMPT_C = (
    "Hence find x2, as a fraction in simplest form.",
    "Use det(M) to find x2, as a fraction in simplest form.",
)
_PROMPT_D = (
    "Using your x1 and x2, predict the response u1 x1 + u2 x2 for a new observation with inputs (u1, u2) = ({u1}, "
    "{u2}). Give your answer as a fraction in simplest form.",
    "A new observation has inputs (u1, u2) = ({u1}, {u2}). Use your coefficients to predict its response, as a "
    "fraction in simplest form.",
)


def _solve(g11, g12, g22, c1, c2, lam):
    """x1, x2 and det for (G + lam I) x = c, by Cramer's rule."""
    det = (g11 + lam) * (g22 + lam) - g12 * g12
    x1 = Fraction((g22 + lam) * c1 - g12 * c2, det)
    x2 = Fraction((g11 + lam) * c2 - g12 * c1, det)
    return det, x1, x2


def _wrong_answers(g11, g12, g22, c1, c2, lam, u1, u2, det, x1, x2):
    """The tagged wrong values for the determinant, x1, x2 and the prediction."""
    det_plain = g11 * g22 - g12 * g12
    x1_plain = x2_plain = None
    if det_plain != 0:
        _, x1_plain, x2_plain = _solve(g11, g12, g22, c1, c2, 0)
    det_once, x1_once, x2_once = _solve_once(g11, g12, g22, c1, c2, lam)
    x1_sign = Fraction((g22 + lam) * c1 + g12 * c2, det)
    x2_sign = Fraction((g11 + lam) * c2 + g12 * c1, det)
    x1_rhs = Fraction((g22 + lam) * c2 - g12 * c1, det)
    x2_rhs = Fraction((g11 + lam) * c1 - g12 * c2, det)
    pred_swapped = u1 * x2 + u2 * x1
    pred_plain = u1 * x1_plain + u2 * x2_plain if x1_plain is not None else None

    det_wrong = [
        (det_plain, "ridge-penalty-omitted"),
        (det_once, "ridge-penalty-added-once"),
        ((g11 + lam) * (g22 + lam), "determinant-formula-slip"),
        ((g11 + lam) * (g22 + lam) + g12 * g12, "determinant-formula-slip"),
    ]
    x1_wrong = [
        (x1_plain, "ridge-penalty-omitted"),
        (x1_once, "ridge-penalty-added-once"),
        (x1_sign, "inverse-off-diagonal-sign-error"),
        (x2, "components-swapped"),
        (x1_rhs, "right-hand-side-swapped"),
    ]
    x2_wrong = [
        (x2_plain, "ridge-penalty-omitted"),
        (x2_once, "ridge-penalty-added-once"),
        (x2_sign, "inverse-off-diagonal-sign-error"),
        (x1, "components-swapped"),
        (x2_rhs, "right-hand-side-swapped"),
    ]
    pred_wrong = [
        (pred_swapped, "components-swapped"),
        (pred_plain, "ridge-penalty-omitted"),
        (u1 * x1_sign + u2 * x2_sign, "inverse-off-diagonal-sign-error"),
        (x1 + x2, "prediction-ignores-input"),
        (u1 * x1_rhs + u2 * x2_rhs, "right-hand-side-swapped"),
    ]

    return det_wrong, x1_wrong, x2_wrong, pred_wrong


def _enough(key, wrong) -> bool:
    """Whether `wrong` holds at least three distinct values that differ from `key`."""
    return len({v for v, _ in wrong if v is not None and v != key}) >= 3


def _build(rng: random.Random) -> dict:
    for _ in attempts(1000, "a ridge problem with non-zero answers"):
        m = rng.choice((2, 3, 3, 4))
        rows = [(rng.randint(-4, 4), rng.randint(-4, 4)) for _ in range(m)]
        b = [rng.randint(-5, 5) for _ in range(m)]
        lam = rng.randint(1, 5)
        u1, u2 = rng.randint(1, 5), rng.randint(1, 5)
        if u1 == u2:
            continue
        g11 = sum(r[0] * r[0] for r in rows)
        g12 = sum(r[0] * r[1] for r in rows)
        g22 = sum(r[1] * r[1] for r in rows)
        c1 = sum(r[0] * bi for r, bi in zip(rows, b))
        c2 = sum(r[1] * bi for r, bi in zip(rows, b))
        det, x1, x2 = _solve(g11, g12, g22, c1, c2, lam)
        pred = u1 * x1 + u2 * x2
        if g11 == 0 or g22 == 0 or g12 == 0 or x1 == 0 or x2 == 0 or pred == 0 or (c1, c2) == (0, 0):
            continue
        if len({tuple(r) for r in rows}) < 2:
            continue
        det_wrong, x1_wrong, x2_wrong, pred_wrong = _wrong_answers(g11, g12, g22, c1, c2, lam, u1, u2, det, x1, x2)
        if all(_enough(k, w) for k, w in ((det, det_wrong), (x1, x1_wrong), (x2, x2_wrong), (pred, pred_wrong))):
            break

    # Independent route: SymPy solves the matrix equation.
    a_mat = sympy.Matrix(rows)
    system = a_mat.T * a_mat + lam * sympy.eye(2)
    solution = system.LUsolve(a_mat.T * sympy.Matrix(b))
    check(system.det() == det, "determinant disagrees with SymPy")
    check(
        Fraction(int(solution[0].p), int(solution[0].q)) == x1
        and Fraction(int(solution[1].p), int(solution[1].q)) == x2,
        "solution disagrees with SymPy",
    )

    row_text = "; ".join(f"({r[0]}, {r[1]})" for r in rows)
    stem = phrase(rng, _STEM, m=m, rows=row_text, b=", ".join(str(v) for v in b), lam=lam)
    ft_b = f"({g22 + lam}*{c1} - ({g12})*({c2}))/part_a"
    ft_c = f"({g11 + lam}*{c2} - ({g12})*({c1}))/part_a"
    ft_d = f"{u1}*part_b + {u2}*part_c"
    parts = [
        build_part(
            phrase(rng, _PROMPT_A),
            Fraction(det),
            kind="number",
            marks=2,
            working=[
                f"A^T A = [[{g11}, {g12}], [{g12}, {g22}]], so M = [[{g11 + lam}, {g12}], [{g12}, {g22 + lam}]].",
                f"det(M) = {g11 + lam} × {g22 + lam} - {pn(g12)}^2 = {det}.",
            ],
            wrong=det_wrong,
        ),
        build_part(
            phrase(rng, _PROMPT_B),
            x1,
            kind="fraction",
            marks=2,
            working=[
                f"A^T b = ({c1}, {c2}). M^(-1) = adj(M)/det(M), so x1 = ({g22 + lam} × {pn(c1)} - {pn(g12)} × {pn(c2)})/{det}.",
                f"x1 = {x1}.",
            ],
            wrong=x1_wrong,
            follow_through=ft_b,
        ),
        build_part(
            phrase(rng, _PROMPT_C),
            x2,
            kind="fraction",
            marks=2,
            working=[f"x2 = ({g11 + lam} × {pn(c2)} - {pn(g12)} × {pn(c1)})/{det} = {x2}."],
            wrong=x2_wrong,
            follow_through=ft_c,
        ),
        build_part(
            phrase(rng, _PROMPT_D, u1=u1, u2=u2),
            pred,
            kind="fraction",
            marks=1,
            working=[f"{u1} × ({x1}) + {u2} × ({x2}) = {pred}."],
            wrong=pred_wrong,
            follow_through=ft_d,
        ),
    ]
    return dict(text=stem, parts=parts)


def _solve_once(g11, g12, g22, c1, c2, lam):
    """The slip of adding lambda to the first diagonal entry only."""
    det = (g11 + lam) * g22 - g12 * g12
    if det == 0:
        return 0, None, None
    return det, Fraction(g22 * c1 - g12 * c2, det), Fraction((g11 + lam) * c2 - g12 * c1, det)


@generator("exd_x_r2h_ridge_chain", level=18, presets={"exam": dict(tier="exam")})
def exd_x_r2h_ridge_chain(rng: random.Random, tier: str = "exam") -> dict:
    """det(A^T A + lambda I), then the two ridge coefficients, then a prediction for a new input."""
    check(tier == "exam", "exd_x_r2h_ridge_chain: only the exam tier exists")
    return _build(rng)
