"""Generates `p14c_inverse_regularised_solve` questions (`ucv1-inverse-regularised-solve`): solve a small
regularised inverse problem, answered as a pair.

**Boundary against the prerequisite ``matrix-rank-small``**
(``skills/linear-algebra/matrix-rank-small.yaml``): that skill computes the
RANK of a small matrix, an integer classification, a different object
entirely from a solved vector -- no render needed, the two answer kinds
cannot be confused.

**Boundary against ``ucv1-ridge-coefficient`` ** (``mathgen/generators/p13f_ridge_coefficient_build.py``):
that skill is the ONE-FEATURE, scalar case, ``beta = Sxy / (Sxx + lambda)``, a single number. This skill is
the TWO-VARIABLE case: given a small matrix ``A`` (``m x 2``, ``m`` in ``{2, 3, 4}``) and vector ``b`` (``m x
1``), solve the regularised normal equations

    (A^T A + lambda*I) x = A^T b

for the pair ``x = (x1, x2)``, never a scalar. ``A^T A`` is a genuine 2x2
matrix here (never a single sum): ``A^T A = [[sum(a_i1^2), sum(a_i1*a_i2)],
[sum(a_i1*a_i2), sum(a_i2^2)]]``, and ``A^T b = [sum(a_i1*b_i),
sum(a_i2*b_i)]``. Solved by Cramer's rule on the resulting symmetric 2x2
system, exact ``Fraction`` arithmetic throughout.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    join_terms,
    phrase,
    pick_distractors,
    rational_text,
)

_MAX_DRAW_ATTEMPTS = 500

# Same tolerance concern as p14c_iterative_linear_solver_step_build.py: the
# typed "pair" grader (mathgen/grading/typed.py's grade_pair) accepts a component
# within +/-0.051 of the expected value, so every distractor must clear that
# margin in at least one component -- checked unconditionally.
_GRADER_TOLERANCE = 0.051


def _clears_grader_tolerance(candidate, answer) -> bool:
    cx, cy = candidate
    ax, ay = answer
    return abs(float(cx) - float(ax)) > _GRADER_TOLERANCE or abs(float(cy) - float(ay)) > _GRADER_TOLERANCE


def _pair_payload(x1, x2) -> tuple[dict, str]:
    x1, x2 = Fraction(x1), Fraction(x2)
    return {"x1": x1, "x2": x2}, f"x1 = {rational_text(x1)}, x2 = {rational_text(x2)}"


def _ata_atb(rows: list[tuple[int, int]], b: list[int]):
    ata11 = sum(a1 * a1 for a1, _a2 in rows)
    ata12 = sum(a1 * a2 for a1, a2 in rows)
    ata22 = sum(a2 * a2 for _a1, a2 in rows)
    atb1 = sum(a1 * bi for (a1, _a2), bi in zip(rows, b))
    atb2 = sum(a2 * bi for (_a1, a2), bi in zip(rows, b))
    return ata11, ata12, ata22, atb1, atb2


def _solve_symmetric_2x2(m11, m12, m22, r1, r2):
    """Solve [[m11, m12], [m12, m22]] x = [r1, r2] by Cramer's rule."""
    det = m11 * m22 - m12 * m12
    if det == 0:
        return None
    x1 = Fraction(r1 * m22 - r2 * m12, det)
    x2 = Fraction(m11 * r2 - m12 * r1, det)
    return x1, x2


def _draw_A_b(rng, m, hi):
    rows = [(rng.randint(-hi, hi), rng.randint(-hi, hi)) for _ in range(m)]
    b = [rng.randint(-hi, hi) for _ in range(m)]
    return rows, b


def _rows_text(rows: list[tuple[int, int]]) -> str:
    return "; ".join(f"({a1}, {a2})" for a1, a2 in rows)


_GIVEN_TEMPLATES = (
    "A has rows {rows_text} (so A^T A and A^T b are computed from these two "
    "columns), and b = {b}. Using ridge penalty lambda = {lam}, solve the "
    "regularised normal equations (A^T A + lambda*I)x = A^T b for x = "
    "(x1, x2).",
    "For the matrix A with rows {rows_text} and vector b = {b}, and penalty "
    "lambda = {lam}, find x = (x1, x2) solving (A^T A + lambda*I)x = A^T b.",
    "A matrix A has rows {rows_text}; b = {b}. Solve the regularised normal "
    "equations (A^T A + lambda*I)x = A^T b for x = (x1, x2), using penalty "
    "lambda = {lam}.",
)

_EXAM_TEMPLATES = (
    "A has rows {rows_text} and b = {b}. The regularised normal equations "
    "(A^T A + lambda*I)x = A^T b have solution x = ({x1}, {x2}) for some "
    "penalty lambda > 0. Find lambda.",
    "For the matrix A with rows {rows_text} and vector b = {b}, the "
    "solution to (A^T A + lambda*I)x = A^T b is x = ({x1}, {x2}) for some "
    "penalty lambda > 0. Find lambda.",
)


def _misconceptions(rows, b, lam, ata11, ata12, ata22, atb1, atb2, x1, x2, *, include_forgets_transpose):
    """The distractor routes shared by every compute tier (easy/medium/hard)."""
    candidates: dict[str, tuple[Fraction, Fraction]] = {}

    plain = _solve_symmetric_2x2(ata11, ata12, ata22, atb1, atb2)
    if plain is not None:
        candidates["omits-lambda"] = plain

    once = _solve_symmetric_2x2(ata11 + lam, ata12, ata22, atb1, atb2)
    if once is not None:
        candidates["adds-lambda-once"] = once

    candidates["swaps-x1-x2"] = (x2, x1)

    if include_forgets_transpose:
        a11, a12 = rows[0]
        a21, a22 = rows[1]
        det_a = a11 * a22 - a12 * a21
        if det_a != 0:
            fx1 = Fraction(b[0] * a22 - b[1] * a12, det_a)
            fx2 = Fraction(a11 * b[1] - a21 * b[0], det_a)
            candidates["forgets-transpose"] = (fx1, fx2)

    return candidates


def _build_compute(rng, *, m, hi, lam_hi, templates, force_fraction=False):
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        rows, b = _draw_A_b(rng, m, hi)
        if any(a1 == 0 and a2 == 0 for a1, a2 in rows):
            continue
        if all(a1 == 0 for a1, _a2 in rows) or all(a2 == 0 for _a1, a2 in rows):
            continue
        lam = rng.randint(1, lam_hi)
        ata11, ata12, ata22, atb1, atb2 = _ata_atb(rows, b)
        m11, m12, m22 = ata11 + lam, ata12, ata22 + lam
        solved = _solve_symmetric_2x2(m11, m12, m22, atb1, atb2)
        if solved is None:
            continue
        x1, x2 = solved
        if force_fraction and x1.denominator == 1 and x2.denominator == 1:
            continue
        # check() the normal equations directly (the system this ANSWER
        # solves), then check_equivalent against a second solve route.
        check(m11 * x1 + m12 * x2 == atb1, "regularised normal-equation row 1 failed")
        check(m12 * x1 + m22 * x2 == atb2, "regularised normal-equation row 2 failed")
        det = m11 * m22 - m12 * m12
        check_equivalent(x1, Fraction(atb1 * m22 - atb2 * m12, det), "x1 recomputation mismatch")
        check_equivalent(x2, Fraction(m11 * atb2 - m12 * atb1, det), "x2 recomputation mismatch")

        candidates = _misconceptions(
            rows, b, lam, ata11, ata12, ata22, atb1, atb2, x1, x2, include_forgets_transpose=(m == 2)
        )
        if not candidates:
            continue
        try:
            check_pairwise_distinct(
                (x1, x2), *candidates.values(), msg="a distractor pair collides with the answer or another distractor"
            )
        except VerificationError:
            continue
        if not all(_clears_grader_tolerance(pair, (x1, x2)) for pair in candidates.values()):
            continue

        payload, answer_text = _pair_payload(x1, x2)
        distractor_strs = [f"x1 = {rational_text(cx1)}, x2 = {rational_text(cx2)}" for cx1, cx2 in candidates.values()]
        text = phrase(rng, templates, rows_text=_rows_text(rows), b=list(b), lam=lam)
        return dict(
            text=text,
            latex=text,
            answer=payload,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"A^T A = [[{ata11}, {ata12}], [{ata12}, {ata22}]], A^T b = [{atb1}, {atb2}].",
                f"Add lambda={lam} to both diagonal entries: [[{m11}, {m12}], [{m12}, {m22}]].",
                f"Solving gives x1 = {rational_text(x1)}, x2 = {rational_text(x2)}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=5),
        )
    raise VerificationError("could not draw a collision-free regularised-inverse question")


def _build_exam(rng):
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        rows, b = _draw_A_b(rng, 2, 5)
        if any(a1 == 0 and a2 == 0 for a1, a2 in rows):
            continue
        if all(a1 == 0 for a1, _a2 in rows) or all(a2 == 0 for _a1, a2 in rows):
            continue
        lam = rng.randint(1, 20)
        ata11, ata12, ata22, atb1, atb2 = _ata_atb(rows, b)
        m11, m12, m22 = ata11 + lam, ata12, ata22 + lam
        solved = _solve_symmetric_2x2(m11, m12, m22, atb1, atb2)
        if solved is None:
            continue
        x1, x2 = solved
        if x1 == 0 or x2 == 0:
            continue

        lam_from_row1 = Fraction(atb1 - ata11 * x1 - ata12 * x2, x1)
        lam_from_row2 = Fraction(atb2 - ata12 * x1 - ata22 * x2, x2)
        check_equivalent(lam_from_row1, lam, "lambda recovered from row 1 does not match the drawn lambda")
        check_equivalent(lam_from_row2, lam, "lambda recovered from row 2 does not match the drawn lambda")

        candidates = {
            "omits-lambda": Fraction(0),
            "wrong-row-division": Fraction(atb1 - ata11 * x1 - ata12 * x2, x2),
            "flips-lambda-sign": Fraction(-lam),
        }
        try:
            check_pairwise_distinct(
                Fraction(lam), *candidates.values(), msg=f"exam-tier collision: rows={rows}, b={b}, lam={lam}"
            )
        except VerificationError:
            continue

        answer_text = f"lambda = {rational_text(Fraction(lam))}"
        answer_latex = rf"\lambda = {rational_text(Fraction(lam))}"
        payload = {"lambda": Fraction(lam)}
        distractor_strs = [f"lambda = {rational_text(v)}" for v in candidates.values()]
        text = phrase(
            rng, _EXAM_TEMPLATES, rows_text=_rows_text(rows), b=list(b), x1=rational_text(x1), x2=rational_text(x2)
        )
        # Row 1's left-hand side ((ata11+lambda)*x1 + ata12*x2) is built with
        # join_terms rather than an f-string "*" concatenation, so a negative
        # ata12 lands as a real subtraction ("- 2x2") instead of a literal
        # "+ -2x2" (text_quality's signed_after_operator) and no bare "*"
        # survives (star_operator). The numerator atb1 - ata11*x1 - ata12*x2
        # is evaluated to a single Fraction and rendered with rational_text
        # instead of spelling out the two numeric products, for the same
        # reason: ata11/ata12 are constants here, not variables, so there is
        # no coefficient*variable pair for fmt_signed to format.
        row1_lhs = join_terms([(1, f"({ata11}+lambda)x1"), (ata12, "x2")])
        numerator = atb1 - ata11 * x1 - ata12 * x2
        return dict(
            text=text,
            latex=text,
            answer=payload,
            answer_text=answer_text,
            answer_latex=answer_latex,
            working=[
                f"Row 1: {row1_lhs} = {atb1}, so lambda = "
                f"{rational_text(numerator)} / {rational_text(x1)} "
                f"= {rational_text(Fraction(lam))}.",
                "Row 2 gives the same lambda, confirming it is consistent with both rows.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=3),
        )
    raise VerificationError("could not draw a collision-free exam-tier regularised-inverse question")


@generator(
    "p14c_inverse_regularised_solve",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def inverse_regularised_solve(rng, tier="easy"):
    """Solve the exact regularised normal equations (A^T A + lambda*I)x =
    A^T b for a supplied tiny matrix A, vector b and ridge penalty lambda,
    the skill's stated contract.

    easy: A is 2x2 (square). medium: A is 3x2 (overdetermined, three-term
    accumulation). hard: A is 4x2, larger integers, generic fraction
    answer. exam: inverse -- A, b and the solution x are stated; recover
    lambda.
    """
    if tier == "easy":
        return _build_compute(rng, m=2, hi=5, lam_hi=6, templates=_GIVEN_TEMPLATES)
    if tier == "medium":
        return _build_compute(rng, m=3, hi=5, lam_hi=8, templates=_GIVEN_TEMPLATES)
    if tier == "hard":
        return _build_compute(rng, m=4, hi=7, lam_hi=10, templates=_GIVEN_TEMPLATES, force_fraction=True)
    if tier == "exam":
        return _build_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
