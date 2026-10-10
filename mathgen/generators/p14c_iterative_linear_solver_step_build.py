"""Generates `p14c_iterative_linear_solver_step` questions (`ucv1-iterative-linear-solver-step`): carry out one
step of an iterative linear solver, answered as a pair.

**Boundary against the prerequisite ``jacobi-iteration-2x2``**
(``mathgen/generators/university_scientific_computing.py``): that skill
performs one JACOBI update, which uses ONLY the old iterate ``(x0, y0)`` in
BOTH update equations. This skill performs one GAUSS-SEIDEL update instead:
the second equation's update uses the JUST-COMPUTED ``x1``, not ``x0`` -- a
genuinely different arithmetic route and, for a generic diagonally dominant
system, a genuinely different result. The prompt states "Gauss-Seidel"
explicitly and spells out the update order (x solved first from x0, y0; y
solved next from x1, y0) so ``wrong-variable-order`` is unambiguously wrong,
never a valid alternative ordering.

Reuses ``jacobi_iteration_2x2``'s own diagonal-dominance discipline exactly:
``off_b`` is drawn from ``[-(diagonal_a-1), diagonal_a-1]`` and ``off_c`` from
``[-(diagonal_d-1), diagonal_d-1]``, so ``|a11| > |a12|`` and ``|a22| > |a21|``
on every draw.

``x1 = (b1 - a12*y0) / a11``; ``y1 = (b2 - a21*x1) / a22`` (uses the
just-updated ``x1``, not ``x0``). Every quantity is stated in the question;
``Fraction`` arithmetic throughout.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    phrase,
    pick_distractors,
    rational_text,
)

_MAX_DRAW_ATTEMPTS = 500

# The typed "pair" grader (mathgen/grading/typed.py's grade_pair) accepts a component
# within +/-0.051 of the expected value (the spoken-transcript tolerance,
# reused for typed answers too). A distractor pair that is numerically THAT
# close to the answer in both components would be silently accepted as
# correct, so every candidate pair must clear this margin in at least one
# component -- checked unconditionally, never assumed from the symbolic
# check_pairwise_distinct alone.
_GRADER_TOLERANCE = 0.051


def _clears_grader_tolerance(candidate: tuple[Fraction, Fraction], answer: tuple[Fraction, Fraction]) -> bool:
    cx, cy = candidate
    ax, ay = answer
    return abs(float(cx) - float(ax)) > _GRADER_TOLERANCE or abs(float(cy) - float(ay)) > _GRADER_TOLERANCE


_ONE_STEP_TEMPLATES = (
    "For {a11}x + ({a12})y = {b1} and ({a21})x + {a22}y = {b2}, perform one "
    "Gauss-Seidel iteration from (x0, y0) = ({x0}, {y0}): solve the first "
    "equation for x1 using y0, then solve the second equation for y1 using "
    "the just-computed x1 (not x0). Give (x1, y1).",
    "A Gauss-Seidel step on {a11}x + ({a12})y = {b1}, ({a21})x + {a22}y = "
    "{b2} starts from (x0, y0) = ({x0}, {y0}). Solve the first equation for "
    "x1 from y0, then solve the second equation for y1 using the "
    "just-computed x1. Give (x1, y1).",
    "Starting from (x0, y0) = ({x0}, {y0}), take one Gauss-Seidel iteration "
    "for {a11}x + ({a12})y = {b1} and ({a21})x + {a22}y = {b2}: first find "
    "x1 from the first equation using y0, then find y1 from the second "
    "equation using the just-computed x1, not x0. Give (x1, y1).",
)

_HARD_TEMPLATES = (
    "For {a11}x + ({a12})y = {b1} and ({a21})x + {a22}y = {b2}, starting "
    "from (x0, y0) = ({x0}, {y0}), perform TWO Gauss-Seidel iterations: "
    "each iteration solves the first equation for the new x using the "
    "current y, then solves the second equation for the new y using the "
    "just-computed x. Give (x2, y2) after both iterations.",
    "A Gauss-Seidel iteration on {a11}x + ({a12})y = {b1}, ({a21})x + "
    "{a22}y = {b2} starts from (x0, y0) = ({x0}, {y0}). Chain two full "
    "iterations (each solves for the new x first, using the current y, "
    "then the new y using the just-computed x) and give the resulting "
    "(x2, y2).",
    "Starting from (x0, y0) = ({x0}, {y0}) for the system {a11}x + "
    "({a12})y = {b1}, ({a21})x + {a22}y = {b2}, apply TWO Gauss-Seidel "
    "updates -- in each, solve for the new x from the current y first, "
    "then for the new y from that just-computed x -- and report the "
    "pair (x2, y2) reached after the second update.",
)


# Both exam-tier equations are rendered in this SAME "both terms
# parenthesised" style regardless of which one is x-dominant, so the
# printed shape never leaks which equation plays which role -- unlike the
# lower tiers, which always print the x-dominant term bare and the
# off-diagonal term in parens (a positional tell that would make the
# identification step trivial to fake).
def _exam_eq_text(cx, cy, rhs) -> str:
    return f"({cx})x + ({cy})y = {rhs}"


_EXAM_TEMPLATES = (
    "For {eq1} and {eq2}, one of these two equations has an x-coefficient "
    "LARGER IN MAGNITUDE than its own y-coefficient -- use that equation to "
    "solve for x1 from y0, then use the OTHER equation to solve for y1 from "
    "the just-computed x1 (a Gauss-Seidel step). Starting point (x0, y0) = "
    "({x0}, {y0}). Give (x1, y1).",
    "Given {eq1} and {eq2} and starting point (x0, y0) = ({x0}, {y0}): "
    "identify which of the two equations has an x-coefficient LARGER IN "
    "MAGNITUDE than its own y-coefficient, use it to solve for x1 from y0, "
    "then use the other equation to solve for y1 from the just-computed x1 "
    "(a Gauss-Seidel step). Give (x1, y1).",
)


def _pair_payload(x, y) -> tuple[dict, str]:
    x, y = Fraction(x), Fraction(y)
    return {"x": x, "y": y}, f"x = {rational_text(x)}, y = {rational_text(y)}"


def _draw_system(rng, coefficient_hi):
    diagonal_a = rng.randint(2, coefficient_hi)
    diagonal_d = rng.randint(2, coefficient_hi)
    off_b = rng.randint(-max(1, diagonal_a - 1), max(1, diagonal_a - 1))
    off_c = rng.randint(-max(1, diagonal_d - 1), max(1, diagonal_d - 1))
    return diagonal_a, off_b, off_c, diagonal_d


def _draw_rhs(rng, diagonal_a, off_b, off_c, diagonal_d, point_hi):
    x_true = rng.randint(-point_hi, point_hi)
    y_true = rng.randint(-point_hi, point_hi)
    b1 = diagonal_a * x_true + off_b * y_true
    b2 = off_c * x_true + diagonal_d * y_true
    return b1, b2


def _draw_fraction(rng, hi=3):
    den = rng.choice((2, 3, 4))
    candidates = [n for n in range(-hi * den, hi * den + 1) if n % den != 0]
    num = rng.choice(candidates)
    return Fraction(num, den)


def _gs_step(a11, a12, b1, a21, a22, b2, x0, y0):
    x1 = Fraction(b1 - a12 * y0, a11)
    y1 = Fraction(b2 - a21 * x1, a22)
    return x1, y1


def _verify_step(a11, a12, b1, a21, a22, b2, x0, y0, x1, y1):
    check(a11 * x1 + a12 * y0 == b1, "Gauss-Seidel x-row residual mismatch")
    check(a21 * x1 + a22 * y1 == b2, "Gauss-Seidel y-row residual mismatch")
    check_equivalent(x1, Fraction(b1, a11) - Fraction(a12, a11) * y0, "Gauss-Seidel x1 recomputation mismatch")
    check_equivalent(y1, Fraction(b2, a22) - Fraction(a21, a22) * x1, "Gauss-Seidel y1 recomputation mismatch")


def _misconceptions(a11, a12, b1, a21, a22, b2, x0, y0, x1, y1):
    """The four required distractor routes for one Gauss-Seidel step, each a
    plausible WRONG procedure -- never a restatement of the correct one."""
    y1_jacobi = Fraction(b2 - a21 * x0, a22)
    x1_wrong_order = Fraction(b1 - a12 * y1_jacobi, a11)
    x1_forgets_divide = Fraction(b1 - a12 * y0)
    y1_forgets_divide = Fraction(b2 - a21 * x1)
    x1_sign = Fraction(b1 + a12 * y0, a11)
    y1_sign = Fraction(b2 + a21 * x1_sign, a22)
    return {
        "uses-jacobi-method": (x1, y1_jacobi),
        "wrong-variable-order": (x1_wrong_order, y1_jacobi),
        "forgets-diagonal-divide": (x1_forgets_divide, y1_forgets_divide),
        "sign-error-on-off-diagonal": (x1_sign, y1_sign),
    }


def _build_one_step(rng, *, coefficient_hi, point_hi, frac_start, templates):
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        a11, a12, a21, a22 = _draw_system(rng, coefficient_hi)
        b1, b2 = _draw_rhs(rng, a11, a12, a21, a22, point_hi)
        if frac_start:
            x0, y0 = _draw_fraction(rng), _draw_fraction(rng)
        else:
            x0 = Fraction(rng.randint(-point_hi, point_hi))
            y0 = Fraction(rng.randint(-point_hi, point_hi))

        x1, y1 = _gs_step(a11, a12, b1, a21, a22, b2, x0, y0)
        _verify_step(a11, a12, b1, a21, a22, b2, x0, y0, x1, y1)

        candidates = _misconceptions(a11, a12, b1, a21, a22, b2, x0, y0, x1, y1)
        try:
            check_pairwise_distinct(
                y1, candidates["uses-jacobi-method"][1], msg="Gauss-Seidel y1 collides with the Jacobi y1"
            )
            all_pairs = [(x1, y1), *candidates.values()]
            check_pairwise_distinct(*all_pairs, msg="a distractor pair collides with the answer or another distractor")
        except VerificationError:
            continue
        if not all(_clears_grader_tolerance(pair, (x1, y1)) for pair in candidates.values()):
            continue

        payload, answer_text = _pair_payload(x1, y1)
        distractor_strs = [f"x = {rational_text(cx)}, y = {rational_text(cy)}" for cx, cy in candidates.values()]
        text = phrase(
            rng, templates, a11=a11, a12=a12, b1=b1, a21=a21, a22=a22, b2=b2, x0=rational_text(x0), y0=rational_text(y0)
        )
        return dict(
            text=text,
            latex=text,
            answer=payload,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"x1 = ({b1} - ({a12})×{rational_text(y0)}) / {a11} = {rational_text(x1)}.",
                f"y1 = ({b2} - ({a21})×{rational_text(x1)}) / {a22} = {rational_text(y1)} "
                "(uses the just-computed x1, not x0).",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=5),
        )
    raise VerificationError("could not draw a collision-free Gauss-Seidel step question")


def _jacobi_step(a11, a12, b1, a21, a22, b2, x0, y0):
    """Both equations use only the OLD (x0, y0) -- the shipping
    ``jacobi-iteration-2x2`` update, reused here for the "did Jacobi
    throughout" hard-tier distractor chain."""
    x1 = Fraction(b1 - a12 * y0, a11)
    y1 = Fraction(b2 - a21 * x0, a22)
    return x1, y1


def _wrong_order_step(a11, a12, b1, a21, a22, b2, x0, y0):
    """Solves y first (from the old x), then x from the just-computed y --
    the reverse of the stated Gauss-Seidel order."""
    y1 = Fraction(b2 - a21 * x0, a22)
    x1 = Fraction(b1 - a12 * y1, a11)
    return x1, y1


def _forgets_divide_step(a11, a12, b1, a21, a22, b2, x0, y0):
    x1 = Fraction(b1 - a12 * y0)
    y1 = Fraction(b2 - a21 * x1)
    return x1, y1


def _sign_error_step(a11, a12, b1, a21, a22, b2, x0, y0):
    x1 = Fraction(b1 + a12 * y0, a11)
    y1 = Fraction(b2 + a21 * x1, a22)
    return x1, y1


def _ignores_dominance_order_step(cx1, cy1, rhs1, cx2, cy2, rhs2, x0, y0):
    """The exam-tier misconception: treats whichever equation is PRINTED
    FIRST as the x-equation regardless of which one is actually x-dominant
    -- the habit that was sufficient at every lower tier, where the
    x-dominant equation is always printed first. `(cx1, cy1, rhs1)` is the
    equation printed first, `(cx2, cy2, rhs2)` the one printed second."""
    x1 = Fraction(rhs1 - cy1 * y0, cx1)
    y1 = Fraction(rhs2 - cx2 * x1, cy2)
    return x1, y1


def _build_exam(rng, *, coefficient_hi, point_hi):
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        a11, a12, a21, a22 = _draw_system(rng, coefficient_hi)
        # off_b (=a12) and off_c (=a21) must both be nonzero: this is the
        # true x-equation's own off-diagonal (y) coefficient and the true
        # y-equation's own off-diagonal (x) coefficient. Guarding both
        # nonzero keeps `_ignores_dominance_order_step` well-defined (never
        # a ZeroDivisionError) whichever equation lands first on the draw
        # below -- if the display order is swapped, that step divides by
        # a21 (=off_c) for x1 and a12 (=off_b) for y1.
        if a12 == 0 or a21 == 0:
            continue
        b1, b2 = _draw_rhs(rng, a11, a12, a21, a22, point_hi)
        x0, y0 = _draw_fraction(rng), _draw_fraction(rng)

        x1, y1 = _gs_step(a11, a12, b1, a21, a22, b2, x0, y0)
        _verify_step(a11, a12, b1, a21, a22, b2, x0, y0, x1, y1)

        # True roles are fixed by construction: |a11| > |a12| within the
        # first equation and |a22| > |a21| within the second, so the
        # x-dominant equation is always (a11, a12, b1) and the y-dominant
        # one is always (a21, a22, b2) -- independent of DISPLAY order,
        # which is chosen below purely for presentation.
        candidates = dict(_misconceptions(a11, a12, b1, a21, a22, b2, x0, y0, x1, y1))

        swapped = rng.choice((False, True))
        if swapped:
            eq_first, eq_second = (a21, a22, b2), (a11, a12, b1)
        else:
            eq_first, eq_second = (a11, a12, b1), (a21, a22, b2)

        # Only meaningful (and only well-defined as a DISTRACTOR) when the
        # display order is swapped -- on a non-swapped draw "trust the
        # printed order" already IS the correct procedure, so it would
        # equal the true answer rather than being a wrong candidate.
        if swapped:
            candidates["ignores-dominance-order"] = _ignores_dominance_order_step(*eq_first, *eq_second, x0, y0)

        try:
            check_pairwise_distinct(
                y1, candidates["uses-jacobi-method"][1], msg="Gauss-Seidel y1 collides with the Jacobi y1"
            )
            all_pairs = [(x1, y1), *candidates.values()]
            check_pairwise_distinct(
                *all_pairs, msg="an exam-tier distractor pair collides with the answer or another distractor"
            )
        except VerificationError:
            continue
        if not all(_clears_grader_tolerance(pair, (x1, y1)) for pair in candidates.values()):
            continue

        payload, answer_text = _pair_payload(x1, y1)
        distractor_strs = [f"x = {rational_text(cx)}, y = {rational_text(cy)}" for cx, cy in candidates.values()]
        text = phrase(
            rng,
            _EXAM_TEMPLATES,
            eq1=_exam_eq_text(*eq_first),
            eq2=_exam_eq_text(*eq_second),
            x0=rational_text(x0),
            y0=rational_text(y0),
        )
        return dict(
            text=text,
            latex=text,
            answer=payload,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"{_exam_eq_text(a11, a12, b1)} has |x-coefficient| = {abs(a11)} > "
                f"|y-coefficient| = {abs(a12)}, so it is the x-equation: "
                f"x1 = ({b1} - ({a12})×{rational_text(y0)}) / {a11} = {rational_text(x1)}.",
                f"{_exam_eq_text(a21, a22, b2)} is then the y-equation, using the "
                f"just-computed x1 (not x0): "
                f"y1 = ({b2} - ({a21})×{rational_text(x1)}) / {a22} = {rational_text(y1)}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=5),
        )
    raise VerificationError("could not draw a collision-free exam-tier Gauss-Seidel step question")


def _build_hard(rng, *, coefficient_hi, point_hi):
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        a11, a12, a21, a22 = _draw_system(rng, coefficient_hi)
        b1, b2 = _draw_rhs(rng, a11, a12, a21, a22, point_hi)
        x0 = Fraction(rng.randint(-point_hi, point_hi))
        y0 = Fraction(rng.randint(-point_hi, point_hi))

        x1, y1 = _gs_step(a11, a12, b1, a21, a22, b2, x0, y0)
        _verify_step(a11, a12, b1, a21, a22, b2, x0, y0, x1, y1)
        x2, y2 = _gs_step(a11, a12, b1, a21, a22, b2, x1, y1)
        _verify_step(a11, a12, b1, a21, a22, b2, x1, y1, x2, y2)

        args = (a11, a12, b1, a21, a22, b2)
        jx1, jy1 = _jacobi_step(*args, x0, y0)
        jx2, jy2 = _jacobi_step(*args, jx1, jy1)
        wx1, wy1 = _wrong_order_step(*args, x0, y0)
        wx2, wy2 = _wrong_order_step(*args, wx1, wy1)
        fx1, fy1 = _forgets_divide_step(*args, x0, y0)
        fx2, fy2 = _forgets_divide_step(*args, fx1, fy1)
        sx1, sy1 = _sign_error_step(*args, x0, y0)
        sx2, sy2 = _sign_error_step(*args, sx1, sy1)

        second_step_candidates = {
            "uses-jacobi-method": (jx2, jy2),
            "wrong-variable-order": (wx2, wy2),
            "forgets-diagonal-divide": (fx2, fy2),
            "sign-error-on-off-diagonal": (sx2, sy2),
            "stopped-after-one-step": (x1, y1),
        }
        try:
            all_pairs = [(x2, y2), *second_step_candidates.values()]
            check_pairwise_distinct(
                *all_pairs, msg="a hard-tier distractor pair collides with the answer or another distractor"
            )
        except VerificationError:
            continue
        if not all(_clears_grader_tolerance(pair, (x2, y2)) for pair in second_step_candidates.values()):
            continue

        payload, answer_text = _pair_payload(x2, y2)
        distractor_strs = [
            f"x = {rational_text(cx)}, y = {rational_text(cy)}" for cx, cy in second_step_candidates.values()
        ]
        text = phrase(
            rng,
            _HARD_TEMPLATES,
            a11=a11,
            a12=a12,
            b1=b1,
            a21=a21,
            a22=a22,
            b2=b2,
            x0=rational_text(x0),
            y0=rational_text(y0),
        )
        return dict(
            text=text,
            latex=text,
            answer=payload,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"Iteration 1: x1 = ({b1} - ({a12})×{rational_text(y0)}) / {a11} = {rational_text(x1)}; "
                f"y1 = ({b2} - ({a21})×{rational_text(x1)}) / {a22} = {rational_text(y1)}.",
                f"Iteration 2: x2 = ({b1} - ({a12})×{rational_text(y1)}) / {a11} = {rational_text(x2)}; "
                f"y2 = ({b2} - ({a21})×{rational_text(x2)}) / {a22} = {rational_text(y2)}.",
            ],
            distractors=pick_distractors(answer_text, distractor_strs, n=5),
        )
    raise VerificationError("could not draw a collision-free two-step Gauss-Seidel question")


@generator(
    "p14c_iterative_linear_solver_step",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def iterative_linear_solver_step(rng, tier="easy"):
    """Compute one (easy/medium/exam) or two chained (hard) Gauss-Seidel
    updates for a supplied diagonally dominant 2x2 linear system, the
    skill's stated contract."""
    if tier == "easy":
        return _build_one_step(rng, coefficient_hi=4, point_hi=3, frac_start=False, templates=_ONE_STEP_TEMPLATES)
    if tier == "medium":
        return _build_one_step(rng, coefficient_hi=8, point_hi=5, frac_start=False, templates=_ONE_STEP_TEMPLATES)
    if tier == "hard":
        return _build_hard(rng, coefficient_hi=8, point_hi=5)
    if tier == "exam":
        return _build_exam(rng, coefficient_hi=6, point_hi=4)
    raise ValueError(f"unknown tier {tier!r}")
