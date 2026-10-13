"""Generates `p14c_linear_system_determinant` questions: for a nonsingular 2x2 system, the determinants that
solve it by Cramer's rule, as a labelled pair.

``ucv1-linear-system-determinant`` (linear-algebra, topic ``p14c_linear_system_determinant``, ``answer:
pair``, labelled).

**Kernel.** Given a 2x2 system ``a11 x + a12 y = b1``, ``a21 x + a22 y = b2``
(all integers, nonsingular: ``D = det(A) = a11*a22 - a12*a21 != 0``), the
graded answer is always a labelled pair of two DETERMINANTS -- never the
solved ``x``/``y`` -- built from Cramer's rule:

    D    = det(A)                       = a11*a22 - a12*a21
    D_x  = det(A, column 1 replaced by b) = b1*a22  - a12*b2
    D_y  = det(A, column 2 replaced by b) = a11*b2  - b1*a21

Boundary against ``matrix-solve-system`` (grades the solved ``(x, y)`` pair
for the exact same kind of system -- method invisible) and
``matrix-determinant-2x2``:
this skill's graded answer is always a pair of the two Cramer determinants
themselves, so a learner who only solved by elimination has not produced
``D_x``/``D_y`` at all. ``|D| >= 2`` is enforced on every draw -- at
``|D| = 1`` Cramer's rule gives ``(D_x, D_y) = D * (x, y)`` with ``D = ±1``,
so the graded pair would numerically equal (or be the exact negation of) the
elimination solution, making this skill swappable into ``matrix-solve-system``
undetected. The exam tier is the one place ``x``/``y`` appear in the prompt
at all -- as GIVEN data, never as the graded answer -- to invert the same
Cramer identity in the other direction (``D_x = x*D``, ``D_y = y*D``).

**Labelled-pair grading.** ``answer`` is a dict (``{"D_x": ..., "D": ...}``
etc), read by ``mathgen/grading/typed.py::_labelled_pair_verdict`` -- order-independent
by label name, decisive only when both sides parse into the same label set
(read that function's docstring before touching ``answer_text``). Labels are
bare identifiers (``D``, ``D_x``, ``D_y``) so they clear
``_LABEL_PREFIX_RE`` and `core.py`'s own ``_MATH_SYMBOL_NAME_RE`` (a
subscripted single-letter symbol). The exact same label text is used in the
prompt and in ``answer_text`` (the words the learner is told to
answer with are the words the grader keys off).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
)

_MAX_DRAW_ATTEMPTS = 500

_MAT_TEXT = "A = [{a11} {a12}; {a21} {a22}]"

_DX_TEMPLATES = (
    "For the linear system A x = b with " + _MAT_TEXT + " and b = ({b1}, {b2}), "
    "find D = det(A) and D_x, the determinant of A with its FIRST column "
    "replaced by b. Give your answer as D_x = ..., D = ... .",
    "A linear system A x = b has " + _MAT_TEXT + " and b = ({b1}, {b2}). Find "
    "D_x -- A with column 1 replaced by b -- together with D = det(A). Give "
    "your answer as D_x = ..., D = ... .",
    "Given the system A x = b where " + _MAT_TEXT + " and b = ({b1}, {b2}), "
    "compute D = det(A) and D_x (A with column 1 replaced by b). Give your "
    "answer as D_x = ..., D = ... .",
)

_DY_TEMPLATES = (
    "For the linear system A x = b with " + _MAT_TEXT + " and b = ({b1}, {b2}), "
    "find D = det(A) and D_y, the determinant of A with its SECOND column "
    "replaced by b. Give your answer as D_y = ..., D = ... .",
    "A linear system A x = b has " + _MAT_TEXT + " and b = ({b1}, {b2}). Find "
    "D_y -- A with column 2 replaced by b -- together with D = det(A). Give "
    "your answer as D_y = ..., D = ... .",
    "Given the system A x = b where " + _MAT_TEXT + " and b = ({b1}, {b2}), "
    "compute D = det(A) and D_y (A with column 2 replaced by b). Give your "
    "answer as D_y = ..., D = ... .",
)

_HARD_TEMPLATES = (
    "For the linear system A x = b with " + _MAT_TEXT + " and b = ({b1}, {b2}), "
    "find D_x (A with column 1 replaced by b) and D_y (A with column 2 "
    "replaced by b). Do not solve for x or y. Give your answer as D_x = ..., "
    "D_y = ... .",
    "A linear system A x = b has " + _MAT_TEXT + " and b = ({b1}, {b2}). "
    "Compute D_x (column 1 of A replaced by b) and D_y (column 2 of A "
    "replaced by b) without solving the system. Give your answer as "
    "D_x = ..., D_y = ... .",
    "Given the system A x = b where " + _MAT_TEXT + " and b = ({b1}, {b2}), "
    "find both D_x and D_y (A with, respectively, column 1 and column 2 "
    "replaced by b) -- not the solved x or y. Give your answer as "
    "D_x = ..., D_y = ... .",
)

_EXAM_TEMPLATES = (
    "A 2x2 linear system A x = b has solution x = {x}, y = {y} and "
    "D = det(A) = {D}. By Cramer's rule x = D_x/D and y = D_y/D. Find "
    "D_x and D_y. Give your answer as D_x = ..., D_y = ... .",
    "For a 2x2 linear system A x = b, det(A) = D = {D} and the solution is "
    "x = {x}, y = {y}, where x = D_x/D and y = D_y/D. Find D_x and D_y. "
    "Give your answer as D_x = ..., D_y = ... .",
    "A linear system A x = b solves to x = {x}, y = {y}, with D = det(A) = "
    "{D}, where x = D_x/D and y = D_y/D. Find D_x and D_y. Give your "
    "answer as D_x = ..., D_y = ... .",
)


def _draw_system(rng, *, hi):
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        x = rng.choice([v for v in range(-6, 7) if v != 0])
        y = rng.choice([v for v in range(-6, 7) if v != 0])
        a11, a12, a21, a22 = (rng.randint(-hi, hi) for _ in range(4))
        D = a11 * a22 - a12 * a21
        if abs(D) < 2:
            continue
        b1 = a11 * x + a12 * y
        b2 = a21 * x + a22 * y

        Dx = b1 * a22 - a12 * b2
        Dy = a11 * b2 - b1 * a21
        # Independent route: Cramer's rule identity (D_x, D_y) = D*(x, y),
        # derived from how b was constructed -- entirely separate from the
        # cofactor-formula arithmetic above.
        check_equivalent(Dx, x * D, "D_x recomputation via Cramer identity mismatch")
        check_equivalent(Dy, y * D, "D_y recomputation via Cramer identity mismatch")
        # Second, differently-ordered route for D itself: expansion along
        # the second row (cofactors C21 = -a12, C22 = a11) rather than the
        # first-row Sarrus product used above.
        D_alt = a21 * (-a12) + a22 * a11
        check(D_alt == D, "D recomputation via second-row expansion mismatch")

        D_sign = a11 * a22 + a12 * a21  # sign error: added, not subtracted
        Dx_sign = b1 * a22 + a12 * b2  # same sign error applied to D_x

        try:
            check_pairwise_distinct(
                D,
                Dx,
                Dy,
                D_sign,
                Dx_sign,
                msg=f"determinant collision: A=[{a11} {a12}; {a21} {a22}], b=({b1},{b2})",
            )
            check(Dx != 0 and Dy != 0, "D_x/D_y must both be nonzero")
        except VerificationError:
            continue
        return dict(
            a11=a11,
            a12=a12,
            a21=a21,
            a22=a22,
            b1=b1,
            b2=b2,
            x=x,
            y=y,
            D=D,
            Dx=Dx,
            Dy=Dy,
            D_sign=D_sign,
            Dx_sign=Dx_sign,
        )
    raise VerificationError("could not draw a collision-free nonsingular 2x2 system")


def _pair_payload(label1, val1, label2, val2):
    answer_text = f"{label1} = {val1}, {label2} = {val2}"
    return {label1: val1, label2: val2}, answer_text


def _build_easy(rng) -> dict:
    d = _draw_system(rng, hi=6)
    payload, answer_text = _pair_payload("D_x", d["Dx"], "D", d["D"])
    text = phrase(rng, _DX_TEMPLATES, **d)
    distractor_strs = pick_distractors(
        answer_text,
        [
            f"D_x = {d['D']}, D = {d['Dx']}",  # swaps-det-order
            f"D_x = {d['Dy']}, D = {d['D']}",  # wrong-column-replaced
            f"D_x = {d['Dx']}, D = {d['D_sign']}",  # sign-error
            f"D_x = {d['D']}, D = {d['D']}",  # uses-plain-determinant-twice
        ],
        n=4,
    )
    return dict(
        text=text,
        latex=text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"D = a11×a22 - a12×a21 = ({d['a11']})({d['a22']}) - ({d['a12']})({d['a21']}) = {d['D']}.",
            f"D_x = b1×a22 - a12×b2 = ({d['b1']})({d['a22']}) - ({d['a12']})({d['b2']}) = {d['Dx']}.",
        ],
        distractors=distractor_strs,
    )


def _build_medium(rng) -> dict:
    d = _draw_system(rng, hi=6)
    payload, answer_text = _pair_payload("D_y", d["Dy"], "D", d["D"])
    text = phrase(rng, _DY_TEMPLATES, **d)
    distractor_strs = pick_distractors(
        answer_text,
        [
            f"D_y = {d['D']}, D = {d['Dy']}",  # swaps-det-order
            f"D_y = {d['Dx']}, D = {d['D']}",  # wrong-column-replaced
            f"D_y = {d['Dy']}, D = {d['D_sign']}",  # sign-error
            f"D_y = {d['D']}, D = {d['D']}",  # uses-plain-determinant-twice
        ],
        n=4,
    )
    return dict(
        text=text,
        latex=text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"D = a11×a22 - a12×a21 = ({d['a11']})({d['a22']}) - ({d['a12']})({d['a21']}) = {d['D']}.",
            f"D_y = a11×b2 - b1×a21 = ({d['a11']})({d['b2']}) - ({d['b1']})({d['a21']}) = {d['Dy']}.",
        ],
        distractors=distractor_strs,
    )


def _build_hard(rng) -> dict:
    d = _draw_system(rng, hi=6)
    payload, answer_text = _pair_payload("D_x", d["Dx"], "D_y", d["Dy"])
    text = phrase(rng, _HARD_TEMPLATES, **d)
    distractor_strs = pick_distractors(
        answer_text,
        [
            f"D_x = {d['Dy']}, D_y = {d['Dx']}",  # swaps-det-order
            f"D_x = {d['Dx_sign']}, D_y = {d['Dy']}",  # sign-error
            f"D_x = {d['D']}, D_y = {d['D']}",  # uses-plain-determinant-twice
        ],
        n=3,
    )
    return dict(
        text=text,
        latex=text,
        answer=payload,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"D_x = b1×a22 - a12×b2 = ({d['b1']})({d['a22']}) - ({d['a12']})({d['b2']}) = {d['Dx']}.",
            f"D_y = a11×b2 - b1×a21 = ({d['a11']})({d['b2']}) - ({d['b1']})({d['a21']}) = {d['Dy']}.",
        ],
        distractors=distractor_strs,
    )


def _build_exam(rng) -> dict:
    """Inverse tier: x, y and D are stated directly (never derived from a
    matrix); the learner recovers D_x = x*D and D_y = y*D from Cramer's
    identity x = D_x/D, y = D_y/D. Each is a direct multiplication -- a
    linear rearrangement of the defining ratio, not a second algebraic root
    -- so both are unique by construction. Proved by round-tripping: divide
    the recovered value back by D and confirm it lands on the stated x/y."""
    for _attempt in range(_MAX_DRAW_ATTEMPTS):
        x = rng.choice([v for v in range(-6, 7) if v != 0])
        y = rng.choice([v for v in range(-6, 7) if v != 0])
        D = rng.choice([v for v in range(-9, 10) if abs(v) >= 2])
        Dx = x * D
        Dy = y * D
        # Round-trip: recompute x, y from the recovered determinants (exact
        # rational division, D != 0 by construction) and confirm they land
        # back on the stated solution -- a route entirely separate from the
        # x*D/y*D multiplication that produced D_x/D_y above.
        recovered_x = Fraction(Dx, D)
        recovered_y = Fraction(Dy, D)
        check(recovered_x == x, "recovered x does not match the stated solution")
        check(recovered_y == y, "recovered y does not match the stated solution")

        candidates = {
            "swaps-det-order": (Dy, Dx),
            "sign-error": (-Dx, Dy),
            "uses-plain-determinant-twice": (D, D),
        }
        try:
            check_pairwise_distinct(
                Dx,
                Dy,
                -Dx,
                D,
                msg=f"exam-tier determinant collision: x={x}, y={y}, D={D}",
            )
        except VerificationError:
            continue

        payload, answer_text = _pair_payload("D_x", Dx, "D_y", Dy)
        text = phrase(rng, _EXAM_TEMPLATES, x=x, y=y, D=D)
        distractor_strs = pick_distractors(
            answer_text, [f"D_x = {vx}, D_y = {vy}" for vx, vy in candidates.values()], n=3
        )
        return dict(
            text=text,
            latex=text,
            answer=payload,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=[
                f"D_x = x*D = ({x})({D}) = {Dx}.",
                f"D_y = y*D = ({y})({D}) = {Dy}.",
            ],
            distractors=distractor_strs,
        )
    raise VerificationError("could not draw a collision-free exam-tier determinant question")


@generator(
    "p14c_linear_system_determinant",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def linear_system_determinant(rng, tier="easy"):
    """Solve a nonsingular 2x2 linear system using determinant ratios (the
    skill's stated contract) -- but the GRADED answer
    is always the pair of Cramer determinants themselves, never the solved
    (x, y) (see the module docstring's boundary against matrix-solve-system).

    easy: (D_x, D) -- first column replaced.
    medium: (D_y, D) -- second column replaced (never "always column 1").
    hard: (D_x, D_y) together -- D itself not part of the graded answer.
    exam: inverse -- x, y and D are stated; recover D_x and D_y.
    """
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    if tier == "hard":
        return _build_hard(rng)
    if tier == "exam":
        return _build_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
