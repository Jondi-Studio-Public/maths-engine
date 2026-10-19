"""Generates `p13f_coordinate_basis` questions (`ucv1-coordinate-basis`): the coordinates of a vector relative to
a stated basis, answered as a pair.

Given a two-dimensional basis {b1, b2} (integer entries, linearly
independent: ``D = b1x*b2y - b1y*b2x != 0``) and a vector v (integer
entries), compute the coordinates ``(c1, c2)`` such that
``c1*b1 + c2*b2 = v``, via Cramer's rule:

    c1 = (vx*b2y - vy*b2x) / D
    c2 = (b1x*vy - b1y*vx) / D

Boundary against ``ucv1-gram-schmidt-2d`` (orthonormalises two vectors,
``answer: list`` of vectors) and ``vector-projection`` (a single
scalar/vector projection): this skill solves a 2x2 LINEAR SYSTEM for
coordinates in a (not-necessarily-orthogonal) basis and returns an ordered
PAIR of coefficients ``(c1, c2)`` -- neither the object type (coefficients,
not vectors) nor the operation (solve a system, not orthonormalise/project)
matches either neighbour.

``grade_pair`` (``mathgen/grading/typed.py``) accepts fraction-shaped pair
components: its positional path runs the given text through
``extract_values`` (``mathgen/grading/spoken.py``), whose digit-token
regex matches ``"p/q"`` as one token directly, and the pair grader then
compares the resulting ``Fraction``s by exact equality (confirmed by a
So the hard tier
is free to draw a genuinely fractional ``(c1, c2)`` rather than being
restricted to integer-only draws.

Difficulty progression:
- easy:   D = 1 exactly, small entries -- integer (c1, c2), no division step.
- medium: |D| = 1 (D may be -1), larger/negative entries -- still integer,
          but the sign of D is no longer fixed.
- hard:   |D| > 1 -- division is now unavoidable; (c1, c2) may be a genuine
          exact fraction.
- exam:   the INVERSE direction -- (c1, c2) and the basis are given; the
          learner computes v = c1*b1 + c2*b2 (a matrix-vector multiply, the
          opposite direction from solving the system).
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Matrix, Rational

from ..core import (
    check,
    check_distinct,
    check_pairwise_distinct,
    generator,
    latex_answer,
    paren_negative,
    phrase,
    pick_distractors,
    rational_text,
    symbolically_equal,
)

_MAX_ATTEMPTS = 500


def _to_fraction(value) -> Fraction:
    r = Rational(value)
    return Fraction(int(r.p), int(r.q))


def _draw_basis(rng, lo, hi, *, exact_d=None, exact_abs_d=None, min_abs_d=None):
    for _attempt in range(_MAX_ATTEMPTS):
        b1x, b1y = rng.randint(lo, hi), rng.randint(lo, hi)
        b2x, b2y = rng.randint(lo, hi), rng.randint(lo, hi)
        d = b1x * b2y - b1y * b2x
        if d == 0:
            continue
        if exact_d is not None and d != exact_d:
            continue
        if exact_abs_d is not None and abs(d) != exact_abs_d:
            continue
        if min_abs_d is not None and abs(d) < min_abs_d:
            continue
        return (b1x, b1y), (b2x, b2y)
    raise RuntimeError("could not draw a usable basis")


def _draw_vector(rng, lo, hi):
    for _attempt in range(_MAX_ATTEMPTS):
        vx, vy = rng.randint(lo, hi), rng.randint(lo, hi)
        if (vx, vy) != (0, 0):
            return vx, vy
    raise RuntimeError("could not draw a usable vector")


def _solve_coordinates(b1, b2, v) -> tuple[Fraction, Fraction, int]:
    """Cramer's-rule route (construction's own algebra): returns (c1, c2, D)."""
    (b1x, b1y), (b2x, b2y) = b1, b2
    vx, vy = v
    d = b1x * b2y - b1y * b2x
    check(d != 0, "basis is not linearly independent")
    num1 = vx * b2y - vy * b2x
    num2 = b1x * vy - b1y * vx
    return Fraction(num1, d), Fraction(num2, d), d


def _independent_solve(b1, b2, v) -> tuple[Fraction, Fraction]:
    """Independent route: sympy's `Matrix.LUsolve` (Gaussian/LU elimination),
    a structurally different algorithm from the Cramer's-rule formula above
    -- never re-deriving the same numerator/denominator expression twice."""
    (b1x, b1y), (b2x, b2y) = b1, b2
    vx, vy = v
    m = Matrix([[b1x, b2x], [b1y, b2y]])
    rhs = Matrix([vx, vy])
    sol = m.LUsolve(rhs)
    return _to_fraction(sol[0]), _to_fraction(sol[1])


def _offer_pair_candidate(answer: tuple, value: tuple, offered: dict, label: str) -> None:
    """Append `value` to `offered` under `label` iff it genuinely differs
    from `answer` and from every candidate already offered -- a misconception
    that happens to coincide with the correct answer on this particular draw
    (e.g. `forgets-divide-by-D` when D == 1) is silently not exercisable here,
    exactly the discipline used elsewhere in this package."""
    if symbolically_equal(answer, value):
        return
    if any(symbolically_equal(value, existing) for existing in offered.values()):
        return
    check_distinct(answer, value, msg=f"{label} collided with the answer")
    for other_label, other in offered.items():
        check_pairwise_distinct(value, other, msg=f"{label} collided with {other_label}")
    offered[label] = value


def _fmt_pair(c1, c2) -> str:
    return f"c1 = {rational_text(c1)}, c2 = {rational_text(c2)}"


_SOLVE_TEMPLATES = (
    "The vectors b1 = ({b1x}, {b1y}) and b2 = ({b2x}, {b2y}) form a basis for "
    "R^2. Find the coordinates (c1, c2) of v = ({vx}, {vy}) in this basis, "
    "i.e. c1×b1 + c2×b2 = v.",
    "b1 = ({b1x}, {b1y}) and b2 = ({b2x}, {b2y}) are a basis for R^2. Express "
    "v = ({vx}, {vy}) as c1×b1 + c2×b2 and find (c1, c2).",
    "Given the basis vectors b1 = ({b1x}, {b1y}) and b2 = ({b2x}, {b2y}), find "
    "c1 and c2 such that c1×b1 + c2×b2 = v, where v = ({vx}, {vy}).",
)

_EXAM_TEMPLATES = (
    "The vectors b1 = ({b1x}, {b1y}) and b2 = ({b2x}, {b2y}) form a basis for "
    "R^2. A vector v has coordinates c1 = {c1}, c2 = {c2} in this basis (so "
    "v = c1×b1 + c2×b2). Find v as an ordered pair (vx, vy).",
    "b1 = ({b1x}, {b1y}) and b2 = ({b2x}, {b2y}) are a basis for R^2. Given "
    "that v = c1×b1 + c2×b2 with c1 = {c1} and c2 = {c2}, find v.",
    "Given the basis vectors b1 = ({b1x}, {b1y}) and b2 = ({b2x}, {b2y}) and "
    "the coordinates c1 = {c1}, c2 = {c2}, reconstruct the vector v = "
    "c1×b1 + c2×b2.",
)


def _build_solve(rng, *, basis_lo, basis_hi, vec_lo, vec_hi, basis_kwargs) -> dict:
    for _attempt in range(_MAX_ATTEMPTS):
        b1, b2 = _draw_basis(rng, basis_lo, basis_hi, **basis_kwargs)
        vx, vy = _draw_vector(rng, vec_lo, vec_hi)

        c1, c2, d = _solve_coordinates(b1, b2, (vx, vy))
        indep_c1, indep_c2 = _independent_solve(b1, b2, (vx, vy))
        check(
            symbolically_equal(c1, indep_c1) and symbolically_equal(c2, indep_c2),
            "Cramer's-rule route and the elimination route disagree",
        )
        check(c1 * b1[0] + c2 * b2[0] == vx, "definitional identity failed on x")
        check(c1 * b1[1] + c2 * b2[1] == vy, "definitional identity failed on y")

        answer = (c1, c2)
        offered: dict[str, tuple] = {}
        _offer_pair_candidate(answer, (c2, c1), offered, "swaps-c1-c2")
        _offer_pair_candidate(answer, (Fraction(vx), Fraction(vy)), offered, "uses-v-directly")
        num1 = vx * b2[1] - vy * b2[0]
        num2 = b1[0] * vy - b1[1] * vx
        _offer_pair_candidate(answer, (Fraction(num1), Fraction(num2)), offered, "forgets-divide-by-D")
        _offer_pair_candidate(answer, (-c1, -c2), offered, "wrong-sign-D")

        if not offered:
            continue

        answer_text = _fmt_pair(c1, c2)
        distractor_strs = [_fmt_pair(*value) for value in offered.values()]
        distractors = pick_distractors(answer_text, distractor_strs)
        if not distractors:
            continue

        text = phrase(rng, _SOLVE_TEMPLATES, b1x=b1[0], b1y=b1[1], b2x=b2[0], b2y=b2[1], vx=vx, vy=vy)
        working = [
            f"c1 = ({paren_negative(vx)} × {paren_negative(b2[1])} - {paren_negative(vy)} × {paren_negative(b2[0])}) / {paren_negative(d)} = {rational_text(c1)}.",
            f"c2 = ({paren_negative(b1[0])} × {paren_negative(vy)} - {paren_negative(b1[1])} × {paren_negative(vx)}) / {paren_negative(d)} = {rational_text(c2)}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer={"c1": c1, "c2": c2},
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("could not draw a usable coordinate-basis question")


def _build_easy(rng) -> dict:
    return _build_solve(
        rng,
        basis_lo=-3,
        basis_hi=3,
        vec_lo=-6,
        vec_hi=6,
        basis_kwargs=dict(exact_d=1),
    )


def _build_medium(rng) -> dict:
    return _build_solve(
        rng,
        basis_lo=-6,
        basis_hi=6,
        vec_lo=-9,
        vec_hi=9,
        basis_kwargs=dict(exact_abs_d=1),
    )


def _build_hard(rng) -> dict:
    return _build_solve(
        rng,
        basis_lo=-6,
        basis_hi=6,
        vec_lo=-9,
        vec_hi=9,
        basis_kwargs=dict(min_abs_d=2),
    )


def _build_exam(rng) -> dict:
    """Inverse tier: (c1, c2) and the basis are given; the learner computes
    v = c1*b1 + c2*b2. Independent check: recover (c1, c2) from the computed
    v via the SOLVING tiers' own Cramer's-rule route
    (`_solve_coordinates`) -- a genuine round trip through the opposite
    direction, not a re-statement of the multiply used to construct v."""
    coef_pool = tuple(v for v in range(-4, 5) if v != 0)
    for _attempt in range(_MAX_ATTEMPTS):
        b1, b2 = _draw_basis(rng, -5, 5, min_abs_d=1)
        c1 = rng.choice(coef_pool)
        c2 = rng.choice(coef_pool)
        vx = c1 * b1[0] + c2 * b2[0]
        vy = c1 * b1[1] + c2 * b2[1]

        recovered_c1, recovered_c2, _d = _solve_coordinates(b1, b2, (vx, vy))
        check(
            symbolically_equal(Fraction(c1), recovered_c1) and symbolically_equal(Fraction(c2), recovered_c2),
            "recovered (c1, c2) via Cramer's rule does not match the drawn coordinates",
        )

        answer = (vx, vy)
        offered: dict[str, tuple] = {}
        swap_v = (c2 * b1[0] + c1 * b2[0], c2 * b1[1] + c1 * b2[1])
        _offer_pair_candidate(answer, swap_v, offered, "swaps-c1-c2")
        _offer_pair_candidate(answer, (c1, c2), offered, "uses-v-directly")

        if not offered:
            continue

        answer_text = f"({vx}, {vy})"
        distractor_strs = [f"({value[0]}, {value[1]})" for value in offered.values()]
        distractors = pick_distractors(answer_text, distractor_strs)
        if not distractors:
            continue

        text = phrase(rng, _EXAM_TEMPLATES, b1x=b1[0], b1y=b1[1], b2x=b2[0], b2y=b2[1], c1=c1, c2=c2)
        working = [
            f"v = c1×b1 + c2×b2 = ({c1})×({b1[0]}, {b1[1]}) + ({c2})×({b2[0]}, {b2[1]}) = {answer_text}.",
        ]
        return dict(
            text=text,
            latex=text,
            answer={"vx": vx, "vy": vy},
            answer_text=answer_text,
            answer_latex=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("could not draw a usable exam-tier question")


@generator(
    "p13f_coordinate_basis",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def coordinate_basis(rng, tier="easy"):
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    if tier == "hard":
        return _build_hard(rng)
    if tier == "exam":
        return _build_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
