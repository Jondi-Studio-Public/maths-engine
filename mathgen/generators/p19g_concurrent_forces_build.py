"""Generates `p19g_concurrent_forces_resultant` questions: the resultant of two or more concurrent forces, found
by genuine 2-D vector resolution.

Skill: senior-understand-motion-of-a-body-under-concurrent-forces
(ACMSM135, topic ``p19g_concurrent_forces_resultant``, ``answer: pair``,
BOTH components integers).

Scope:
    Given two or more force vectors, stated by magnitude and direction,
    acting concurrently on a body of stated mass, compute the resultant
    force's magnitude and direction (or the resulting acceleration's
    magnitude and direction).

**GRADER-KIND, load-bearing.** Every stated force's magnitude/direction and every graded answer's
magnitude/direction is rounded to the nearest whole newton / whole degree with `round_half_up` before display,
so `answer_text`'s two literals ("(52, 137)") carry no decimal point at all -- the real grader
(`mathgen/grading/typed.py:_component_target`) then grades each component EXACTLY (zero tolerance for an
integer-literal segment), trivially inside the 0.051 window. No decimal-precision variant of this skill is
built here.

Reference convention, fixed for every tier and stated in every prompt:
direction is measured ANTICLOCKWISE FROM EAST (the positive x-axis), as a
whole number of degrees in [0, 360).

**BOUNDARY (load-bearing):** the shipping
``senior-examine-momentum-force-resultant-force-action-and-reaction``
skill's own forces act along a single straight line (signed 1-D addition,
``answer: number``); this skill's forces act at a point from genuinely
different 2-D directions, requiring real Fx/Fy component resolution before
recombining into a magnitude+direction result (``answer: pair``). Never
build a question here that could be answered with 1-D signed addition
alone -- ``_check_noncollinear`` asserts this directly against the actual
stated angles, every seed, every difficulty (not merely by construction).
Also distinct from the prerequisite-shaped ``vector-addition`` skill: that
skill adds/subtracts already-COMPONENT-form 3-D vectors ("u = (-5, 0, 1)");
this skill states forces by MAGNITUDE AND DIRECTION, so the learner's first
step is resolving into components -- a genuinely different first move.

Four tiers, each a genuinely different DECISION (never a bigger number
standing in for a harder question):
- easy: exactly two forces, "nice" angles (multiples of 30 or 45 degrees),
  asks for the resultant FORCE (magnitude + direction); mass not needed.
- medium: exactly two forces at arbitrary (non-multiple-of-15) angles, so
  the sin/cos resolution needs a genuine calculator value, never a
  memorised special angle.
- hard: THREE concurrent forces (summing three Fx/Fy pairs, not two -- a
  genuinely new step) AND asks for the resulting ACCELERATION (dividing
  the resultant force by the stated mass), not the bare resultant force --
two new decisions together, both stated here as required
- exam: one force's magnitude is UNKNOWN (only its direction is given);
  instead, the resultant's own component in the 90-degree direction is
  given directly. Because the unknown force's direction is drawn away from
  0/180 degrees (its sine is never near zero), this is a single linear
  equation in one unknown with exactly one solution -- an inverse-
  resolution step the learner performs BEFORE resolving the full
  resultant's magnitude and direction, a genuine kind escalation over
  hard's forward-only resolution.

Independent check, every tier: ``_resolve`` sums each STATED force's
``(F*cos(theta), F*sin(theta))`` via a plain trig sum; ``_resolve_independent``
recomputes the same total via ``cmath.exp`` complex-exponential summation --
a different code path sharing no arithmetic with ``_resolve`` -- and the two
are compared to within 1e-6 before any rounding happens. ``exam``'s own
extra unknown-magnitude step is independently re-solved via a pre-solved
``sympy`` symbolic expression (solved once at import time, substituted per
question -- the same speed discipline
``mathgen/generators/p13b_mechanics_force_build.py`` uses for its own
per-question ``sympy`` checks), a third, symbolic route distinct from both
of the above.
"""

from __future__ import annotations

from functools import cache

import cmath
import math
from fractions import Fraction

from sympy import Eq, solve, symbols

from ..core import (
    GRADER_TOLERANCE,
    VerificationError,
    check,
    check_distinct,
    check_pairwise_distinct,
    generator,
    phrase,
    pick_distractors,
    round_half_up,
    rounds_unambiguously,
)

_RETRY_LIMIT = 300

# Multiples of 30 or 45 degrees in [0, 360) -- "nice" angles for the easy
#: tier, so the resolved components land close to friendly values before
#: rounding.
_NICE_ANGLES = tuple(sorted({a for a in range(0, 360, 15) if a % 30 == 0 or a % 45 == 0}))


def _draw_arbitrary_angle(rng) -> int:
    """An integer degree in [1, 359] that is NOT a multiple of 15 -- the
    medium/hard/exam tiers' "arbitrary angle" requirement, forcing a genuine
    calculator sin/cos value rather than a memorised special angle."""
    while True:
        a = rng.randint(1, 359)
        if a % 15 != 0:
            return a


def _resolve(forces) -> tuple[float, float]:
    """Sum each stated ``(magnitude, angle_degrees)`` force into
    ``(Fx, Fy)`` via a plain trig sum -- construction's own resolving
    route."""
    fx = math.fsum(f * math.cos(math.radians(a)) for f, a in forces)
    fy = math.fsum(f * math.sin(math.radians(a)) for f, a in forces)
    return fx, fy


def _resolve_independent(forces) -> tuple[float, float]:
    """The same sum via complex-exponential arithmetic (`cmath`) -- a
    genuinely different code path from `_resolve`, sharing no arithmetic
    with it, used only inside `_check_resolution` below."""
    total = sum(f * cmath.exp(1j * math.radians(a)) for f, a in forces)
    return total.real, total.imag


def _check_resolution(forces, fx: float, fy: float) -> None:
    ix, iy = _resolve_independent(forces)
    check(abs(ix - fx) < 1e-6, f"independent Fx mismatch: {ix} vs {fx}")
    check(abs(iy - fy) < 1e-6, f"independent Fy mismatch: {iy} vs {fy}")


def _check_noncollinear(angles) -> None:
    """At least one pair of the STATED force directions differs by neither
    0 nor 180 degrees (mod 360) -- so this question is never answerable by
    the shipping 1-D skill's own signed scalar addition (see module
    docstring BOUNDARY). Asserted every seed, every difficulty."""
    ok = any(abs(a - b) % 180 != 0 for i, a in enumerate(angles) for b in angles[i + 1 :])
    check(ok, f"every stated force direction is collinear: {angles}")


def _round_int(value: float, label: str) -> int:
    check(rounds_unambiguously(value, 0), f"{label} sits on a rounding boundary: {value}")
    frac = Fraction(*float(value).as_integer_ratio())
    rounded = round_half_up(frac, 0)
    check(rounded.denominator == 1, f"{label} did not round to a whole number: {rounded}")
    return int(rounded)


def _round_direction(value: float) -> int:
    return _round_int(value % 360.0, "direction") % 360


def _rounded_pair(fx: float, fy: float) -> tuple[int, int]:
    """Round `(Fx, Fy)` to `(magnitude, direction)`, both whole numbers,
    raising `VerificationError` (via `check`) if either sits on a rounding
    boundary -- the caller retries the draw in that case."""
    mag = math.hypot(fx, fy)
    direction = math.degrees(math.atan2(fy, fx))
    return _round_int(mag, "magnitude"), _round_direction(direction)


def _pair_text(mag: int, direction: int) -> str:
    return f"({mag}, {direction % 360})"


def _scalar_sum_pair(forces) -> tuple[int, int]:
    """The BOUNDARY misconception, quoted in the module docstring: sums the
    bare magnitudes as if the forces were collinear (1-D signed addition,
    the shipping momentum/force skill's own kernel), keeping the first
    force's own stated direction."""
    total = sum(f for f, _ in forces)
    return int(round(total)), forces[0][1] % 360


def _naive_atan_pair(fx: float, fy: float) -> tuple[int, int]:
    """Uses a bare `atan(Fy/Fx)` with no quadrant correction, instead of
    `atan2` -- wrong whenever Fx < 0."""
    if fx == 0:
        rad = math.pi / 2 if fy >= 0 else -math.pi / 2
    else:
        rad = math.atan(fy / fx)
    mag = int(round(math.hypot(fx, fy)))
    direction = int(round(math.degrees(rad))) % 360
    return mag, direction


def _swap_pair(mag: int, direction: int) -> tuple[int, int]:
    """Writes direction where magnitude belongs and vice versa."""
    return direction, mag


_DIR_NOTE = "direction measured anticlockwise from east, 0-360 degrees"

# --- easy: two forces, nice angles, asks for the resultant FORCE ----------

_EASY_PATTERNS = (
    "Two forces act at the same point on a body: {F1} N at {th1} degrees "
    "and {F2} N at {th2} degrees ({dir_note}). Find the magnitude (to the "
    "nearest newton) and direction (to the nearest degree) of the "
    "resultant force.",
    "A body has two concurrent forces acting on it: {F1} N at {th1} "
    "degrees and {F2} N at {th2} degrees ({dir_note}). Determine the "
    "resultant force's magnitude (nearest newton) and direction (nearest "
    "degree).",
    "Two forces, {F1} N at {th1} degrees and {F2} N at {th2} degrees "
    "({dir_note}), act concurrently on a body. Find the magnitude "
    "(nearest N) and direction (nearest degree) of the resultant force.",
)


def _build_easy(rng):
    for _ in range(_RETRY_LIMIT):
        th1 = rng.choice(_NICE_ANGLES)
        th2 = rng.choice(_NICE_ANGLES)
        if abs(th1 - th2) % 180 == 0:
            continue
        F1 = rng.randint(10, 90)
        F2 = rng.randint(10, 90)
        forces = [(F1, th1), (F2, th2)]
        fx, fy = _resolve(forces)
        _check_resolution(forces, fx, fy)
        try:
            mag, direction = _rounded_pair(fx, fy)
        except VerificationError:
            continue
        _check_noncollinear([th1, th2])

        answer = (mag, direction)
        m1 = _scalar_sum_pair(forces)  # 1-D BOUNDARY misconception
        m2 = (F1, th1 % 360)  # ignores the second force entirely
        m3 = _naive_atan_pair(fx, fy)  # atan, no quadrant correction
        m4 = _swap_pair(mag, direction)  # magnitude/direction swapped
        candidates = [m1, m2, m3, m4]
        try:
            check_distinct(answer, *candidates, msg="an easy distractor equals the answer")
            check_pairwise_distinct(*candidates, msg="two easy distractors collide")
        except VerificationError:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor easy question")

    text = phrase(rng, _EASY_PATTERNS, F1=F1, F2=F2, th1=th1, th2=th2, dir_note=_DIR_NOTE)
    answer_text = _pair_text(mag, direction)
    working = [
        f"Fx = {F1} x cos({th1}) + {F2} x cos({th2}) = {fx:.2f} N.",
        f"Fy = {F1} x sin({th1}) + {F2} x sin({th2}) = {fy:.2f} N.",
        f"Magnitude = sqrt(Fx^2 + Fy^2) = {mag} N (nearest N). "
        f"Direction = the angle of (Fx, Fy) from the positive x-axis = {direction} degrees (nearest degree).",
    ]
    cand_texts = [_pair_text(*c) for c in candidates]
    return dict(
        text=text,
        answer={"magnitude": mag, "direction": direction},
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, cand_texts, tolerance=GRADER_TOLERANCE),
    )


# --- medium: two forces, arbitrary angles, resultant FORCE ----------------

_MEDIUM_PATTERNS = (
    "Two forces act at the same point on a body: {F1} N at {th1} degrees "
    "and {F2} N at {th2} degrees ({dir_note}). Calculate the magnitude "
    "(to the nearest newton) and direction (to the nearest degree) of the "
    "resultant force.",
    "A body experiences two concurrent forces, {F1} N at {th1} degrees and "
    "{F2} N at {th2} degrees ({dir_note}). Work out the resultant force's "
    "magnitude (nearest newton) and direction (nearest degree).",
    "At a single point on a body, two forces act: {F1} N at {th1} degrees "
    "and {F2} N at {th2} degrees ({dir_note}). Find the resultant force's "
    "magnitude (nearest N) and direction (nearest degree).",
)


def _build_medium(rng):
    for _ in range(_RETRY_LIMIT):
        th1 = _draw_arbitrary_angle(rng)
        th2 = _draw_arbitrary_angle(rng)
        if abs(th1 - th2) % 180 == 0:
            continue
        F1 = rng.randint(10, 90)
        F2 = rng.randint(10, 90)
        forces = [(F1, th1), (F2, th2)]
        fx, fy = _resolve(forces)
        _check_resolution(forces, fx, fy)
        try:
            mag, direction = _rounded_pair(fx, fy)
        except VerificationError:
            continue
        _check_noncollinear([th1, th2])

        answer = (mag, direction)
        m1 = _scalar_sum_pair(forces)
        m2 = (F1, th1 % 360)
        m3 = _naive_atan_pair(fx, fy)
        m4 = _swap_pair(mag, direction)
        candidates = [m1, m2, m3, m4]
        try:
            check_distinct(answer, *candidates, msg="a medium distractor equals the answer")
            check_pairwise_distinct(*candidates, msg="two medium distractors collide")
        except VerificationError:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor medium question")

    text = phrase(rng, _MEDIUM_PATTERNS, F1=F1, F2=F2, th1=th1, th2=th2, dir_note=_DIR_NOTE)
    answer_text = _pair_text(mag, direction)
    working = [
        f"Fx = {F1} x cos({th1}) + {F2} x cos({th2}) = {fx:.2f} N.",
        f"Fy = {F1} x sin({th1}) + {F2} x sin({th2}) = {fy:.2f} N.",
        f"Magnitude = sqrt(Fx^2 + Fy^2) = {mag} N (nearest N). "
        f"Direction = the angle of (Fx, Fy) from the positive x-axis = {direction} degrees (nearest degree).",
    ]
    cand_texts = [_pair_text(*c) for c in candidates]
    return dict(
        text=text,
        answer={"magnitude": mag, "direction": direction},
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, cand_texts, tolerance=GRADER_TOLERANCE),
    )


# --- hard: three forces + mass, asks for the resulting ACCELERATION -------

_HARD_PATTERNS = (
    "Three forces act at a single point on a body of mass {m} kg: {F1} N "
    "at {th1} degrees, {F2} N at {th2} degrees, and {F3} N at {th3} degrees "
    "({dir_note}). Find the magnitude (to the nearest whole m/s^2) and "
    "direction (to the nearest degree) of the body's resulting "
    "acceleration.",
    "A body of mass {m} kg has three concurrent forces acting on it: "
    "{F1} N at {th1} degrees, {F2} N at {th2} degrees, and {F3} N at "
    "{th3} degrees ({dir_note}). Determine the magnitude (nearest whole "
    "m/s^2) and direction (nearest degree) of the resulting acceleration.",
    "Three forces, {F1} N at {th1} degrees, {F2} N at {th2} degrees, and "
    "{F3} N at {th3} degrees ({dir_note}), act at one point on a body of "
    "mass {m} kg. Find the resulting acceleration's magnitude (nearest "
    "whole m/s^2) and direction (nearest degree).",
)


def _build_hard(rng):
    for _ in range(_RETRY_LIMIT):
        thetas = [_draw_arbitrary_angle(rng) for _ in range(3)]
        if any(abs(a - b) % 180 == 0 for i, a in enumerate(thetas) for b in thetas[i + 1 :]):
            continue
        mags = [rng.randint(10, 90) for _ in range(3)]
        forces = list(zip(mags, thetas))
        m = rng.randint(2, 10)
        fx, fy = _resolve(forces)
        _check_resolution(forces, fx, fy)
        try:
            fmag, fdirection = _rounded_pair(fx, fy)  # resultant FORCE -- used only as a distractor
        except VerificationError:
            continue
        ax, ay = fx / m, fy / m
        try:
            mag, direction = _rounded_pair(ax, ay)
        except VerificationError:
            continue
        # Bound the resulting acceleration to a physically plausible value for an unqualified
        # "body": three forces up to 90 N each on a body as light as 2 kg could otherwise resolve to
        # |a| over 90 m/s^2 (>9g), absurd with no named context to justify it. Resample rather than
        # ship it. Lower bound: |a| under 2 m/s^2 is a near-cancelling resultant, a degenerate edge
        # case for this framing. Resample too.
        if mag > 50 or mag < 2:
            continue
        check(
            abs(ax * m - fx) < 1e-9 and abs(ay * m - fy) < 1e-9,
            "acceleration does not re-multiply back to the resultant force",
        )
        _check_noncollinear(thetas)

        answer = (mag, direction)
        fx2, fy2 = _resolve(forces[:-1])
        ax2, ay2 = fx2 / m, fy2 / m
        m1 = _scalar_sum_pair(forces)  # 1-D BOUNDARY misconception
        m2 = (fmag, fdirection)  # forgot to divide by mass
        m3 = (
            int(round(math.hypot(ax2, ay2))),
            int(round(math.degrees(math.atan2(ay2, ax2)))) % 360,
        )  # dropped the third force
        m4 = _naive_atan_pair(ax, ay)  # atan, no quadrant correction
        candidates = [m1, m2, m3, m4]
        try:
            check_distinct(answer, *candidates, msg="a hard distractor equals the answer")
            check_pairwise_distinct(*candidates, msg="two hard distractors collide")
        except VerificationError:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor hard question")

    F1, F2, F3 = mags
    th1, th2, th3 = thetas
    text = phrase(
        rng,
        _HARD_PATTERNS,
        m=m,
        F1=F1,
        F2=F2,
        F3=F3,
        th1=th1,
        th2=th2,
        th3=th3,
        dir_note=_DIR_NOTE,
    )
    answer_text = _pair_text(mag, direction)
    working = [
        f"Fx = {F1} x cos({th1}) + {F2} x cos({th2}) + {F3} x cos({th3}) = {fx:.2f} N.",
        f"Fy = {F1} x sin({th1}) + {F2} x sin({th2}) + {F3} x sin({th3}) = {fy:.2f} N.",
        f"Acceleration = resultant force / mass = ({fx:.2f}, {fy:.2f}) / {m} "
        f"m/s^2. Magnitude = {mag} m/s^2 (nearest whole), direction = "
        f"{direction} degrees (nearest degree).",
    ]
    cand_texts = [_pair_text(*c) for c in candidates]
    return dict(
        text=text,
        answer={"magnitude": mag, "direction": direction},
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, cand_texts, tolerance=GRADER_TOLERANCE),
    )


# --- exam: one force's magnitude unknown, inverse-resolution --------------

# Solved ONCE at import time, symbolically -- each question only substitutes
# numbers, matching `p13b_mechanics_force_build.py`'s own speed discipline
# for a per-question `sympy` check (`scripts/check_generator_speed.py` flags
# a fresh `solve()` call per draw).
_A_sym, _sinA_sym, _sinB_sym, _Ry_sym, _X_sym = symbols("A sinA sinB Ry X")


@cache
def _exam_x_solution():
    return solve(Eq(_A_sym * _sinA_sym + _X_sym * _sinB_sym, _Ry_sym), _X_sym)[0]


_EXAM_PATTERNS = (
    "Two forces act at the same point on a body. The first, {A} N, acts at "
    "{thA} degrees. The second acts at {thB} degrees, but its magnitude is "
    "not given directly. The resultant of the two forces has a component "
    "of {Ry} N in the 90-degree direction ({dir_note}; this fixes the "
    "second force's magnitude uniquely). Find the magnitude (to the "
    "nearest newton) and direction (to the nearest degree) of the "
    "resultant force.",
    "A body has two concurrent forces acting on it: {A} N at {thA} "
    "degrees, and a second force, of unknown magnitude, at {thB} degrees "
    "({dir_note}). The resultant's component in the 90-degree direction is "
    "{Ry} N, which determines the second force's magnitude uniquely. "
    "Determine the resultant force's magnitude (nearest N) and direction "
    "(nearest degree).",
    "At one point on a body, a force of {A} N acts at {thA} degrees, and a "
    "second force of unknown magnitude acts at {thB} degrees ({dir_note}). "
    "Given that the resultant's component in the 90-degree direction is "
    "{Ry} N (fixing the second force's magnitude uniquely), find the "
    "resultant force's magnitude (nearest N) and direction (nearest "
    "degree).",
)


def _build_exam(rng):
    for _ in range(_RETRY_LIMIT):
        thA = _draw_arbitrary_angle(rng)
        thB = _draw_arbitrary_angle(rng)
        if abs(thA - thB) % 180 == 0:
            continue
        sinB = math.sin(math.radians(thB))
        if abs(sinB) < 0.35:
            continue  # keep the inverse-solve well-conditioned
        A = rng.randint(15, 70)
        Ry = rng.choice([v for v in range(-90, 91) if v != 0])
        sinA = math.sin(math.radians(thA))
        X = (Ry - A * sinA) / sinB
        if not (10 <= X <= 150):
            continue

        # Independent re-derivation of X: the pre-solved symbolic expression,
        # substituted -- a genuinely different route from the plain-algebra
        # division line directly above.
        solved_X = float(_exam_x_solution().subs({_A_sym: A, _sinA_sym: sinA, _sinB_sym: sinB, _Ry_sym: Ry}))
        check(abs(solved_X - X) < 1e-6, f"exam: X mismatch, {solved_X} vs {X}")

        cosA, cosB = math.cos(math.radians(thA)), math.cos(math.radians(thB))
        Fx = A * cosA + X * cosB
        forces = [(A, thA), (X, thB)]
        fx_check, fy_check = _resolve(forces)
        if abs(fy_check - Ry) > 1e-6 or abs(fx_check - Fx) > 1e-6:
            continue
        _check_resolution(forces, Fx, Ry)

        try:
            mag, direction = _rounded_pair(Fx, Ry)
        except VerificationError:
            continue
        _check_noncollinear([thA, thB])

        answer = (mag, direction)
        m1 = (A, thA % 360)  # ignores the unknown force entirely
        X_wrong = (Ry + A * sinA) / sinB  # sign error solving for X
        Fx_wrong = A * cosA + X_wrong * cosB
        m2 = (
            int(round(math.hypot(Fx_wrong, Ry))),
            int(round(math.degrees(math.atan2(Ry, Fx_wrong)))) % 360,
        )
        m3 = _naive_atan_pair(Fx, Ry)  # atan, no quadrant correction
        m4 = _swap_pair(mag, direction)  # magnitude/direction swapped
        candidates = [m1, m2, m3, m4]
        try:
            check_distinct(answer, *candidates, msg="an exam distractor equals the answer")
            check_pairwise_distinct(*candidates, msg="two exam distractors collide")
        except VerificationError:
            continue
        break
    else:
        raise RuntimeError("could not draw a distinct-distractor exam question")

    text = phrase(rng, _EXAM_PATTERNS, A=A, thA=thA, thB=thB, Ry=Ry, dir_note=_DIR_NOTE)
    answer_text = _pair_text(mag, direction)
    working = [
        f"Resolve in the 90-degree direction: {A} x sin({thA}) + X x sin({thB}) = {Ry}, so X = {X:.2f} N.",
        f"Fx = {A} x cos({thA}) + X x cos({thB}) = {Fx:.2f} N; Fy = {Ry} N (given).",
        f"Magnitude = sqrt(Fx^2 + Fy^2) = {mag} N (nearest N). Direction = "
        f"the angle of (Fx, Fy) from the positive x-axis = {direction} degrees (nearest degree).",
    ]
    cand_texts = [_pair_text(*c) for c in candidates]
    return dict(
        text=text,
        answer={"magnitude": mag, "direction": direction},
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, cand_texts, tolerance=GRADER_TOLERANCE),
    )


_BUILDERS = {
    "easy": _build_easy,
    "medium": _build_medium,
    "hard": _build_hard,
    "exam": _build_exam,
}


@generator(
    "p19g_concurrent_forces_resultant",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19g_concurrent_forces_resultant(rng, tier="easy"):
    """See module docstring for the tier ladder. Every tier's graded answer
    is the unlabelled, positionally-ordered pair
    ``(magnitude, direction)``, both whole numbers, matching
    ``p13a_divide_ratio_build.py``'s own unlabelled `answer: pair` shape."""
    builder = _BUILDERS.get(tier)
    if builder is None:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(_BUILDERS)}")
    return builder(rng)
