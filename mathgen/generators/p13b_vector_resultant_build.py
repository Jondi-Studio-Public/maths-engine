"""Resultant of vectors at a non-right angle.

Skill: ``senior-solve-problems-involving-displacement-force-and-velocity-
involving`` (ACMSM023, domain ``differential-equations``,
framework ``acara-senior-v8-4``). Skill name, quoted verbatim:
"Solve problems involving displacement, force and velocity involving the
above concepts." ACMSM023 sits in Specialist Mathematics Unit 1, topic
"Vectors in the plane"; confirmed
against that raw text before building -- "the above concepts" is vector
addition/subtraction/components, not Unit-3/4 mechanics (which is a
different pair of rows).

**Duplicate-risk note**: ``senior-examine-examples-of-vectors-including-displacement-and-velocity`` already
ships a Pythagoras-on-PERPENDICULAR-legs kernel (magnitude of a displacement/velocity from two perpendicular
components, every variant). Every question this module builds instead involves two (or three) vectors at a
stated angle drawn from a pool that EXCLUDES 90 degrees for the two-vector tiers, so Pythagoras alone cannot
answer easy or medium -- the learner must resolve components with sine/cosine, or apply the cosine/sine rule.
The hard tier does not compute a magnitude at all (a chained rate problem). The exam tier sums THREE
non-perpendicular vectors by full component resolution, a method the two-vector shortcuts cannot reach. "Uses
Pythagoras" (ignoring the angle) is deliberately built and tested as a REJECTED distractor wherever a
two-vector magnitude is asked for (easy, exam), since it is the exact "looks like the shipping neighbour"
error this skill exists to not reward.

Single generator, topic ``p13b_vector_resultant_angle``, four tiers via a
``tier`` param:

- **easy** -- resultant MAGNITUDE of two forces at a stated non-right angle,
  via the cosine rule: ``R = sqrt(F1^2 + F2^2 + 2*F1*F2*cos(theta))``.
- **medium** -- resultant DIRECTION: same two-force setup, but the unknown is
  the angle the resultant makes with the first force's own direction, via
  the sine rule: ``alpha = asin(F2*sin(theta)/R)``. A different unknown needs
  a different law -- genuinely different reasoning from easy, not a bigger
  number.
- **hard** -- river-crossing drift: a craft crosses a channel of width w at
  its own speed u (straight across); a current c carries it downstream.
  Find the drift when it reaches the far bank (``drift = c*(w/u)``). This
  chains a TIME derived from one perpendicular pair (w, u) into a SECOND,
  independent perpendicular quantity (c) -- genuinely different from a
  single cosine/sine-rule evaluation, and it never combines u and c into a
  magnitude at all, so Pythagoras is not merely wrong here, it is not even
  the right SHAPE of mistake (tested as a distractor anyway, since a learner
  might reach for it out of habit).
- **exam** -- resultant of THREE vectors at stated angles from one common
  reference direction (not all equal, not all perpendicular to each other),
  by full component resolution: ``Fx = sum(Fi*cos(angle_i))``,
  ``Fy = sum(Fi*sin(angle_i))``, ``R = sqrt(Fx^2+Fy^2)``. The two-vector
  cosine/sine-rule shortcuts of easy/medium no longer apply -- a genuinely
  new method.

Every irrational-valued answer (easy, medium, exam) is rounded to a STATED
1 decimal place (``mathgen.core.stated_precision_string``) and guarded
against landing within 1e-6 of the .x5 rounding boundary before being
accepted, re-drawing otherwise (the same discipline
``p12c_exp_log_derivative_apply_build.py`` and ``p8b_trig_rules_build.py``
use). The hard tier constructs its inputs so the drift is an EXACT integer
(``w = u*t``, ``drift = c*t`` for integer crossing time ``t``), preferring
constructing w/u to divide exactly -- so its
``answer_text`` carries no decimal point and is graded EXACTLY.

Independent check: every tier is re-derived via ``sympy``'s own trig at the
EXACT drawn angle (``sympy.cos``/``sympy.sin`` on a ``Rational`` degree value
through ``sympy.pi/180``, evaluated at high precision) -- a route wholly
independent of the ``math.cos``/``math.radians`` construction used to build
the question, per ``check_equivalent``.

Each tier draws from THREE structurally distinct phrasing templates (not
merely a noun/unit swap) so ``test_prompt_variety``'s masked-shape count is
met by genuine sentence-structure variety, not an artefact of the slot-bank
masking.
"""

from __future__ import annotations

import math
from fractions import Fraction

from sympy import Rational, cos as sym_cos, pi as sym_pi, sin as sym_sin, sqrt as sym_sqrt

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_RESAMPLE_ATTEMPTS = 500
_DP = 1
_TOL = stated_precision_tolerance(_DP)

# Angle pool for the TWO-vector tiers (easy, medium): excludes 90 degrees by
# construction -- Pythagoras alone must never answer either tier.
_ANGLE_POOL_2V = (30, 45, 60, 120, 135, 150)

# Angle pool for the exam THREE-vector tier's non-reference vectors. 90 is a
# valid draw here: resolving a sum of three non-collinear vectors by
# component is never reducible to a single Pythagoras step even if one pair
# happens to be perpendicular, unlike the two-vector case.
_ANGLE_POOL_3V = (30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330)

_FORCE_POOL = tuple(range(8, 61))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fmt(value: Fraction) -> str:
    return stated_precision_string(value, _DP)


def _round(value) -> Fraction:
    return round_half_up(Fraction(value).limit_denominator(10**9), _DP)


def _near_rounding_boundary(value: float, dp: int = _DP, eps: float = 1e-6) -> bool:
    """True when `value` sits within `eps` of the .x5 halfway point at `dp`
    decimal places -- the point where this generator's own float arithmetic
    and the independent sympy oracle could legitimately round opposite ways."""
    scaled = value * (10**dp)
    frac = scaled - math.floor(scaled)
    return abs(frac - 0.5) < eps


def _gap_ok(values: list[Fraction], tol: Fraction = _TOL) -> bool:
    """Every pair in `values` (already rounded to the stated precision)
    differs by more than `tol` -- checked directly on the values the real
    grader will see."""
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if abs(values[i] - values[j]) <= tol:
                return False
    return True


def _sym_cos_deg(theta_deg: int):
    return sym_cos(Rational(theta_deg, 1) * sym_pi / 180)


def _sym_sin_deg(theta_deg: int):
    return sym_sin(Rational(theta_deg, 1) * sym_pi / 180)


# ---------------------------------------------------------------------------
# easy: resultant MAGNITUDE of two forces at a non-right angle (cosine rule)
# ---------------------------------------------------------------------------

_EASY_TEMPLATES = (
    "Two forces act on an object: a force of {F1} N and a force of {F2} N, "
    "with an angle of {theta} degrees between their directions. Find the "
    "magnitude of the resultant force, in N, to 1 decimal place.",
    "A crate is pulled by two ropes exerting forces of {F1} N and {F2} N. "
    "The ropes make an angle of {theta} degrees with each other. Calculate "
    "the size of the combined (resultant) force on the crate, in N, correct "
    "to 1 decimal place.",
    "Two tugboats push a barge with forces of {F1} N and {F2} N respectively, "
    "at an angle of {theta} degrees to each other. Determine the magnitude of "
    "the net force on the barge, in N, to 1 decimal place.",
)


def _build_easy(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        F1 = rng.choice(_FORCE_POOL)
        F2 = rng.choice(_FORCE_POOL)
        theta = rng.choice(_ANGLE_POOL_2V)

        cos_t = math.cos(math.radians(theta))
        R_raw = math.sqrt(F1 * F1 + F2 * F2 + 2 * F1 * F2 * cos_t)
        if _near_rounding_boundary(R_raw):
            continue
        R = _round(R_raw)

        # Misconception 1 (uses-pythagoras): ignores theta entirely -- the
        # exact "looks like the shipping neighbour" error.
        pyth = math.sqrt(F1 * F1 + F2 * F2)
        # Misconception 2 (wrong-sign-cosine-rule): subtracts instead of
        # adds the cross term.
        inside = F1 * F1 + F2 * F2 - 2 * F1 * F2 * cos_t
        wrong_sign = math.sqrt(inside) if inside > 0 else None
        # Misconception 3 (adds-the-magnitudes): treats the two forces as
        # collinear.
        added = float(F1 + F2)

        cand_floats = [v for v in (pyth, wrong_sign, added) if v is not None]
        if any(_near_rounding_boundary(v) for v in cand_floats):
            continue
        cands = [_round(v) for v in cand_floats]
        if R <= 0 or not _gap_ok([R] + cands):
            continue
        break
    else:
        raise RuntimeError("p13b_vector_resultant_angle: no collision-free easy draw found")

    check_distinct(R, *cands, msg=f"easy: a distractor equals R (F1={F1}, F2={F2}, theta={theta})")

    # Independent check: sympy's own trig at the exact drawn angle.
    R_sym = sym_sqrt(F1 * F1 + F2 * F2 + 2 * F1 * F2 * _sym_cos_deg(theta))
    R_sym_val = float(R_sym.evalf(30))
    check(abs(R_sym_val - R_raw) < 1e-9, f"easy: sympy route {R_sym_val} disagrees with math route {R_raw}")
    check_equivalent(
        Rational(R.numerator, R.denominator),
        Rational(_round(R_sym_val).numerator, _round(R_sym_val).denominator),
        "easy: sympy-derived rounded answer disagrees with the math-derived one",
    )

    template = rng.choice(_EASY_TEMPLATES)
    text = template.format(F1=F1, F2=F2, theta=theta)
    answer_text = _fmt(R)
    distractors = pick_distractors(answer_text, [_fmt(c) for c in cands], n=3)
    check(len(distractors) >= 1, "easy tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(R),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"R = sqrt({F1}^2 + {F2}^2 + 2·{F1}·{F2}·cos({theta}°)) = {answer_text} N."],
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# medium: resultant DIRECTION of two forces at a non-right angle (sine rule)
# ---------------------------------------------------------------------------

_MEDIUM_TEMPLATES = (
    "Two forces act on an object: a force of {F1} N and a force of {F2} N, "
    "with an angle of {theta} degrees between their directions. Find the "
    "angle, in degrees, that the resultant force makes with the direction "
    "of the {F1} N force, to 1 decimal place.",
    "A crate is pulled by two ropes exerting forces of {F1} N and {F2} N, "
    "making an angle of {theta} degrees with each other. Find the angle, in "
    "degrees, between the combined (resultant) force and the {F1} N rope's "
    "own direction, correct to 1 decimal place.",
    "Two tugboats push a barge with forces of {F1} N and {F2} N respectively, "
    "at an angle of {theta} degrees to each other. Find the angle, in "
    "degrees, between the net force on the barge and the direction of the "
    "{F1} N push, to 1 decimal place.",
)


def _build_medium(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        F1 = rng.choice(_FORCE_POOL)
        F2 = rng.choice(_FORCE_POOL)
        theta = rng.choice(_ANGLE_POOL_2V)

        cos_t = math.cos(math.radians(theta))
        sin_t = math.sin(math.radians(theta))
        x_comp = F1 + F2 * cos_t
        if x_comp <= 1.0:
            # Keep well clear of the branch boundary where the sine rule's
            # principal asin() value stops matching the true geometric angle.
            continue
        y_comp = F2 * sin_t
        R_raw = math.sqrt(x_comp * x_comp + y_comp * y_comp)

        ratio = F2 * sin_t / R_raw
        if not (-0.97 <= ratio <= 0.97):
            continue
        alpha_raw = math.degrees(math.asin(ratio))
        alpha_true = math.degrees(math.atan2(y_comp, x_comp))
        if abs(alpha_raw - alpha_true) > 1e-6:
            continue
        if alpha_raw < 3.0:
            continue
        if _near_rounding_boundary(alpha_raw):
            continue
        alpha = _round(alpha_raw)

        # Misconception 1 (wrong-side-of-sine-rule): swaps the F1/F2 roles
        # in the sine-rule ratio.
        ratio_swap = F1 * sin_t / R_raw
        swap_val = math.degrees(math.asin(ratio_swap)) if -1 < ratio_swap < 1 else None
        # Misconception 2 (halves-the-angle): naive theta/2.
        half_val = theta / 2
        # Misconception 3 (ignores-the-angle / pythagoras-echo): treats
        # F1, F2 as perpendicular legs and reports atan(F2/F1) -- the
        # direction a right-angle-triangle reasoning would give, ignoring
        # theta entirely.
        ignore_val = math.degrees(math.atan2(F2, F1))

        cand_floats = [v for v in (swap_val, half_val, ignore_val) if v is not None]
        if any(_near_rounding_boundary(v) for v in cand_floats):
            continue
        cands = [_round(v) for v in cand_floats]
        if alpha <= 0 or not _gap_ok([alpha] + cands):
            continue
        break
    else:
        raise RuntimeError("p13b_vector_resultant_angle: no collision-free medium draw found")

    check_distinct(alpha, *cands, msg=f"medium: a distractor equals alpha (F1={F1}, F2={F2}, theta={theta})")

    R_sym = sym_sqrt(F1 * F1 + F2 * F2 + 2 * F1 * F2 * _sym_cos_deg(theta))
    R_sym_val = float(R_sym.evalf(30))
    check(abs(R_sym_val - R_raw) < 1e-9, "medium: sympy resultant disagrees with math route")
    sin_alpha_sym = float((F2 * _sym_sin_deg(theta) / R_sym).evalf(30))
    check(abs(sin_alpha_sym - ratio) < 1e-9, "medium: sympy sine-rule ratio disagrees with math route")

    template = rng.choice(_MEDIUM_TEMPLATES)
    text = template.format(F1=F1, F2=F2, theta=theta)
    answer_text = _fmt(alpha)
    distractors = pick_distractors(answer_text, [_fmt(c) for c in cands], n=3)
    check(len(distractors) >= 1, "medium tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(alpha),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"R = sqrt({F1}^2 + {F2}^2 + 2·{F1}·{F2}·cos({theta}°)) N.",
            f"sin(alpha)/{F2} = sin({theta}°)/R, so alpha = asin({round(ratio, 4)}) = {answer_text}°.",
        ],
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# hard: river-crossing drift -- a chained time-then-scale problem, no
# resultant magnitude is ever computed.
# ---------------------------------------------------------------------------

_U_POOL = tuple(range(2, 11))
_T_POOL = tuple(range(1, 6))
_C_POOL = tuple(range(1, 11))

_HARD_TEMPLATES = (
    "A boat's own speed through the water is {u} km/h, directed straight "
    "across a river {w} km wide. The current flows parallel to the bank at "
    "{c} km/h. Find how far downstream the boat has drifted by the time it "
    "reaches the far bank, in km.",
    "A swimmer, swimming at {u} km/h relative to the water, heads straight "
    "across a channel {w} km wide. A current of {c} km/h runs parallel to "
    "the bank. Find the swimmer's downstream drift when they reach the "
    "opposite side, in km.",
    "A ferry crosses a river {w} km wide, travelling straight across at "
    "{u} km/h through the water while a current of {c} km/h flows along the "
    "bank. Find the total sideways drift of the ferry when it reaches the "
    "other bank, in km.",
)


def _build_hard(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        u = rng.choice(_U_POOL)
        t = rng.choice(_T_POOL)
        w = u * t
        c = rng.choice(_C_POOL)

        drift = c * t

        # Misconception 1 (uses-pythagoras): computes the boat's resultant
        # SPEED via Pythagoras and (wrongly) multiplies it by the crossing
        # time, as if drift were the resultant magnitude scaled by time --
        # the exact "looks like the shipping neighbour" error, tested here
        # even though this tier's own reasoning never needs a magnitude.
        pyth_speed = math.sqrt(u * u + c * c)
        # Rounded to the SAME exact-integer grid the answer itself sits on
        # (0 decimal places), so no option is identifiable by carrying a
        # decimal point the others lack.
        cand_pyth = round_half_up(Fraction(pyth_speed * t).limit_denominator(10**9), 0)
        # Misconception 2 (swapped-rates): uses the across-river speed u in
        # place of the current c when scaling by time.
        cand_swapped = Fraction(u * t)
        # Misconception 3 (forgot-the-time): reports the current speed
        # itself, never multiplying by the crossing time at all.
        cand_forgot_time = Fraction(c)

        cands = [cand_pyth, cand_swapped, cand_forgot_time]
        ans = Fraction(drift)
        if ans <= 0 or not _gap_ok([ans] + cands, tol=stated_precision_tolerance(0)):
            continue
        break
    else:
        raise RuntimeError("p13b_vector_resultant_angle: no collision-free hard draw found")

    check_distinct(ans, *cands, msg=f"hard: a distractor equals drift (u={u}, w={w}, c={c})")

    # Independent check: re-derive the crossing time from w/u exactly (a
    # Fraction division, never re-using the constructed t directly), then
    # scale c by that time -- an independent route to the same value.
    t_independent = Fraction(w, u)
    check(t_independent.denominator == 1, f"hard: w/u did not divide exactly ({w}/{u})")
    check_equivalent(
        Rational(int(t_independent) * c),
        Rational(int(ans)),
        "hard: independent w/u*c route disagrees with construction",
    )

    template = rng.choice(_HARD_TEMPLATES)
    text = template.format(u=u, w=w, c=c)
    check(
        all(Fraction(v).denominator == 1 for v in cands),
        "hard: every candidate must sit on the exact-integer grid, same as the answer",
    )
    answer_text = str(int(ans))
    distractors = pick_distractors(answer_text, [str(int(v)) for v in cands], n=3)
    check(len(distractors) >= 1, "hard tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(ans),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Crossing time = {w} / {u} = {t} h.",
            f"Drift = current speed · time = {c} · {t} = {answer_text} km.",
        ],
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# exam: resultant of THREE vectors at stated angles, full component
# resolution -- the two-vector shortcuts no longer apply.
# ---------------------------------------------------------------------------

_EXAM_TEMPLATES = (
    "Three forces act on a point: {F1} N along a fixed reference direction, "
    "{F2} N at {a2} degrees anticlockwise from that direction, and {F3} N "
    "at {a3} degrees anticlockwise from it. Find the magnitude of the "
    "resultant force, in N, to 1 decimal place.",
    "Three ropes pull on a ring: one rope pulls with {F1} N along a fixed "
    "reference direction, a second with {F2} N at {a2} degrees anticlockwise "
    "from that direction, and a third with {F3} N at {a3} degrees "
    "anticlockwise from it. Find the magnitude of the net force on the "
    "ring, in N, to 1 decimal place.",
    "A junction is pulled by three cables: {F1} N along a fixed reference "
    "direction, {F2} N at {a2} degrees anticlockwise from that direction, "
    "and {F3} N at {a3} degrees anticlockwise from it. Find the magnitude "
    "of the combined (resultant) pull on the junction, in N, to 1 decimal "
    "place.",
)


def _components(F, angle_deg):
    return F * math.cos(math.radians(angle_deg)), F * math.sin(math.radians(angle_deg))


def _build_exam(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        F1 = rng.choice(_FORCE_POOL)
        F2 = rng.choice(_FORCE_POOL)
        F3 = rng.choice(_FORCE_POOL)
        a2 = rng.choice(_ANGLE_POOL_3V)
        a3 = rng.choice(_ANGLE_POOL_3V)
        if a2 == a3:
            continue

        x1, y1 = F1, 0.0
        x2, y2 = _components(F2, a2)
        x3, y3 = _components(F3, a3)
        Fx = x1 + x2 + x3
        Fy = y1 + y2 + y3
        R_raw = math.hypot(Fx, Fy)
        if _near_rounding_boundary(R_raw):
            continue
        R = _round(R_raw)

        # Misconception 1 (dropped-a-vector): drops F3 entirely.
        R_drop3 = math.hypot(x1 + x2, y1 + y2)
        # Misconception 2 (component-sign-error): subtracts instead of adds
        # the second vector's y-component.
        R_sign = math.hypot(x1 + x2 + x3, y1 - y2 + y3)
        # Misconception 3 (uses-pythagoras): sums the three MAGNITUDES as if
        # mutually perpendicular, ignoring every angle.
        R_pyth = math.sqrt(F1 * F1 + F2 * F2 + F3 * F3)

        cand_floats = [R_drop3, R_sign, R_pyth]
        if any(_near_rounding_boundary(v) for v in cand_floats):
            continue
        cands = [_round(v) for v in cand_floats]
        if R <= 0 or not _gap_ok([R] + cands):
            continue
        break
    else:
        raise RuntimeError("p13b_vector_resultant_angle: no collision-free exam draw found")

    check_distinct(R, *cands, msg=f"exam: a distractor equals R (F1={F1}, F2={F2}, F3={F3}, a2={a2}, a3={a3})")

    # Independent check: sympy's own trig at the exact drawn angles.
    Fx_sym = F1 + F2 * _sym_cos_deg(a2) + F3 * _sym_cos_deg(a3)
    Fy_sym = F2 * _sym_sin_deg(a2) + F3 * _sym_sin_deg(a3)
    R_sym_val = float(sym_sqrt(Fx_sym**2 + Fy_sym**2).evalf(30))
    check(abs(R_sym_val - R_raw) < 1e-9, f"exam: sympy route {R_sym_val} disagrees with math route {R_raw}")

    template = rng.choice(_EXAM_TEMPLATES)
    text = template.format(F1=F1, F2=F2, F3=F3, a2=a2, a3=a3)
    answer_text = _fmt(R)
    distractors = pick_distractors(answer_text, [_fmt(c) for c in cands], n=3)
    check(len(distractors) >= 1, "exam tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=numeric_answer(R),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Fx = {F1} + {F2}·cos({a2}°) + {F3}·cos({a3}°) = {round(Fx, 4)} N.",
            f"Fy = {F2}·sin({a2}°) + {F3}·sin({a3}°) = {round(Fy, 4)} N.",
            f"R = sqrt(Fx^2 + Fy^2) = {answer_text} N.",
        ],
        distractors=distractors,
    )


@generator(
    "p13b_vector_resultant_angle",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13b_vector_resultant_angle(rng, tier="easy"):
    """Resultant of vectors (forces/displacements/velocities) at a stated
    angle that is NEVER 90 degrees for the two-vector tiers (easy, medium),
    so Pythagoras alone cannot answer them. easy: resultant magnitude via
    the cosine rule. medium: resultant direction via the sine rule. hard:
    river-crossing drift, a chained time-then-scale problem with no
    resultant magnitude at all. exam: resultant of THREE vectors by full
    component resolution."""
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    if tier == "hard":
        return _build_hard(rng)
    if tier == "exam":
        return _build_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
