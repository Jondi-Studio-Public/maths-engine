"""Extend the LIVE `sine-rule` and `cosine-rule` skills with four new pooled sources.

Four new topics, each registered here and each a single reasoning shape (no
internal easy/medium/hard branching -- the *skill* YAML picks which topic to
draw from at which tier, matching `sine_rule`/`cosine_rule`'s own existing
single-shape kinds inside `mathgen/generators/geometry.py`'s `trigonometry`
topic, which this module never imports from):

- ``sine_rule_angle`` -- SSA with a > b (the unambiguous case): given angle A
  and sides a (opposite A), b (opposite B), find angle B.
- ``sine_rule_ambiguous`` -- the genuine two-solution SSA case
  (b sin A < a < b): the question text names which of the two triangles
  (always the OBTUSE one) it wants.
- ``sine_rule_solve`` -- AAS: given A, B and side a, find side c via the
  two-step C = 180 - A - B, then c = a sin(C) / sin(A).
- ``cosine_rule_angle`` -- SSS: given a, b, c, find angle C via
  cos(C) = (a^2 + b^2 - c^2) / (2ab).

Every topic's rejection-and-redraw loop lives entirely inside a `_draw_once_*` function: it draws the raw
geometry, computes every misconception candidate, and returns `None` (redraw) unless the sanity sweep passes.
Every `@generator` function is then a thin `while True` loop plus the independent check and text formatting --
see `tests/test_p8b_trig_rules_build.py` for the "red first" proof that the filter is load-bearing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core import check, check_distinct, generator, pick_distractors, GRADER_TOLERANCE

# "Nice" degree pool for the given angle in the unambiguous SSA case.
SINE_ANGLE_POOL = [30, 35, 40, 45, 50, 55, 60, 65, 70]

# Degree pool for the genuinely ambiguous SSA case -- kept narrow: only A in this range leaves room
# for an obtuse B2 landing inside [30, 150] together with a positive C2 also inside [30, 150].
AMBIGUOUS_ANGLE_POOL = list(range(30, 46))

# Degree pool for the two given angles in the AAS "solve" case.
SOLVE_ANGLE_POOL = list(range(30, 121))


def _rounding_safe(exact: float, eps: float = 0.02) -> bool:
    """False when `exact` sits within `eps` of an "x.x5" 1-d.p. rounding
    boundary, where float noise between this generator's own `round()` and
    an independent oracle's separate computation could round the two ways.
    """
    frac = (exact * 10) % 1
    return abs(frac - 0.5) > eps


def _far_enough(candidate: float | None, answer: float, min_gap: float = 0.051) -> bool:
    """Whether a (still-unrounded) misconception candidate is far enough from
    the true answer that the real `grade_number` grader -- whose typed-answer
    tolerance at 1 stated decimal place is 0.05 -- could never accept it.
    `None` (an out-of-domain candidate) always passes; a caller only offers
    the candidate at all when it is not None. Checked as part of the
    rejection-and-redraw loop (not after it) so an accepted question always
    ships every candidate its misconception list names."""
    if candidate is None:
        return True
    return abs(candidate - answer) > min_gap


# =============================================================================
# Topic 1: sine_rule_angle -- unambiguous SSA (a > b), find angle B.
# =============================================================================


@dataclass
class _SineAngleDraw:
    A: int
    a: int
    b: int
    B: float
    B_exact: float
    C: float
    wrong_side_ratio: float | None
    linear_scale: float
    other_c_branch: float | None
    used_cos: float


def _draw_once_sine_rule_angle(rng) -> _SineAngleDraw | None:
    A = rng.choice(SINE_ANGLE_POOL)
    a = rng.randint(8, 25)
    b = rng.randint(8, 25)
    if not a > b:
        return None
    sinA = math.sin(math.radians(A))
    ratio = b * sinA / a
    if not (0 < ratio <= 0.95):
        return None
    B_exact = math.degrees(math.asin(ratio))
    if not _rounding_safe(B_exact):
        return None
    B = round(B_exact, 1)
    C = round(180 - A - B, 1)
    if not (30 <= A <= 150 and 30 <= B <= 150 and 30 <= C <= 150):
        return None

    # Misconceptions, each guarded to its own valid domain:
    #  - wrong_side_ratio: put a and A on the wrong side of the ratio
    #    (asin(a sin A / b) instead of asin(b sin A / a)).
    #  - linear_scale: never converted the ratio at all, scaled A directly
    #    by the raw side ratio a/b.
    #  - other_c_branch: the arithmetic slip 180 - A - wrong_side_ratio, a
    #    genuinely different value from B rather than a rounding wobble.
    #  - used_cos: applied acos instead of asin to the SAME correct ratio --
    #    always in-domain here since `ratio` is already constrained to
    #    (0, 0.95] above, unlike `wrong_side_ratio`'s swapped ratio.
    wrong_side_ratio = None
    swap_ratio = a * sinA / b
    if 0 < swap_ratio < 1:
        wrong_side_ratio = round(math.degrees(math.asin(swap_ratio)), 1)
    linear_scale = round((a / b) * A, 1)
    other_c_branch = round(180 - A - wrong_side_ratio, 1) if wrong_side_ratio is not None else None
    used_cos = round(math.degrees(math.acos(ratio)), 1)

    for cand in (wrong_side_ratio, linear_scale, other_c_branch, used_cos):
        if not _far_enough(cand, B):
            return None

    return _SineAngleDraw(
        A=A,
        a=a,
        b=b,
        B=B,
        B_exact=B_exact,
        C=C,
        wrong_side_ratio=wrong_side_ratio,
        linear_scale=linear_scale,
        other_c_branch=other_c_branch,
        used_cos=used_cos,
    )


def _verify_sine_rule_angle(draw: _SineAngleDraw) -> None:
    """Independent cross-check via the cosine-rule quadratic for the third
    side, never calling `asin` -- a genuinely different equation from the one
    that produced `draw.B`. Compared against the UNROUNDED `B_exact` (not the
    1-d.p. `B` shown to the learner), matching `geometry.py`'s own comment on
    why: comparing to the rounded value risks float noise crossing the
    rounding boundary."""
    A, a, b, B_exact = draw.A, draw.a, draw.b, draw.B_exact
    cosA = math.cos(math.radians(A))
    sinA = math.sin(math.radians(A))
    disc = a * a - (b * sinA) ** 2
    check(disc > 0, "sine_rule_angle: no real triangle for drawn a, b, A")
    sqrt_disc = math.sqrt(disc)
    positive_roots = [x for x in (b * cosA + sqrt_disc, b * cosA - sqrt_disc) if x > 1e-9]
    check(len(positive_roots) == 1, f"sine_rule_angle: expected a unique triangle for a>b, got {positive_roots}")
    c_side = positive_roots[0]
    B_check = math.degrees(math.acos((a * a + c_side * c_side - b * b) / (2 * a * c_side)))
    check(
        abs(B_check - B_exact) < 1e-6, f"sine_rule_angle: cosine-rule cross-check gives B={B_check}, expected {B_exact}"
    )


@generator("sine_rule_angle", level=11)
def sine_rule_angle(rng):
    while True:
        draw = _draw_once_sine_rule_angle(rng)
        if draw is not None:
            break
    A, a, b, B, C = draw.A, draw.a, draw.b, draw.B, draw.C
    _verify_sine_rule_angle(draw)

    present = [
        c for c in (draw.wrong_side_ratio, draw.linear_scale, draw.other_c_branch, draw.used_cos) if c is not None
    ]
    check_distinct(B, *present, msg=f"sine_rule_angle: a distractor collides with B={B} (A={A}, a={a}, b={b})")

    answer_text = f"B = {B}°"
    candidates = [f"B = {c}°" for c in present]
    sinA = math.sin(math.radians(A))
    return dict(
        text=(
            f"In triangle ABC, angle A = {A}°, side a = {a} and side b = {b} "
            f"(a is opposite A, b is opposite B). Use the sine rule to find "
            f"angle B (1 d.p.)."
        ),
        latex=rf"A = {A}^\circ,\ a = {a},\ b = {b}.\ \text{{Find }} B\ \text{{(1 d.p.)}}",
        answer=B,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"sin(B)/b = sin(A)/a, so sin(B) = {b}·sin({A}°)/{a}.", f"B = sin⁻¹({round(b * sinA / a, 4)}) = {B}°"],
        distractors=pick_distractors(answer_text, candidates, tolerance=GRADER_TOLERANCE),
    )


# =============================================================================
# Topic 2: sine_rule_ambiguous -- the genuine two-solution SSA case.
# =============================================================================


@dataclass
class _SineAmbiguousDraw:
    A: int
    a: int
    b: int
    B1: float
    C1: float
    B2: float
    B2_exact: float
    C2: float
    wrong_ninety_minus: float


def _draw_once_sine_rule_ambiguous(rng) -> _SineAmbiguousDraw | None:
    A = rng.choice(AMBIGUOUS_ANGLE_POOL)
    b = rng.randint(15, 60)
    a = rng.randint(15, 60)
    sinA = math.sin(math.radians(A))
    margin = 0.4
    if not (b * sinA + margin < a < b - margin):
        return None
    ratio = b * sinA / a
    if not (0 < ratio < 1):
        return None
    B1_exact = math.degrees(math.asin(ratio))
    B2_exact = 180 - B1_exact
    if not (_rounding_safe(B1_exact) and _rounding_safe(B2_exact)):
        return None
    B1 = round(B1_exact, 1)
    B2 = round(B2_exact, 1)
    C1 = round(180 - A - B1, 1)
    C2 = round(180 - A - B2, 1)
    # The chosen (obtuse) triangle must be plausible; the rejected (acute)
    # triangle must be a genuinely distinct alternative, not the same
    # triangle by a tangent-case coincidence.
    if not (30 <= A <= 150 and 30 <= B2 <= 150 and 30 <= C2 <= 150):
        return None
    if C1 <= 0 or B1 == B2 or C1 == C2:
        return None

    # Misconceptions: gave the acute solution (B1) when the obtuse one (B2)
    # was asked for; forgot the second solution existed and reported B1
    # unconditionally (same candidate value, different reasoning framing);
    # arithmetic slip computing 90 - B1 instead of 180 - B1.
    wrong_ninety_minus = round(90 - B1, 1)

    for cand in (B1, wrong_ninety_minus):
        if not _far_enough(cand, B2):
            return None

    return _SineAmbiguousDraw(
        A=A, a=a, b=b, B1=B1, C1=C1, B2=B2, B2_exact=B2_exact, C2=C2, wrong_ninety_minus=wrong_ninety_minus
    )


def _verify_sine_rule_ambiguous(draw: _SineAmbiguousDraw) -> None:
    """Independent cross-check via the cosine-rule quadratic for the third
    side of the CHOSEN (obtuse) triangle -- never calls `asin`. Compared
    against the UNROUNDED `B2_exact`, for the same reason as
    `_verify_sine_rule_angle`."""
    A, a, b, B2_exact = draw.A, draw.a, draw.b, draw.B2_exact
    cosA = math.cos(math.radians(A))
    sinA = math.sin(math.radians(A))
    disc = a * a - (b * sinA) ** 2
    check(disc > 0, "sine_rule_ambiguous: no real triangle for drawn a, b, A")
    sqrt_disc = math.sqrt(disc)
    roots = [b * cosA + sqrt_disc, b * cosA - sqrt_disc]
    positive_roots = [x for x in roots if x > 1e-9]
    # The ambiguous case has exactly two valid triangles for this data; the
    # obtuse one is the SMALLER positive root of the two (its third side is
    # shorter, since the obtuse angle B2 leaves less room opposite it than
    # the acute B1 does for the far vertex) -- checked directly below rather
    # than assumed.
    check(len(positive_roots) == 2, f"sine_rule_ambiguous: expected two triangles, got {positive_roots}")
    c_side = min(positive_roots)
    B_check = math.degrees(math.acos((a * a + c_side * c_side - b * b) / (2 * a * c_side)))
    check(
        abs(B_check - B2_exact) < 1e-6,
        f"sine_rule_ambiguous: cosine-rule cross-check gives B={B_check}, expected {B2_exact}",
    )


@generator("sine_rule_ambiguous", level=11)
def sine_rule_ambiguous(rng):
    while True:
        draw = _draw_once_sine_rule_ambiguous(rng)
        if draw is not None:
            break
    A, a, b, B1, B2 = draw.A, draw.a, draw.b, draw.B1, draw.B2
    _verify_sine_rule_ambiguous(draw)

    check_distinct(
        B2,
        B1,
        draw.wrong_ninety_minus,
        msg=f"sine_rule_ambiguous: a distractor collides with B2={B2} (A={A}, a={a}, b={b})",
    )

    answer_text = f"B = {B2}°"
    candidates = [f"B = {B1}°", f"B = {draw.wrong_ninety_minus}°"]
    return dict(
        text=(
            f"In triangle ABC, angle A = {A}°, side a = {a} and side b = {b} "
            f"(a is opposite A, b is opposite B). Two triangles satisfy this "
            f"data. In the OBTUSE-angled triangle, find angle B (1 d.p.)."
        ),
        latex=rf"A = {A}^\circ,\ a = {a},\ b = {b}.\ \text{{Obtuse triangle: find }} B\ \text{{(1 d.p.)}}",
        answer=B2,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"sin(B)/b = sin(A)/a, so sin(B) = {b}·sin({A}°)/{a}, giving two solutions: "
            f"B = {B1}° (acute) or B = {B2}° (obtuse).",
            f"The obtuse-angled triangle wants B = {B2}°.",
        ],
        distractors=pick_distractors(answer_text, candidates, tolerance=GRADER_TOLERANCE),
    )


# =============================================================================
# Topic 3: sine_rule_solve -- AAS "solve the triangle" for one final side.
# =============================================================================


@dataclass
class _SineSolveDraw:
    A: int
    B: int
    a: int
    C: int
    c: float
    swapped_ratio: float
    used_b_instead: float
    proportional_nonsense: float


def _draw_once_sine_rule_solve(rng) -> _SineSolveDraw | None:
    A = rng.choice(SOLVE_ANGLE_POOL)
    B = rng.choice(SOLVE_ANGLE_POOL)
    C = 180 - A - B
    if not (30 <= A <= 150 and 30 <= B <= 150 and 30 <= C <= 150):
        return None
    a = rng.randint(5, 30)
    sinA = math.sin(math.radians(A))
    sinB = math.sin(math.radians(B))
    sinC = math.sin(math.radians(C))
    c_exact = a * sinC / sinA
    if not _rounding_safe(c_exact):
        return None
    c = round(c_exact, 1)
    if c <= 0:
        return None

    # Misconceptions: swapped A and C in the ratio (used a sin(A)/sin(C));
    # used B in place of the derived C (reports the OTHER unknown side, b);
    # forgot the angle-sum step and treated the final side as directly
    # proportional to the two GIVEN angles.
    swapped_ratio = round(a * sinA / sinC, 1)
    used_b_instead = round(a * sinB / sinA, 1)
    proportional_nonsense = round(a * B / A, 1)

    for cand in (swapped_ratio, used_b_instead, proportional_nonsense):
        if not _far_enough(cand, c):
            return None

    return _SineSolveDraw(
        A=A,
        B=B,
        a=a,
        C=C,
        c=c,
        swapped_ratio=swapped_ratio,
        used_b_instead=used_b_instead,
        proportional_nonsense=proportional_nonsense,
    )


def _verify_sine_rule_solve(draw: _SineSolveDraw) -> None:
    """Independent cross-check via the OTHER two-sided sine ratio: derive b
    from a, A, B (a genuinely different pair of angles from the one used to
    derive c), then re-derive c from b, B, C -- a different equation route
    from `a sin(C) / sin(A)`."""
    A, B, a, C, c = draw.A, draw.B, draw.a, draw.C, draw.c
    sinA = math.sin(math.radians(A))
    sinB = math.sin(math.radians(B))
    sinC = math.sin(math.radians(C))
    b_side = a * sinB / sinA
    c_check = b_side * sinC / sinB
    check(abs(c_check - c) < 0.05, f"sine_rule_solve: cross-check via b gives c={c_check}, expected {c}")


@generator("sine_rule_solve", level=11)
def sine_rule_solve(rng):
    while True:
        draw = _draw_once_sine_rule_solve(rng)
        if draw is not None:
            break
    A, B, a, C, c = draw.A, draw.B, draw.a, draw.C, draw.c
    _verify_sine_rule_solve(draw)

    check_distinct(
        c,
        draw.swapped_ratio,
        draw.used_b_instead,
        draw.proportional_nonsense,
        msg=f"sine_rule_solve: a distractor collides with c={c} (A={A}, B={B}, a={a})",
    )

    answer_text = f"c = {c}"
    candidates = [f"c = {draw.swapped_ratio}", f"c = {draw.used_b_instead}", f"c = {draw.proportional_nonsense}"]
    return dict(
        text=(
            f"In triangle ABC, angle A = {A}°, angle B = {B}° and side a = {a} "
            f"(a is opposite A). Solve the triangle to find side c (1 d.p.)."
        ),
        latex=rf"A = {A}^\circ,\ B = {B}^\circ,\ a = {a}.\ \text{{Find }} c\ \text{{(1 d.p.)}}",
        answer=c,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"First find the missing angle: C = 180 - {A} - {B} = {C}°.",
            f"Then apply the sine rule: c = a·sin(C)/sin(A) = {a}·sin({C}°)/sin({A}°) = {c}.",
        ],
        distractors=pick_distractors(answer_text, candidates, tolerance=GRADER_TOLERANCE),
    )


# =============================================================================
# Topic 4: cosine_rule_angle -- SSS, find the angle opposite the third side.
# =============================================================================


@dataclass
class _CosineAngleDraw:
    a: int
    b: int
    c: int
    A: float
    B: float
    C: float
    sign_error: float
    wrong_side: float
    sine_misuse: float | None


def _draw_once_cosine_rule_angle(rng) -> _CosineAngleDraw | None:
    a = rng.randint(5, 20)
    b = rng.randint(5, 20)
    c = rng.randint(5, 20)
    margin = 1
    if not (abs(a - b) + margin < c < a + b - margin):
        return None
    cosA = (b * b + c * c - a * a) / (2 * b * c)
    cosB = (a * a + c * c - b * b) / (2 * a * c)
    cosC = (a * a + b * b - c * c) / (2 * a * b)
    if not (-1 < cosA < 1 and -1 < cosB < 1 and -1 < cosC < 1):
        return None
    C_exact = math.degrees(math.acos(cosC))
    if not _rounding_safe(C_exact):
        return None
    A = round(math.degrees(math.acos(cosA)), 1)
    B = round(math.degrees(math.acos(cosB)), 1)
    C = round(C_exact, 1)
    if not (30 <= A <= 150 and 30 <= B <= 150 and 30 <= C <= 150):
        return None
    # Keep the sign-error distractor (180 - C) comfortably distinct: it
    # collides with C only when C == 90 exactly.
    if abs(C - 90) < 1.0:
        return None

    # Misconceptions: sign error isolating cos(C) (used +2ab instead of
    # -2ab, which flips the sign of cos(C) and so gives 180 - C); used the
    # wrong pair of sides (reported the angle opposite a instead of c);
    # applied the sine-rule ratio to SSS data as if it were valid (treated
    # c/a as sin(C) directly).
    sign_error = round(180 - C, 1)
    wrong_side = A
    sine_misuse = round(math.degrees(math.asin(c / a)), 1) if 0 < c / a < 1 else None

    for cand in (sign_error, wrong_side, sine_misuse):
        if not _far_enough(cand, C):
            return None

    return _CosineAngleDraw(
        a=a, b=b, c=c, A=A, B=B, C=C, sign_error=sign_error, wrong_side=wrong_side, sine_misuse=sine_misuse
    )


def _verify_cosine_rule_angle(draw: _CosineAngleDraw) -> None:
    """Independent cross-check: reconstruct c from a, b and the derived angle
    C via the cosine rule FORWARD -- a different equation from the inverse
    (acos) route used to derive C."""
    a, b, c, C = draw.a, draw.b, draw.c, draw.C
    c_forward = math.sqrt(a * a + b * b - 2 * a * b * math.cos(math.radians(C)))
    check(abs(c_forward - c) < 0.05, f"cosine_rule_angle: forward cosine rule gives c={c_forward}, expected {c}")


@generator("cosine_rule_angle", level=11)
def cosine_rule_angle(rng):
    while True:
        draw = _draw_once_cosine_rule_angle(rng)
        if draw is not None:
            break
    a, b, c, A, B, C = draw.a, draw.b, draw.c, draw.A, draw.B, draw.C
    _verify_cosine_rule_angle(draw)

    present = [cand for cand in (draw.sign_error, draw.wrong_side, draw.sine_misuse) if cand is not None]
    check_distinct(C, *present, msg=f"cosine_rule_angle: a distractor collides with C={C} (a={a}, b={b}, c={c})")

    answer_text = f"C = {C}°"
    candidates = [f"C = {cand}°" for cand in present]
    return dict(
        text=(
            f"Triangle ABC has side a = {a}, side b = {b} and side c = {c}. "
            f"Use the cosine rule to find angle C, the angle opposite side c "
            f"(1 d.p.)."
        ),
        latex=rf"a = {a},\ b = {b},\ c = {c}.\ \text{{Find }} C\ \text{{(1 d.p.)}}",
        answer=C,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"cos(C) = (a² + b² - c²) / (2ab) = ({a}² + {b}² - {c}²) / (2×{a}×{b})",
            f"C = cos⁻¹({round((a * a + b * b - c * c) / (2 * a * b), 4)}) = {C}°",
        ],
        distractors=pick_distractors(answer_text, candidates, tolerance=GRADER_TOLERANCE),
    )
