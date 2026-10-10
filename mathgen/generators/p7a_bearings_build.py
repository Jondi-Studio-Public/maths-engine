"""Generates `bearings_problem` questions: solve a right-angled bearings problem for a distance or an angle.

Solve problems involving bearings.

- ``bearings_problem``
  (``senior-solve-problems-involving-bearings``, ACMEM120, domain senior-mathematics,
  answer: number) -- Solve problems involving bearings, restricted to right-angle
  triangles (legs due-north/south and due-east/west, or bearings 90 degrees apart)
  so this skill never needs the cosine or sine rule (which already exist as
  ``cosine-rule``/``sine-rule``).

Three tiers:
- easy (``back_bearing``): back-bearing of a stated bearing, pure mod-360
  arithmetic, no trigonometry.
- medium (``leg_distance``): a two-leg right-angle journey (legs on two
  perpendicular bearings); find the straight-line distance A-B via
  Pythagoras.
- hard (``leg_bearing``): the same right-angle journey; find the compass
  bearing of the straight line A-B (or B-A), which needs the extra step of
  converting a triangle-relative inverse-tangent angle into an absolute
  compass bearing. Swept across all four ways two perpendicular bearings can
  pair: 000+090, 090+180, 180+270, 270+000.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, generator, pick_distractors, VerificationError

# The four ways two bearings 90 degrees apart (clockwise) can pair: leg 1 runs on the first bearing, leg 2 on the second.
BEARING_PAIRS = [(0, 90), (90, 180), (180, 270), (270, 0)]

LEG_MIN, LEG_MAX = 4, 40


def _format_bearing(deg: int) -> str:
    """Three-digit compass convention, e.g. 45 -> "045deg"."""
    return f"{deg % 360:03d}°"


def _fmt_answer_bearing(n: int) -> str:
    """A bearing answer/distractor option, zero-padded to three digits when it
    is a valid bearing (0..359). A misconception candidate that falls outside
    that range (negative, or >=360 from forgetting to normalise) is left as a
    plain integer -- it is not a valid bearing at all, which is the point of
    that distractor."""
    if 0 <= n < 360:
        return f"{n:03d}"
    return str(n)


# --- easy: back bearing -----------------------------------------------------


def _check_back_bearing(bearing_ab: int, claimed: int) -> None:
    """Independent re-derivation via the reversed unit vector + atan2 --
    a different route from construction's ``(bearing_ab + 180) % 360``
    arithmetic."""
    theta = math.radians(bearing_ab)
    dx, dy = math.sin(theta), math.cos(theta)
    recomputed = round(math.degrees(math.atan2(-dx, -dy))) % 360
    check(
        recomputed == claimed % 360,
        f"back_bearing: recomputed {recomputed} != claimed {claimed} for bearing_ab={bearing_ab}",
    )


def _back_bearing_variant(rng):
    bearing_ab = rng.randint(0, 359)
    correct = (bearing_ab + 180) % 360
    answer_text = _fmt_answer_bearing(correct)
    answer = correct

    question_text = (
        f"The bearing from point A to point B is {_format_bearing(bearing_ab)}. "
        f"What is the bearing from point B to point A?"
    )

    _check_back_bearing(bearing_ab, correct)

    candidates = []
    # Misconception: reports the same bearing unchanged (forgets to reverse).
    candidates.append(_fmt_answer_bearing(bearing_ab))
    # Misconception: confuses the reverse with a perpendicular turn.
    candidates.append(_fmt_answer_bearing((bearing_ab + 90) % 360))
    # Misconception: applies +/-180 without normalising past 0/360.
    if bearing_ab < 180:
        candidates.append(_fmt_answer_bearing(bearing_ab - 180))
    elif bearing_ab > 180:
        candidates.append(_fmt_answer_bearing(bearing_ab + 180))

    distractors = pick_distractors(answer_text, candidates)

    working = [
        f"A bearing and its reverse are 180° apart: back bearing = "
        f"({_format_bearing(bearing_ab)} + 180°) mod 360° = {_format_bearing(correct)}.",
    ]

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


# --- medium: straight-line distance via Pythagoras --------------------------


def _check_leg_distance(L1: int, L2: int, hyp_full: float) -> None:
    """Independent re-derivation: verify the Pythagorean identity on the
    full-precision hypotenuse, then recompute L1 from (hyp, L2) via sqrt --
    a different formula from construction's ``math.hypot(L1, L2)``."""
    residual = abs(L1**2 + L2**2 - hyp_full**2)
    check(residual < 1e-6, f"leg_distance: pythagorean residual {residual} for L1={L1}, L2={L2}")
    recomputed_L1 = math.sqrt(max(hyp_full**2 - L2**2, 0.0))
    check(abs(recomputed_L1 - L1) < 1e-6, f"leg_distance: recomputed L1 {recomputed_L1} != {L1}")


def _leg_distance_variant(rng):
    for _ in range(200):
        b1, b2 = rng.choice(BEARING_PAIRS)
        L1 = rng.randint(LEG_MIN, LEG_MAX)
        L2 = rng.randint(LEG_MIN, LEG_MAX)
        if L1 == L2:
            continue
        hyp = math.hypot(L1, L2)
        # Resample if the hypotenuse sits close enough to a rounding boundary
        # (an "x.x5" value) that a learner's own calculator could round the
        # other way -- the same guard the hard tier uses for degrees, applied
        # here to tenths.
        if abs(hyp * 10 - round(hyp * 10)) > 0.45:
            continue
        answer_text = f"{hyp:.1f}"
        if answer_text.endswith(".0"):
            continue  # avoid the trailing-zero-on-a-triple defect class
        break
    else:
        raise VerificationError("leg_distance: could not sample a non-trailing-zero leg pair")

    answer = float(answer_text)

    question_text = (
        f"Starting at point A, a hiker walks on a bearing of {_format_bearing(b1)} for {L1} km, "
        f"arriving at point M, then turns and walks on a bearing of {_format_bearing(b2)} for {L2} km, "
        f"arriving at point B. Find the straight-line distance from A to B, to 1 decimal place."
    )

    _check_leg_distance(L1, L2, hyp)

    candidates = [
        f"{float(L1 + L2):.1f}",  # adds the two legs instead of Pythagoras
        f"{float(abs(L1 - L2)):.1f}",  # subtracts the two legs
        f"{float(L1):.1f}",  # reports a single leg, forgetting the other
    ]
    distractors = pick_distractors(answer_text, candidates, tolerance=Fraction(1, 20))

    working = [
        f"The two legs meet at a right angle, so AB is the hypotenuse of a right triangle: "
        f"AB = √({L1}² + {L2}²) = √{L1**2 + L2**2} = {answer_text} km.",
    ]

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


# --- hard: resultant bearing -------------------------------------------------


def _check_leg_bearing(b1: int, L1: int, L2: int, claimed_ab: int, claimed_ba: int) -> None:
    """Independent re-derivation via an explicit per-quadrant reference-bearing
    + inverse-tan lookup -- a different route from construction's full 2D
    vector-sum + atan2."""
    alpha = math.degrees(math.atan2(L2, L1))  # triangle's interior angle at A, 0..90
    expected_ab = round((b1 + alpha) % 360) % 360
    check(
        expected_ab == claimed_ab,
        f"leg_bearing: expected forward bearing {expected_ab}, got {claimed_ab} (b1={b1}, L1={L1}, L2={L2})",
    )
    expected_ba = (expected_ab + 180) % 360
    check(
        expected_ba == claimed_ba,
        f"leg_bearing: expected back bearing {expected_ba}, got {claimed_ba}",
    )


def _leg_bearing_variant(rng):
    for _ in range(200):
        b1, b2 = rng.choice(BEARING_PAIRS)
        L1 = rng.randint(LEG_MIN, LEG_MAX)
        L2 = rng.randint(LEG_MIN, LEG_MAX)
        if L1 == L2:
            continue

        theta1, theta2 = math.radians(b1), math.radians(b2)
        dx = L1 * math.sin(theta1) + L2 * math.sin(theta2)
        dy = L1 * math.cos(theta1) + L2 * math.cos(theta2)
        bearing_ab_raw = math.degrees(math.atan2(dx, dy)) % 360

        # Resample if the raw bearing sits close enough to a half-degree
        # boundary that rounding could disagree between construction and
        # the independent check's separately-computed float.
        if abs(bearing_ab_raw - round(bearing_ab_raw)) > 0.48:
            continue
        break
    else:
        raise VerificationError("leg_bearing: could not sample a non-boundary leg pair")

    bearing_ab = round(bearing_ab_raw) % 360
    bearing_ba = (bearing_ab + 180) % 360

    ask_forward = rng.choice([True, False])
    if ask_forward:
        correct = bearing_ab
        direction_phrase = "from A to B"
        other_direction = bearing_ba
    else:
        correct = bearing_ba
        direction_phrase = "from B to A"
        other_direction = bearing_ab

    answer_text = _fmt_answer_bearing(correct)
    answer = correct

    question_text = (
        f"Starting at point A, a surveyor walks on a bearing of {_format_bearing(b1)} for {L1} m, "
        f"arriving at point M, then turns and walks on a bearing of {_format_bearing(b2)} for {L2} m, "
        f"arriving at point B. Find the bearing {direction_phrase}, to the nearest degree."
    )

    _check_leg_bearing(b1, L1, L2, bearing_ab, bearing_ba)

    alpha = math.degrees(math.atan2(L2, L1))
    alpha_rounded = round(alpha)

    # Misconception: reports the raw triangle angle, never converting it to
    # an absolute compass bearing.
    raw_angle = alpha_rounded % 360
    # Misconception: subtracts the triangle angle from the reference bearing
    # instead of adding it (wrong turn direction).
    wrong_ref = round((b1 - alpha) % 360) % 360
    # Misconception: reports the untouched opposite-direction bearing
    # (forward vs back bearing confusion).
    opposite_direction = other_direction
    # Misconception: swaps opposite/adjacent legs in the inverse-tan ratio.
    alpha_swapped = math.degrees(math.atan2(L1, L2))
    swapped_ab = round((b1 + alpha_swapped) % 360) % 360
    swapped_final = (swapped_ab + 180) % 360 if not ask_forward else swapped_ab

    candidates = [
        _fmt_answer_bearing(raw_angle),
        _fmt_answer_bearing(wrong_ref),
        _fmt_answer_bearing(opposite_direction),
        _fmt_answer_bearing(swapped_final),
    ]
    distractors = pick_distractors(answer_text, candidates)

    working = [
        f"The angle inside the triangle at A, between the {_format_bearing(b1)} leg and the "
        f"line to B, is arctan({L2}/{L1}) = {alpha_rounded}°.",
        f"Bearing from A to B = {_format_bearing(b1)} + {alpha_rounded}° = {_format_bearing(bearing_ab)}.",
    ]
    if not ask_forward:
        working.append(f"Bearing from B to A = {_format_bearing(bearing_ab)} + 180° = {_format_bearing(bearing_ba)}.")

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


@generator(
    "bearings_problem",
    level=13,
    presets={
        "easy": dict(variant="back_bearing"),
        "medium": dict(variant="leg_distance"),
        "hard": dict(variant="leg_bearing"),
    },
)
def bearings_problem(rng, variant="back_bearing"):
    """Solve problems involving bearings, restricted to right-angle triangles.

    Every journey's two legs run on bearings 90 degrees apart, so every
    computation is Pythagoras plus a basic sine/cosine/tan ratio -- never the
    cosine or sine rule.
    """
    if variant == "back_bearing":
        return _back_bearing_variant(rng)
    elif variant == "leg_distance":
        return _leg_distance_variant(rng)
    elif variant == "leg_bearing":
        return _leg_bearing_variant(rng)
    else:
        raise ValueError(f"Unknown variant: {variant}")
