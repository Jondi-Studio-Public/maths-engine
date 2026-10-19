"""Apply cosine and sine ratios to find unknowns.

- ``senior_apply_the_cosine_and_sine_ratios_to_find``
  (``senior-apply-the-cosine-and-sine-ratios-to-find``, ACMEM119, domain senior-mathematics,
  answer: number) -- Apply the cosine and sine ratios to find unknown angles and
  sides in right-angled triangles.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, generator, pick_distractors, VerificationError


# Pythagorean triples (leg, leg, hyp) for medium tier
PYTHAGOREAN_TRIPLES = [
    (3, 4, 5),
    (5, 12, 13),
    (8, 15, 17),
    (7, 24, 25),
    (20, 21, 29),
    (9, 40, 41),
    (12, 35, 37),
    (28, 45, 53),
    (11, 60, 61),
]

# Theta values excluding the 40..50 band (45-degree collision guard)
THETA_RANGE = [t for t in range(10, 81) if not (40 <= t <= 50)]


def _tolerance_1dp() -> Fraction:
    """Tolerance for 1 decimal place: 0.05"""
    return Fraction(1, 20)


def _tolerance_integer() -> Fraction:
    """Tolerance for integer: 0 (exact match required)"""
    return Fraction(0)


def _values_differ_enough(v1_str: str, v2_str: str, tol: Fraction) -> bool:
    """Check if two string values differ by more than tolerance."""
    try:
        v1 = float(v1_str)
        v2 = float(v2_str)
        return abs(v1 - v2) > float(tol)
    except:
        return False


def _check_side_from_angle(hyp: float, theta_deg: int, queried_leg_str: str, variant: str) -> None:
    """Independent check for side_from_angle using Pythagorean theorem.

    Construction computes one leg directly. Check computes both legs via sin/cos,
    then verifies they satisfy Pythagoras, proving triangle consistency.
    """
    theta_rad = math.radians(theta_deg)

    # Compute both legs independently
    opp_check = hyp * math.sin(theta_rad)
    adj_check = hyp * math.cos(theta_rad)

    # Pythagorean check: opp^2 + adj^2 = hyp^2
    pythag_residual = abs(opp_check**2 + adj_check**2 - hyp**2)
    check(
        pythag_residual < 1e-6,
        f"side_from_angle/{variant}: Pythagorean residual {pythag_residual} for H={hyp}, theta={theta_deg}",
    )

    # Check the queried leg matches (compare formatted strings for tolerance)
    expected = opp_check if variant == "opposite" else adj_check
    expected_str = f"{expected:.1f}"
    check(
        queried_leg_str == expected_str,
        f"side_from_angle/{variant}: queried_leg {queried_leg_str} doesn't match expected {expected_str}",
    )


def _check_angle_from_sides(leg: float, hyp: float, theta_deg_str: str) -> None:
    """Independent check for angle_from_sides using atan2.

    Construction computes theta via asin/acos. Check independently derives
    the other leg via Pythagoras, then recomputes theta via atan2 (different
    trig function), and verifies agreement.
    """
    theta_deg = int(theta_deg_str)

    # Derive the other leg via Pythagoras (no trig)
    other_leg_check = math.sqrt(hyp**2 - leg**2)

    # The given leg could be opposite or adjacent. We know theta, so:
    # if sin(theta) = leg / hyp, then leg is opposite, and opposite_leg = leg, adjacent_leg = other_leg_check
    # if cos(theta) = leg / hyp, then leg is adjacent, and opposite_leg = other_leg_check, adjacent_leg = leg

    # Try both interpretations and see which matches theta (use integer rounding for comparison)
    theta_rad_1 = math.atan2(leg, other_leg_check)
    theta_rad_2 = math.atan2(other_leg_check, leg)

    theta_computed_1_int = round(math.degrees(theta_rad_1))
    theta_computed_2_int = round(math.degrees(theta_rad_2))

    # One of these should match theta_deg
    match_1 = theta_computed_1_int == theta_deg
    match_2 = theta_computed_2_int == theta_deg

    check(
        match_1 or match_2,
        f"angle_from_sides: atan2 recomputation {theta_computed_1_int} or {theta_computed_2_int} doesn't match {theta_deg}",
    )


def _check_chained(adj: float, theta_deg: int, opposite_full: float, H: float) -> None:
    """Independent check for chained using Pythagorean theorem.

    Construction computes H via cosine, then opposite via sine. Check verifies
    the final triangle (adj, opposite, H) satisfies Pythagoras using no trig,
    then separately checks opposite against sqrt(H^2 - adj^2).
    """

    # Verify triangle is consistent with Pythagoras (using full precision values)
    pythag_residual = abs(adj**2 + opposite_full**2 - H**2)
    check(pythag_residual < 1e-6, f"chained: Pythagorean residual {pythag_residual} for adj={adj}, theta={theta_deg}")

    # Independently compute opposite via Pythagoras (no trig)
    opposite_via_pythag = math.sqrt(H**2 - adj**2)
    check(
        abs(opposite_full - opposite_via_pythag) < 1e-6,
        f"chained: opposite {opposite_full} doesn't match Pythagoras-derived {opposite_via_pythag}",
    )


@generator(
    "sine_cosine_ratio_triangle",
    level=13,
    presets={
        "easy": dict(variant="side_from_angle"),
        "medium": dict(variant="angle_from_sides"),
        "hard": dict(variant="chained"),
    },
)
def sine_cosine_ratio_triangle(rng, variant="side_from_angle"):
    """Apply sine and cosine ratios to find unknown sides and angles in right-angled triangles.

    This skill uses only sine and cosine ratios. Tangent is never used.
    Every triangle is fully specified in prose (no diagram).
    """

    if variant == "side_from_angle":
        # Easy: Given hypotenuse H and acute angle theta, find one leg (opposite or adjacent)

        # Sample theta from the restricted range (exclude 40..50)
        theta = rng.choice(THETA_RANGE)
        theta_rad = math.radians(theta)

        # Sample hypotenuse: integer, 5..50
        hyp = rng.randint(5, 50)

        # Choose whether to ask for opposite or adjacent
        ask_opposite = rng.choice([True, False])

        if ask_opposite:
            # Ask for opposite side
            opposite = hyp * math.sin(theta_rad)
            answer_text = f"{opposite:.1f}"
            which_side = "opposite"
            side_name = "opposite"
        else:
            # Ask for adjacent side
            adjacent = hyp * math.cos(theta_rad)
            answer_text = f"{adjacent:.1f}"
            which_side = "adjacent"
            side_name = "adjacent"

        answer = float(answer_text)

        # Build question text
        question_text = (
            f"A right-angled triangle has a hypotenuse of {hyp} units and one acute angle of {theta} degrees. "
            f"Find the length of the {side_name} side to this angle, to 1 decimal place."
        )

        # Independent check: Pythagorean verification
        _check_side_from_angle(hyp, theta, answer_text, which_side)

        # Build distractors: only include those that differ from correct answer after rounding
        candidates = []
        tol = _tolerance_1dp()

        # Misconception 1: swaps sine and cosine
        if ask_opposite:
            swapped = hyp * math.cos(theta_rad)
        else:
            swapped = hyp * math.sin(theta_rad)
        swapped_str = f"{swapped:.1f}"
        if _values_differ_enough(answer_text, swapped_str, tol):
            candidates.append(swapped_str)

        # Misconception 2: reads hypotenuse as the leg
        hyp_str = f"{hyp:.1f}"
        if _values_differ_enough(answer_text, hyp_str, tol):
            candidates.append(hyp_str)

        # Misconception 3: uses tangent (note: only applies to opposite case)
        if ask_opposite:
            tangent = hyp * math.tan(theta_rad)
            tangent_str = f"{tangent:.1f}"
            if _values_differ_enough(answer_text, tangent_str, tol):
                candidates.append(tangent_str)

        # Pick distractors from available candidates
        distractors = pick_distractors(answer_text, candidates)

        working = [
            f"The {side_name} side is {'opposite' if ask_opposite else 'adjacent'} to the {theta}° angle.",
            f"Using {'sine' if ask_opposite else 'cosine'}: {side_name} = {hyp} × {'sin' if ask_opposite else 'cos'}({theta}°) = {answer_text}",
        ]

    elif variant == "angle_from_sides":
        # Medium: Given one leg and hypotenuse (Pythagorean triple), find the angle

        # Pick a Pythagorean triple
        a, b, c = rng.choice(PYTHAGOREAN_TRIPLES)

        # Choose which leg is given: a (opposite) or b (opposite)
        given_opposite = rng.choice([True, False])

        if given_opposite:
            given_leg = a
            other_leg = b
        else:
            given_leg = b
            other_leg = a

        hyp = c

        # Compute the angle
        # The angle asked for: the one the given opposite leg is opposite to (sine), or the one at the end of the
        # hypotenuse that the given adjacent leg touches (cosine).
        if given_opposite:
            theta_rad = math.asin(given_leg / hyp)
        else:
            theta_rad = math.acos(given_leg / hyp)
        theta_deg = math.degrees(theta_rad)
        answer_text = str(round(theta_deg))
        answer = int(answer_text)

        # Build question text
        direction = "opposite" if given_opposite else "adjacent"
        if given_opposite:
            which_angle = "Find the acute angle that the opposite side is opposite to"
        else:
            which_angle = "Find the acute angle at the vertex where the adjacent side meets the hypotenuse"
        question_text = (
            f"A right-angled triangle has an {direction} side of {given_leg} units and a hypotenuse of {hyp} units. "
            f"{which_angle}, to the nearest degree."
        )

        # Independent check: atan2-based recomputation
        _check_angle_from_sides(given_leg, hyp, answer_text)

        # Build distractors: only include those distinct after rounding
        candidates = []
        tol = _tolerance_integer()

        # Misconception 1: swaps opposite and adjacent (uses the other of sine and cosine)
        if given_opposite:
            swapped = math.degrees(math.acos(given_leg / hyp))
        else:
            swapped = math.degrees(math.asin(given_leg / hyp))
        swapped_str = str(round(swapped))
        if swapped_str != answer_text:
            candidates.append(swapped_str)

        # Misconception 2: forgets to invert - just reports the ratio
        ratio = given_leg / hyp
        ratio_str = str(round(ratio))
        if ratio_str != answer_text:
            candidates.append(ratio_str)

        # Misconception 3: uses atan2 on both legs (different trig function, but not independent enough)
        # Actually, atan2(opp, adj) should give the same angle, so this isn't really a distractor
        # Instead, use: reports the complementary angle (90 - theta)
        complementary = 90 - answer
        complementary_str = str(complementary)
        if complementary_str != answer_text:
            candidates.append(complementary_str)

        # Pick distractors
        distractors = pick_distractors(answer_text, candidates)

        working = [
            (
                f"The {direction} side is opposite the angle, so sin(angle) = {given_leg}/{hyp}: "
                f"angle = arcsin({given_leg}/{hyp}) = arcsin({given_leg / hyp:.3f}) = {answer}°"
                if given_opposite
                else f"The {direction} side touches the angle, so cos(angle) = {given_leg}/{hyp}: "
                f"angle = arccos({given_leg}/{hyp}) = arccos({given_leg / hyp:.3f}) = {answer}°"
            ),
        ]

    elif variant == "chained":
        # Hard: Given adjacent side and angle, find opposite side via hypotenuse

        # Sample theta from restricted range
        theta = rng.choice(THETA_RANGE)
        theta_rad = math.radians(theta)

        # Sample adjacent side: integer, 3..40
        adj = rng.randint(3, 40)

        # Compute hypotenuse using cosine: H = adj / cos(theta)
        hyp = adj / math.cos(theta_rad)

        # Compute opposite using sine: opp = H * sin(theta)
        opposite = hyp * math.sin(theta_rad)

        answer_text = f"{opposite:.1f}"
        answer = float(answer_text)

        # Build question text
        question_text = (
            f"A right-angled triangle has an adjacent side of {adj} units at an acute angle of {theta} degrees. "
            f"Find the opposite side to this angle, to 1 decimal place."
        )

        # Independent check: Pythagorean verification (use full-precision opposite)
        _check_chained(adj, theta, opposite, hyp)

        # Build distractors: only include those distinct after rounding
        candidates = []
        tol = _tolerance_1dp()

        # Misconception 1: skips the hypotenuse step - applies sine directly to adjacent
        skip_hyp = adj * math.sin(theta_rad)
        skip_hyp_str = f"{skip_hyp:.1f}"
        if _values_differ_enough(answer_text, skip_hyp_str, tol):
            candidates.append(skip_hyp_str)

        # Misconception 2: uses tangent directly (mathematically correct, but violates skill constraint)
        # In practice, this collides with correct answer after rounding in many cases, so it's only
        # included if it clears the tolerance check
        tangent = adj * math.tan(theta_rad)
        tangent_str = f"{tangent:.1f}"
        if _values_differ_enough(answer_text, tangent_str, tol):
            candidates.append(tangent_str)

        # Misconception 3: stops after computing hypotenuse (reports H instead of opposite)
        hyp_str = f"{hyp:.1f}"
        if _values_differ_enough(answer_text, hyp_str, tol):
            candidates.append(hyp_str)

        # Pick distractors
        distractors = pick_distractors(answer_text, candidates)

        working = [
            f"First, find the hypotenuse: H = {adj} / cos({theta}°) = {hyp:.4f}",
            f"Then, find the opposite: opposite = {hyp:.4f} × sin({theta}°) = {answer_text}",
        ]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
