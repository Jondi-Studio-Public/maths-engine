"""Arc length/sector area + angle of inclination.

- ``senior_calculate_lengths_of_arcs_and_areas_of_sectors``
  (``senior-calculate-lengths-of-arcs-and-areas-of-sectors``,
  ACMMM033, domain trigonometry, answer: expression) -- calculate lengths of arcs
  and areas of sectors in circles, given bounded fully specified inputs.

- ``senior_examine_the_relationship_between_the_angle_of_inclination``
  (``senior-examine-the-relationship-between-the-angle-of-inclination``,
  ACMMM030, domain trigonometry, answer: expression) -- examine the relationship
  between the angle of inclination of a line and its gradient.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from ..core import check, format_pi_multiple, generator, latex_answer, latex_pi_multiple, pick_distractors


def _make_sympy_parseable(answer_text: str) -> str:
    """Convert an answer_text (with no bare *) to a form sympy can parse.
    E.g. '6pi' -> '6*pi', 'pi/2' -> 'pi/2', 'pi' -> 'pi', '5pi/6' -> '5*pi/6'."""
    # Insert * between digit and pi (e.g., "6pi" -> "6*pi", "5pi/6" -> "5*pi/6")
    return re.sub(r"(\d)(pi)", r"\1*\2", answer_text)


# ============================================================================
# Skill 1: senior_calculate_lengths_of_arcs_and_areas_of_sectors
# ============================================================================


@generator(
    "arc_length_sector_area",
    level=12,
    presets={
        "easy": dict(variant="arc_length"),
        "medium": dict(variant="sector_area"),
        "hard": dict(variant="find_radius"),
    },
)
def arc_length_sector_area(rng, variant="arc_length"):
    """Calculate arc lengths, sector areas, or recover the radius from an arc."""

    # Angle pool: (numerator, denominator) pairs for theta = Fraction(num, den) * pi
    angle_pool = [(1, 6), (1, 4), (1, 3), (1, 2), (2, 3), (3, 4), (5, 6)]

    # Radius pool
    radius_pool = range(2, 12)

    if variant == "arc_length":
        # Easy: forward arc length calculation
        # s = r * theta
        r = rng.choice(radius_pool)
        num, den = rng.choice(angle_pool)
        theta_coeff = Fraction(num, den)
        theta_str = format_pi_multiple(theta_coeff)

        s_coeff = r * theta_coeff
        s_str = format_pi_multiple(s_coeff)

        question_text = (
            f"A circle has radius {r} cm. An arc subtends an angle of {theta_str} "
            f"radians at the centre. Find the exact length of the arc, in cm."
        )
        answer_text = s_str
        answer_latex = latex_pi_multiple(s_coeff)
        # For the answer field, use explicit multiplication for sympy parsing
        answer_for_grader = _make_sympy_parseable(s_str)

        # Independent check: floating-point cross-check
        float_s = float(s_coeff) * math.pi
        float_check = r * (num / den) * math.pi
        check(abs(float_s - float_check) < 1e-9, f"arc_length: float check failed for r={r}, theta={theta_coeff}")

        # Distractors
        # halves: s_coeff / 2 (uses the sector-area formula's 1/2 factor on arc length)
        # Always distinct since s_coeff != 0 for every r>=2, theta_coeff>0
        halves = s_coeff / 2
        halves_str = format_pi_multiple(halves)
        check(halves != s_coeff, f"halves distractor equals answer for r={r}, theta={theta_coeff}")

        # doubles: s_coeff * 2 (uses diameter instead of radius)
        # Always distinct since s_coeff != 0
        doubles = s_coeff * 2
        doubles_str = format_pi_multiple(doubles)
        check(doubles != s_coeff, f"doubles distractor equals answer for r={r}, theta={theta_coeff}")

        candidates = [halves_str, doubles_str]
        distractors = pick_distractors(answer_text, candidates)

    elif variant == "sector_area":
        # Medium: forward sector area calculation
        # A = (1/2) * r * r * theta
        r = rng.choice(radius_pool)
        num, den = rng.choice(angle_pool)
        theta_coeff = Fraction(num, den)
        theta_str = format_pi_multiple(theta_coeff)

        a_coeff = Fraction(1, 2) * r * r * theta_coeff
        a_str = format_pi_multiple(a_coeff)

        question_text = (
            f"A circle has radius {r} cm. A sector has an angle of {theta_str} "
            f"radians at the centre. Find the exact area of the sector, in cm^2."
        )
        answer_text = a_str
        answer_latex = latex_pi_multiple(a_coeff)
        # For the answer field, use explicit multiplication for sympy parsing
        answer_for_grader = _make_sympy_parseable(a_str)

        # Independent check: floating-point cross-check
        float_a = float(a_coeff) * math.pi
        float_check = 0.5 * r * r * (num / den) * math.pi
        check(abs(float_a - float_check) < 1e-9, f"sector_area: float check failed for r={r}, theta={theta_coeff}")

        # Distractors
        # drops_half: a_coeff * 2 (forgets the 1/2 factor)
        # Always distinct since a_coeff != 0
        drops_half = a_coeff * 2
        drops_half_str = format_pi_multiple(drops_half)
        check(drops_half != a_coeff, f"drops_half distractor equals answer for r={r}, theta={theta_coeff}")

        # uses_diameter: a_coeff * 4 (uses diameter squared in place of radius squared)
        # Always distinct since 4*a_coeff != a_coeff for nonzero a_coeff
        uses_diameter = a_coeff * 4
        uses_diameter_str = format_pi_multiple(uses_diameter)
        check(uses_diameter != a_coeff, f"uses_diameter distractor equals answer for r={r}, theta={theta_coeff}")

        candidates = [drops_half_str, uses_diameter_str]
        distractors = pick_distractors(answer_text, candidates)

    elif variant == "find_radius":
        # Hard: inverse -- recover radius from arc length and angle
        # Given s = r * theta, find r
        # Answer: r = s / theta
        r = rng.choice(radius_pool)
        num, den = rng.choice(angle_pool)
        theta_coeff = Fraction(num, den)
        theta_str = format_pi_multiple(theta_coeff)

        # s_coeff is the coefficient of pi in the given arc length
        s_coeff = r * theta_coeff
        s_str = format_pi_multiple(s_coeff)

        question_text = (
            f"An arc of length {s_str} cm is subtended by an angle of {theta_str} "
            f"radians at the centre of a circle. Find the radius of the circle, in cm."
        )
        answer_text = str(r)
        answer_latex = answer_text
        answer_for_grader = answer_text

        # Independent check: recompute r via division (different route)
        r_check = s_coeff / theta_coeff
        check(r_check == r, f"find_radius: r_check={r_check} != r={r} for s={s_coeff}, theta={theta_coeff}")

        # Distractors
        # multiplies: s_coeff * theta_coeff (multiplies instead of divides)
        # Equals r * theta_coeff^2, always < r since 0 < theta_coeff < 1 for all pool members
        multiplies = s_coeff * theta_coeff
        multiplies_str = str(multiplies)
        check(multiplies != r, f"multiplies distractor equals answer for s={s_coeff}, theta={theta_coeff}")

        # doubles: 2 * r_check = 2 * r (double-counts)
        # Always distinct since 2r != r for r > 0
        doubles = 2 * r
        doubles_str = str(doubles)
        check(doubles != r, f"doubles distractor equals answer for r={r}")

        candidates = [multiplies_str, doubles_str]
        distractors = pick_distractors(answer_text, candidates)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer_for_grader,
        "answer_text": answer_text,
        "answer_latex": answer_latex,
        "distractors": distractors,
    }


# ============================================================================
# Skill 2: senior_examine_the_relationship_between_the_angle_of_inclination
# ============================================================================

# Lookup tables for angle of inclination
ANGLE_TO_GRADIENT = {
    30: "sqrt(3)/3",
    45: "1",
    60: "sqrt(3)",
    120: "-sqrt(3)",
    135: "-1",
    150: "-sqrt(3)/3",
}
GRADIENT_TO_ANGLE = {v: k for k, v in ANGLE_TO_GRADIENT.items()}
SIN_AT = {30: "1/2", 45: "sqrt(2)/2", 60: "sqrt(3)/2"}
COS_AT = {30: "sqrt(3)/2", 45: "sqrt(2)/2", 60: "1/2"}


@generator(
    "angle_of_inclination_gradient",
    level=12,
    presets={
        "easy": dict(variant="forward"),
        "medium": dict(variant="inverse"),
        "hard": dict(variant="from_points"),
    },
)
def angle_of_inclination_gradient(rng, variant="forward"):
    """Examine the relationship between angle of inclination and gradient."""

    if variant == "forward":
        # Easy: direct angle -> gradient lookup (acute angles only)
        theta = rng.choice([30, 45, 60])
        answer_text = ANGLE_TO_GRADIENT[theta]

        question_text = (
            f"A line makes an angle of {theta} degrees with the positive x-axis "
            f"(its angle of inclination). Find the gradient of the line as an exact value."
        )

        # Independent check: re-lookup from the fixed dict (this is a lookup table skill)
        check(
            answer_text in ANGLE_TO_GRADIENT.values(), f"forward: answer {answer_text} not in ANGLE_TO_GRADIENT values"
        )
        check(ANGLE_TO_GRADIENT[theta] == answer_text, f"forward: lookup mismatch for theta={theta}")

        # Distractors (always distinct from answer and each other)
        # sin_distractor: used sine instead of tangent
        sin_distractor = SIN_AT[theta]
        check(sin_distractor != answer_text, f"forward: sin distractor equals answer for theta={theta}")

        # cos_distractor: used cosine instead of tangent
        # Note: For theta=45, sin(45) == cos(45) == sqrt(2)/2
        cos_distractor = COS_AT[theta]
        check(cos_distractor != answer_text, f"forward: cos distractor equals answer for theta={theta}")

        # Additional distractors to ensure 2+ distinct ones even when sin==cos
        # Use tan values of other standard angles
        if theta == 30:
            extra_distractor_1 = ANGLE_TO_GRADIENT[60]  # tan(60)
            extra_distractor_2 = ANGLE_TO_GRADIENT[45]  # tan(45)
        elif theta == 45:
            extra_distractor_1 = ANGLE_TO_GRADIENT[30]  # tan(30)
            extra_distractor_2 = ANGLE_TO_GRADIENT[60]  # tan(60)
        else:  # theta == 60
            extra_distractor_1 = ANGLE_TO_GRADIENT[30]  # tan(30)
            extra_distractor_2 = ANGLE_TO_GRADIENT[45]  # tan(45)

        candidates = [sin_distractor, cos_distractor, extra_distractor_1, extra_distractor_2]
        distractors = pick_distractors(answer_text, candidates)

    elif variant == "inverse":
        # Medium: gradient -> angle lookup (all six angles/both signs)
        m_str = rng.choice(list(GRADIENT_TO_ANGLE))
        theta = GRADIENT_TO_ANGLE[m_str]
        answer_text = str(theta)

        question_text = (
            f"A line has gradient {m_str}. Find its angle of inclination, in degrees, "
            f"measured from the positive x-axis, where 0 <= angle < 180."
        )

        # Independent check: re-derive theta via a freshly-constructed reverse mapping
        fresh_reverse = {v: k for k, v in ANGLE_TO_GRADIENT.items()}
        theta_check = fresh_reverse[m_str]
        check(theta_check == theta, f"inverse: theta_check={theta_check} != theta={theta} for m={m_str}")

        # Distractors
        # supplementary_angle: 180 - theta
        # Always distinct since no angle in the set is 90
        supplementary = 180 - theta
        supplementary_str = str(supplementary)
        check(supplementary != theta, f"inverse: supplementary angle equals answer for theta={theta}")

        # mistakes_gradient_for_angle: m_str reported as if it were the degree value
        # Always distinct: magnitude of every gradient value is <= sqrt(3) ~= 1.73,
        # every angle in the set is >= 30, so they never coincide numerically
        # State this argument explicitly
        gradient_as_angle = m_str
        check(
            gradient_as_angle != answer_text,
            f"inverse: gradient-as-angle distractor equals answer for m={m_str}, theta={theta}",
        )

        candidates = [supplementary_str, gradient_as_angle]
        distractors = pick_distractors(answer_text, candidates)

    elif variant == "from_points":
        # Hard: derive gradient from coordinates first, then look up angle
        # Restricted to rational-slope members (45/135, gradient +-1)
        # because integer coordinates give rational rise/run
        theta = rng.choice([45, 135])
        m = 1 if theta == 45 else -1

        x1, y1 = rng.randint(-8, 8), rng.randint(-8, 8)
        run = rng.randint(1, 6)
        rise = run * m
        x2, y2 = x1 + run, y1 + rise

        answer_text = str(theta)

        question_text = (
            f"A line passes through the points ({x1}, {y1}) and ({x2}, {y2}). "
            f"Find its angle of inclination, in degrees, measured from the positive x-axis, "
            f"where 0 <= angle < 180."
        )

        # Independent check: recompute gradient from points via Fraction (different route)
        m_check = Fraction(y2 - y1, x2 - x1)
        check(m_check in (1, -1), f"from_points: m_check={m_check} not in (1, -1)")
        theta_check = 45 if m_check == 1 else 135
        check(theta_check == theta, f"from_points: theta_check={theta_check} != theta={theta}")

        # Distractors (same two families as medium, applied to this seed's theta)
        # supplementary_angle: 180 - theta
        # Always the other member of {45, 135}, always distinct from this seed's theta
        supplementary = 180 - theta
        supplementary_str = str(supplementary)
        check(supplementary != theta, f"from_points: supplementary angle equals answer")

        # gradient_as_angle: m reported as the angle ("1" or "-1")
        # Always distinct from 45/135 by magnitude
        gradient_as_angle = str(m)
        check(
            gradient_as_angle != answer_text,
            f"from_points: gradient-as-angle distractor equals answer for m={m}, theta={theta}",
        )

        candidates = [supplementary_str, gradient_as_angle]
        distractors = pick_distractors(answer_text, candidates)

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer_text,  # For angle questions, answer_text is already in correct form
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text),
        "distractors": distractors,
    }
