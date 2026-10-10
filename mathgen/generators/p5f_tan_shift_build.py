"""Generators for tan(x+c) and co-function identities (sin, cos shifts).

Three difficulty tiers:
- easy: apply sin(x + pi/2) = cos(x)
- medium: apply cos(x - pi/2) = sin(x)
- hard: find the smallest positive vertical asymptote of y = tan(x + c)
"""

from __future__ import annotations

from fractions import Fraction
import math

from ..core import check, format_pi_multiple, generator, latex_answer, pick_distractors


THETA_POOL = [(1, 6), (1, 4), (1, 3), (1, 2), (2, 3), (3, 4), (5, 6)]

COS_STR = {
    (1, 6): "sqrt(3)/2",
    (1, 4): "sqrt(2)/2",
    (1, 3): "1/2",
    (1, 2): "0",
    (2, 3): "-1/2",
    (3, 4): "-sqrt(2)/2",
    (5, 6): "-sqrt(3)/2",
}
SIN_STR = {
    (1, 6): "1/2",
    (1, 4): "sqrt(2)/2",
    (1, 3): "sqrt(3)/2",
    (1, 2): "1",
    (2, 3): "sqrt(3)/2",
    (3, 4): "sqrt(2)/2",
    (5, 6): "1/2",
}

NEIGHBOR = {
    (1, 6): (1, 4),
    (1, 4): (1, 3),
    (1, 3): (1, 2),
    (1, 2): (2, 3),
    (2, 3): (3, 4),
    (3, 4): (5, 6),
    (5, 6): (1, 6),
}


def _negate_str(s: str) -> str:
    """Negate a string representation of a value."""
    if s == "0":
        return "0"
    return s[1:] if s.startswith("-") else "-" + s


@generator(
    "tan_shift_complementary_identity",
    level=12,
    presets={
        "easy": dict(variant="sin_shift"),
        "medium": dict(variant="cos_shift"),
        "hard": dict(variant="tan_asymptote"),
    },
)
def tan_shift_complementary_identity(rng, variant="sin_shift"):
    """Apply co-function identities or tan shift to evaluate exact trig values."""

    if variant == "sin_shift":
        # Easy: apply sin(x + pi/2) = cos(x)
        theta = rng.choice(THETA_POOL)
        theta_str = format_pi_multiple(Fraction(*theta))
        answer_text = COS_STR[theta]

        text = f"Using the identity sin(x + pi/2) = cos(x), evaluate sin({theta_str} + pi/2) exactly."

        # Independent check via direct numeric verification
        theta_rad = (theta[0] / theta[1]) * math.pi
        check(
            abs(math.sin(theta_rad + math.pi / 2) - math.cos(theta_rad)) < 1e-9,
            f"sin({theta_str} + pi/2) should equal cos({theta_str})",
        )

        # Distractor candidates
        distractor_1 = SIN_STR[theta]  # Forgets identity, evaluates sin(theta)
        distractor_2 = _negate_str(COS_STR[theta])  # Sign error on cos value
        other = NEIGHBOR[theta]
        distractor_3 = COS_STR[other]  # Uses co-function value of neighbouring angle
        distractor_4 = SIN_STR[other]  # Uses sine value of neighbouring angle

        distractors = pick_distractors(answer_text, [distractor_1, distractor_2, distractor_3, distractor_4])

    elif variant == "cos_shift":
        # Medium: apply cos(x - pi/2) = sin(x)
        theta = rng.choice(THETA_POOL)
        theta_str = format_pi_multiple(Fraction(*theta))
        answer_text = SIN_STR[theta]

        text = f"Using the identity cos(x - pi/2) = sin(x), evaluate cos({theta_str} - pi/2) exactly."

        # Independent check via direct numeric verification
        theta_rad = (theta[0] / theta[1]) * math.pi
        check(
            abs(math.cos(theta_rad - math.pi / 2) - math.sin(theta_rad)) < 1e-9,
            f"cos({theta_str} - pi/2) should equal sin({theta_str})",
        )

        # Distractor candidates
        distractor_1 = COS_STR[theta]  # Forgets identity, evaluates cos(theta)
        distractor_2 = _negate_str(SIN_STR[theta])  # Sign error on sin value
        other = NEIGHBOR[theta]
        distractor_3 = SIN_STR[other]  # Uses sine value of neighbouring angle
        distractor_4 = COS_STR[other]  # Uses co-function value of neighbouring angle

        distractors = pick_distractors(answer_text, [distractor_1, distractor_2, distractor_3, distractor_4])

    elif variant == "tan_asymptote":
        # Hard: find smallest positive x where tan(x+c) is undefined
        c = rng.choice(THETA_POOL)
        c_coeff = Fraction(*c)
        c_str = format_pi_multiple(c_coeff)

        text = f"For y = tan(x + {c_str}), find the smallest positive value of x at which the graph has a vertical asymptote. Give your answer as an exact value in terms of pi."

        # tan(x+c) is undefined where x+c = pi/2 + k*pi
        # So x = pi/2 - c + k*pi
        # Find the smallest positive x
        answer_coeff = Fraction(1, 2) - c_coeff
        while answer_coeff <= 0:
            answer_coeff += 1

        answer_text = format_pi_multiple(answer_coeff)

        # Independent check: verify the asymptote location
        total = answer_coeff + c_coeff - Fraction(1, 2)
        check(total.denominator == 1, "Not an integer number of pi's -- not a genuine asymptote location")
        check(0 < answer_coeff <= 1, "Not the smallest positive representative")

        # Distractor candidates
        distractor_1 = format_pi_multiple(Fraction(1, 2))  # Ignores shift c entirely
        distractor_2_coeff = Fraction(1, 2) + c_coeff  # Uses + instead of -
        while distractor_2_coeff <= 0:
            distractor_2_coeff += 1
        distractor_2 = format_pi_multiple(distractor_2_coeff)

        distractors = pick_distractors(answer_text, [distractor_1, distractor_2])

    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Ensure no '*' appears in text/answer_text/distractors (the user-facing fields)
    check("*" not in text, "bare * found in text")
    check("*" not in answer_text, "bare * found in answer_text")
    for distractor in distractors:
        check("*" not in distractor, f"bare * found in distractor: {distractor}")

    # For answer field: add explicit multiplication for sympy parsing where needed
    # e.g., "2pi/3" -> "2*pi/3", "pi/4" -> "pi/4" (already valid)
    if "pi" in answer_text:
        import re

        # Replace digit+pi with digit+*pi (e.g., "2pi" -> "2*pi")
        answer_for_grader = re.sub(r"(\d)pi", r"\1*pi", answer_text)
    else:
        answer_for_grader = answer_text

    return dict(
        text=text,
        latex=text,  # For now, text and latex are the same
        answer=answer_for_grader,  # Sympifiable version for the grader
        answer_text=answer_text,  # Display version (no bare *)
        answer_latex=latex_answer(answer_text),
        distractors=distractors,
    )
