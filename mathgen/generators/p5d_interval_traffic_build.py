"""Interval estimate for proportions + traffic-light probability.

- ``interval_estimate_proportion_margin``
  (``senior-the-concept-of-an-interval-estimate-for-a``,
  ACMMM177/ACMSM140, domain statistics-probability, answer: number)
  -- margin of error for a proportion, given bounded fully specified inputs.

- ``traffic_light_probability``
  (``senior-determine-the-probabilities-of-occurrence-of-simple-traffic``,
  ACMEM158, domain statistics-probability, answer: fraction)
  -- traffic-light probability problems using the product and complement rules.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, generator, pick_distractors, rounds_unambiguously


# ============================================================================
# Skill 1: interval_estimate_proportion_margin
# ============================================================================

Z_VALUES = {"90%": 1.645, "95%": 1.96, "99%": 2.576}
PHAT_HUNDREDTHS = range(20, 81)  # 0.20 to 0.80 in increments of 0.01


def _ceils_unambiguously(value: float, margin: float = 0.01) -> bool:
    """True when ``value`` is far enough from a whole number that ceil is
    determined. Same reasoning as :func:`_rounds_unambiguously`."""
    return abs(value - round(value)) > margin


@generator(
    "interval_estimate_proportion_margin",
    level=12,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="from_counts"),
        "hard": dict(variant="solve_for_n"),
    },
)
def interval_estimate_proportion_margin(rng, variant="direct"):
    """Calculate margin of error for proportions, given fully specified inputs."""

    if variant == "direct":
        # Easy: direct E calculation
        # E = z * sqrt(phat * (1 - phat) / n)
        # Resample until the margin of error rounds to 3 dp unambiguously.
        for _attempt in range(500):
            confidence = rng.choice(list(Z_VALUES.keys()))
            z = Z_VALUES[confidence]
            phat_hundredths = rng.choice(PHAT_HUNDREDTHS)
            phat = phat_hundredths / 100.0
            n = rng.randint(30, 201)
            variance_term = phat * (1 - phat) / n
            E = z * math.sqrt(variance_term)
            if rounds_unambiguously(E, 3):
                break
        else:
            raise RuntimeError(
                "interval_estimate_proportion_margin/direct: no draw gave a "
                "margin of error that rounds to 3 dp unambiguously in 500 attempts"
            )
        E_rounded = round(E, 3)
        answer_text = f"{E_rounded:.3f}"

        # Independent check: use Fraction for phat computation
        phat_frac = Fraction(phat_hundredths, 100)
        variance_frac = phat_frac * (1 - phat_frac) / n
        variance_float = float(variance_frac)
        E_check = z * math.sqrt(variance_float)
        E_check_rounded = round(E_check, 3)
        check(
            abs(E_rounded - E_check_rounded) < 1e-9,
            f"direct: E_rounded={E_rounded} != E_check_rounded={E_check_rounded}",
        )

        question_text = (
            f"A sample of size {n} is drawn from a large population. "
            f"The sample proportion is {phat:.2f}. "
            f"Calculate the margin of error at the {confidence} confidence level "
            f"(z = {z}). Round to 3 decimal places."
        )

        working = [
            f"Margin of error = z times the square root of (phat times (1 - phat) divided by n).",
            f"= {z} times the square root of ({phat:.2f} times {1 - phat:.2f} divided by {n}) = {answer_text}.",
        ]

        # Distractors for easy tier
        # forgot-sqrt: z * phat * (1 - phat) / n
        forgot_sqrt_val = z * phat * (1 - phat) / n
        forgot_sqrt_rounded = round(forgot_sqrt_val, 3)
        forgot_sqrt_str = f"{forgot_sqrt_rounded:.3f}"
        check(forgot_sqrt_rounded != E_rounded, f"direct: forgot-sqrt distractor equals answer")

        # forgot-division-by-n: z * sqrt(phat * (1 - phat))
        forgot_div_val = z * math.sqrt(phat * (1 - phat))
        forgot_div_rounded = round(forgot_div_val, 3)
        forgot_div_str = f"{forgot_div_rounded:.3f}"
        check(forgot_div_rounded != E_rounded, f"direct: forgot-division distractor equals answer")

        # forgot-complement: z * sqrt(phat / n)
        forgot_comp_val = z * math.sqrt(phat / n)
        forgot_comp_rounded = round(forgot_comp_val, 3)
        forgot_comp_str = f"{forgot_comp_rounded:.3f}"
        check(forgot_comp_rounded != E_rounded, f"direct: forgot-complement distractor equals answer")

        candidates = [forgot_sqrt_str, forgot_div_str, forgot_comp_str]
        distractors = pick_distractors(answer_text, candidates)

    elif variant == "from_counts":
        # Medium: given raw counts, compute phat first
        # E = z * sqrt(phat * (1 - phat) / n)
        # n is drawn from the multiples of 100 // gcd(100, phat_hundredths) so
        # that x / n is exactly the stated two-decimal proportion. Drawing n
        # freely left phat = x / n unrounded while the working displayed it to
        # 2 dp, so a learner following the working line got a different third
        # decimal from the answer.
        for _attempt in range(500):
            confidence = rng.choice(list(Z_VALUES.keys()))
            z = Z_VALUES[confidence]
            phat_hundredths = rng.choice(PHAT_HUNDREDTHS)
            step = 100 // math.gcd(100, phat_hundredths)
            n = rng.choice([m for m in range(30, 202) if m % step == 0])
            x = phat_hundredths * n // 100
            phat_actual = x / n
            variance_term = phat_actual * (1 - phat_actual) / n
            E = z * math.sqrt(variance_term)
            if 0 < x < n and rounds_unambiguously(E, 3):
                break
        else:
            raise RuntimeError(
                "interval_estimate_proportion_margin/from_counts: no draw gave an "
                "exact two-decimal proportion with an unambiguous 3 dp margin of "
                "error in 500 attempts"
            )
        E_rounded = round(E, 3)
        answer_text = f"{E_rounded:.3f}"

        # Independent check: use Fraction
        phat_frac = Fraction(x, n)
        variance_frac = phat_frac * (1 - phat_frac) / n
        variance_float = float(variance_frac)
        E_check = z * math.sqrt(variance_float)
        E_check_rounded = round(E_check, 3)
        check(
            abs(E_rounded - E_check_rounded) < 1e-9,
            f"from_counts: E_rounded={E_rounded} != E_check_rounded={E_check_rounded}",
        )

        question_text = (
            f"In a sample of size {n}, a total of {x} respondents answered 'yes'. "
            f"Calculate the margin of error at the {confidence} confidence level "
            f"(z = {z}). Round to 3 decimal places."
        )

        working = [
            f"Sample proportion phat = {x} divided by {n} = {phat_actual:.2f}.",
            f"Margin of error = z times the square root of (phat times (1 - phat) divided by n).",
            f"= {z} times the square root of ({phat_actual:.2f} times {1 - phat_actual:.2f} divided by {n}) = {answer_text}.",
        ]

        # Distractors
        forgot_sqrt_val = z * phat_actual * (1 - phat_actual) / n
        forgot_sqrt_rounded = round(forgot_sqrt_val, 3)
        forgot_sqrt_str = f"{forgot_sqrt_rounded:.3f}"
        check(forgot_sqrt_rounded != E_rounded, f"from_counts: forgot-sqrt distractor equals answer")

        forgot_div_val = z * math.sqrt(phat_actual * (1 - phat_actual))
        forgot_div_rounded = round(forgot_div_val, 3)
        forgot_div_str = f"{forgot_div_rounded:.3f}"
        check(forgot_div_rounded != E_rounded, f"from_counts: forgot-division distractor equals answer")

        forgot_comp_val = z * math.sqrt(phat_actual / n)
        forgot_comp_rounded = round(forgot_comp_val, 3)
        forgot_comp_str = f"{forgot_comp_rounded:.3f}"
        check(forgot_comp_rounded != E_rounded, f"from_counts: forgot-complement distractor equals answer")

        candidates = [forgot_sqrt_str, forgot_div_str, forgot_comp_str]
        distractors = pick_distractors(answer_text, candidates)

    elif variant == "solve_for_n":
        # Hard: solve for n given E_target
        # n = ceil(z**2 * phat * (1 - phat) / E_target**2)
        # Resample until n_exact is clear of a whole number, so the ceiling is
        # determined rather than decided by floating-point noise.
        E_target_options = [0.025, 0.030, 0.035, 0.040, 0.050]
        for _attempt in range(500):
            confidence = rng.choice(list(Z_VALUES.keys()))
            z = Z_VALUES[confidence]
            phat_hundredths = rng.choice(PHAT_HUNDREDTHS)
            phat = phat_hundredths / 100.0
            E_target = rng.choice(E_target_options)
            numerator = z**2 * phat * (1 - phat)
            n_exact = numerator / (E_target**2)
            if _ceils_unambiguously(n_exact):
                break
        else:
            raise RuntimeError(
                "interval_estimate_proportion_margin/solve_for_n: no draw put "
                "n_exact clear of a whole number in 500 attempts"
            )
        n_required = math.ceil(n_exact)
        answer_text = str(n_required)

        # Independent check: use Fraction for phat
        phat_frac = Fraction(phat_hundredths, 100)
        numerator_frac = phat_frac * (1 - phat_frac)
        numerator_float = float(numerator_frac)
        n_exact_check = (z**2 * numerator_float) / (E_target**2)
        n_check = math.ceil(n_exact_check)
        check(n_check == n_required, f"solve_for_n: n_check={n_check} != n_required={n_required}")

        question_text = (
            f"A researcher wants a margin of error of {E_target:.3f} at the {confidence} confidence level. "
            f"If the sample proportion is estimated to be {phat:.2f}, "
            f"what is the minimum sample size required? (z = {z})"
        )

        working = [
            f"Rearranging the margin of error formula to solve for n: n = z squared times phat times (1 - phat) divided by E squared.",
            f"= {z} squared times {phat:.2f} times {1 - phat:.2f} divided by {E_target:.3f} squared = {answer_text}.",
        ]

        # Distractors for hard tier (all related to the formula rearrangement)
        # forgot-z-squared: n = phat * (1 - phat) / E_target**2 (omits z**2)
        forgot_z2_val = phat * (1 - phat) / (E_target**2)
        forgot_z2 = math.ceil(forgot_z2_val)
        forgot_z2_str = str(forgot_z2)
        check(forgot_z2 != n_required, f"solve_for_n: forgot-z-squared distractor equals answer")

        # forgot-E-squared: n = z**2 * phat * (1 - phat) / E_target (omits squaring E_target)
        forgot_E2_val = z**2 * phat * (1 - phat) / E_target
        forgot_E2 = math.ceil(forgot_E2_val)
        forgot_E2_str = str(forgot_E2)
        check(forgot_E2 != n_required, f"solve_for_n: forgot-E-squared distractor equals answer")

        # floor-instead-of-ceil: floor(n_exact) instead of ceil
        floor_val = math.floor(n_exact)
        floor_str = str(floor_val)
        check(floor_val != n_required, f"solve_for_n: floor distractor equals answer")

        candidates = [forgot_z2_str, forgot_E2_str, floor_str]
        distractors = pick_distractors(answer_text, candidates)

    return {
        "text": question_text,
        "latex": f"plain:{question_text}",
        "answer": answer_text,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
    }


# ============================================================================
# Skill 2: traffic_light_probability
# ============================================================================

LIGHT_PROBS = (
    Fraction(1, 2),
    Fraction(1, 3),
    Fraction(2, 3),
    Fraction(1, 4),
    Fraction(3, 4),
    Fraction(2, 5),
    Fraction(3, 5),
)


@generator(
    "traffic_light_probability",
    level=12,
    presets={
        "easy": dict(variant="both_green"),
        "medium": dict(variant="exactly_one"),
        "hard": dict(variant="at_least_one"),
    },
)
def traffic_light_probability(rng, variant="both_green"):
    """Determine probabilities in simple traffic-light problems."""

    if variant == "both_green":
        # Easy: P(both green) = p1 * p2
        p1 = rng.choice(LIGHT_PROBS)
        p2 = rng.choice(LIGHT_PROBS)
        answer = p1 * p2

        question_text = (
            f"A commuter passes two independent traffic lights. "
            f"Light 1 is green with probability {p1}; light 2 is green with probability {p2}. "
            f"What is the probability both lights are green?"
        )
        answer_text = str(answer)

        working = [f"P(both green) = {p1} times {p2} = {answer_text}."]

        # Independent check: explicit enumeration
        # All 4 combinations: (G,G), (G,R), (R,G), (R,R)
        both_green_enum = p1 * p2
        check(both_green_enum == answer, f"both_green: enumeration check failed for p1={p1}, p2={p2}")

        # Distractors
        # adds: p1 + p2
        adds = p1 + p2
        check(adds != answer, f"both_green: adds distractor equals answer")

        # forgets_p2: p1 alone
        forgets_p2 = p1
        check(forgets_p2 != answer, f"both_green: forgets_p2 distractor equals answer")

        # adds can exceed 1 (an impossible probability, eliminated at sight), so it is offered only when it
        # is a real probability. The union (P(A or B)) and the other light alone are in-range stand-ins.
        either_green = p1 + p2 - p1 * p2
        candidates = [
            str(v) for v in (adds, forgets_p2, either_green, p2, (1 - p1) * (1 - p2)) if 0 < v < 1 and v != answer
        ]
        distractors = pick_distractors(answer_text, candidates, n=2)
        check(len(distractors) == 2, "both_green: need two in-range distractors")

    elif variant == "exactly_one":
        # Medium: P(exactly one green) = p1*(1-p2) + (1-p1)*p2
        p1 = rng.choice(LIGHT_PROBS)
        p2 = rng.choice(LIGHT_PROBS)
        answer = p1 * (1 - p2) + (1 - p1) * p2

        question_text = (
            f"A commuter passes two independent traffic lights. "
            f"Light 1 is green with probability {p1}; light 2 is green with probability {p2}. "
            f"What is the probability exactly one light is green?"
        )
        answer_text = str(answer)

        p_light1_only = p1 * (1 - p2)
        p_light2_only = (1 - p1) * p2
        working = [
            f"P(exactly one green) = P(light 1 only) + P(light 2 only) = {p_light1_only} + {p_light2_only} = {answer_text}."
        ]

        # Independent check: explicit enumeration over all 4 combinations
        p_both_green = p1 * p2
        p_neither_green = (1 - p1) * (1 - p2)
        exactly_one_enum = p_light1_only + p_light2_only
        check(exactly_one_enum == answer, f"exactly_one: enumeration check failed for p1={p1}, p2={p2}")

        # Distractors
        # only_light1_green: p1*(1-p2), forgets the second case
        only_light1 = p1 * (1 - p2)
        check(only_light1 != answer, f"exactly_one: only_light1 distractor equals answer")

        # not_both_green: 1 - p1*p2, wrong complement
        not_both = 1 - p1 * p2
        check(not_both != answer, f"exactly_one: not_both distractor equals answer")

        candidates = [str(only_light1), str(not_both)]
        distractors = pick_distractors(answer_text, candidates)

    elif variant == "at_least_one":
        # Hard: P(at least one green) = 1 - (1-p1)*(1-p2)*(1-p3)
        p1 = rng.choice(LIGHT_PROBS)
        p2 = rng.choice(LIGHT_PROBS)
        p3 = rng.choice(LIGHT_PROBS)
        answer = 1 - (1 - p1) * (1 - p2) * (1 - p3)

        question_text = (
            f"A commuter passes three independent traffic lights. "
            f"Light 1 is green with probability {p1}; light 2 is green with probability {p2}; "
            f"light 3 is green with probability {p3}. "
            f"What is the probability at least one light is green?"
        )
        answer_text = str(answer)

        all_red = (1 - p1) * (1 - p2) * (1 - p3)
        working = [
            f"P(at least one green) = 1 - P(all red) = 1 - ({1 - p1} times {1 - p2} times {1 - p3}) = 1 - {all_red} = {answer_text}."
        ]

        # Independent check: explicit enumeration over all 8 combinations
        probs_by_outcome = []
        for i1 in [0, 1]:  # 0 = red, 1 = green
            for i2 in [0, 1]:
                for i3 in [0, 1]:
                    prob_this = (p1 if i1 else (1 - p1)) * (p2 if i2 else (1 - p2)) * (p3 if i3 else (1 - p3))
                    at_least_one = (i1 == 1) or (i2 == 1) or (i3 == 1)
                    if at_least_one:
                        probs_by_outcome.append(prob_this)
        at_least_one_enum = sum(probs_by_outcome)
        check(at_least_one_enum == answer, f"at_least_one: enumeration check failed for p1={p1}, p2={p2}, p3={p3}")

        # Distractors
        # all_three_green: p1 * p2 * p3
        all_three = p1 * p2 * p3
        check(all_three != answer, f"at_least_one: all_three distractor equals answer")

        # all_red: (1-p1)*(1-p2)*(1-p3) (no final 1-)
        check(all_red != answer, f"at_least_one: all_red distractor equals answer")

        candidates = [str(all_three), str(all_red)]
        distractors = pick_distractors(answer_text, candidates)

    return {
        "text": question_text,
        "latex": f"plain:{question_text}",
        "answer": answer_text,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
    }
