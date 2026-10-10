"""Derives correct answers from rendered question text alone. Never imports p6d_fitted_line_build."""

import re
from fractions import Fraction
from mathgen.core import make


def extract_precision(answer_text):
    """Extract precision from answer text.
    Integer: precision 0.
    N decimal places: precision N.
    Returns (precision_places, precision_value) tuple.
    """
    text = answer_text.strip()
    if "." in text:
        # Count decimal places
        decimal_part = text.split(".")[-1].rstrip()
        places = len(decimal_part)
        return places, Fraction(10) ** (-places)
    else:
        # Integer
        return 0, Fraction(1)


def extract_question_components(text):
    """Extract m, c, input_var, output_var, and given_value from question text.
    Returns dict with keys: m, c, input_var, output_var, given_value, is_hard.
    """
    # Pattern: "{output_var} = {m} x {input_var} + {c}"
    # Match decimal or integer numbers: optional sign, digits, optional decimal point and more digits
    number_pattern = r"[-+]?(?:\d+\.?\d*|\d*\.\d+)"
    equation_pattern = rf"(\w+)\s*=\s*({number_pattern})\s*x\s*(\w+)\s*\+\s*({number_pattern})"
    match = re.search(equation_pattern, text)
    if not match:
        return None

    output_var = match.group(1)
    m_str = match.group(2)
    input_var = match.group(3)
    c_str = match.group(4)

    # Parse m and c as Fraction
    try:
        m = Fraction(m_str)
        c = Fraction(c_str)
    except (ValueError, ZeroDivisionError):
        return None

    # Check if this is a hard (inverse) problem
    hard_pattern = rf"What value of {re.escape(input_var)}\s+does\s+this\s+fitted\s+line\s+predict\s+will\s+give\s+a\s+{re.escape(output_var)}\s+of\s+({number_pattern})"
    hard_match = re.search(hard_pattern, text)

    if hard_match:
        # Hard: find input given output
        given_value_str = hard_match.group(1)
        is_hard = True
    else:
        # Easy/Medium: find output given input
        # Pattern: "when {input_var} is {value}"
        easy_pattern = rf"when\s+{re.escape(input_var)}\s+is\s+({number_pattern})"
        easy_match = re.search(easy_pattern, text)
        if not easy_match:
            return None
        given_value_str = easy_match.group(1)
        is_hard = False

    try:
        given_value = Fraction(given_value_str)
    except (ValueError, ZeroDivisionError):
        return None

    return {
        "m": m,
        "c": c,
        "input_var": input_var,
        "output_var": output_var,
        "given_value": given_value,
        "is_hard": is_hard,
    }


def compute_oracle_answer(components):
    """Compute the correct answer using the extracted components.
    Returns Fraction.
    """
    m = components["m"]
    c = components["c"]
    given_value = components["given_value"]
    is_hard = components["is_hard"]

    if is_hard:
        # Hard: compute input from output
        # (y0 - c) / m
        if m == 0:
            return None
        answer = (given_value - c) / m
    else:
        # Easy/Medium: compute output from input
        # m*x + c
        answer = m * given_value + c

    return answer


def precision_tolerance(precision_places):
    """Calculate the tolerance for comparison.
    Integer: exact match (tolerance 0).
    N decimal places: within half a unit of the Nth place.
    """
    if precision_places == 0:
        return Fraction(0)
    else:
        return Fraction(1) / (2 * (10**precision_places))


def matches_with_precision(computed, answer_text, precision_places):
    """Check if computed value matches answer_text within precision tolerance."""
    try:
        answer_fraction = Fraction(answer_text.strip())
    except (ValueError, ZeroDivisionError):
        return False

    tolerance = precision_tolerance(precision_places)

    if precision_places == 0:
        # Exact match for integers
        return computed == answer_fraction
    else:
        # Within tolerance for decimals
        return abs(computed - answer_fraction) <= tolerance


def distractor_is_distinct(distractor_text, computed_value, precision_places):
    """Check that distractor differs from computed value by more than tolerance."""
    try:
        distractor = Fraction(distractor_text.strip())
    except (ValueError, ZeroDivisionError):
        # If we can't parse a distractor, it fails the test
        return False

    tolerance = precision_tolerance(precision_places)
    gap = abs(distractor - computed_value)

    # Distractor should be outside the tolerance band
    return gap > tolerance


def test_fitted_line_oracle():
    """Sweep 40+ seeds per tier and verify oracle independently."""
    seeds = list(range(1, 51))  # 50 seeds per tier
    tiers = ["easy", "medium", "hard"]

    min_gaps = {tier: float("inf") for tier in tiers}
    total_questions = 0
    challenged_questions = []

    for tier in tiers:
        for seed in seeds:
            q = make("use_the_equation_of_a_fitted_line_to", seed=seed, difficulty=tier)
            total_questions += 1

            # Extract components
            components = extract_question_components(q.text)
            if components is None:
                msg = f"Could not extract components from seed {seed}, tier {tier}. Text:\n{q.text}"
                challenged_questions.append(msg)
                assert False, msg

            # Compute oracle answer
            computed = compute_oracle_answer(components)
            if computed is None:
                msg = f"Could not compute answer for seed {seed}, tier {tier}. Text:\n{q.text}"
                challenged_questions.append(msg)
                assert False, msg

            # Check precision and format
            precision_places, _ = extract_precision(q.answer_text)

            # Verify oracle answer matches question answer
            assert matches_with_precision(computed, q.answer_text, precision_places), (
                f"Seed {seed} ({tier}): computed {computed} != answer_text {q.answer_text}"
            )

            # Verify all distractors are distinct from the correct answer
            for distractor in q.distractors:
                assert distractor_is_distinct(distractor, computed, precision_places), (
                    f"Seed {seed} ({tier}): distractor {distractor} too close to computed {computed}"
                )

                # Track minimum gap to closest distractor
                try:
                    dist_frac = Fraction(distractor.strip())
                    gap = abs(dist_frac - computed)
                    if gap < min_gaps[tier]:
                        min_gaps[tier] = gap
                except (ValueError, ZeroDivisionError):
                    pass

    print(f"\nOracle test passed: {total_questions} questions")
    print(f"Minimum gaps by tier:")
    for tier in tiers:
        min_gap = min_gaps[tier]
        if min_gap == float("inf"):
            print(f"  {tier}: no distractors found")
        else:
            print(f"  {tier}: {float(min_gap):.6f}")

    if challenged_questions:
        print(f"\nChallenged questions ({len(challenged_questions)}):")
        for msg in challenged_questions:
            print(msg)
    else:
        print("\nNo questions challenged.")


if __name__ == "__main__":
    test_fitted_line_oracle()
