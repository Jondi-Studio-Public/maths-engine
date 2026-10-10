"""Independent oracle for integration skills.

Oracle that re-derives the correct answer to each rendered question using
sympy.integrate() on parsed question text, compared against mathgen's answer.
"""

import re
from sympy import symbols, sympify, integrate, pi, log
import pytest
from mathgen import make
from mathgen.core import symbolically_equal

INTEGRATE_BY_PARTS_TOPIC = "integrate_by_parts"
PARTIAL_FRACTIONS_TOPIC = "use_partial_fractions_where_necessary_for_integration_in"


def parse_question_text(question_text):
    """Parse question.text to extract (lo, hi, integrand) as sympy expressions.

    Input format: "Find ∫ from x = <LO> to x = <HI> of <INTEGRAND> dx"
    Replaces π→pi, ·→*, ^→** before sympify.
    Returns (lo_expr, hi_expr, integrand_expr).
    """
    # Match the template: "Find ∫ from x = <LO> to x = <HI> of <INTEGRAND> dx"
    pattern = r"Find ∫ from x = (.+?) to x = (.+?) of (.+?) dx"
    match = re.match(pattern, question_text)
    if not match:
        raise ValueError(f"Cannot parse question text: {question_text}")

    lo_str, hi_str, integrand_str = match.groups()

    # Define x symbol for parsing
    x = symbols("x")

    # Perform replacements: π→pi, ·→*, ^→**
    lo_str = lo_str.replace("π", "pi").replace("·", "*").replace("^", "**")
    hi_str = hi_str.replace("π", "pi").replace("·", "*").replace("^", "**")
    integrand_str = integrand_str.replace("π", "pi").replace("·", "*").replace("^", "**")

    # Parse using sympify with x in locals
    lo = sympify(lo_str, locals={"x": x, "pi": pi})
    hi = sympify(hi_str, locals={"x": x, "pi": pi})
    integrand = sympify(integrand_str, locals={"x": x, "pi": pi, "log": log})

    return lo, hi, integrand


def parse_answer_text(answer_text):
    """Parse question.answer_text to extract answer as sympy expression.

    Input is bare sympy-syntax expression, replaces ^→** before sympify.
    Returns the sympified expression.
    """
    # Perform replacement: ^→**
    answer_text = answer_text.replace("^", "**")

    # Parse using sympify
    answer = sympify(answer_text, locals={"pi": pi, "log": log})
    return answer


@pytest.mark.parametrize("topic", [INTEGRATE_BY_PARTS_TOPIC, PARTIAL_FRACTIONS_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(60))
def test_oracle_integration_accuracy(topic, difficulty, seed):
    """Oracle: re-derive the correct answer for each question.

    For every seed/difficulty/topic combination:
    1. Generate the question via mathgen.make()
    2. Parse question.text to extract integrand and bounds
    3. Compute the definite integral using sympy.integrate()
    4. Compare oracle_value to question.answer_text via symbolically_equal
    """
    # Generate question
    question = make(topic, seed=seed, difficulty=difficulty)

    # Parse question text
    try:
        lo, hi, integrand = parse_question_text(question.text)
    except (ValueError, SyntaxError) as e:
        pytest.fail(
            f"Failed to parse question text for seed={seed}, "
            f"difficulty={difficulty}, topic={topic}: {e}\n"
            f"Question text: {question.text}"
        )

    # Parse expected answer
    try:
        expected_answer = parse_answer_text(question.answer_text)
    except (ValueError, SyntaxError) as e:
        pytest.fail(
            f"Failed to parse answer text for seed={seed}, "
            f"difficulty={difficulty}, topic={topic}: {e}\n"
            f"Answer text: {question.answer_text}"
        )

    # Compute oracle value using sympy.integrate
    x = symbols("x")
    try:
        oracle_value = integrate(integrand, (x, lo, hi))
    except Exception as e:
        pytest.fail(
            f"sympy.integrate() failed for seed={seed}, "
            f"difficulty={difficulty}, topic={topic}: {e}\n"
            f"Question text: {question.text}\n"
            f"Integrand: {integrand}, bounds: ({lo}, {hi})"
        )

    # Check if oracle_value is unevaluated (undet integral)
    if oracle_value.has(integrate):
        pytest.fail(
            f"sympy.integrate() could not resolve for seed={seed}, "
            f"difficulty={difficulty}, topic={topic}\n"
            f"Question text: {question.text}\n"
            f"Integrand: {integrand}, bounds: ({lo}, {hi})\n"
            f"Oracle value (unevaluated): {oracle_value}"
        )

    # Verify equivalence
    if not symbolically_equal(oracle_value, expected_answer):
        pytest.fail(
            f"Oracle mismatch for seed={seed}, "
            f"difficulty={difficulty}, topic={topic}\n"
            f"Question text: {question.text}\n"
            f"Answer text: {question.answer_text}\n"
            f"Oracle value: {oracle_value}\n"
            f"Expected: {expected_answer}"
        )
