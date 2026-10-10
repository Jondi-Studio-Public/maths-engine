"""
Independent oracle for integration techniques B (arcsin/arctan & volumes).

Parses rendered questions without reading mathgen/generators/p6c_integration_techniques_b_build.py
and verifies answers independently using sympy.integrate.
"""

import re
import pytest
from sympy import symbols, sympify, integrate, pi, simplify
from mathgen import make
from mathgen.core import symbolically_equal


x = symbols("x")
y = symbols("y")


def parse_integral_question(text):
    """Parse Shape 1: Find ∫ from x = <LO> to x = <HI> of <INTEGRAND> dx"""
    # Match: Find ∫ from x = <LO> to x = <HI> of <INTEGRAND> dx
    pattern = r"Find ∫ from x = (.+?) to x = (.+?) of (.+?) dx"
    match = re.match(pattern, text)
    if not match:
        raise ValueError(f"Could not parse integral question: {text}")

    lo_str, hi_str, integrand_str = match.groups()

    # Replace unicode and arithmetic symbols
    def normalize(s):
        return s.replace("π", "pi").replace("·", "*").replace("^", "**")

    lo_str = normalize(lo_str)
    hi_str = normalize(hi_str)
    integrand_str = normalize(integrand_str)

    lo = sympify(lo_str)
    hi = sympify(hi_str)
    integrand = sympify(integrand_str, locals={"x": x})

    return lo, hi, integrand


def parse_answer(answer_text):
    """Parse answer text: normalize and sympify."""
    normalized = answer_text.replace("^", "**")
    return sympify(normalized)


def parse_volume_question(text):
    """Parse Shape 2: The region bounded by the curve <CURVE>, the <AXIS>, and the lines <LINE1> and <LINE2>, is rotated about the <AXIS>. Find the volume of the resulting solid."""

    # Extract axis (x-axis or y-axis)
    if "x-axis" in text:
        axis = "x"
    elif "y-axis" in text:
        axis = "y"
    else:
        raise ValueError(f"Could not determine axis from: {text}")

    # Extract the curve
    curve_match = re.search(r"The region bounded by the curve (.+?),", text)
    if not curve_match:
        raise ValueError(f"Could not parse curve from: {text}")
    curve_str = curve_match.group(1).strip()

    # Extract the two boundary lines
    # Lines come after "the <AXIS>, and the lines"
    if axis == "x":
        # Lines are "x = <number>" and "x = <number>"
        lines_match = re.search(r"the lines (x = .+?) and (x = .+?),", text)
    else:
        # Lines are "y = <number>" and "y = <number>"
        lines_match = re.search(r"the lines (y = .+?) and (y = .+?),", text)

    if not lines_match:
        raise ValueError(f"Could not parse lines from: {text}")

    line1_str, line2_str = lines_match.groups()

    # Extract numeric values
    def extract_value(line_str):
        # "x = 0" or "y = 4"
        match = re.search(r"[xy] = (.+)$", line_str.strip())
        if not match:
            raise ValueError(f"Could not parse line: {line_str}")
        val_str = match.group(1).strip()
        # Normalize
        val_str = val_str.replace("π", "pi").replace("·", "*").replace("^", "**")
        return sympify(val_str)

    bound1 = extract_value(line1_str)
    bound2 = extract_value(line2_str)
    lo = min(bound1, bound2)
    hi = max(bound1, bound2)

    # Parse curve
    curve_str = curve_str.replace("π", "pi").replace("·", "*").replace("^", "**")

    if axis == "x":
        # Curve is "y = <expr in x>"
        curve_match = re.match(r"y = (.+)$", curve_str)
        if not curve_match:
            raise ValueError(f"Could not parse y-expr from: {curve_str}")
        expr_str = curve_match.group(1)
        curve_expr = sympify(expr_str, locals={"x": x})
        # Disk method: π * ∫ f(x)² dx
        return axis, lo, hi, curve_expr
    else:
        # For y-axis, curve is always "x = sqrt(y) (that is, y = x^2 for x >= 0)"
        # This means x² = y, so disk radius is x = sqrt(y)
        if "x = sqrt(y)" not in curve_str:
            # Check if it's the fixed literal string
            expected = "x = sqrt(y) (that is, y = x^2 for x >= 0)"
            if curve_str != expected:
                raise ValueError(f"Expected fixed curve string '{expected}', got '{curve_str}'")
        # For y-axis with x = sqrt(y), disk method integrand is y
        return axis, lo, hi, None  # None signals to use y as integrand


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(60))
def test_integrate_expressions_of_the_form_frac_pm1_sqrt(seed, difficulty):
    """Test Shape 1: arcsin and arctan integrals."""
    q = make("integrate_expressions_of_the_form_frac_pm1_sqrt", seed=seed, difficulty=difficulty)

    try:
        lo, hi, integrand = parse_integral_question(q.text)
    except Exception as e:
        pytest.fail(f"Parsing failed for seed={seed}, difficulty={difficulty}: {e}\nText: {q.text}")

    try:
        oracle_value = integrate(integrand, (x, lo, hi))
    except Exception as e:
        pytest.fail(
            f"Integration failed for seed={seed}, difficulty={difficulty}: {e}\nIntegrand: {integrand}\nBounds: {lo} to {hi}"
        )

    parsed_answer = parse_answer(q.answer_text)

    if not symbolically_equal(oracle_value, parsed_answer):
        pytest.fail(
            f"Mismatch for seed={seed}, difficulty={difficulty}\n"
            f"Text: {q.text}\n"
            f"Expected answer: {q.answer_text}\n"
            f"Oracle computed: {oracle_value}\n"
            f"Simplified oracle: {simplify(oracle_value)}\n"
            f"Simplified expected: {simplify(parsed_answer)}"
        )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", range(60))
def test_determine_volumes_of_solids_of_revolution_about_either(seed, difficulty):
    """Test Shape 2: volumes of revolution."""
    q = make("determine_volumes_of_solids_of_revolution_about_either", seed=seed, difficulty=difficulty)

    try:
        axis, lo, hi, curve_expr = parse_volume_question(q.text)
    except Exception as e:
        pytest.fail(f"Parsing failed for seed={seed}, difficulty={difficulty}: {e}\nText: {q.text}")

    try:
        if axis == "x":
            # Disk method for x-axis rotation: π * ∫ f(x)² dx
            oracle_value = pi * integrate(curve_expr**2, (x, lo, hi))
        else:  # axis == "y"
            # Disk method for y-axis rotation with x = sqrt(y): π * ∫ y dy
            oracle_value = pi * integrate(y, (y, lo, hi))
    except Exception as e:
        pytest.fail(
            f"Integration failed for seed={seed}, difficulty={difficulty}: {e}\nAxis: {axis}\nBounds: {lo} to {hi}"
        )

    parsed_answer = parse_answer(q.answer_text)

    if not symbolically_equal(oracle_value, parsed_answer):
        pytest.fail(
            f"Mismatch for seed={seed}, difficulty={difficulty}\n"
            f"Text: {q.text}\n"
            f"Expected answer: {q.answer_text}\n"
            f"Oracle computed: {oracle_value}\n"
            f"Simplified oracle: {simplify(oracle_value)}\n"
            f"Simplified expected: {simplify(parsed_answer)}"
        )
