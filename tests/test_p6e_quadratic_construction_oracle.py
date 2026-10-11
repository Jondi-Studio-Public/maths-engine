"""
Independent oracle for quadratic construction skills.

This test independently computes answers from question text using sympy,
without reading the generator code. Its purpose is to verify that shipped
answers are correct.
"""

import re
from typing import List, Tuple
import sympy as sp
from sympy import Symbol, solve, Eq, expand, simplify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from mathgen.skills import load_skills
import mathgen


def extract_vertex_and_point(text: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Extract turning point and point from vertex form question."""
    # Pattern: "turning point (h, k) and passes through the point (x1, y1)"
    match = re.search(
        r"turning point \((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\) and passes through the point \((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)",
        text,
    )
    if match:
        h, k, x1, y1 = match.groups()
        return (float(h), float(k)), (float(x1), float(y1))
    raise ValueError(f"Could not parse vertex form question: {text}")


def extract_roots_and_point(text: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Extract roots and point from roots form question."""
    # Pattern: "roots x = r1 and x = r2, and passes through the point (x1, y1)"
    match = re.search(
        r"roots x = (-?\d+(?:\.\d+)?)\s+and x = (-?\d+(?:\.\d+)?), and passes through the point \((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)",
        text,
    )
    if match:
        r1, r2, x1, y1 = match.groups()
        return (float(r1), float(r2)), (float(x1), float(y1))
    raise ValueError(f"Could not parse roots form question: {text}")


def extract_three_points(text: str) -> List[Tuple[float, float]]:
    """Extract three points from the question text."""
    # Pattern: "passes through the points (x1, y1), (x2, y2) and (x3, y3)"
    pattern = r"passes through the points \((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\), \((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\) and \((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)"
    match = re.search(pattern, text)
    if match:
        x1, y1, x2, y2, x3, y3 = match.groups()
        return [(float(x1), float(y1)), (float(x2), float(y2)), (float(x3), float(y3))]
    raise ValueError(f"Could not parse three points question: {text}")


def solve_vertex_form(vertex: Tuple[float, float], point: Tuple[float, float]) -> sp.Poly:
    """Solve for quadratic equation given vertex and a point."""
    h, k = vertex
    x1, y1 = point
    x = Symbol("x")

    # y = a(x - h)^2 + k
    # Substitute point to find a
    # y1 = a(x1 - h)^2 + k
    a_val = (y1 - k) / ((x1 - h) ** 2)

    # Expand: y = a(x - h)^2 + k
    y = a_val * (x - h) ** 2 + k
    y_expanded = expand(y)

    return y_expanded


def solve_roots_form(roots: Tuple[float, float], point: Tuple[float, float]) -> sp.Poly:
    """Solve for quadratic equation given roots and a point."""
    r1, r2 = roots
    x1, y1 = point
    x = Symbol("x")

    # y = a(x - r1)(x - r2)
    # Substitute point to find a
    # y1 = a(x1 - r1)(x1 - r2)
    a_val = y1 / ((x1 - r1) * (x1 - r2))

    # Expand: y = a(x - r1)(x - r2)
    y = a_val * (x - r1) * (x - r2)
    y_expanded = expand(y)

    return y_expanded


def solve_three_points(points: List[Tuple[float, float]]) -> sp.Poly:
    """Solve for quadratic equation given three points."""
    x = Symbol("x")
    a, b, c = sp.symbols("a b c")

    # y = ax^2 + bx + c
    # Set up three equations
    equations = []
    for px, py in points:
        eq = Eq(a * px**2 + b * px + c, py)
        equations.append(eq)

    # Solve the system
    solution = solve(equations, [a, b, c])

    # Build the polynomial
    a_val = solution[a]
    b_val = solution[b]
    c_val = solution[c]

    y = a_val * x**2 + b_val * x + c_val
    y_expanded = expand(y)

    return y_expanded


def extract_factors(text: str) -> List[str]:
    """Extract factors from expansion question."""
    # Pattern: "Expand (factor1)(factor2)..."
    match = re.search(r"Expand\s+((?:\([^)]+\))+)", text)
    if match:
        factors_str = match.group(1)
        # Extract each factor
        factors = re.findall(r"\(([^)]+)\)", factors_str)
        return factors
    raise ValueError(f"Could not parse expansion question: {text}")


def expand_factors(factors: List[str]) -> sp.Poly:
    """Multiply out the factors."""
    x = Symbol("x")
    result = 1

    transformations = standard_transformations + (implicit_multiplication_application,)

    for factor in factors:
        # Parse the factor: could be like "x - 6", "-3x + 6", etc.
        # Convert to sympy expression
        factor_text = factor.replace("^", "**")
        expr = parse_expr(factor_text, transformations=transformations)
        result *= expr

    result_expanded = expand(result)
    return result_expanded


def normalize_polynomial(poly_expr) -> str:
    """Convert polynomial expression to normalized form for comparison."""
    x = Symbol("x")
    # Expand and simplify
    expanded = expand(poly_expr)
    # Extract coefficients
    return str(expanded)


def parse_answer_text(answer_text: str) -> sp.Expr:
    """Parse the shipped answer text to a sympy expression."""
    # Remove "y = " prefix if present
    text = answer_text.strip()
    if text.startswith("y = "):
        text = text[4:]

    # Replace ^ with ** for sympy
    text = text.replace("^", "**")

    # Parse as sympy expression with transformations
    transformations = standard_transformations + (implicit_multiplication_application,)
    expr = parse_expr(text, transformations=transformations)
    return expand(expr)


def expressions_equal(expr1, expr2) -> bool:
    """Check if two symbolic expressions are equal."""
    x = Symbol("x")
    diff = expand(expr1 - expr2)
    return simplify(diff) == 0


def test_quadratic_construction():
    """Test all quadratic construction questions."""
    _domains, skills_list = load_skills()
    by_id = {skill.id: skill for skill in skills_list}
    skill = by_id["senior-find-the-equation-of-a-quadratic-given-sufficient"]

    question_count = 0
    disagreements = []

    for source in skill.sources:
        if source.difficulty == "exam":
            continue  # the exam source is covered by this skill's tests/test_exd_x_* file
        topic = source.topic
        params = source.params
        difficulty = source.difficulty

        # Test at least 40 seeds per source
        for seed in range(40):
            question = mathgen.make(topic, seed=seed, difficulty=difficulty, **params)

            question_count += 1
            shipped_answer = parse_answer_text(question.answer_text)

            # Independently derive the answer
            try:
                if "vertex_point" in params.get("variant", ""):
                    vertex, point = extract_vertex_and_point(question.text)
                    derived_answer = solve_vertex_form(vertex, point)
                elif "roots_point" in params.get("variant", ""):
                    roots, point = extract_roots_and_point(question.text)
                    derived_answer = solve_roots_form(roots, point)
                elif "three_points" in params.get("variant", ""):
                    points = extract_three_points(question.text)
                    derived_answer = solve_three_points(points)
                else:
                    disagreements.append(
                        {
                            "question": question.text,
                            "shipped": question.answer_text,
                            "error": f"Unknown variant in params: {params}",
                        }
                    )
                    continue

                # Check if they match
                if not expressions_equal(derived_answer, shipped_answer):
                    disagreements.append(
                        {
                            "question": question.text,
                            "shipped": question.answer_text,
                            "derived": str(expand(derived_answer)),
                            "variant": params.get("variant", "unknown"),
                            "seed": seed,
                        }
                    )

            except Exception as e:
                disagreements.append({"question": question.text, "shipped": question.answer_text, "error": str(e)})

    # Report results
    print(f"\nQuadratic construction: tested {question_count} questions")

    if disagreements:
        print(f"FOUND {len(disagreements)} DISAGREEMENTS:")
        for i, d in enumerate(disagreements, 1):
            print(f"\n--- Disagreement {i} ---")
            print(f"Question: {d['question']}")
            print(f"Shipped answer: {d['shipped']}")
            if "derived" in d:
                print(f"Derived answer: {d['derived']}")
            if "error" in d:
                print(f"Error: {d['error']}")
            if "variant" in d:
                print(f"Variant: {d['variant']}")
            if "seed" in d:
                print(f"Seed: {d['seed']}")
    else:
        print("No disagreements found.")

    assert len(disagreements) == 0, f"Found {len(disagreements)} disagreements"


def test_polynomial_expansion():
    """Test all polynomial expansion questions."""
    _domains, skills_list = load_skills()
    by_id = {skill.id: skill for skill in skills_list}
    skill = by_id["senior-expand-quadratic-and-cubic-polynomials-from-factors"]

    question_count = 0
    disagreements = []

    for source in skill.sources:
        if source.difficulty == "exam":
            continue  # the exam source is covered by this skill's tests/test_exd_x_* file
        topic = source.topic
        params = source.params
        difficulty = source.difficulty

        # Test at least 40 seeds per source
        for seed in range(40):
            question = mathgen.make(topic, seed=seed, difficulty=difficulty, **params)

            question_count += 1
            shipped_answer = parse_answer_text(question.answer_text)

            # Independently derive the answer
            try:
                factors = extract_factors(question.text)
                derived_answer = expand_factors(factors)

                # Check if they match
                if not expressions_equal(derived_answer, shipped_answer):
                    disagreements.append(
                        {
                            "question": question.text,
                            "shipped": question.answer_text,
                            "derived": str(expand(derived_answer)),
                            "variant": params.get("variant", "unknown"),
                            "seed": seed,
                        }
                    )

            except Exception as e:
                disagreements.append({"question": question.text, "shipped": question.answer_text, "error": str(e)})

    # Report results
    print(f"\nPolynomial expansion: tested {question_count} questions")

    if disagreements:
        print(f"FOUND {len(disagreements)} DISAGREEMENTS:")
        for i, d in enumerate(disagreements, 1):
            print(f"\n--- Disagreement {i} ---")
            print(f"Question: {d['question']}")
            print(f"Shipped answer: {d['shipped']}")
            if "derived" in d:
                print(f"Derived answer: {d['derived']}")
            if "error" in d:
                print(f"Error: {d['error']}")
            if "variant" in d:
                print(f"Variant: {d['variant']}")
            if "seed" in d:
                print(f"Seed: {d['seed']}")
    else:
        print("No disagreements found.")

    assert len(disagreements) == 0, f"Found {len(disagreements)} disagreements"


if __name__ == "__main__":
    test_quadratic_construction()
    test_polynomial_expansion()
