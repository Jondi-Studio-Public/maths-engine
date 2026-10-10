"""Solve quadratic equations using the quadratic formula and by completing the square.

Senior Mathematical Methods, exact surd answers for non-factorable quadratics.
Three tiers:
- easy: monic (a=1), squarefree discriminant
- medium: monic (a=1), discriminant with perfect square factors
- hard: non-monic (a in {2,3}), discriminant with perfect square factors
"""

import math
from sympy import symbols, expand, solve, Eq, Rational, sqrt, simplify, latex

from ..core import generator, check_equivalent, check_distinct

x = symbols("x")


def _is_perfect_square(n):
    """Check if n is a perfect square."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _poly_display(expr) -> str:
    """Format polynomial expression for display, replacing ** with ^ and removing *."""
    return str(expr).replace("**", "^").replace("*", "")


def _generate_quadratic_formula(rng, variant="monic_squarefree"):
    """
    Generate a quadratic equation and solve it exactly using the quadratic formula.

    variant:
    - "monic_squarefree": a=1, D is squarefree (no perfect square factors > 1)
    - "monic_surd_simplifies": a=1, D has perfect square factors
    - "non_monic_surd_simplifies": a in {2,3}, D has perfect square factors
    """

    # Define parameter sets based on variant
    if variant == "monic_squarefree":
        a = 1
        D_set = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23]
    elif variant == "monic_surd_simplifies":
        a = 1
        D_set = [8, 12, 18, 20, 24, 27, 28, 32, 44, 45, 48, 50, 52, 63, 72, 75]
    elif variant == "non_monic_surd_simplifies":
        a = rng.choice([2, 3])
        D_set = [8, 12, 18, 20, 24, 27, 28, 32, 44, 45, 48, 50, 52, 63, 72, 75]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Resample until we get valid c
    # Create list of non-zero b values
    b_values = list(range(-9, 0)) + list(range(1, 10))  # [-9, ..., -1, 1, ..., 9]

    while True:
        # Draw b from the list
        b = rng.choice(b_values)

        # Try each D in the set in order, take the first that gives integer c
        c = None
        selected_D = None
        for D in D_set:
            # Need: c = Rational(b*b - D, 4*a) to be an integer
            # So: b^2 - D must be divisible by 4*a
            if (b * b - D) % (4 * a) == 0:
                c = Rational(b * b - D, 4 * a)
                selected_D = D
                break

        if c is None:
            # None of the D values worked for this b, resample b
            continue

        # Check discriminant is positive and not perfect square
        D = b * b - 4 * a * c
        if D > 0 and not _is_perfect_square(D):
            break

    # Construct the quadratic
    expr = a * x**2 + b * x + c

    # Solve using quadratic formula (sympy's solve auto-simplifies surds)
    roots_all = solve(Eq(expr, 0), x)
    root_expr = max(roots_all, key=lambda r: float(r.evalf()))  # larger root

    # Independent check via completing the square
    # a*x^2 + b*x + c = 0  =>  a*(x + b/(2a))^2 + (c - b^2/(4a)) = 0
    vertex_x = -Rational(b, 2 * a)
    vertex_y = c - Rational(b, 2 * a) ** 2 * a

    # Solve: a*(x - vertex_x)^2 + vertex_y = 0
    # (x - vertex_x)^2 = -vertex_y/a
    # x = vertex_x +/- sqrt(-vertex_y/a)
    offset = sqrt(-vertex_y / a)
    root_cts_larger = simplify(vertex_x + offset)

    # Verify the two routes agree
    check_equivalent(root_cts_larger, root_expr, "quadratic-formula and completing-the-square roots disagree")

    # Misconceptions
    # 1. wrong-sign-b: (b + sqrt(D)) / (2*a)
    wrong_sign_b = simplify((b + sqrt(D)) / (2 * a))
    check_distinct(root_expr, wrong_sign_b, msg="wrong-sign-b distractor equals correct answer")

    # 2. forgot-double-a: (-b + sqrt(D)) / a
    forgot_double_a = simplify((-b + sqrt(D)) / a)
    check_distinct(root_expr, forgot_double_a, msg="forgot-double-a distractor equals correct answer")

    # 3. half-divided-surd: -b/(2*a) + sqrt(D) (divides only -b term, forgets to divide sqrt term)
    half_divided_surd = simplify(-Rational(b, 2 * a) + sqrt(D))
    check_distinct(root_expr, half_divided_surd, msg="half-divided-surd distractor equals correct answer")

    # Verify the answer is a genuine surd (not plain integer or terminating decimal)
    answer_text = f"x = {_poly_display(root_expr)}"
    # The answer should contain sqrt and not be a plain number
    assert "sqrt" in str(root_expr), f"Answer should be a surd: {answer_text}"

    # Verify discriminant is positive and not a perfect square
    assert D > 0, f"Discriminant {D} is not positive"
    assert not _is_perfect_square(D), f"Discriminant {D} is a perfect square"

    # Build the question
    lhs = expand(a * x**2 + b * x + c)
    prompt = f"Solve {_poly_display(lhs)} = 0. Give the larger root exactly."

    # For LaTeX, convert the polynomial expression properly
    # Convert root expression to LaTeX
    latex_root = latex(root_expr)
    latex_answer_text = f"x = {latex_root}"

    return dict(
        text=prompt,
        latex=rf"\text{{Solve}} \quad {latex(lhs)} = 0",
        answer=root_expr,
        answer_text=answer_text,
        answer_latex=latex_answer_text,
        distractors=[
            f"x = {_poly_display(wrong_sign_b)}",
            f"x = {_poly_display(forgot_double_a)}",
            f"x = {_poly_display(half_divided_surd)}",
        ],
    )


@generator(
    "solve_quadratic_equations_using_the_quadratic_formula_and",
    level=12,
    presets={
        "easy": dict(variant="monic_squarefree"),
        "medium": dict(variant="monic_surd_simplifies"),
        "hard": dict(variant="non_monic_surd_simplifies"),
    },
)
def solve_quadratic_equations_using_the_quadratic_formula_and(rng, variant="monic_squarefree"):
    """Solve quadratic equations using the quadratic formula with exact surd answers."""
    return _generate_quadratic_formula(rng, variant=variant)
