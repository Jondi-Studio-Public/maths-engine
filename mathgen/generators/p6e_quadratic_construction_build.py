"""Generates `find_the_equation_of_a_quadratic_given_sufficient` and
`expand_quadratic_and_cubic_polynomials_from_factors` questions: build or expand a quadratic or cubic from
given information.

Quadratic construction cluster.

- ``senior_find_the_equation_of_a_quadratic_given_sufficient``
  (``senior-find-the-equation-of-a-quadratic-given-sufficient``,
  ACMMM009, domain quadratics, answer: expression) -- given bounded, fully
  specified inputs, find the equation of a quadratic given sufficient
  information. Return the resulting value, expression, relation, or
  classification.

- ``senior_expand_quadratic_and_cubic_polynomials_from_factors``
  (``senior-expand-quadratic-and-cubic-polynomials-from-factors``,
  ACMMM016, domain quadratics, answer: expression) -- given bounded, fully
  specified inputs, expand quadratic and cubic polynomials from factors.
  Return the resulting value, expression, relation, or classification.
"""

from __future__ import annotations

from fractions import Fraction
from sympy import symbols, expand, solve, linsolve, Eq, Rational

from ..core import check, check_distinct, check_equivalent, generator, pick_distractors, fmt_signed

x = symbols("x")


def _poly_display(expr) -> str:
    """Convert expression to display form: ** -> ^, remove *."""
    return str(expr).replace("**", "^").replace("*", "")


# ==============================================================================
# Skill 1: Find the equation of a quadratic given sufficient information
# ==============================================================================


@generator(
    "find_the_equation_of_a_quadratic_given_sufficient",
    level=13,
    presets={
        "easy": dict(variant="vertex_point"),
        "medium": dict(variant="roots_point"),
        "hard": dict(variant="three_points"),
    },
)
def find_the_equation_of_a_quadratic_given_sufficient(rng, variant="vertex_point"):
    """
    Construct a quadratic equation from sufficient information.

    easy: vertex and one point (one unknown: a)
    medium: two roots and one point (two-step: factor form, then expand)
    hard: three points (3x3 system for a, b, c)
    """

    if variant == "vertex_point":
        # Easy: turning point (h, k) and point (x1, y1)
        # Constraints: h != 0, k != 0
        h_pool = list(range(-6, 0)) + list(range(1, 7))
        k_pool = list(range(-6, 0)) + list(range(1, 7))
        h = rng.choice(h_pool)
        k = rng.choice(k_pool)
        a = rng.choice([-3, -2, -1, 1, 2, 3])

        expr = expand(a * (x - h) ** 2 + k)

        x1 = rng.randint(-6, 6)
        while x1 == h:
            x1 = rng.randint(-6, 6)
        y1 = expr.subs(x, x1)

        question_text = f"A quadratic has turning point ({h}, {k}) and passes through the point ({x1}, {y1}). Find its equation in the form y = ax^2 + bx + c."

        # Independent check: reconstruct a from the given information
        a_check = Rational(int(y1) - k, (x1 - h) ** 2)
        expr_check = expand(a_check * (x - h) ** 2 + k)
        check_equivalent(expr_check, expr, f"vertex_point: reconstruction disagrees for h={h}, k={k}")

        # Distractors with guaranteed distinctness
        forgot_negate_h = expand(a * (x + h) ** 2 + k)
        sign_error_k = expand(a * (x - h) ** 2 - k)
        sign_error_a = expand(-a * (x - h) ** 2 + k)

        # Verify distinctness (guards: h != 0, k != 0, both already ensured)
        check_distinct(
            expr, forgot_negate_h, sign_error_k, sign_error_a, msg=f"easy distractor equals answer for h={h}, k={k}"
        )

        candidates = [_poly_display(forgot_negate_h), _poly_display(sign_error_k), _poly_display(sign_error_a)]
        distractors = [f"y = {d}" for d in pick_distractors(_poly_display(expr), candidates)]

        return {
            "text": question_text,
            "latex": question_text,
            "answer": str(expr),
            "answer_text": f"y = {_poly_display(expr)}",
            "answer_latex": f"y = {_poly_display(expr)}",
            "distractors": distractors,
        }

    elif variant == "roots_point":
        # Medium: two roots r1, r2 and point (x2, y2)
        # Constraints: r1 != r2, r1+r2 != 0
        root_pool = list(range(-6, 0)) + list(range(1, 7))
        r1 = rng.choice(root_pool)
        r2 = rng.choice(root_pool)
        while r2 == r1 or r1 + r2 == 0:
            r2 = rng.choice(root_pool)

        a = rng.choice([-3, -2, -1, 1, 2, 3])
        expr = expand(a * (x - r1) * (x - r2))

        x2 = rng.randint(-6, 6)
        while x2 == r1 or x2 == r2:
            x2 = rng.randint(-6, 6)
        y2 = expr.subs(x, x2)

        question_text = f"A quadratic has roots x = {r1} and x = {r2}, and passes through the point ({x2}, {y2}). Find its equation in the form y = ax^2 + bx + c."

        # Independent check: solve for a from point (x2, y2)
        a_check = Rational(int(y2), (x2 - r1) * (x2 - r2))
        expr_check = expand(a_check * (x - r1) * (x - r2))
        check_equivalent(expr_check, expr, f"roots_point: reconstruction disagrees for r1={r1}, r2={r2}")

        # Distractors (guards: r2 != 0, r1+r2 != 0, both already ensured)
        sign_error_one_root = expand(a * (x - r1) * (x + r2))
        sign_error_both_roots = expand(a * (x + r1) * (x + r2))
        sign_error_a = expand(-a * (x - r1) * (x - r2))

        check_distinct(
            expr,
            sign_error_one_root,
            sign_error_both_roots,
            sign_error_a,
            msg=f"medium distractor equals answer for r1={r1}, r2={r2}",
        )

        candidates = [
            _poly_display(sign_error_one_root),
            _poly_display(sign_error_both_roots),
            _poly_display(sign_error_a),
        ]
        distractors = [f"y = {d}" for d in pick_distractors(_poly_display(expr), candidates)]

        return {
            "text": question_text,
            "latex": question_text,
            "answer": str(expr),
            "answer_text": f"y = {_poly_display(expr)}",
            "answer_latex": f"y = {_poly_display(expr)}",
            "distractors": distractors,
        }

    elif variant == "three_points":
        # Hard: three points, solve 3x3 system for a, b, c
        # Constraints: b != 0, c != 0
        a_pool = [-3, -2, -1, 1, 2, 3]
        b_pool = list(range(-6, 0)) + list(range(1, 7))
        c_pool = list(range(-6, 0)) + list(range(1, 7))

        a = rng.choice(a_pool)
        b = rng.choice(b_pool)
        c = rng.choice(c_pool)

        expr = a * x**2 + b * x + c

        # Draw three distinct x-coordinates
        x_vals = []
        x_pool = list(range(-5, 6))
        x1 = rng.choice(x_pool)
        x_vals.append(x1)
        x_pool.remove(x1)
        x2 = rng.choice(x_pool)
        x_vals.append(x2)
        x_pool.remove(x2)
        x3 = rng.choice(x_pool)
        x_vals.append(x3)

        y1 = expr.subs(x, x1)
        y2 = expr.subs(x, x2)
        y3 = expr.subs(x, x3)

        question_text = f"A quadratic passes through the points ({x1}, {y1}), ({x2}, {y2}) and ({x3}, {y3}). Find its equation in the form y = ax^2 + bx + c."

        # Independent check: solve the 3x3 system directly
        A, B, C = symbols("A B C")
        eq1 = Eq(A * x1**2 + B * x1 + C, y1)
        eq2 = Eq(A * x2**2 + B * x2 + C, y2)
        eq3 = Eq(A * x3**2 + B * x3 + C, y3)

        sol = linsolve([eq1, eq2, eq3], [A, B, C])
        check(len(sol) > 0, f"hard: no solution for points ({x1},{y1}),({x2},{y2}),({x3},{y3})")
        a_sol, b_sol, c_sol = next(iter(sol))

        expr_check = a_sol * x**2 + b_sol * x + c_sol
        check_equivalent(expr_check, expr, f"hard: system solution disagrees for points")

        # Distractors (guards: b != 0, c != 0, both already ensured)
        dropped_constant = a * x**2 + b * x + 0
        sign_error_b = a * x**2 - b * x + c
        off_by_one_constant = a * x**2 + b * x + (c + 1)

        check_distinct(
            expr, dropped_constant, sign_error_b, off_by_one_constant, msg=f"hard distractor equals answer for points"
        )

        candidates = [_poly_display(dropped_constant), _poly_display(sign_error_b), _poly_display(off_by_one_constant)]
        distractors = [f"y = {d}" for d in pick_distractors(_poly_display(expr), candidates)]

        return {
            "text": question_text,
            "latex": question_text,
            "answer": str(expr),
            "answer_text": f"y = {_poly_display(expr)}",
            "answer_latex": f"y = {_poly_display(expr)}",
            "distractors": distractors,
        }


# ==============================================================================
# Skill 2: Expand quadratic and cubic polynomials from factors
# ==============================================================================


@generator(
    "expand_quadratic_and_cubic_polynomials_from_factors",
    level=13,
    presets={
        "easy": dict(variant="two_monic"),
        "medium": dict(variant="three_monic"),
        "hard": dict(variant="non_monic_cubic"),
    },
)
def expand_quadratic_and_cubic_polynomials_from_factors(rng, variant="two_monic"):
    """
    Expand polynomial products.

    easy: (x+p)(x+q) with monic binomials
    medium: (x+p)(x+q)(x+r) with three monic binomials
    hard: (ax+b)(x+p)(x+q) with one non-monic factor
    """

    if variant == "two_monic":
        # Easy: (x+p)(x+q)
        # Constraints: p != 0, q != 0, p+q != 0
        factor_pool = list(range(-6, 0)) + list(range(1, 7))
        p = rng.choice(factor_pool)
        q = rng.choice(factor_pool)
        while q == p or p + q == 0:
            q = rng.choice(factor_pool)

        expr = expand((x + p) * (x + q))

        # Format the question using sign-aware display
        if p >= 0:
            factor_p = f"(x + {p})"
        else:
            factor_p = f"(x - {abs(p)})"
        if q >= 0:
            factor_q = f"(x + {q})"
        else:
            factor_q = f"(x - {abs(q)})"

        question_text = f"Expand {factor_p}{factor_q}"

        # Independent check: numeric evaluation at multiple points
        probe_points = [2, -3, 5]
        for px in probe_points:
            prod_form = (px + p) * (px + q)
            expanded_form = expr.subs(x, px)
            check_equivalent(prod_form, expanded_form, f"easy: numeric probe failed at x={px}")

        # Verify constant term
        const_product = p * q
        const_term = expr.as_poly(x).nth(0)
        check_equivalent(const_product, const_term, "easy: constant term mismatch")

        # Distractors (guards: p != 0, q != 0, p+q != 0, all already ensured)
        sign_flip_one = expand((x - p) * (x + q))
        dropped_cross_term = x**2 + p * q
        sign_flip_const_only = x**2 + (p + q) * x - p * q

        check_distinct(
            expr,
            sign_flip_one,
            dropped_cross_term,
            sign_flip_const_only,
            msg=f"easy distractor equals answer for p={p}, q={q}",
        )

        candidates = [
            _poly_display(sign_flip_one),
            _poly_display(dropped_cross_term),
            _poly_display(sign_flip_const_only),
        ]
        distractors = pick_distractors(_poly_display(expr), candidates)

        return {
            "text": question_text,
            "latex": question_text,
            "answer": str(expr),
            "answer_text": _poly_display(expr),
            "answer_latex": _poly_display(expr),
            "distractors": distractors,
            "answer_form": "expanded",
        }

    elif variant == "three_monic":
        # Medium: (x+p)(x+q)(x+r)
        # Constraints: p, q, r all nonzero and pairwise-distinct
        factor_pool = list(range(-6, 0)) + list(range(1, 7))
        p = rng.choice(factor_pool)
        q = rng.choice(factor_pool)
        while q == p:
            q = rng.choice(factor_pool)
        r = rng.choice(factor_pool)
        while r == p or r == q:
            r = rng.choice(factor_pool)

        expr = expand((x + p) * (x + q) * (x + r))

        # Format the question
        p_str = fmt_signed(p, "")
        q_str = fmt_signed(q, "")
        r_str = fmt_signed(r, "")
        if p >= 0:
            factor_p = f"(x + {p})"
        else:
            factor_p = f"(x - {abs(p)})"
        if q >= 0:
            factor_q = f"(x + {q})"
        else:
            factor_q = f"(x - {abs(q)})"
        if r >= 0:
            factor_r = f"(x + {r})"
        else:
            factor_r = f"(x - {abs(r)})"

        question_text = f"Expand {factor_p}{factor_q}{factor_r}"

        # Independent check: numeric evaluation at multiple points
        probe_points = [2, -3, 5]
        for px in probe_points:
            prod_form = (px + p) * (px + q) * (px + r)
            expanded_form = expr.subs(x, px)
            check_equivalent(prod_form, expanded_form, f"medium: numeric probe failed at x={px}")

        # Verify constant term
        const_product = p * q * r
        const_term = expr.as_poly(x).nth(0)
        check_equivalent(const_product, const_term, "medium: constant term mismatch")

        # Distractors (guard: p, q, r all nonzero, already ensured)
        sign_flip_one = expand((x - p) * (x + q) * (x + r))
        sign_flip_all = expand((x - p) * (x - q) * (x - r))
        dropped_linear_cross = x**3 + (p + q + r) * x**2 + (p * q + p * r) * x + p * q * r

        check_distinct(
            expr,
            sign_flip_one,
            sign_flip_all,
            dropped_linear_cross,
            msg=f"medium distractor equals answer for p={p}, q={q}, r={r}",
        )

        candidates = [_poly_display(sign_flip_one), _poly_display(sign_flip_all), _poly_display(dropped_linear_cross)]
        distractors = pick_distractors(_poly_display(expr), candidates)

        return {
            "text": question_text,
            "latex": question_text,
            "answer": str(expr),
            "answer_text": _poly_display(expr),
            "answer_latex": _poly_display(expr),
            "distractors": distractors,
            "answer_form": "expanded",
        }

    elif variant == "non_monic_cubic":
        # Hard: (ax+b)(x+p)(x+q)
        # Constraints: a, b, p, q all nonzero
        a = rng.choice([2, 3, -2, -3])
        b_pool = list(range(-6, 0)) + list(range(1, 7))
        b = rng.choice(b_pool)

        factor_pool = list(range(-6, 0)) + list(range(1, 7))
        p = rng.choice(factor_pool)
        q = rng.choice(factor_pool)
        while q == p:
            q = rng.choice(factor_pool)

        expr = expand((a * x + b) * (x + p) * (x + q))

        # Format the question
        ax_str = fmt_signed(a, "x")
        b_str = fmt_signed(b, "")
        if b >= 0:
            factor_a = f"({ax_str} + {b})"
        else:
            factor_a = f"({ax_str} - {abs(b)})"

        if p >= 0:
            factor_p = f"(x + {p})"
        else:
            factor_p = f"(x - {abs(p)})"
        if q >= 0:
            factor_q = f"(x + {q})"
        else:
            factor_q = f"(x - {abs(q)})"

        question_text = f"Expand {factor_a}{factor_p}{factor_q}"

        # Independent check: numeric evaluation at multiple points
        probe_points = [2, -3, 5]
        for px in probe_points:
            prod_form = (a * px + b) * (px + p) * (px + q)
            expanded_form = expr.subs(x, px)
            check_equivalent(prod_form, expanded_form, f"hard: numeric probe failed at x={px}")

        # Verify constant term
        const_product = b * p * q
        const_term = expr.as_poly(x).nth(0)
        check_equivalent(const_product, const_term, "hard: constant term mismatch")

        # Distractors (all guard: a, b, p, q all nonzero, already ensured)
        sign_flip_b = expand((a * x - b) * (x + p) * (x + q))
        sign_flip_one_root = expand((a * x + b) * (x - p) * (x + q))
        dropped_const = expr - b * p * q

        check_distinct(
            expr,
            sign_flip_b,
            sign_flip_one_root,
            dropped_const,
            msg=f"hard distractor equals answer for a={a}, b={b}, p={p}, q={q}",
        )

        candidates = [_poly_display(sign_flip_b), _poly_display(sign_flip_one_root), _poly_display(dropped_const)]
        distractors = pick_distractors(_poly_display(expr), candidates)

        return {
            "text": question_text,
            "latex": question_text,
            "answer": str(expr),
            "answer_text": _poly_display(expr),
            "answer_latex": _poly_display(expr),
            "distractors": distractors,
            "answer_form": "expanded",
        }
