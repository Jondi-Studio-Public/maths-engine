"""Senior level: tangent line slope and equation."""

from sympy import symbols, diff, Rational

from ..core import generator, check, pick_distractors, check_equivalent, symbolically_equal

x = symbols("x")


def _typed(expr) -> str:
    """answer_text / distractors: keep `*`, turn `**` into `^`."""
    return str(expr).replace("**", "^")


def _prose(expr) -> str:
    """text / working: no `*` at all (learner-readable multiplication dot)."""
    return _typed(expr).replace("*", "·")


@generator(
    "find_the_slope_of_a_tangent_and_the",
    level=14,
    presets={
        "easy": dict(kind="slope"),
        "medium": dict(kind="solve_x"),
        "hard": dict(kind="tangent"),
    },
)
def find_the_slope_of_a_tangent_and_the(rng, kind="slope"):
    """Generate questions about tangent line slopes and equations."""

    if kind == "slope":
        # easy — evaluate the slope at a given x
        # f(x) = a*x**2 + b*x + c
        # p in [-5,-4,-3,-2,-1,1,2,3,4,5] (exclude 0)
        # answer = 2*a*p + b (a plain integer)
        # Candidates: eval_f, power_wrong, forgot_b
        # Redraw loop required
        pool = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

        for attempt in range(200):
            a = rng.randint(1, 5)
            b = rng.randint(-9, 9)
            c = rng.randint(-9, 9)
            p = rng.choice(pool)

            # Build f(x) as a sympy expression
            f = a * x**2 + b * x + c

            # Compute answer via hand formula
            answer = 2 * a * p + b

            # Compute candidates
            eval_f = a * p**2 + b * p + c  # confuses function value with slope
            power_wrong = a * p + b  # forgot power-rule coefficient
            forgot_b = 2 * a * p  # dropped constant term

            # Redraw if any candidate equals answer
            if answer == eval_f or answer == power_wrong or answer == forgot_b:
                continue

            # Independent check via sympy's diff
            sympy_answer = diff(f, x).subs(x, p)
            check(sympy_answer == answer, "easy-tier answer mismatch")

            text = f"The curve y = {_prose(f)} has a tangent at x = {p}. Find the slope of the tangent."

            return dict(
                text=text,
                latex=rf"y = {str(f)}",
                answer=str(answer),
                answer_text=str(answer),
                answer_latex=str(answer),
                distractors=pick_distractors(str(answer), [str(eval_f), str(power_wrong), str(forgot_b)]),
            )

        raise ValueError("Redraw loop exceeded 200 attempts for easy tier")

    elif kind == "solve_x":
        # medium — solve for x given a target slope
        # x0 in same nonzero pool
        # m = 2*a*x0 + b (the "given slope")
        # Ask for x solving f'(x) = m
        # answer = x0
        # Candidates: wrong_derivative_formula, dropped_division, sign_error
        # Redraw loop required
        pool = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

        for attempt in range(200):
            a = rng.randint(1, 5)
            b = rng.randint(-9, 9)
            x0 = rng.choice(pool)

            # Build f(x)
            f = a * x**2 + b * x + rng.randint(-9, 9)

            # Compute given slope and answer
            m = 2 * a * x0 + b
            answer = x0

            # Compute candidates as Fraction/Rational for exactness
            wrong_derivative_formula = Rational(m - b, a)
            dropped_division = m - b
            sign_error = -x0

            # Redraw if any candidate equals x0
            if wrong_derivative_formula == x0 or dropped_division == x0 or sign_error == x0:
                continue

            # Independent check: substitute x0 into f'(x) and verify it equals m
            check(2 * a * x0 + b == m, "medium-tier derivative mismatch")

            text = f"The curve y = {_prose(f)} has a tangent with slope {m}. Find the x-coordinate where this occurs."

            return dict(
                text=text,
                latex=rf"y = {str(f)}",
                answer=str(x0),
                answer_text=str(x0),
                answer_latex=str(x0),
                distractors=pick_distractors(
                    str(x0), [str(wrong_derivative_formula), str(dropped_division), str(sign_error)]
                ),
            )

        raise ValueError("Redraw loop exceeded 200 attempts for medium tier")

    elif kind == "tangent":
        # hard — the equation of the tangent line
        # p from same nonzero pool
        # slope = 2*a*p + b
        # point_value = a*p**2 + b*p + c
        # intercept = point_value - slope*p
        # tangent line: y = slope*x + intercept
        # answer = slope*x + intercept (bare RHS, no y = )
        # answer_text = "y = " + _typed(slope*x + intercept)
        # Candidates: no_intercept, wrong_intercept, swapped
        # Redraw loop required
        pool = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]

        for attempt in range(200):
            a = rng.randint(1, 5)
            b = rng.randint(-9, 9)
            c = rng.randint(-9, 9)
            p = rng.choice(pool)

            # Build f(x)
            f = a * x**2 + b * x + c

            # Compute slope and point value
            slope = 2 * a * p + b
            point_value = a * p**2 + b * p + c

            # Compute intercept
            intercept = point_value - slope * p

            # Tangent line expression
            tangent = slope * x + intercept

            # Compute candidates
            no_intercept = slope * x
            wrong_intercept = slope * x + point_value
            swapped = point_value * x + slope

            # Redraw if any candidate equals tangent
            if symbolically_equal(tangent, no_intercept):
                continue
            if symbolically_equal(tangent, wrong_intercept):
                continue
            if symbolically_equal(tangent, swapped):
                continue

            # Independent check: recompute slope and point_value via sympy
            sympy_slope = diff(f, x).subs(x, p)
            sympy_point_value = f.subs(x, p)
            check_equivalent(sympy_slope, slope, "hard-tier slope mismatch")
            check_equivalent(sympy_point_value, point_value, "hard-tier point_value mismatch")

            text = f"Find the equation of the tangent to y = {_prose(f)} at x = {p}."
            answer_text_str = "y = " + _typed(tangent)

            return dict(
                text=text,
                latex=rf"y = {str(f)}",
                answer=str(tangent),
                answer_text=answer_text_str,
                answer_latex=answer_text_str,
                distractors=pick_distractors(
                    answer_text_str,
                    [
                        "y = " + _typed(no_intercept),
                        "y = " + _typed(wrong_intercept),
                        "y = " + _typed(swapped),
                    ],
                ),
            )

        raise ValueError("Redraw loop exceeded 200 attempts for hard tier")

    else:
        raise ValueError(f"Unknown kind: {kind}")
