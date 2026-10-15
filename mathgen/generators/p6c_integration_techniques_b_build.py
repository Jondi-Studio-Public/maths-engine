"""Integration techniques (arcsin/arctan) and volumes of revolution.

- ``integrate_expressions_of_the_form_frac_pm1_sqrt``
  (``senior-integrate-expressions-of-the-form-frac-pm1-sqrt``,
  ACMSM121, domain integration, answer: expression) -- given bounded, fully
  specified inputs, integrate expressions of the form 1/sqrt(a^2-x^2) and a/(a^2+x^2).
  Return the resulting value, expression, relation, or classification.
- ``determine_volumes_of_solids_of_revolution_about_either``
  (``senior-determine-volumes-of-solids-of-revolution-about-either``,
  ACMSM125, domain integration, answer: expression) -- given bounded, fully
  specified inputs, determine volumes of solids of revolution about either axis.
  Return the resulting value, expression, relation, or classification.
"""

from __future__ import annotations

from sympy import symbols, sqrt, asin, atan, pi, diff, Rational, simplify

from ..calc_cache import integrate

from ..core import check, check_equivalent, check_distinct, generator, latex_answer, pick_distractors


# Display helpers (^/* convention): convert ** to ^ for typed, ** to ^ and * to · for prose
def _typed(expr) -> str:
    """Convert expression to learner-facing typed form: ** -> ^."""
    return str(expr).replace("**", "^")


def _prose(expr) -> str:
    """Convert expression to prose form: ** -> ^, * to ·."""
    typed_str = str(expr).replace("**", "^")
    return typed_str.replace("*", "·")


# Module-scope symbols
x = symbols("x")
y = symbols("y")


@generator(
    "integrate_expressions_of_the_form_frac_pm1_sqrt",
    level=17,
    presets={
        "easy": dict(kind="arcsin_single"),
        "medium": dict(kind="arcsin_two_bound"),
        "hard": dict(kind="arctan"),
    },
)
def integrate_expressions_of_the_form_frac_pm1_sqrt(rng, kind="arcsin_single"):
    """
    Integrate expressions of the form 1/sqrt(a^2-x^2) and a/(a^2+x^2).

    easy: arcsin_single with zero lower bound
    medium: arcsin_two_bound with two nonzero bounds
    hard: arctan, different antiderivative family
    """

    if kind == "arcsin_single":
        # Standard: ∫ 1/sqrt(a² - x²) dx = asin(x/a) + C
        a = rng.randint(2, 6)
        s = rng.choice([Rational(1, 2), 1])
        k = rng.randint(2, 4)
        f = k / sqrt(a**2 - x**2)
        F = k * asin(x / a)
        lo = 0
        hi = a * s
        ans = F.subs(x, hi) - F.subs(x, lo)

        # Independent checks
        check_equivalent(diff(F, x), f, msg="Derivative of antiderivative")
        check_equivalent(ans, integrate(f, (x, lo, hi)), msg="Definite integral")

        # Misconceptions
        extra_factor_a = a * ans
        sign_flip = -ans
        dropped_k = ans / k

        check_distinct(ans, extra_factor_a, sign_flip, dropped_k, msg="Arcsin single evaluation distractors")

        candidates = [
            _typed(extra_factor_a),
            _typed(sign_flip),
            _typed(dropped_k),
        ]
        distractors = pick_distractors(_typed(ans), candidates)

        return {
            "text": f"Find ∫ from x = 0 to x = {_prose(hi)} of {_prose(f)} dx. Give your answer as an exact value in terms of pi.",
            "latex": f"Find ∫ from x = 0 to x = {_prose(hi)} of {_prose(f)} dx. Give your answer as an exact value in terms of pi.",
            "answer": str(ans),
            "answer_text": _typed(ans),
            "answer_latex": latex_answer(ans),
            "distractors": distractors,
        }

    elif kind == "arcsin_two_bound":
        # Arcsin with two nonzero bounds
        a = rng.randint(2, 6)
        k = rng.randint(2, 4)
        f = k / sqrt(a**2 - x**2)
        F = k * asin(x / a)

        pair = rng.choice([(Rational(-1, 2), Rational(1, 2)), (Rational(1, 2), 1), (-1, Rational(-1, 2))])
        s1, s2 = pair
        lo = a * s1
        hi = a * s2
        ans = F.subs(x, hi) - F.subs(x, lo)

        # Independent checks
        check_equivalent(diff(F, x), f, msg="Derivative of antiderivative")
        check_equivalent(ans, integrate(f, (x, lo, hi)), msg="Definite integral")

        # Misconceptions
        dropped_k = ans / k
        sign_flip = -ans
        only_upper_bound = F.subs(x, hi) - F.subs(x, 0)

        check_distinct(ans, dropped_k, sign_flip, only_upper_bound, msg="Arcsin two-bound distractors")

        candidates = [
            _typed(dropped_k),
            _typed(sign_flip),
            _typed(only_upper_bound),
        ]
        distractors = pick_distractors(_typed(ans), candidates)

        return {
            "text": f"Find ∫ from x = {_prose(lo)} to x = {_prose(hi)} of {_prose(f)} dx. Give your answer as an exact value in terms of pi.",
            "latex": f"Find ∫ from x = {_prose(lo)} to x = {_prose(hi)} of {_prose(f)} dx. Give your answer as an exact value in terms of pi.",
            "answer": str(ans),
            "answer_text": _typed(ans),
            "answer_latex": latex_answer(ans),
            "distractors": distractors,
        }

    elif kind == "arctan":
        # Standard: ∫ a/(a² + x²) dx = atan(x/a) + C
        a = rng.randint(2, 9)
        # Exclude k=2 to avoid collision where dropped_k equals only_upper_bound
        k = rng.choice([3, 4, 5])
        f = k * a / (a**2 + x**2)
        F = k * atan(x / a)
        lo = -a
        hi = a
        ans = F.subs(x, hi) - F.subs(x, lo)

        # Independent checks
        check_equivalent(diff(F, x), f, msg="Derivative of antiderivative")
        check_equivalent(ans, integrate(f, (x, lo, hi)), msg="Definite integral")

        # Ensure bounds never place x/a at 0
        check(lo != 0 and hi != 0, msg="Arctan bounds never at x/a = 0")

        # Misconceptions
        dropped_k = ans / k
        sign_flip = -ans
        only_upper_bound = F.subs(x, hi) - F.subs(x, 0)

        check_distinct(ans, dropped_k, sign_flip, only_upper_bound, msg="Arctan distractors")

        candidates = [
            _typed(dropped_k),
            _typed(sign_flip),
            _typed(only_upper_bound),
        ]
        distractors = pick_distractors(_typed(ans), candidates)

        return {
            "text": f"Find ∫ from x = {_prose(lo)} to x = {_prose(hi)} of {_prose(f)} dx. Give your answer as an exact value in terms of pi.",
            "latex": f"Find ∫ from x = {_prose(lo)} to x = {_prose(hi)} of {_prose(f)} dx. Give your answer as an exact value in terms of pi.",
            "answer": str(ans),
            "answer_text": _typed(ans),
            "answer_latex": latex_answer(ans),
            "distractors": distractors,
        }


@generator(
    "determine_volumes_of_solids_of_revolution_about_either",
    level=17,
    presets={
        "easy": dict(kind="line_x_axis"),
        "medium": dict(kind="parabola_x_axis"),
        "hard": dict(kind="parabola_y_axis"),
    },
)
def determine_volumes_of_solids_of_revolution_about_either(rng, kind="line_x_axis"):
    """
    Determine volumes of solids of revolution about either axis using disk method.

    easy: line y = kx about x-axis with zero lower bound
    medium: parabola y = x^2 about x-axis with two nonzero bounds
    hard: parabola y = x^2 about y-axis, integrated in y
    """

    if kind == "line_x_axis":
        # Disk method: y = kx rotated about x-axis
        k = rng.randint(1, 6)
        # Exclude b=3 to avoid collision where wrong_power equals answer
        b = rng.choice([2, 4, 5, 6, 7])
        integrand = (k * x) ** 2
        F = k**2 * x**3 / 3
        lo = 0
        hi = b
        V = pi * (F.subs(x, hi) - F.subs(x, lo))

        # Independent checks
        check_equivalent(diff(F, x), integrand, msg="Derivative of antiderivative")
        check_equivalent(V, pi * integrate(integrand, (x, lo, hi)), msg="Volume integral")

        # Misconceptions
        forgot_pi = V / pi
        forgot_square = pi * integrate(k * x, x).subs(x, hi) - pi * integrate(k * x, x).subs(x, lo)
        wrong_power = pi * k**2 * (b**2 - 0)  # used x^2 as antiderivative of (kx)^2

        check_distinct(V, forgot_pi, forgot_square, wrong_power, msg="Line x-axis volume distractors")

        candidates = [
            _typed(forgot_pi),
            _typed(forgot_square),
            _typed(wrong_power),
        ]
        distractors = pick_distractors(_typed(V), candidates)

        curve = f"y = {_prose(k)}·x"
        axis_name = "x-axis"
        line1 = f"x = {lo}"
        line2 = f"x = {hi}"
        prompt = f"The region bounded by the curve {curve}, the {axis_name}, and the lines {line1} and {line2}, is rotated about the {axis_name}. Find the volume of the resulting solid. Give your answer as an exact value in terms of pi."

        return {
            "text": prompt,
            "latex": prompt,
            "answer": str(V),
            "answer_text": _typed(V),
            "answer_latex": latex_answer(V),
            "distractors": distractors,
        }

    elif kind == "parabola_x_axis":
        # Disk method: y = x^2 rotated about x-axis with two nonzero bounds
        a_ = rng.randint(1, 6)
        gap = rng.randint(1, 5)
        b_ = a_ + gap
        integrand = (x**2) ** 2
        F = x**5 / 5
        lo = a_
        hi = b_
        V = pi * (F.subs(x, hi) - F.subs(x, lo))

        # Independent checks
        check_equivalent(diff(F, x), integrand, msg="Derivative of antiderivative")
        check_equivalent(V, pi * integrate(integrand, (x, lo, hi)), msg="Volume integral")

        # Misconceptions
        forgot_pi = V / pi
        forgot_square = pi * integrate(x**2, x).subs(x, hi) - pi * integrate(x**2, x).subs(x, lo)
        only_upper_bound = pi * (F.subs(x, hi) - 0)

        check_distinct(V, forgot_pi, forgot_square, only_upper_bound, msg="Parabola x-axis volume distractors")

        candidates = [
            _typed(forgot_pi),
            _typed(forgot_square),
            _typed(only_upper_bound),
        ]
        distractors = pick_distractors(_typed(V), candidates)

        curve = "y = x^2"
        axis_name = "x-axis"
        line1 = f"x = {lo}"
        line2 = f"x = {hi}"
        prompt = f"The region bounded by the curve {curve}, the {axis_name}, and the lines {line1} and {line2}, is rotated about the {axis_name}. Find the volume of the resulting solid. Give your answer as an exact value in terms of pi."

        return {
            "text": prompt,
            "latex": prompt,
            "answer": str(V),
            "answer_text": _typed(V),
            "answer_latex": latex_answer(V),
            "distractors": distractors,
        }

    elif kind == "parabola_y_axis":
        # Disk method about y-axis: y = x^2 (x >= 0), so x = sqrt(y)
        # Integrate in y: V = π ∫ y dy
        # Ensure c_ > 0 to avoid collision with only_upper_bound misconception
        c_ = rng.randint(1, 6)
        gap = rng.randint(2, 6)
        d_ = c_ + gap
        integrand_y = y
        F_y = y**2 / 2
        lo = c_
        hi = d_
        V = pi * (F_y.subs(y, hi) - F_y.subs(y, lo))

        # Independent checks
        check_equivalent(diff(F_y, y), integrand_y, msg="Derivative of antiderivative")
        check_equivalent(V, pi * integrate(integrand_y, (y, lo, hi)), msg="Volume integral")

        # Misconceptions
        forgot_pi = V / pi
        used_x_axis_formula = pi * integrate((x**2) ** 2, (x, sqrt(c_), sqrt(d_)))
        only_upper_bound = pi * (F_y.subs(y, hi) - 0)

        check_distinct(V, forgot_pi, used_x_axis_formula, only_upper_bound, msg="Parabola y-axis volume distractors")

        candidates = [
            _typed(forgot_pi),
            _typed(used_x_axis_formula),
            _typed(only_upper_bound),
        ]
        distractors = pick_distractors(_typed(V), candidates)

        # Explicitly state curve without unicode characters
        curve = "x = sqrt(y) (that is, y = x^2 for x >= 0)"
        axis_name = "y-axis"
        line1 = f"y = {lo}"
        line2 = f"y = {hi}"
        prompt = f"The region bounded by the curve {curve}, the {axis_name}, and the lines {line1} and {line2}, is rotated about the {axis_name}. Find the volume of the resulting solid. Give your answer as an exact value in terms of pi."

        return {
            "text": prompt,
            "latex": prompt,
            "answer": str(V),
            "answer_text": _typed(V),
            "answer_latex": latex_answer(V),
            "distractors": distractors,
        }
