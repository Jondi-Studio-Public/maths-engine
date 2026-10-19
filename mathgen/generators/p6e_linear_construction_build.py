"""Skill 1: Find the equation of a straight line given sufficient information;
parallel and perpendicular lines. Skill 2: Develop a linear formula from a word
description."""

import math

from sympy import symbols, expand, Rational, simplify

from ..core import (
    article_for_number,
    generator,
    check,
    check_distinct,
    check_equivalent,
    join_terms,
    pick_distractors,
)

x, y = symbols("x y")


def _poly_display(expr) -> str:
    """Format a polynomial expression for display: replace ** with ^ and remove *."""
    return str(expr).replace("**", "^").replace("*", "")


def _standard_form_from_two_points(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int]:
    """Integer `(a, b, c)` for the line through two integer points, in
    `ax + by + c = 0` form, with `a > 0` and `gcd(|a|, |b|, |c|) == 1`. Pure integer cross-multiplication --
    `(y2-y1)(x-x1) = (x2-x1)(y-y1)` -- never float rounding: `x1 != x2` is
    guaranteed by every caller (a vertical line never arises from this
    generator's draws), so `a` and `b` are never simultaneously zero and
    `g` below is always a positive integer.
    """
    a = y2 - y1
    b = -(x2 - x1)
    c = -(a * x1 + b * y1)
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    a, b, c = a // g, b // g, c // g
    if a < 0:
        a, b, c = -a, -b, -c
    return a, b, c


def _standard_form_text(a: int, b: int, c: int) -> str:
    """`"ax + by + c = 0"` the way a student would write it -- `fmt_signed`/
    `join_terms` handle the sign display, the dropped-when-zero constant
    term, and `a == 1`/`b == 1` (rendered as bare `x`/`y`), so this never
    hand-rolls any of that."""
    return f"{join_terms([(a, 'x'), (b, 'y'), (c, '')])} = 0"


# ============================================================================
# SKILL 1: senior-find-the-equation-of-a-straight-line-given
# ============================================================================


@generator(
    "find_the_equation_of_a_straight_line_given",
    level=11,
    presets={
        "easy": dict(variant="two_points"),
        "medium": dict(variant="parallel"),
        "hard": dict(variant="perpendicular"),
        "exam": dict(variant="standard_form"),
    },
)
def find_the_equation_of_a_straight_line_given(rng, variant="two_points"):
    """Find the equation of a straight line given two points, or via parallel/perpendicular construction.

    Four variants:
    - two_points: given two points, find y = mx + c
    - parallel: given a reference line and a point, find the parallel line through that point
    - perpendicular: given a reference line and a point, find the perpendicular line through that point
    - standard_form: given two points, find ax + by + c = 0 (integer a > 0, fully reduced) --
      exam source
    """

    if variant == "two_points":
        # easy: two points -> straight line equation
        # Resample-on-collision loop for m_wrong2 != m condition
        while True:
            # Draw m from {-4,-3,-2,-1,1,2,3,4}, excluding {-1,1} for distractor safety
            m_choices = [-4, -3, -2, 2, 3, 4]
            m = rng.choice(m_choices)
            c = rng.randint(-8, 8)
            expr = expand(m * x + c)

            # Draw two distinct x-coordinates from {-6..6}, x1 != x2
            x1 = rng.randint(-6, 6)
            x2 = rng.randint(-6, 6)
            while x2 == x1:
                x2 = rng.randint(-6, 6)

            y1 = expr.subs(x, x1)
            y2 = expr.subs(x, x2)

            # Guard: x2 != 0 for used-total-not-change distractor
            if x2 == 0:
                continue

            # Guard: c != 0 for swapped-intercept-sign distractor
            if c == 0:
                continue

            # Check that m_wrong2 != m for used-total-not-change to be distinct
            m_wrong2 = Rational(y2, x2)
            if m_wrong2 == m:
                continue

            # All conditions met, break out
            break

        text = f"A straight line passes through ({x1}, {y1}) and ({x2}, {y2}). Give the equation of the line, as y in terms of x."
        answer = expr
        answer_text = f"y = {_poly_display(expand(m * x + c))}"

        # Independent check (different route): compute gradient from two points, then intercept
        m_check = Rational(y2 - y1, x2 - x1)
        c_check = y1 - m_check * x1
        check_equivalent(m_check * x + c_check, expr, "independent check: gradient from two points")

        # Misconceptions:
        # 1. reversed-rise-run: swapped rise and run
        m_wrong1 = Rational(x2 - x1, y2 - y1)  # only if y2 != y1, which is guaranteed since m != 0
        distractor1 = expand(m_wrong1 * x + (y1 - m_wrong1 * x1))
        check_distinct(expr, distractor1, msg="reversed-rise-run must differ from answer")

        # 2. used-total-not-change: use y2 / x2 as gradient instead of (y2-y1)/(x2-x1)
        distractor2 = expand(m_wrong2 * x + (y2 - m_wrong2 * x2))
        check_distinct(expr, distractor2, msg="used-total-not-change must differ from answer")

        # 3. swapped-intercept-sign: negate c but keep m
        distractor3 = expand(m * x - c)
        check_distinct(expr, distractor3, msg="swapped-intercept-sign must differ from answer")

        distractors = [
            f"y = {_poly_display(distractor1)}",
            f"y = {_poly_display(distractor2)}",
            f"y = {_poly_display(distractor3)}",
        ]

    elif variant == "parallel":
        # medium: parallel line through a point
        # Resample loop to ensure distractors are distinct
        while True:
            # Draw reference line: m from {-4,-3,-2,-1,1,2,3,4}, b0 from {-8..8}
            m_choices = [-4, -3, -2, -1, 1, 2, 3, 4]
            m = rng.choice(m_choices)
            b0 = rng.randint(-8, 8)

            # Draw target point (x1, y1) not on reference line
            x1 = rng.randint(-6, 6)
            y1 = rng.randint(-8, 8)
            while y1 == m * x1 + b0:
                y1 = rng.randint(-8, 8)

            # Target line: same gradient m, passing through (x1, y1)
            c = y1 - m * x1
            expr = expand(m * x + c)

            # Ensure c != b0 (for reused-reference-intercept to be distinct)
            if c == b0:
                continue

            # All conditions met, break out
            break

        text = f"A line is parallel to y = {_poly_display(expand(m * x + b0))} and passes through the point ({x1}, {y1}). Give the equation of this line, as y in terms of x."
        answer = expr
        answer_text = f"y = {_poly_display(expand(m * x + c))}"

        # Independent check (different route):
        # 1. confirm expr passes through (x1, y1) by direct substitution
        check(expr.subs(x, x1) == y1, "constructed line must pass through the given point")
        # 2. confirm gradient equals m by sampling two x-values and computing rise/run
        y_at_5 = expr.subs(x, 5)
        y_at_2 = expr.subs(x, 2)
        m_check = (y_at_5 - y_at_2) / Rational(5 - 2)
        check_equivalent(m_check, m, "gradient check from two sampled points")

        # Misconceptions:
        # 1. used-perpendicular-gradient: use -1/m instead of m
        m_perp = Rational(-1, m)
        distractor1 = expand(m_perp * x + (y1 - m_perp * x1))
        check_distinct(expr, distractor1, msg="used-perpendicular-gradient must differ from answer")

        # 2. reused-reference-intercept: used b0 instead of computing c
        # This is distinct whenever c != b0, which holds because (x1,y1) was drawn off the reference line
        # y1 != m*x1 + b0 => y1 - m*x1 != b0 => c != b0
        distractor2 = expand(m * x + b0)
        check_distinct(expr, distractor2, msg="reused-reference-intercept must differ from answer")

        distractors = [
            f"y = {_poly_display(distractor1)}",
            f"y = {_poly_display(distractor2)}",
        ]

    elif variant == "perpendicular":
        # hard: perpendicular line through a point
        # Resample loop to ensure distractors are distinct
        while True:
            # Draw reference line with m != 0
            m_choices = [-4, -3, -2, -1, 1, 2, 3, 4]
            m = rng.choice(m_choices)
            b0 = rng.randint(-8, 8)

            # Draw target point (x1, y1) not on reference line
            x1 = rng.randint(-6, 6)
            y1 = rng.randint(-8, 8)
            while y1 == m * x1 + b0:
                y1 = rng.randint(-8, 8)

            # Perpendicular gradient
            m_perp = Rational(-1, m)
            c = y1 - m_perp * x1
            expr = expand(m_perp * x + c)

            # Ensure c != b0 (for reused-reference-intercept to be distinct)
            if c == b0:
                continue

            # All conditions met, break out
            break

        text = f"A line is perpendicular to y = {_poly_display(expand(m * x + b0))} and passes through the point ({x1}, {y1}). Give the equation of this line, as y in terms of x."
        answer = expr
        answer_text = f"y = {_poly_display(expand(m_perp * x + c))}"

        # Independent check (different route):
        # 1. confirm expr passes through (x1, y1) by direct substitution
        check(expr.subs(x, x1) == y1, "constructed line must pass through the given point")
        # 2. confirm gradient equals m_perp by sampling two x-values and computing rise/run
        y_at_5 = expr.subs(x, 5)
        y_at_2 = expr.subs(x, 2)
        m_check = (y_at_5 - y_at_2) / Rational(5 - 2)
        check_equivalent(m_check, m_perp, "perpendicular gradient check from two sampled points")
        # 3. confirm perpendicularity: m * m_perp == -1 (recomputed from expr)
        m_from_expr = (expr.subs(x, 5) - expr.subs(x, 2)) / Rational(3)
        check_equivalent(m * m_from_expr, -1, "perpendicular gradient product must equal -1")

        # Misconceptions:
        # 1. used-same-gradient: parallel instead of perpendicular (used m instead of -1/m)
        distractor1 = expand(m * x + (y1 - m * x1))
        check_distinct(expr, distractor1, msg="used-same-gradient must differ from answer")

        # 2. reused-reference-intercept: used b0 instead of computing c
        distractor2 = expand(m_perp * x + b0)
        check_distinct(expr, distractor2, msg="reused-reference-intercept must differ from answer")

        distractors = [
            f"y = {_poly_display(distractor1)}",
            f"y = {_poly_display(distractor2)}",
        ]

    elif variant == "standard_form":
        # exam: two points -> the equation of the line in ax + by + c = 0
        # form, integer a > 0, gcd(|a|, |b|, |c|) == 1. Reuses the two_points variant's point-drawing /
        # gradient-guard loop above -- only the final rendering and
        # answer_form differ.
        while True:
            m_choices = [-4, -3, -2, 2, 3, 4]
            m = rng.choice(m_choices)
            c_int = rng.randint(-8, 8)
            expr = expand(m * x + c_int)

            x1 = rng.randint(-6, 6)
            x2 = rng.randint(-6, 6)
            while x2 == x1:
                x2 = rng.randint(-6, 6)

            y1 = int(expr.subs(x, x1))
            y2 = int(expr.subs(x, x2))

            # Guard: c_int != 0 so the constant term is never trivially zero
            # -- needed for the swapped-constant-sign distractor below to be
            # a genuinely different value.
            if c_int == 0:
                continue

            break

        templates = [
            f"A straight line passes through ({x1}, {y1}) and ({x2}, {y2}). "
            "Write the equation of the line in the form ax + by + c = 0, "
            "where a, b and c are integers, a > 0, and the equation shares "
            "no common factor.",
            f"Find the equation of the line through the points ({x1}, {y1}) "
            f"and ({x2}, {y2}). Give your answer in general form "
            "ax + by + c = 0, with integer coefficients, a > 0, fully "
            "reduced.",
            f"Two points on a straight line are ({x1}, {y1}) and "
            f"({x2}, {y2}). Express the equation of this line as "
            "ax + by + c = 0, using integers with a > 0 that share no "
            "common factor.",
        ]
        text = rng.choice(templates)

        # Primary derivation: cross-multiplication directly from the two
        # points, integer arithmetic throughout (never float rounding).
        a, b, c = _standard_form_from_two_points(x1, y1, x2, y2)
        answer = a * x + b * y + c
        answer_text = _standard_form_text(a, b, c)

        # Independent check (different route, not trust in the derivation
        # above): both ORIGINAL points substituted directly into
        # a*x + b*y + c must give exactly 0.
        check(a * x1 + b * y1 + c == 0, "constructed standard-form line must pass through the first point")
        check(a * x2 + b * y2 + c == 0, "constructed standard-form line must pass through the second point")
        # Second independent route: the already-known gradient/intercept
        # (m, c_int) used to draw the points must describe the EXACT same
        # reduced line -- not merely a proportional one.
        if m > 0:
            check(
                (a, b, c) == (m, -1, c_int),
                "reduced standard form disagrees with the known gradient/intercept",
            )
        else:
            check(
                (a, b, c) == (-m, 1, -c_int),
                "reduced standard form disagrees with the known gradient/intercept",
            )

        # Misconceptions:
        # 1. reversed-rise-run: swap rise and run when computing the
        # gradient from the two points, then express THAT (wrong) line in
        # standard form. y2 != y1 is guaranteed since m != 0.
        m_wrong = Rational(x2 - x1, y2 - y1)
        p, q = m_wrong.p, m_wrong.q  # q > 0 by sympy's own Rational convention
        a1, b1, c1 = _standard_form_from_two_points(x1, y1, x1 + q, y1 + p)
        distractor1 = a1 * x + b1 * y + c1
        check_distinct(answer, distractor1, msg="reversed-rise-run must differ from answer")

        # 2. swapped-constant-sign: negate c but keep a, b (c != 0 is
        # guaranteed by the c_int != 0 draw guard, carried through the
        # reduction exactly, since gcd(|a|,|b|,|c|) == gcd(|a|,|b|,|-c|)).
        a2, b2, c2 = a, b, -c
        distractor2 = a2 * x + b2 * y + c2
        check_distinct(answer, distractor2, msg="swapped-constant-sign must differ from answer")

        distractors = [
            _standard_form_text(a1, b1, c1),
            _standard_form_text(a2, b2, c2),
        ]

    else:
        raise ValueError(f"unknown variant: {variant!r}")

    return dict(
        text=text,
        latex=text,  # no special LaTeX formatting for now
        answer=answer,
        answer_text=answer_text,
        distractors=distractors,
        answer_form="line_standard_form" if variant == "standard_form" else None,
    )


# ============================================================================
# SKILL 2: senior-develop-a-linear-formula-from-a-word-description
# ============================================================================


@generator(
    "develop_a_linear_formula_from_a_word_description",
    level=11,
    presets={
        "easy": dict(variant="increasing"),
        "medium": dict(variant="decreasing"),
        "hard": dict(variant="back_solved_intercept"),
    },
)
def develop_a_linear_formula_from_a_word_description(rng, variant="increasing"):
    """Develop a linear formula from a word description.

    Three variants:
    - increasing: fixed starting amount + constant rate added per unit
    - decreasing: fixed starting amount - constant rate subtracted per unit
    - back_solved_intercept: initial value not stated, but value at a later point is given
    """

    # Scenario templates: (template_string, input_var_letter, output_var_letter, rate_range, intercept_range)
    # Each tuple: (prompt_template, rate_range, intercept_range)

    if variant == "increasing":
        # easy: increasing scenario
        # Guard: rate != intercept so swapped-rate-and-intercept can't coincide with the answer
        while True:
            rate = rng.randint(1, 15)
            intercept = rng.randint(1, 40)
            if rate != intercept:
                break

        # Choose template and format with proper singular/plural
        template_choice = rng.choice([0, 1, 2, 3])
        if template_choice == 0:
            # Gym template
            var_in, var_out = "w", "C"
            text = f"A gym charges {article_for_number(intercept)} ${intercept} joining fee, then ${rate} per week of membership. Write a formula for the total cost C (in dollars) after w weeks."
        elif template_choice == 1:
            # Tank template with singular/plural handling
            var_in, var_out = "t", "V"
            intercept_unit = "litre" if intercept == 1 else "litres"
            rate_unit = "litre" if rate == 1 else "litres"
            text = f"A tank starts with {intercept} {intercept_unit} and is filled at {rate} {rate_unit} per minute. Write a formula for the volume V (in litres) after t minutes."
        elif template_choice == 2:
            # Storage unit template: "charges a $N ... then $N per" with a different business
            var_in, var_out = "m", "C"
            text = f"A storage unit charges {article_for_number(intercept)} ${intercept} setup fee, then ${rate} per month. Write a formula for the total cost C (in dollars) after m months."
        else:
            # Reservoir template: "starts with N ... is filled at N per" with a different container
            var_in, var_out = "h", "V"
            intercept_unit = "litre" if intercept == 1 else "litres"
            rate_unit = "litre" if rate == 1 else "litres"
            text = f"A reservoir starts with {intercept} {intercept_unit} and is filled at {rate} {rate_unit} per hour. Write a formula for the volume V (in litres) after h hours."

        var_in_sym = symbols(var_in)
        formula = expand(rate * var_in_sym + intercept)

        answer = formula
        answer_text = f"{var_out} = {_poly_display(expand(rate * var_in_sym + intercept))}"

        # Independent checks
        # 1. confirm formula.subs(var, 0) == intercept
        check(formula.subs(var_in_sym, 0) == intercept, f"formula at {var_in}=0 must equal {intercept}")
        # 2. confirm rate of change equals the signed rate (positive for increasing)
        val_at_2 = formula.subs(var_in_sym, 2)
        val_at_5 = formula.subs(var_in_sym, 5)
        rate_check = (val_at_5 - val_at_2) / Rational(3)
        check_equivalent(rate_check, rate, "rate of change must equal stated rate (positive for increasing)")

        # Misconceptions, all built from the rate/intercept that the question text actually shows:
        # 1. swapped-rate-and-intercept: swapped rate and intercept
        distractor1 = expand(intercept * var_in_sym + rate)
        check_distinct(formula, distractor1, msg="swapped-rate-and-intercept must differ from answer")

        # 2. dropped-intercept: forgot the starting amount
        distractor2 = expand(rate * var_in_sym)
        check_distinct(formula, distractor2, msg="dropped-intercept must differ from answer (intercept != 0)")

        # 3. wrong-sign-direction: subtracted instead of added
        distractor3 = expand(intercept - rate * var_in_sym)
        check_distinct(formula, distractor3, msg="wrong-sign-direction must differ from answer (rate != 0)")

        distractors = [
            f"{var_out} = {_poly_display(distractor1)}",
            f"{var_out} = {_poly_display(distractor2)}",
            f"{var_out} = {_poly_display(distractor3)}",
        ]

    elif variant == "decreasing":
        # medium: decreasing scenario
        # Guard: rate != intercept so swapped-rate-and-intercept can't coincide with the answer
        while True:
            rate = rng.randint(1, 15)
            intercept = rng.randint(1, 40)
            if rate != intercept:
                break

        # Choose template and format with proper singular/plural
        template_choice = rng.choice([0, 1, 2, 3])
        if template_choice == 0:
            # Car template
            var_in, var_out = "y", "V"
            text = f"A car loses value at ${rate} per year. Initially, it is worth ${intercept}. Write a formula for the value V (in dollars) after y years."
        elif template_choice == 1:
            # Tank template with singular/plural handling
            var_in, var_out = "h", "D"
            intercept_unit = "litre" if intercept == 1 else "litres"
            rate_unit = "litre" if rate == 1 else "litres"
            text = f"A tank drains at {rate} {rate_unit} per hour and starts with {intercept} {intercept_unit}. Write a formula for the volume D (in litres) after h hours."
        elif template_choice == 2:
            # Header tank template: same "drains at/starts with" style, different vessel noun
            var_in, var_out = "h", "D"
            intercept_unit = "litre" if intercept == 1 else "litres"
            rate_unit = "litre" if rate == 1 else "litres"
            text = f"A header tank drains at {rate} {rate_unit} per hour and starts with {intercept} {intercept_unit}. Write a formula for the volume D (in litres) after h hours."
        else:
            # Company car template: same "loses value at/initially worth" style, different vehicle noun
            var_in, var_out = "y", "V"
            text = f"A company car loses value at ${rate} per year. Initially, it is worth ${intercept}. Write a formula for the value V (in dollars) after y years."

        var_in_sym = symbols(var_in)
        # Decreasing: formula = intercept - rate*var
        formula = expand(intercept - rate * var_in_sym)

        answer = formula
        answer_text = f"{var_out} = {_poly_display(expand(intercept - rate * var_in_sym))}"

        # Independent checks
        # 1. confirm formula.subs(var, 0) == intercept
        check(formula.subs(var_in_sym, 0) == intercept, f"formula at {var_in}=0 must equal {intercept}")
        # 2. confirm rate of change equals the negative signed rate (decreasing)
        val_at_2 = formula.subs(var_in_sym, 2)
        val_at_5 = formula.subs(var_in_sym, 5)
        rate_check = (val_at_5 - val_at_2) / Rational(3)
        check_equivalent(rate_check, -rate, "rate of change must equal negative rate (decreasing)")

        # Misconceptions:
        # 1. wrong-sign-direction: added instead of subtracted (used + instead of -)
        distractor1 = expand(intercept + rate * var_in_sym)
        check_distinct(formula, distractor1, msg="wrong-sign-direction must differ from answer (rate != 0)")

        # 2. dropped-intercept: forgot the starting amount
        distractor2 = expand(-rate * var_in_sym)
        check_distinct(formula, distractor2, msg="dropped-intercept must differ from answer (intercept != 0)")

        # 3. swapped-rate-and-intercept: swap the roles of rate and intercept
        distractor3 = expand(rate - intercept * var_in_sym)
        check_distinct(formula, distractor3, msg="swapped-rate-and-intercept must differ from answer")

        distractors = [
            f"{var_out} = {_poly_display(distractor1)}",
            f"{var_out} = {_poly_display(distractor2)}",
            f"{var_out} = {_poly_display(distractor3)}",
        ]

    else:  # variant == "back_solved_intercept"
        # hard: back-solved intercept
        # Rejection sampling: resample rate, t0, and intercept until value >= 5
        # and rate != intercept (so swapped-rate-and-intercept can't coincide with the answer)
        while True:
            rate = rng.randint(1, 15)
            t0 = rng.randint(1, 5)
            intercept = rng.randint(1, 40)
            # Compute value at t0: value = intercept - rate*t0
            value = intercept - rate * t0
            if value >= 5 and rate != intercept:
                break

        # Choose template and format with proper singular/plural
        template_choice = rng.choice([0, 1, 2, 3])
        if template_choice == 0:
            # Car template
            var_in, var_out = "t", "V"
            year_unit = "year" if t0 == 1 else "years"
            text = f"After {t0} {year_unit}, a car that depreciates at ${rate} per year is worth ${value}. Write a formula for the value V (in dollars) after t years, measured from when the car started depreciating."
        elif template_choice == 1:
            # Tank template with singular/plural handling
            var_in, var_out = "t", "V"
            rate_unit = "litre" if rate == 1 else "litres"
            value_unit = "litre" if value == 1 else "litres"
            hour_unit = "hour" if t0 == 1 else "hours"
            text = f"After {t0} {hour_unit}, a tank that drains at {rate} {rate_unit} per hour contains {value} {value_unit}. Write a formula for the volume V (in litres) after t hours, measured from when the tank started draining."
        elif template_choice == 2:
            # Rental car template: same structure, different vehicle noun
            var_in, var_out = "t", "V"
            year_unit = "year" if t0 == 1 else "years"
            text = f"After {t0} {year_unit}, a rental car that depreciates at ${rate} per year is worth ${value}. Write a formula for the value V (in dollars) after t years, measured from when the car started depreciating."
        else:
            # Cooling tank template: same structure, different vessel noun
            var_in, var_out = "t", "V"
            rate_unit = "litre" if rate == 1 else "litres"
            value_unit = "litre" if value == 1 else "litres"
            hour_unit = "hour" if t0 == 1 else "hours"
            text = f"After {t0} {hour_unit}, a cooling tank that drains at {rate} {rate_unit} per hour contains {value} {value_unit}. Write a formula for the volume V (in litres) after t hours, measured from when the tank started draining."

        var_in_sym = symbols(var_in)
        # Formula: V = intercept - rate*t
        formula = expand(intercept - rate * var_in_sym)

        answer = formula
        answer_text = f"{var_out} = {_poly_display(expand(intercept - rate * var_in_sym))}"

        # Independent checks
        # 1. confirm formula.subs(var, t0) == value
        check(formula.subs(var_in_sym, t0) == value, f"formula at {var_in}={t0} must equal {value}")
        # 2. confirm intercept derived from value and t0 matches the constructed intercept
        intercept_check = value + rate * t0
        check_equivalent(intercept_check, intercept, "back-solved intercept must match construction")

        # Misconceptions:
        # 1. used-later-value-as-intercept: used value as if it were the initial value
        distractor1 = expand(value - rate * var_in_sym)
        check_distinct(formula, distractor1, msg="used-later-value-as-intercept must differ from answer (rate*t0 != 0)")

        # 2. wrong-sign-direction: used + instead of - for decreasing
        distractor2 = expand(intercept + rate * var_in_sym)
        check_distinct(formula, distractor2, msg="wrong-sign-direction must differ from answer (rate != 0)")

        # 3. swapped-rate-and-intercept: swap the roles of rate and intercept
        distractor3 = expand(rate - intercept * var_in_sym)
        check_distinct(formula, distractor3, msg="swapped-rate-and-intercept must differ from answer")

        distractors = [
            f"{var_out} = {_poly_display(distractor1)}",
            f"{var_out} = {_poly_display(distractor2)}",
            f"{var_out} = {_poly_display(distractor3)}",
        ]

    return dict(
        text=text,
        latex=text,  # no special LaTeX formatting for now
        answer=answer,
        answer_text=answer_text,
        distractors=distractors,
    )
