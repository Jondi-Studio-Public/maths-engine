"""Vector equations of curves (ACMSM104) and position as function of time (ACMSM111).

SKILL 1: Vector equations of curves in two dimensions
Eliminates parameter t from the curve r(t) = (a*t, b/t) to find a Cartesian relation.
Three tiers: easy (positive a, b), medium (mixed signs), hard (y-value evaluation).
All answers are numeric values (the constant k in xy=k, or a y-value).

SKILL 2: Position vectors as functions of time
Evaluates a stated position vector at a stated time, or finds the displacement
between two stated times by direct substitution and subtraction.
Three tiers: easy (linear position), medium (quadratic position), hard (displacement).
Answers are evaluated numeric pairs (x, y), never symbolic derivatives/integrals.

CRITICAL BOUNDARY WITH ACMSM113 (differentiate-and-integrate-a-vector-function):
This generator NEVER calls diff() or any integration operation.
ACMSM113 (already built in p5e_vector_calculus_build.py) owns all rate-of-change
content. Skill 2 is evaluation only.
"""

from __future__ import annotations

from fractions import Fraction
from sympy import symbols, simplify

from ..core import check, check_distinct, check_equivalent, generator, pick_distractors


def _fmt_signed_term(coef: int, var: str) -> str:
    """Format a single term with its sign word.

    Returns strings like "+ 3t", "- 2", etc., or empty string for zero coef.
    """
    if coef == 0:
        return ""
    sign_word = "+ " if coef > 0 else "- "
    abs_coef = abs(coef)
    if var and abs_coef == 1:
        return f"{sign_word}{var}"
    if var:
        return f"{sign_word}{abs_coef}{var}"
    return f"{sign_word}{abs_coef}"


def _fmt_component_linear(a: int, b: int) -> str:
    """Format one linear component ``a*t + b``.

    Returns strings like "4t + 2", "6t - 1", "t + 3", "t", "-t".
    """
    if a == 0:
        return str(b)
    lead = "t" if a == 1 else ("-t" if a == -1 else f"{a}t")
    if b == 0:
        return lead
    return f"{lead} {'+' if b > 0 else '-'} {abs(b)}"


def _fmt_component_quadratic(a: int, b: int, p: int) -> str:
    """Format one component of r(t) = (a*t^2 + b*t + p, ...).

    Returns strings like "3t^2 + 2t + 1", "t^2 - 5t", etc.
    """
    parts = []
    if a != 0:
        if a == 1:
            parts.append("t^2")
        else:
            parts.append(f"{a}t^2")

    if b != 0:
        if b > 0:
            if b == 1:
                parts.append("+ t")
            else:
                parts.append(f"+ {b}t")
        else:  # b < 0
            if b == -1:
                parts.append("- t")
            else:
                parts.append(f"- {abs(b)}t")

    if p != 0:
        if p > 0:
            parts.append(f"+ {p}")
        else:
            parts.append(f"- {abs(p)}")

    return " ".join(parts) if parts else "0"


# ============================================================================
# SKILL 1: Use vector equations of curves in two dimensions (ACMSM104)
# ============================================================================


@generator(
    "use_vector_equations_of_curves_in_two_or",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def use_vector_equations_of_curves_in_two_or(rng, tier="easy"):
    """Eliminate parameter from curve to find Cartesian equation or evaluate it."""

    if tier == "easy":
        # Easy: xy = k where k = a*b (positive only)
        for attempt in range(200):
            a = rng.randint(2, 9)
            b = rng.randint(2, 9)
            k = a * b

            # Independent check via sympy: build x = a*t, y = b/t, verify product simplifies to k
            t_sym = symbols("t")
            x_expr = a * t_sym
            y_expr = b / t_sym
            product = simplify(x_expr * y_expr)
            check_equivalent(product, k, f"easy: xy product check failed")

            # Distractors: common misconceptions
            adds_instead = a + b
            squares_wrong = b * b
            off_by_one = a * b + 1

            # Check all distractors are distinct from answer (no try/except to maintain determinism)
            if (
                k != adds_instead
                and k != squares_wrong
                and k != off_by_one
                and adds_instead != squares_wrong
                and adds_instead != off_by_one
                and squares_wrong != off_by_one
            ):
                # All distinct, proceed with final checks
                check_distinct(k, adds_instead, msg="easy: adds-instead-of-multiplies")
                check_distinct(k, squares_wrong, msg="easy: squares-wrong-variable")
                check_distinct(k, off_by_one, msg="easy: off-by-one-product")
                break

        answer_text = str(k)
        distractor_strs = [str(adds_instead), str(squares_wrong), str(off_by_one)]

        text = (
            f"A curve has vector equation r(t) = ({a}t, {b}/t) for t not equal to 0. "
            f"Eliminating the parameter t gives a Cartesian equation of the form xy = k. Find k."
        )

        return dict(
            text=text,
            latex=rf"r(t) = ({a}t, {b}/t); \text{{find }} k \text{{ where }} xy = k",
            answer=k,
            answer_text=answer_text,
            working=[
                f"r(t) = ({a}t, {b}/t)",
                f"From x = {a}t, we get t = x/{a}",
                f"Substituting into y = {b}/t: y = {b} / (x/{a}) = {b} × {a}/x = {k}/x",
                f"So xy = {k}",
            ],
            distractors=pick_distractors(answer_text, distractor_strs),
        )

    elif tier == "medium":
        # Medium: xy = k where k = a*b (can be negative)
        for attempt in range(200):
            a = rng.choice([v for v in range(-9, 10) if v != 0])
            b = rng.choice([v for v in range(-9, 10) if v != 0])
            k = a * b

            # Independent check via sympy
            t_sym = symbols("t")
            x_expr = a * t_sym
            y_expr = b / t_sym
            product = simplify(x_expr * y_expr)
            check_equivalent(product, k, f"medium: xy product check failed")

            # Distractors: sign-mix escalation
            adds_instead = a + b
            squares_wrong = b * b
            off_by_one = a * b + 1
            sign_dropped = abs(a * b)

            # Check distinctness without try/except
            if k < 0:
                # Include sign_dropped when k is negative
                if (
                    k != adds_instead
                    and k != squares_wrong
                    and k != off_by_one
                    and k != sign_dropped
                    and adds_instead != squares_wrong
                    and adds_instead != off_by_one
                    and adds_instead != sign_dropped
                    and squares_wrong != off_by_one
                    and squares_wrong != sign_dropped
                    and off_by_one != sign_dropped
                ):
                    check_distinct(k, adds_instead, msg="medium: adds-instead-of-multiplies")
                    check_distinct(k, squares_wrong, msg="medium: squares-wrong-variable")
                    check_distinct(k, off_by_one, msg="medium: off-by-one-product")
                    check_distinct(k, sign_dropped, msg="medium: sign-dropped")
                    break
            else:
                # Don't include sign_dropped when k is positive
                if (
                    k != adds_instead
                    and k != squares_wrong
                    and k != off_by_one
                    and adds_instead != squares_wrong
                    and adds_instead != off_by_one
                    and squares_wrong != off_by_one
                ):
                    check_distinct(k, adds_instead, msg="medium: adds-instead-of-multiplies")
                    check_distinct(k, squares_wrong, msg="medium: squares-wrong-variable")
                    check_distinct(k, off_by_one, msg="medium: off-by-one-product")
                    break

        answer_text = str(k)
        distractor_strs = [str(adds_instead), str(squares_wrong), str(off_by_one)]
        if k < 0:
            distractor_strs.append(str(sign_dropped))

        # Format the coefficient of t correctly: "t" for a==1, "-t" for a==-1, "{a}t" otherwise
        a_coeff = _fmt_component_linear(a, 0)

        text = (
            f"A curve has vector equation r(t) = ({a_coeff}, {b}/t) for t not equal to 0. "
            f"Eliminating the parameter t gives a Cartesian equation of the form xy = k. Find k."
        )

        return dict(
            text=text,
            latex=rf"r(t) = ({a_coeff}, {b}/t); \text{{find }} k \text{{ where }} xy = k",
            answer=k,
            answer_text=answer_text,
            working=[
                f"r(t) = ({a_coeff}, {b}/t)",
                f"From x = {a_coeff}, we get t = x/{a}",
                f"Substituting into y = {b}/t: y = {b} / (x/{a}) = {b} × {a}/x = {k}/x",
                f"So xy = {k}",
            ],
            distractors=pick_distractors(answer_text, distractor_strs),
        )

    elif tier == "hard":
        # Hard: evaluate y at a given x, where x(t) = a*t, y(t) = b/t + c
        for attempt in range(200):
            a = rng.randint(2, 9)
            t0 = rng.randint(2, 4)
            m = rng.randint(2, 9)
            b = t0 * m
            c = rng.choice([v for v in range(-9, 10) if v != 0])
            x0 = a * t0

            # Correct y: y0 = b/t0 + c = m + c
            y0 = m + c

            # Independent check via sympy
            t_sym = symbols("t")
            x_sym = a * t_sym
            y_sym = b / t_sym + c
            t0_val = Fraction(x0, a)
            check_equivalent(t0_val, t0, f"hard: t0 recomputation check failed")
            y0_val = b / t0_val + c
            check_equivalent(y0_val, y0, f"hard: y0 evaluation check failed")

            # Distractors
            forgot_offset = m
            sign_error_offset = m - c
            off_by_one_on_m = (m + 1) + c

            # Check distinctness without try/except
            if (
                y0 != forgot_offset
                and y0 != sign_error_offset
                and y0 != off_by_one_on_m
                and forgot_offset != sign_error_offset
                and forgot_offset != off_by_one_on_m
                and sign_error_offset != off_by_one_on_m
            ):
                check_distinct(y0, forgot_offset, msg="hard: forgot-offset")
                check_distinct(y0, sign_error_offset, msg="hard: sign-error-offset")
                check_distinct(y0, off_by_one_on_m, msg="hard: off-by-one-on-m")
                break

        answer_text = str(y0)
        distractor_strs = [str(forgot_offset), str(sign_error_offset), str(off_by_one_on_m)]

        c_signed = _fmt_signed_term(c, "")
        text = (
            f"A curve has vector equation r(t) = ({a}t, {b}/t {c_signed}) for t not equal to 0. Find y when x = {x0}."
        )

        return dict(
            text=text,
            latex=rf"r(t) = ({a}t, {b}/t{_fmt_signed_term(c, '')}); \text{{find }} y \text{{ when }} x = {x0}",
            answer=y0,
            answer_text=answer_text,
            working=[
                f"r(t) = ({a}t, {b}/t {_fmt_signed_term(c, '')})" if c != 0 else f"r(t) = ({a}t, {b}/t)",
                f"From x = {a}t with x = {x0}: t = {x0}/{a} = {t0}",
                f"Substituting: y = {b}/{t0} {_fmt_signed_term(c, '')} = {m} {_fmt_signed_term(c, '')} = {y0}",
            ],
            distractors=pick_distractors(answer_text, distractor_strs),
        )

    else:
        raise ValueError(f"unknown tier: {tier}")


# ============================================================================
# SKILL 2: Consider position of vectors as a function of time (ACMSM111)
# ============================================================================


@generator(
    "consider_position_of_vectors_as_a_function_of",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def consider_position_of_vectors_as_a_function_of(rng, tier="easy"):
    """Evaluate position vector at stated time or find displacement between times."""

    if tier == "easy":
        # Easy: linear position r(t) = (a*t + p, c*t + q), evaluate at t0
        for attempt in range(200):
            a = rng.randint(1, 9)
            c = rng.randint(1, 9)
            p = rng.randint(-9, 9)
            q = rng.randint(-9, 9)
            t0 = rng.randint(1, 6)

            x0 = a * t0 + p
            y0 = c * t0 + q

            # Independent check via sympy
            t_sym = symbols("t")
            x_component = a * t_sym + p
            y_component = c * t_sym + q
            x0_val = x_component.subs(t_sym, t0)
            y0_val = y_component.subs(t_sym, t0)
            check_equivalent(x0_val, x0, "easy: x0 evaluation check failed")
            check_equivalent(y0_val, y0, "easy: y0 evaluation check failed")

            # Distractors
            dropped_constant_x = a * t0
            dropped_constant_y = c * t0
            swapped_x = y0
            swapped_y = x0
            sign_error_x = a * (-t0) + p
            sign_error_y = c * (-t0) + q

            # Check distinctness (manual check, no need to call check_distinct for tuples)
            if (
                (x0, y0) != (dropped_constant_x, dropped_constant_y)
                and (x0, y0) != (swapped_x, swapped_y)
                and (x0, y0) != (sign_error_x, sign_error_y)
                and (dropped_constant_x, dropped_constant_y) != (swapped_x, swapped_y)
                and (dropped_constant_x, dropped_constant_y) != (sign_error_x, sign_error_y)
                and (swapped_x, swapped_y) != (sign_error_x, sign_error_y)
            ):
                break

        answer_text = f"x = {x0}, y = {y0}"
        distractor_strs = [
            f"x = {dropped_constant_x}, y = {dropped_constant_y}",
            f"x = {swapped_x}, y = {swapped_y}",
            f"x = {sign_error_x}, y = {sign_error_y}",
        ]

        p_signed = _fmt_signed_term(p, "")
        q_signed = _fmt_signed_term(q, "")
        time_unit = "second" if t0 == 1 else "seconds"

        x_comp = _fmt_component_linear(a, p)
        y_comp = _fmt_component_linear(c, q)

        text = (
            f"A particle's position vector is r(t) = ({x_comp}, {y_comp}), "
            f"in metres, t in seconds. Find the particle's position vector at t = {t0} {time_unit}."
        )

        return dict(
            text=text,
            latex=rf"r(t) = ({x_comp}, {y_comp}); \text{{find }} r({t0})",
            answer={"x": float(x0), "y": float(y0)},
            answer_text=answer_text,
            working=[
                f"r(t) = ({x_comp}, {y_comp})",
                f"At t = {t0}: r({t0}) = ({a} × {t0} {p_signed}, {c} × {t0} {q_signed})",
                f"= ({x0}, {y0})",
            ],
            distractors=pick_distractors(answer_text, distractor_strs),
        )

    elif tier == "medium":
        # Medium: quadratic position r(t) = (a*t^2 + b*t + p, c*t^2 + d*t + q), evaluate at t0
        for attempt in range(200):
            a = rng.randint(1, 4)
            b = rng.randint(-6, 6)
            p = rng.randint(-9, 9)
            c = rng.randint(1, 4)
            d = rng.randint(-6, 6)
            q = rng.randint(-9, 9)
            t0 = rng.randint(1, 4)

            x0 = a * t0**2 + b * t0 + p
            y0 = c * t0**2 + d * t0 + q

            # Independent check via sympy
            t_sym = symbols("t")
            x_component = a * t_sym**2 + b * t_sym + p
            y_component = c * t_sym**2 + d * t_sym + q
            x0_val = x_component.subs(t_sym, t0)
            y0_val = y_component.subs(t_sym, t0)
            check_equivalent(x0_val, x0, "medium: x0 evaluation check failed")
            check_equivalent(y0_val, y0, "medium: y0 evaluation check failed")

            # Distractors
            forgot_square_x = (a + b) * t0 + p
            forgot_square_y = (c + d) * t0 + q
            swapped_x = y0
            swapped_y = x0
            sign_error_x = a * (-t0) ** 2 + b * (-t0) + p
            sign_error_y = c * (-t0) ** 2 + d * (-t0) + q

            # Check distinctness (manual check, no need to call check_distinct for tuples)
            if (
                (x0, y0) != (forgot_square_x, forgot_square_y)
                and (x0, y0) != (swapped_x, swapped_y)
                and (x0, y0) != (sign_error_x, sign_error_y)
                and (forgot_square_x, forgot_square_y) != (swapped_x, swapped_y)
                and (forgot_square_x, forgot_square_y) != (sign_error_x, sign_error_y)
                and (swapped_x, swapped_y) != (sign_error_x, sign_error_y)
            ):
                break

        answer_text = f"x = {x0}, y = {y0}"
        distractor_strs = [
            f"x = {forgot_square_x}, y = {forgot_square_y}",
            f"x = {swapped_x}, y = {swapped_y}",
            f"x = {sign_error_x}, y = {sign_error_y}",
        ]

        x_comp = _fmt_component_quadratic(a, b, p)
        y_comp = _fmt_component_quadratic(c, d, q)
        time_unit = "second" if t0 == 1 else "seconds"

        text = (
            f"A particle's position vector is r(t) = ({x_comp}, {y_comp}), "
            f"in metres, t in seconds. Find the particle's position vector at t = {t0} {time_unit}."
        )

        return dict(
            text=text,
            latex=rf"r(t) = ({x_comp}, {y_comp}); \text{{find }} r({t0})",
            answer={"x": float(x0), "y": float(y0)},
            answer_text=answer_text,
            working=[
                f"r(t) = ({x_comp}, {y_comp})",
                f"At t = {t0}: r({t0}) = ({a} × {t0}^2 {_fmt_signed_term(b, str(t0))} {_fmt_signed_term(p, '')}, "
                f"{c} × {t0}^2 {_fmt_signed_term(d, str(t0))} {_fmt_signed_term(q, '')})",
                f"= ({x0}, {y0})",
            ],
            distractors=pick_distractors(answer_text, distractor_strs),
        )

    elif tier == "hard":
        # Hard: displacement between two times; r(t) = (a*t^2 + b*t + p, c*t^2 + d*t + q)
        for attempt in range(200):
            a = rng.randint(1, 4)
            b = rng.randint(-6, 6)
            p = rng.randint(-9, 9)
            c = rng.randint(1, 4)
            d = rng.randint(-6, 6)
            q = rng.randint(-9, 9)
            t1 = rng.randint(1, 4)
            t2 = rng.randint(1, 4)
            if t1 == t2:
                continue

            x_t1 = a * t1**2 + b * t1 + p
            y_t1 = c * t1**2 + d * t1 + q
            x_t2 = a * t2**2 + b * t2 + p
            y_t2 = c * t2**2 + d * t2 + q
            dx = x_t2 - x_t1
            dy = y_t2 - y_t1

            # Independent check via sympy
            t_sym = symbols("t")
            x_component = a * t_sym**2 + b * t_sym + p
            y_component = c * t_sym**2 + d * t_sym + q
            x_t1_val = x_component.subs(t_sym, t1)
            y_t1_val = y_component.subs(t_sym, t1)
            x_t2_val = x_component.subs(t_sym, t2)
            y_t2_val = y_component.subs(t_sym, t2)
            dx_val = x_t2_val - x_t1_val
            dy_val = y_t2_val - y_t1_val
            check_equivalent(dx_val, dx, "hard: dx evaluation check failed")
            check_equivalent(dy_val, dy, "hard: dy evaluation check failed")

            # Distractors
            position_not_displacement_x = x_t2
            position_not_displacement_y = y_t2
            wrong_direction_x = x_t1 - x_t2
            wrong_direction_y = y_t1 - y_t2
            swapped_x = dy
            swapped_y = dx

            # Check distinctness (manual check, no need to call check_distinct for tuples)
            if (
                (dx, dy) != (position_not_displacement_x, position_not_displacement_y)
                and (dx, dy) != (wrong_direction_x, wrong_direction_y)
                and (dx, dy) != (swapped_x, swapped_y)
                and (position_not_displacement_x, position_not_displacement_y) != (wrong_direction_x, wrong_direction_y)
                and (position_not_displacement_x, position_not_displacement_y) != (swapped_x, swapped_y)
                and (wrong_direction_x, wrong_direction_y) != (swapped_x, swapped_y)
            ):
                break

        answer_text = f"x = {dx}, y = {dy}"
        distractor_strs = [
            f"x = {position_not_displacement_x}, y = {position_not_displacement_y}",
            f"x = {wrong_direction_x}, y = {wrong_direction_y}",
            f"x = {swapped_x}, y = {swapped_y}",
        ]

        x_comp = _fmt_component_quadratic(a, b, p)
        y_comp = _fmt_component_quadratic(c, d, q)

        t1_unit = "second" if t1 == 1 else "seconds"
        t2_unit = "second" if t2 == 1 else "seconds"

        text = (
            f"A particle's position vector is r(t) = ({x_comp}, {y_comp}), "
            f"in metres, t in seconds. Find the particle's displacement vector "
            f"from t = {t1} {t1_unit} to t = {t2} {t2_unit}."
        )

        return dict(
            text=text,
            latex=rf"r(t) = ({x_comp}, {y_comp}); \text{{find displacement from }} t={t1} \text{{ to }} t={t2}",
            answer={"x": float(dx), "y": float(dy)},
            answer_text=answer_text,
            working=[
                f"r(t) = ({x_comp}, {y_comp})",
                f"At t = {t1}: r({t1}) = ({x_t1}, {y_t1})",
                f"At t = {t2}: r({t2}) = ({x_t2}, {y_t2})",
                f"Displacement = r({t2}) - r({t1}) = ({dx}, {dy})",
            ],
            distractors=pick_distractors(answer_text, distractor_strs),
        )

    else:
        raise ValueError(f"unknown tier: {tier}")
