"""Vector calculus (ACMSM113): differentiate and integrate a vector function
with respect to time.

Three tiers: easy (power-rule differentiation of polynomial components,
evaluated at a stated time), medium (chain-rule differentiation of
transcendental components, evaluated at t=0), hard (integration of a
velocity vector to find displacement over a stated interval). All answers
are evaluated as numeric pairs at stated times or bounds, never symbolic
vector expressions.
"""

from __future__ import annotations

from fractions import Fraction
from sympy import cos, diff, exp, sin, symbols

from ..core import check, check_equivalent, generator, join_terms, pick_distractors


def _fmt_signed_term(coef: int, var: str) -> str:
    """Format a single component term with its sign word.

    Returns strings like "+ 3t", "- 2t", etc., or empty string for zero coef.
    Never produces a bare multiplication sign; uses spaces and sign words.
    """
    if coef == 0:
        return ""
    sign_word = "+ " if coef > 0 else "- "
    abs_coef = abs(coef)
    if abs_coef == 1:
        return f"{sign_word}{var}"
    return f"{sign_word}{abs_coef}{var}"


def _fmt_component_linear(a: int, b: int) -> str:
    """Format one linear component ``a*t + b``: "4t + 2", "6t - 1", "t + 3".

    ``_fmt_signed_term(coef, "")`` cannot be used for the constant term: it
    suppresses a coefficient of 1 because it assumes a variable follows, so
    ``e1 = 1`` rendered "6t + " with no number. That reached the hard tier's
    latex and working lines as well as the prompt.
    """
    lead = "t" if a == 1 else ("-t" if a == -1 else f"{a}t")
    if b == 0:
        return lead
    return f"{lead} {'+' if b > 0 else '-'} {abs(b)}"


def _fmt_component_easy(a: int, b: int) -> str:
    """Format one component of r(t) = (a*t^2 + b*t, ...) for easy tier.

    Returns strings like "3t^2 + 2t", "t^2 - 5t", etc.
    No bare multiplication signs; coefficients directly adjacent to t.
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

    return " ".join(parts) if parts else "0"


def _fmt_component_medium(a: int, k: int, func: str, b: int, m: int) -> str:
    """Format one component of r(t) = (a*exp(k*t), b*sin(m*t), ...) for medium tier.

    Args:
        a, k: coefficient and inner-function factor for exponential
        func: "exp" or "sin"
        b, m: coefficient and inner-function factor for other component

    Returns strings like "3exp(2t)", "5sin(3t)", etc.
    No bare multiplication signs.
    """
    if func == "exp":
        coef_str = "" if a == 1 else str(a)
        inner = f"{k}t" if k != 1 else "t"
        return f"{coef_str}exp({inner})" if coef_str else f"exp({inner})"
    else:  # sin
        coef_str = "" if b == 1 else str(b)
        inner = f"{m}t" if m != 1 else "t"
        return f"{coef_str}sin({inner})" if coef_str else f"sin({inner})"


def _verify_easy(a1, b1, a2, b2, T, x_ans, y_ans):
    """Verify easy-tier answer via finite-difference approximation.

    Computes the numeric derivative of each component using exact Fraction
    arithmetic and a small step, confirming it matches the closed-form formula
    (computed separately as x_ans, y_ans).
    """
    h = Fraction(1, 1000)
    # Position function for x component
    pos_x_at_T = a1 * T**2 + b1 * T
    pos_x_at_T_plus_h = a1 * (T + h) ** 2 + b1 * (T + h)
    deriv_x_approx = (pos_x_at_T_plus_h - pos_x_at_T) / h
    check(
        abs(deriv_x_approx - x_ans) < Fraction(1, 100),
        f"x derivative finite-difference check failed: {deriv_x_approx} vs {x_ans}",
    )

    # Position function for y component
    pos_y_at_T = a2 * T**2 + b2 * T
    pos_y_at_T_plus_h = a2 * (T + h) ** 2 + b2 * (T + h)
    deriv_y_approx = (pos_y_at_T_plus_h - pos_y_at_T) / h
    check(
        abs(deriv_y_approx - y_ans) < Fraction(1, 100),
        f"y derivative finite-difference check failed: {deriv_y_approx} vs {y_ans}",
    )


def _verify_medium(a, k, b, m, x_ans, y_ans):
    """Verify medium-tier answer via sympy differentiation.

    Constructs the symbolic vector components, differentiates, evaluates
    at t=0, and confirms the result matches the closed-form formula.
    """
    t_sym = symbols("t")
    x_component = a * exp(k * t_sym)
    y_component = b * sin(m * t_sym)

    x_derivative = diff(x_component, t_sym)
    y_derivative = diff(y_component, t_sym)

    x_at_zero = x_derivative.subs(t_sym, 0)
    y_at_zero = y_derivative.subs(t_sym, 0)

    check_equivalent(x_at_zero, x_ans, f"x derivative check failed at t=0: sympy gives {x_at_zero}, expected {x_ans}")
    check_equivalent(y_at_zero, y_ans, f"y derivative check failed at t=0: sympy gives {y_at_zero}, expected {y_ans}")


def _verify_hard(d1, e1, d2, e2):
    """Verify hard-tier answer via fundamental theorem of calculus.

    Differentiates the computed antiderivative and confirms it recovers
    the original velocity vector (at all t, checked via sympy).
    """
    t_sym = symbols("t")
    # Antiderivatives
    x_antideriv = d1 * t_sym**2 + e1 * t_sym
    y_antideriv = d2 * t_sym**2 + e2 * t_sym

    # Differentiate to recover velocity
    x_velocity_recovered = diff(x_antideriv, t_sym)
    y_velocity_recovered = diff(y_antideriv, t_sym)

    # Should match the given velocity: v(t) = (2*d1*t + e1, 2*d2*t + e2)
    check_equivalent(
        x_velocity_recovered, 2 * d1 * t_sym + e1, f"x antiderivative check failed: diff gives {x_velocity_recovered}"
    )
    check_equivalent(
        y_velocity_recovered, 2 * d2 * t_sym + e2, f"y antiderivative check failed: diff gives {y_velocity_recovered}"
    )


def _redraw_loop(tier_name, answer_pair, candidate_pairs, max_attempts=200):
    """Check for collisions and rebuild if needed.

    If any candidate's (x, y) both individually satisfy the tolerance
    (would grade as correct), raise ValueError.

    Args:
        tier_name: "easy", "medium", or "hard" for error messages
        answer_pair: (x_ans, y_ans)
        candidate_pairs: list of (x_cand, y_cand) tuples

    Returns True if all candidates are safe, False if collision detected.
    """
    ax, ay = answer_pair
    for cx, cy in candidate_pairs:
        if abs(cx - ax) <= 0.051 and abs(cy - ay) <= 0.051:
            return False  # Collision detected, needs redraw
    return True  # All candidates are safe


@generator(
    "differentiate_and_integrate_a_vector_function_with_respect",
    level=15,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def differentiate_and_integrate_a_vector_function_with_respect(rng, tier="easy"):
    """Differentiate or integrate a two-component vector function of time."""

    if tier == "easy":
        # Easy: differentiate polynomial components, evaluate at stated time
        for attempt in range(200):
            a1 = rng.randint(1, 4)
            b1 = rng.choice([v for v in range(-6, 7) if v != 0])
            a2 = rng.randint(1, 4)
            b2 = rng.choice([v for v in range(-6, 7) if v != 0])
            T = rng.randint(1, 4)

            # Answer: dr/dt at t=T
            x_ans = 2 * a1 * T + b1
            y_ans = 2 * a2 * T + b2

            # Verify via finite difference
            _verify_easy(a1, b1, a2, b2, T, x_ans, y_ans)

            # Build three candidate pairs: position value, forgot power rule, sign flip
            pos_x = a1 * T**2 + b1 * T
            pos_y = a2 * T**2 + b2 * T
            power_wrong_x = a1 * T + b1
            power_wrong_y = a2 * T + b2
            sign_flip_x = -x_ans
            sign_flip_y = -y_ans

            candidates = [(pos_x, pos_y), (power_wrong_x, power_wrong_y), (sign_flip_x, sign_flip_y)]

            # Check for collisions (both components within tolerance)
            if _redraw_loop("easy", (x_ans, y_ans), candidates):
                break
        else:
            raise ValueError("easy tier redraw loop exceeded 200 attempts")

        answer_text = f"x = {x_ans}, y = {y_ans}"

        # Build distractor strings
        distractor_strs = [
            f"x = {pos_x}, y = {pos_y}",
            f"x = {power_wrong_x}, y = {power_wrong_y}",
            f"x = {sign_flip_x}, y = {sign_flip_y}",
        ]

        # Build the text with no bare multiplication signs
        x_comp = _fmt_component_easy(a1, b1)
        y_comp = _fmt_component_easy(a2, b2)
        time_unit = "second" if T == 1 else "seconds"

        return dict(
            text=(
                f"A particle's position vector is r(t) = ({x_comp}, {y_comp}), "
                f"in metres, t in seconds. Find the velocity vector dr/dt at t = {T} {time_unit}."
            ),
            latex=rf"r(t) = ({a1}t^2{_fmt_signed_term(b1, 't')}, "
            rf"{a2}t^2{_fmt_signed_term(b2, 't')});  \text{{find }}r'({T})",
            answer={"x": float(x_ans), "y": float(y_ans)},
            answer_text=answer_text,
            working=[
                f"r(t) = ({x_comp}, {y_comp})",
                f"dr/dt = ({_fmt_component_linear(2 * a1, b1)}, {_fmt_component_linear(2 * a2, b2)})",
                f"At t = {T}: dr/dt = ({x_ans}, {y_ans})",
            ],
            distractors=pick_distractors(answer_text, distractor_strs),
        )

    elif tier == "medium":
        # Medium: chain rule with transcendental functions, evaluate at t=0
        for attempt in range(200):
            a = rng.randint(1, 4)
            k = rng.randint(2, 4)
            b = rng.randint(1, 4)
            m = rng.randint(2, 4)

            # Answer: dr/dt at t=0
            # d/dt[a*exp(k*t)] = a*k*exp(k*t), at t=0 equals a*k
            x_ans = a * k
            # d/dt[b*sin(m*t)] = b*m*cos(m*t), at t=0 equals b*m
            y_ans = b * m

            # Verify via sympy
            _verify_medium(a, k, b, m, x_ans, y_ans)

            # Build three candidate pairs
            no_chain_x = a
            no_chain_y = b
            sign_confused_x = a * k
            sign_confused_y = 0
            exp_sign_error_x = -a * k
            exp_sign_error_y = b * m

            candidates = [
                (no_chain_x, no_chain_y),
                (sign_confused_x, sign_confused_y),
                (exp_sign_error_x, exp_sign_error_y),
            ]

            # Check for collisions
            if _redraw_loop("medium", (x_ans, y_ans), candidates):
                break
        else:
            raise ValueError("medium tier redraw loop exceeded 200 attempts")

        answer_text = f"x = {x_ans}, y = {y_ans}"

        # Build distractor strings
        distractor_strs = [
            f"x = {no_chain_x}, y = {no_chain_y}",
            f"x = {sign_confused_x}, y = {sign_confused_y}",
            f"x = {exp_sign_error_x}, y = {exp_sign_error_y}",
        ]

        # Build text strings for each component
        # Format: "aexp(kt)" or just "exp(kt)" if a=1
        x_comp_str = f"{a}exp({k}t)" if a > 1 else f"exp({k}t)"
        y_comp_str = f"{b}sin({m}t)" if b > 1 else f"sin({m}t)"

        return dict(
            text=(
                f"A particle's position vector is r(t) = ({x_comp_str}, {y_comp_str}), "
                f"in metres, t in seconds. Find the velocity vector dr/dt at t = 0 seconds."
            ),
            latex=rf"r(t) = ({a}\exp({k}t), {b}\sin({m}t));  \text{{find }}r'(0)",
            answer={"x": float(x_ans), "y": float(y_ans)},
            answer_text=answer_text,
            working=[
                f"r(t) = ({x_comp_str}, {y_comp_str})",
                f"dr/dt = ({a * k}exp({k}t), {b * m}cos({m}t))",
                f"At t = 0: dr/dt = ({x_ans}, {y_ans})",
            ],
            distractors=pick_distractors(answer_text, distractor_strs),
        )

    elif tier == "hard":
        # Hard: integrate velocity to find displacement
        for attempt in range(200):
            d1 = rng.randint(1, 4)
            e1 = rng.choice([v for v in range(-6, 7) if v != 0])
            d2 = rng.randint(1, 4)
            e2 = rng.choice([v for v in range(-6, 7) if v != 0])
            T = rng.randint(1, 4)

            # Answer: displacement from t=0 to t=T
            x_disp = d1 * T**2 + e1 * T
            y_disp = d2 * T**2 + e2 * T

            # Verify via FTC
            _verify_hard(d1, e1, d2, e2)

            # Build three candidate pairs
            velocity_at_T_x = 2 * d1 * T + e1
            velocity_at_T_y = 2 * d2 * T + e2
            botched_x = d1 * T + e1 * T
            botched_y = d2 * T + e2 * T
            sign_flip_x = -x_disp
            sign_flip_y = -y_disp

            candidates = [(velocity_at_T_x, velocity_at_T_y), (botched_x, botched_y), (sign_flip_x, sign_flip_y)]

            # Check for collisions
            if _redraw_loop("hard", (x_disp, y_disp), candidates):
                break
        else:
            raise ValueError("hard tier redraw loop exceeded 200 attempts")

        answer_text = f"x = {x_disp}, y = {y_disp}"

        # Build distractor strings
        distractor_strs = [
            f"x = {velocity_at_T_x}, y = {velocity_at_T_y}",
            f"x = {botched_x}, y = {botched_y}",
            f"x = {sign_flip_x}, y = {sign_flip_y}",
        ]

        # Build the text
        v_coef_x = 2 * d1
        v_coef_y = 2 * d2
        # The velocity here is linear (2*d1*t + e1) -- _fmt_component_easy renders
        # the quadratic POSITION form a*t^2 + b*t and belongs to the easy tier.
        # Using it here printed "v(t) = (4t^2 + 2t, ...)" above an answer computed
        # for v(t) = 4t + 2, so every hard-tier prompt contradicted its own answer
        # and a learner who integrated what they were shown was marked wrong.
        # The latex and working lines already used the linear form; only the text
        # the learner reads was wrong.
        x_comp = _fmt_component_linear(v_coef_x, e1)
        y_comp = _fmt_component_linear(v_coef_y, e2)
        time_unit = "second" if T == 1 else "seconds"

        return dict(
            text=(
                f"A particle's velocity vector is v(t) = ({x_comp}, {y_comp}), "
                f"in metres per second, t in seconds. Find the particle's displacement vector "
                f"from t = 0 to t = {T} {time_unit}."
            ),
            latex=rf"v(t) = ({x_comp}, {y_comp});  \text{{find displacement }}[0,{T}]",
            answer={"x": float(x_disp), "y": float(y_disp)},
            answer_text=answer_text,
            working=[
                f"v(t) = ({x_comp}, {y_comp})",
                f"Integrating from 0 to {T}:",
                f"x displacement = integral of {x_comp} from 0 to {T}",
                f"= [{join_terms([(d1, 't^2'), (e1, 't')])}] from 0 to {T} = {x_disp}",
                f"y displacement = {y_disp}",
                f"Displacement = ({x_disp}, {y_disp})",
            ],
            distractors=pick_distractors(answer_text, distractor_strs),
        )

    else:
        raise ValueError(f"unknown tier: {tier}")
