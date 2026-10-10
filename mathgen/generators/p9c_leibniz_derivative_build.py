"""Leibniz derivative notation + increments-formula estimate.

- ``p9c_leibniz_derivative_correspondence``
  (``senior-use-the-leibniz-notation-for-the-derivative-frac``, ACMMM083,
  domain differentiation, answer: number) -- given bounded, fully specified
  inputs, use the Leibniz notation dy/dx = lim_(dx->0) dy/dx and the
  correspondence dy/dx = f'(x) where y = f(x). Return the resulting value.

Boundary quotes, both required:
  - vs `differentiate-polynomial`: that skill only bare-differentiates a
    polynomial; it never evaluates at a point and never names dy/dx. Every
    tier here evaluates the derivative formula at a stated numeric point x
    (or inverts it) and names BOTH notations dy/dx and f'(x) together.
  - vs `senior-define-the-derivative-f-left-x-right-as` (ACMMM082,
    `p5e_first_principles_build.py`): that skill forms/simplifies the
    DIFFERENCE QUOTIENT as an expression in x, h (or, at its hard tier,
    takes the limit to get an expression in x alone) -- `answer:
    expression`, never a number, never at a stated point. This skill never
    forms a difference quotient and never returns an expression in x or
    x, h; it always evaluates the already-known derivative formula
    2*A*x + B at a NUMERIC point, or inverts that formula to recover the
    point, always returning a plain number.

- ``p9c_increments_formula_estimate``
  (``senior-use-the-increments-formula-delta-y-cong-frac``, ACMMM107,
  domain differentiation, answer: fraction) -- given bounded, fully
  specified inputs, use the increments formula dy = dy/dx * dx to estimate
  the change in y resulting from a change in x.

  Boundary quotes:
  - vs `differentiate-polynomial`: bare differentiation only, never
    evaluated or used inside a formula.
  - vs the Leibniz-correspondence skill above: that skill CONNECTS dy/dx to
    f'(a), evaluated or inverted, and answers with the EXACT derivative
    value or the point that produces a stated derivative. This skill never
    reports the exact derivative value or the exact change in y; every
    tier multiplies the derivative by dx to ESTIMATE a change (or a new
    value), and every rendered question says "estimate" -- a candidate
    that computes the true exact change f(x0+dx) - f(x0) is a
    misconception here, never the target.
  - vs ACMMM082 (`p5e_first_principles_build.py`): that skill forms the
    difference quotient itself; this skill assumes the derivative is
    already known and only ever uses it inside the linear approximation.
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Eq, diff, solve, symbols

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    latex_answer,
    pick_distractors,
    rational_text,
)

x = symbols("x")

A_POOL = list(range(1, 6))  # [1, 5]
P_POOL = [v for v in range(-6, 7) if v != 0]  # nonzero integers in [-6, 6]
C_POOL = list(range(-9, 10))  # [-9, 9]
B_POOL = [v for v in range(-9, 10) if v != 0]  # nonzero integers in [-9, 9]
DX_POS_POOL = [Fraction(1, 10), Fraction(1, 5), Fraction(1, 4), Fraction(1, 2)]
DX_NEG_POOL = [Fraction(-1, 10), Fraction(-1, 5), Fraction(-1, 4), Fraction(-1, 2)]

_MAX_RESAMPLE = 30


@generator(
    "p9c_leibniz_derivative_correspondence",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p9c_leibniz_derivative_correspondence(rng, tier="easy"):
    """Connect dy/dx to f'(x) at a stated numeric point (ACMMM083).

    easy: f(x) = A*x^2 + C (B = 0). dy/dx at x = p: answer 2*A*p.
    medium: f(x) = A*x^2 + B*x + C (B nonzero). dy/dx at x = p: answer
        2*A*p + B -- one more term through the same correspondence.
    hard: same f as medium; genuinely different operation -- INVERT the
        correspondence: given dy/dx = T, find the x that produces it.
        Answer p0 (the hidden point T was built from).
    """
    A = rng.choice(A_POOL)
    C = rng.choice(C_POOL)

    if tier == "easy":
        B = 0
        p = rng.choice(P_POOL)
        hand_answer = 2 * A * p

        power_rule_drop = A * p
        sign_error = -hand_answer
        confuses_value = A * p**2 + C
        attempts = 0
        while confuses_value == hand_answer and attempts < _MAX_RESAMPLE:
            C = rng.choice(C_POOL)
            confuses_value = A * p**2 + C
            attempts += 1

        f_expr = A * x**2 + C
        check_equivalent(diff(f_expr, x).subs(x, p), hand_answer, f"easy: diff check failed A={A}, C={C}, p={p}")
        check_distinct(hand_answer, power_rule_drop, msg=f"easy: power_rule_drop collides A={A}, p={p}")
        check_distinct(hand_answer, sign_error, msg=f"easy: sign_error collides A={A}, p={p}")
        check_distinct(hand_answer, confuses_value, msg=f"easy: confuses_value collides A={A}, p={p}, C={C}")

        f_text = join_terms([(A, "x^2"), (C, "")])
        question_text = (
            f"y = f(x) = {f_text}. Using the correspondence dy/dx = f'(x), find the value of dy/dx when x = {p}."
        )
        candidates = [str(power_rule_drop), str(sign_error), str(confuses_value)]
        distractors = pick_distractors(str(hand_answer), candidates)
        answer = hand_answer

    elif tier == "medium":
        B = rng.choice(B_POOL)
        p = rng.choice(P_POOL)
        hand_answer = 2 * A * p + B
        power_rule_drop = A * p
        sign_error = -hand_answer
        dropped_linear_term = 2 * A * p
        # power_rule_drop collides iff B == -A*p (the "distinct
        # always since A*p != 0" claim holds for easy, where B == 0, but
        # not for medium where B ranges over all of B_POOL); sign_error
        # collides iff hand_answer == 0. dropped_linear_term is always
        # distinct here since B != 0 unconditionally.
        attempts = 0
        while (power_rule_drop == hand_answer or sign_error == hand_answer) and attempts < _MAX_RESAMPLE:
            p = rng.choice(P_POOL)
            hand_answer = 2 * A * p + B
            power_rule_drop = A * p
            sign_error = -hand_answer
            dropped_linear_term = 2 * A * p
            attempts += 1

        f_expr = A * x**2 + B * x + C
        check_equivalent(
            diff(f_expr, x).subs(x, p), hand_answer, f"medium: diff check failed A={A}, B={B}, C={C}, p={p}"
        )
        check_distinct(hand_answer, power_rule_drop, msg=f"medium: power_rule_drop collides A={A}, B={B}, p={p}")
        check_distinct(hand_answer, sign_error, msg=f"medium: sign_error collides A={A}, B={B}, p={p}")
        check_distinct(
            hand_answer, dropped_linear_term, msg=f"medium: dropped_linear_term collides A={A}, B={B}, p={p}"
        )

        f_text = join_terms([(A, "x^2"), (B, "x"), (C, "")])
        question_text = (
            f"y = f(x) = {f_text}. Using the correspondence dy/dx = f'(x), find the value of dy/dx when x = {p}."
        )
        candidates = [str(power_rule_drop), str(sign_error), str(dropped_linear_term)]
        distractors = pick_distractors(str(hand_answer), candidates)
        answer = hand_answer

    else:  # hard
        # B is drawn from the multiples of A within B_POOL so the
        # sign-error-on-solve distractor below is guaranteed integral
        # ("fall back to a different T draw that keeps it integral") -- always non-empty since A itself is a nonzero member.
        b_hard_pool = [v for v in B_POOL if v % A == 0]
        B = rng.choice(b_hard_pool)
        p0 = rng.choice(P_POOL)
        T = 2 * A * p0 + B
        evaluates_instead_of_solves = 2 * A * T + B
        sign_error_on_solve = Fraction(-T - B, 2 * A)
        ignored_b_in_solve = Fraction(T, 2 * A)

        attempts = 0
        while (
            evaluates_instead_of_solves == p0 or sign_error_on_solve == p0 or ignored_b_in_solve == p0
        ) and attempts < _MAX_RESAMPLE:
            p0 = rng.choice(P_POOL)
            T = 2 * A * p0 + B
            evaluates_instead_of_solves = 2 * A * T + B
            sign_error_on_solve = Fraction(-T - B, 2 * A)
            ignored_b_in_solve = Fraction(T, 2 * A)
            attempts += 1

        # Independent check: a symbolic equation solver -- a genuinely
        # different route from the Fraction division construction used to
        # derive p0 (here, p0 is simply carried forward from the draw).
        solutions = solve(Eq(2 * A * x + B, T), x)
        check(len(solutions) == 1, f"hard: expected a unique solution, got {solutions}")
        check_equivalent(solutions[0], p0, f"hard: solve check failed A={A}, B={B}, T={T}, p0={p0}")
        check_distinct(
            p0, evaluates_instead_of_solves, msg=f"hard: evaluates_instead_of_solves collides A={A}, B={B}, T={T}"
        )
        check_distinct(p0, sign_error_on_solve, msg=f"hard: sign_error_on_solve collides A={A}, B={B}, T={T}")
        check_distinct(p0, ignored_b_in_solve, msg=f"hard: ignored_b_in_solve collides A={A}, B={B}, T={T}")

        def _fmt(v):
            v = Fraction(v)
            return str(v.numerator) if v.denominator == 1 else rational_text(v)

        f_text = join_terms([(A, "x^2"), (B, "x"), (C, "")])
        question_text = (
            f"y = f(x) = {f_text}. dy/dx = f'(x) = {T} when x takes some value. "
            f"Using the correspondence dy/dx = f'(x), find that value of x."
        )
        candidates = [
            _fmt(evaluates_instead_of_solves),
            _fmt(sign_error_on_solve),
            _fmt(ignored_b_in_solve),
        ]
        distractors = pick_distractors(str(p0), candidates)
        answer = p0

    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer,
        "answer_text": str(answer),
        "answer_latex": latex_answer(answer),
        "distractors": distractors,
    }


@generator(
    "p9c_increments_formula_estimate",
    level=13,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p9c_increments_formula_estimate(rng, tier="easy"):
    """Estimate a change (or new value) via dy = dy/dx * dx (ACMMM107).

    easy: f(x) = A*x^2 + C (B = 0), positive dx. dy ~= 2*A*x0 * dx.
    medium: f(x) = A*x^2 + B*x + C (B nonzero), positive dx.
        dy ~= (2*A*x0 + B) * dx -- one more term inside the derivative.
    hard: same f as medium, NEGATIVE dx (estimating a decrease) AND the
        target quantity changes -- the ESTIMATED NEW VALUE
        f(x0) + (2*A*x0 + B) * dx, not the increment alone.
    """
    A = rng.choice(A_POOL)
    C = rng.choice(C_POOL)

    if tier == "easy":
        B = 0
        dx = rng.choice(DX_POS_POOL)
        x0 = rng.choice(P_POOL)
        deriv_value = 2 * A * x0
        estimate = deriv_value * dx
        sign_error = deriv_value * (-dx)
        forgot_multiply = deriv_value
        attempts = 0
        while sign_error == estimate and attempts < _MAX_RESAMPLE:
            x0 = rng.choice(P_POOL)
            deriv_value = 2 * A * x0
            estimate = deriv_value * dx
            sign_error = deriv_value * (-dx)
            forgot_multiply = deriv_value
            attempts += 1

        f_expr = A * x**2 + C
        check_equivalent(diff(f_expr, x).subs(x, x0), deriv_value, f"easy: diff check failed A={A}, C={C}, x0={x0}")

        f_x0 = A * x0**2 + C
        exact_change = (A * (x0 + dx) ** 2 + C) - f_x0
        check(
            exact_change - estimate == A * dx**2, f"easy: exact-minus-estimate identity failed A={A}, dx={dx}, x0={x0}"
        )

        check_distinct(estimate, exact_change, msg=f"easy: exact_change collides A={A}, x0={x0}, dx={dx}, C={C}")
        check_distinct(estimate, forgot_multiply, msg=f"easy: forgot_multiply collides A={A}, x0={x0}, dx={dx}")
        check_distinct(estimate, sign_error, msg=f"easy: sign_error collides A={A}, x0={x0}, dx={dx}")

        f_text = join_terms([(A, "x^2"), (C, "")])
        question_text = (
            f"y = f(x) = {f_text}. Estimate the change in y, δy, using the "
            f"increments formula δy ≅ dy/dx × δx, for a change of δx = "
            f"{rational_text(dx)} from x0 = {x0}."
        )
        candidates = [rational_text(exact_change), rational_text(forgot_multiply), rational_text(sign_error)]
        distractors = pick_distractors(rational_text(estimate), candidates)
        value = estimate

    elif tier == "medium":
        B = rng.choice(B_POOL)
        dx = rng.choice(DX_POS_POOL)
        x0 = rng.choice(P_POOL)
        deriv_value = 2 * A * x0 + B
        estimate = deriv_value * dx
        sign_error = deriv_value * (-dx)
        forgot_multiply = deriv_value
        attempts = 0
        while sign_error == estimate and attempts < _MAX_RESAMPLE:
            x0 = rng.choice(P_POOL)
            deriv_value = 2 * A * x0 + B
            estimate = deriv_value * dx
            sign_error = deriv_value * (-dx)
            forgot_multiply = deriv_value
            attempts += 1

        f_expr = A * x**2 + B * x + C
        check_equivalent(
            diff(f_expr, x).subs(x, x0), deriv_value, f"medium: diff check failed A={A}, B={B}, C={C}, x0={x0}"
        )

        f_x0 = A * x0**2 + B * x0 + C
        exact_change = (A * (x0 + dx) ** 2 + B * (x0 + dx) + C) - f_x0
        check(
            exact_change - estimate == A * dx**2,
            f"medium: exact-minus-estimate identity failed A={A}, dx={dx}, x0={x0}",
        )

        check_distinct(estimate, exact_change, msg=f"medium: exact_change collides A={A}, x0={x0}, dx={dx}")
        check_distinct(estimate, forgot_multiply, msg=f"medium: forgot_multiply collides A={A}, x0={x0}, dx={dx}")
        check_distinct(estimate, sign_error, msg=f"medium: sign_error collides A={A}, x0={x0}, dx={dx}")

        f_text = join_terms([(A, "x^2"), (B, "x"), (C, "")])
        question_text = (
            f"y = f(x) = {f_text}. Estimate the change in y, δy, using the "
            f"increments formula δy ≅ dy/dx × δx, for a change of δx = "
            f"{rational_text(dx)} from x0 = {x0}."
        )
        candidates = [rational_text(exact_change), rational_text(forgot_multiply), rational_text(sign_error)]
        distractors = pick_distractors(rational_text(estimate), candidates)
        value = estimate

    else:  # hard
        B = rng.choice(B_POOL)
        dx = rng.choice(DX_NEG_POOL)
        x0 = rng.choice(P_POOL)
        deriv_value = 2 * A * x0 + B
        f_x0 = A * x0**2 + B * x0 + C
        increment = deriv_value * dx
        estimate = f_x0 + increment
        sign_error = f_x0 + deriv_value * (-dx)
        forgot_multiply = f_x0 + deriv_value
        forgot_add = increment
        attempts = 0
        while (sign_error == estimate or forgot_add == estimate) and attempts < _MAX_RESAMPLE:
            x0 = rng.choice(P_POOL)
            C = rng.choice(C_POOL)
            deriv_value = 2 * A * x0 + B
            f_x0 = A * x0**2 + B * x0 + C
            increment = deriv_value * dx
            estimate = f_x0 + increment
            sign_error = f_x0 + deriv_value * (-dx)
            forgot_multiply = f_x0 + deriv_value
            forgot_add = increment
            attempts += 1

        f_expr = A * x**2 + B * x + C
        check_equivalent(
            diff(f_expr, x).subs(x, x0), deriv_value, f"hard: diff check failed A={A}, B={B}, C={C}, x0={x0}"
        )
        # Second, independent substitution path for f(x0) -- separate from
        # whatever construction used, confirmed before it is added in.
        check_equivalent(f_expr.subs(x, x0), f_x0, f"hard: f(x0) independent check failed A={A}, B={B}, C={C}, x0={x0}")

        check_distinct(estimate, sign_error, msg=f"hard: sign_error collides A={A}, B={B}, C={C}, x0={x0}, dx={dx}")
        check_distinct(
            estimate, forgot_multiply, msg=f"hard: forgot_multiply collides A={A}, B={B}, C={C}, x0={x0}, dx={dx}"
        )
        check_distinct(estimate, forgot_add, msg=f"hard: forgot_add collides A={A}, B={B}, C={C}, x0={x0}, dx={dx}")

        f_text = join_terms([(A, "x^2"), (B, "x"), (C, "")])
        question_text = (
            f"y = f(x) = {f_text}. Estimate the new value of y when x changes "
            f"from x0 = {x0} to x0 + δx, using the increments formula "
            f"δy ≅ dy/dx × δx, given δx = {rational_text(dx)}."
        )
        candidates = [rational_text(sign_error), rational_text(forgot_multiply), rational_text(forgot_add)]
        distractors = pick_distractors(rational_text(estimate), candidates)
        value = estimate

    value = Fraction(value)
    answer = {"p": value.numerator, "q": value.denominator}
    return {
        "text": question_text,
        "latex": question_text,
        "answer": answer,
        "answer_text": rational_text(value),
        "answer_latex": latex_answer(value),
        "distractors": distractors,
    }
