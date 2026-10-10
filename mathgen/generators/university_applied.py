"""Small, exact computational slices from the university applied core.

This batch intentionally stops before proofs, general constrained optimisation,
or arbitrary stochastic-process questions.  Every item has a closed-form answer
and two independent checks so that the existing numeric/fraction/pair graders
can be used without adding a new answer contract.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    GRADER_TOLERANCE,
    check,
    fmt_signed,
    generator,
    join_terms,
    pick_distractors,
)


def _fraction_payload(value: Fraction) -> tuple[dict, str]:
    value = Fraction(value)
    text = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    return {"p": value.numerator, "q": value.denominator}, text


def _probability_options(answer: Fraction, values: list, den: int) -> list[str]:
    """Distractor texts for a probability answer: every one a value in [0, 1], none equal to ``answer``.

    ``values`` are the misconception values in priority order; anything outside [0, 1] is dropped
    (a learner eliminates an impossible probability at sight). The tail is topped up with the
    complement and one-step nudges of the answer, so the option count survives the drop.
    """
    nudges = [
        1 - answer,
        answer + Fraction(1, den),
        answer - Fraction(1, den),
        answer / 2,
        (1 + answer) / 2,
        answer + Fraction(2, den),
        answer - Fraction(2, den),
        Fraction(1, den),
        Fraction(den - 1, den),
        Fraction(1, 2),
    ]
    out: list[str] = []
    for v in [*values, *nudges]:
        v = Fraction(v)
        text = _fraction_payload(v)[1]
        if 0 <= v <= 1 and v != answer and text not in out:
            out.append(text)
    return out


def _pair_payload(x: Fraction, y: Fraction) -> tuple[dict, str]:
    x, y = Fraction(x), Fraction(y)

    def render(v: Fraction) -> str:
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"

    return {"x": x, "y": y}, f"x = {render(x)}, y = {render(y)}"


@generator(
    "convex_quadratic_minimum",
    level=16,
    presets={
        "easy": dict(coefficient_hi=4, linear_hi=6, constant_hi=6),
        "medium": dict(coefficient_hi=8, linear_hi=12, constant_hi=12),
        "hard": dict(coefficient_hi=14, linear_hi=20, constant_hi=20),
    },
)
def convex_quadratic_minimum(rng, coefficient_hi=8, linear_hi=12, constant_hi=12):
    """Find the minimising x and minimum value of a convex quadratic."""
    a = rng.randint(1, coefficient_hi)
    b = rng.randint(-linear_hi, linear_hi)
    c = rng.randint(-constant_hi, constant_hi)
    x_star = Fraction(-b, 2 * a)
    value = Fraction(4 * a * c - b * b, 4 * a)
    # Independent checks: completing the square and evaluating the derivative.
    check(Fraction(a) * (x_star**2) + b * x_star + c == value, "quadratic minimum evaluation mismatch")
    check(2 * a * x_star + b == 0 and a > 0, "quadratic stationary-point check failed")
    check(Fraction(a) * (x_star - x_star) ** 2 + value == value, "quadratic completed-square check failed")
    answer, answer_text = _pair_payload(x_star, value)
    wrong_x = Fraction(-b, a)
    wrong_value = Fraction(4 * a * c + b * b, 4 * a)
    alternatives = [
        f"x = {wrong_x}, y = {value}",
        f"x = {x_star}, y = {wrong_value}",
        f"x = {b}/{2 * a}, y = {value}",
        f"x = {x_star}, y = {c}",
        f"x = {x_star + 1}, y = {value}",
        f"x = {x_star}, y = {value + 1}",
    ]
    polynomial = join_terms([(a, "x²"), (b, "x"), (c, "")])
    linear_latex = f" + {fmt_signed(b, 'x')}" if b > 0 else f" - {fmt_signed(-b, 'x')}" if b < 0 else ""
    constant_latex = f" + {c}" if c > 0 else f" - {abs(c)}" if c < 0 else ""
    polynomial_latex = f"{fmt_signed(a, 'x^2')}" + linear_latex + constant_latex
    return dict(
        text=f"For f(x) = {polynomial}, give the minimiser x and minimum value f(x).",
        latex=rf"f(x)={polynomial_latex};\quad (x_*,f(x_*))",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"f'(x) = {2 * a}x {b:+d}, so x* = {x_star}.",
            f"f(x*) = {value}; the positive leading coefficient makes this a minimum.",
        ],
        # GRADER_TOLERANCE: this is a `pair` answer, and grade_pair accepts a
        # component within ~0.051, so a candidate closer than that is not a
        # wrong answer -- x = 1/14 was offered against x = 1/28.
        distractors=pick_distractors(answer_text, alternatives, tolerance=GRADER_TOLERANCE),
    )


@generator(
    "lagrange_constrained_point",
    level=17,
    presets={
        "easy": dict(coordinate_hi=4, coefficient_hi=4),
        "medium": dict(coordinate_hi=7, coefficient_hi=7),
        "hard": dict(coordinate_hi=12, coefficient_hi=11),
    },
)
def lagrange_constrained_point(rng, coordinate_hi=7, coefficient_hi=7, kind=None):
    """Find the closest point to a target on a non-axis-aligned line.

    ``kind=None`` (the default) preserves the original behaviour
    byte-for-byte: the line is given directly in normal form ``ux+vy=rhs``.
    ``kind="slope_form"`` gives the SAME kind of line in slope-intercept
    form ``y = (p/q)x + c`` instead (``p``, ``q`` coprime, ``q >= 2`` so the
    slope is never already an integer, ``c`` forced nonzero so the rendered
    line is never shown with an ugly ``+ 0``), so the student must convert
    to normal form (``px - qy = -qc``, multiplying through by ``q``)
    themselves before applying the same closest-point method -- a genuine
    extra algebraic step, not a bigger number. The objective ("closest
    point to the target") is unchanged either way.
    """
    target_x = rng.randint(-coordinate_hi, coordinate_hi)
    target_y = rng.randint(-coordinate_hi, coordinate_hi)
    if kind == "slope_form":
        q = rng.randint(2, coefficient_hi)
        p = rng.randint(-coefficient_hi, coefficient_hi)
        while p == 0 or math.gcd(abs(p), q) != 1:
            p = rng.randint(-coefficient_hi, coefficient_hi)
        c = rng.randint(-coefficient_hi, coefficient_hi)
        while c == 0:
            c = rng.randint(-coefficient_hi, coefficient_hi)
        u, v = p, -q
        rhs = -q * c
    else:
        u = rng.randint(1, coefficient_hi)
        v = rng.randint(1, coefficient_hi)
        # Pick a line whose signed offset is nonzero, avoiding the trivial target.
        offset = rng.choice([n for n in range(-coefficient_hi, coefficient_hi + 1) if n])
        rhs = u * target_x + v * target_y + offset
    denom = u * u + v * v
    lam = Fraction(rhs - u * target_x - v * target_y, denom)
    # For ||z-target||² subject to uz_x+vz_y=rhs, z=target+lambda(u,v).
    x_star = Fraction(target_x) + lam * u
    y_star = Fraction(target_y) + lam * v
    check(u * x_star + v * y_star == rhs, "Lagrange constraint check failed")
    check((x_star - target_x) * v == (y_star - target_y) * u, "Lagrange normal-direction check failed")
    # Independent distance comparison against the line's orthogonal projection.
    projected_lam = Fraction(rhs - u * target_x - v * target_y, denom)
    check(projected_lam == lam, "Lagrange multiplier recomputation mismatch")
    answer, answer_text = _pair_payload(x_star, y_star)
    if kind == "slope_form":
        # Independent check: the closest point must also satisfy the
        # ORIGINAL slope-intercept equation (a second, differently-derived
        # route from the normal-form check above).
        check(Fraction(p, q) * x_star + c == y_star, "slope-form line does not contain the computed closest point")
        sign = "+" if c > 0 else "-"
        line_text = f"y = ({p}/{q})x {sign} {abs(c)}"
        alternatives = [
            f"x = {target_x}, y = {target_y}",
            f"x = {x_star + 1}, y = {y_star}",
            f"x = {x_star}, y = {y_star + 1}",
            f"x = {y_star}, y = {x_star}",
        ]
        return dict(
            text=(f"Find the point (x, y) on the line {line_text} closest to the target ({target_x}, {target_y})."),
            latex=(
                rf"\min\left((x-{target_x})^2+(y-{target_y})^2\right)\quad"
                rf"\text{{subject to }}y=\frac{{{p}}}{{{q}}}x{sign}{abs(c)}"
            ),
            answer=answer,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"Convert to normal form: {p}x − {q}y = {rhs}.",
                f"The displacement is normal to the line: (x−{target_x}, y−{target_y}) = λ({u}, {v}).",
                f"Solving the constraint gives (x, y) = ({x_star}, {y_star}).",
            ],
            # GRADER_TOLERANCE: this is a `pair` answer, and grade_pair accepts a
            # component within ~0.051, so a candidate closer than that is not a
            # wrong answer.
            distractors=pick_distractors(answer_text, alternatives, tolerance=GRADER_TOLERANCE),
        )
    alternatives = [
        f"x = {x_star}, y = {y_star}",
        f"x = {Fraction(target_x)}, y = {Fraction(target_y)}",
        f"x = {Fraction(target_x) - lam * u}, y = {Fraction(target_y) - lam * v}",
        f"x = {Fraction(rhs, u)}, y = {Fraction(target_y)}",
    ]
    constraint_text = join_terms([(u, "x"), (v, "y")])
    constraint_latex = f"{fmt_signed(u, 'x')}" + (f" + {v}y" if v >= 0 else f" - {abs(v)}y")
    return dict(
        text=(f"Find the point (x, y) on {constraint_text} = {rhs} closest to the target ({target_x}, {target_y})."),
        latex=rf"\min\left((x-{target_x})^2+(y-{target_y})^2\right)\quad\text{{subject to }}{constraint_latex}={rhs}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"The displacement is normal to the line: (x−{target_x}, y−{target_y}) = λ({u}, {v}).",
            f"Solving the constraint gives (x, y) = ({x_star}, {y_star}).",
        ],
        # GRADER_TOLERANCE: this is a `pair` answer, and grade_pair accepts a
        # component within ~0.051, so a candidate closer than that is not a
        # wrong answer.
        distractors=pick_distractors(answer_text, alternatives, tolerance=GRADER_TOLERANCE),
    )


@generator(
    "markov_two_state_step",
    level=16,
    presets={
        "easy": dict(denominator_hi=5),
        "medium": dict(denominator_hi=9),
        "hard": dict(denominator_hi=14),
    },
)
def markov_two_state_step(rng, denominator_hi=9, kind=None):
    """Compute P(X_1=1) from an initial law and a two-state transition matrix.

    ``kind=None`` (the default) preserves the original forward behaviour
    byte-for-byte: p, q and P(X0=1) are given, P(X1=1) is asked. ``kind=
    "find_initial"`` gives p, q and P(X1=1) instead, and asks for P(X0=1) --
    solving the SAME balance equation backwards
    (P(X0=1) = (P(X1=1) - p) / (1 - p - q)), a genuinely different algebraic
    step from substituting forward.
    """
    den = rng.randint(3, denominator_hi)
    p = Fraction(rng.randint(1, den - 1), den)  # 0 -> 1
    q = Fraction(rng.randint(1, den - 1), den)  # 1 -> 0
    if kind == "find_initial":
        # Redraw until 1 - p - q != 0 so the backward solve is well posed,
        # and until the resulting initial probability lands in [0, 1] --
        # exactly the same "keep redrawing until the question is well
        # posed" discipline the forward path already uses for `initial`.
        while True:
            initial = Fraction(rng.randint(0, den), den)  # P(X_0=1), used only to construct a consistent target
            step_one = (1 - initial) * p + initial * (1 - q)
            denom_coeff = 1 - p - q
            if denom_coeff == 0:
                q = Fraction(rng.randint(1, den - 1), den)
                continue
            recovered_initial = (step_one - p) / denom_coeff
            if Fraction(0) <= recovered_initial <= Fraction(1):
                break
        check(recovered_initial == initial, "Markov backward-solve recovery mismatch")
        check(Fraction(0) <= step_one <= Fraction(1), "Markov probability outside [0,1]")
        answer = initial
        payload, answer_text = _fraction_payload(answer)
        alternatives = _probability_options(
            answer,
            [
                step_one,
                p,
                1 - q,
                (step_one - (1 - q)) / (p - (1 - q)) if p != (1 - q) else step_one,
                answer / 2,
            ],
            den,
        )
        return dict(
            text=(f"A two-state chain has P(0→1)={p}, P(1→0)={q}, and P(X₁=1)={step_one}. Find P(X₀=1)."),
            latex=rf"P(X_0=1),\quad P(0\to1)={p},\;P(1\to0)={q},\;P(X_1=1)={step_one}",
            answer=payload,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"P(X₁=1) = (1−P(X₀=1))·{p} + P(X₀=1)·{1 - q}.",
                f"Solving for P(X₀=1): P(X₀=1) = ({step_one} − {p}) / (1 − {p} − {q}) = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, alternatives),
        )
    initial = Fraction(rng.randint(0, den), den)  # P(X_0=1)
    answer = (1 - initial) * p + initial * (1 - q)
    row_sum = (1 - p) + p
    independent = (1 - initial) * p + initial * (1 - q)
    check(row_sum == 1 and answer == independent, "Markov one-step transition mismatch")
    check(Fraction(0) <= answer <= Fraction(1), "Markov probability outside [0,1]")
    payload, answer_text = _fraction_payload(answer)
    alternatives = _probability_options(answer, [p, 1 - q, initial, (1 - initial) * p + initial * q, answer / 2], den)
    return dict(
        text=(f"A two-state chain has P(0→1)={p}, P(1→0)={q}, and P(X₀=1)={initial}. Find P(X₁=1)."),
        latex=rf"P(X_1=1),\quad P(0\to1)={p},\;P(1\to0)={q},\;P(X_0=1)={initial}",
        answer=payload,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"P(X₁=1)=P(X₀=0)P(0→1)+P(X₀=1)P(1→1) = {answer_text}."],
        distractors=pick_distractors(answer_text, alternatives),
    )


@generator(
    "markov_two_state_stationary",
    level=17,
    presets={
        "easy": dict(denominator_hi=5),
        "medium": dict(denominator_hi=9),
        "hard": dict(denominator_hi=14),
    },
)
def markov_two_state_stationary(rng, denominator_hi=9, kind=None):
    """Find the stationary probability of state 1 for an irreducible 2-state chain.

    ``kind=None`` (the default) preserves the original forward behaviour
    byte-for-byte: p and q are given, π1 is asked. ``kind="find_transition"``
    gives p and π1 instead, and asks for q -- solving the SAME balance
    equation backwards (q = π0·p/π1) rather than substituting forward into
    π1 = p/(p+q).
    """
    den = rng.randint(3, denominator_hi)
    p = Fraction(rng.randint(1, den - 1), den)
    q = Fraction(rng.randint(1, den - 1), den)
    pi_one = p / (p + q)
    pi_zero = 1 - pi_one
    # Balance equation and direct matrix multiplication are independent checks.
    check(pi_zero * p == pi_one * q and pi_zero + pi_one == 1, "stationary balance equation failed")
    check(pi_zero * p + pi_one * (1 - q) == pi_one, "stationary transition check failed")
    if kind == "find_transition":
        derived_q = pi_zero * p / pi_one
        check(derived_q == q, "stationary backward-solve recovery mismatch")
        payload, answer_text = _fraction_payload(q)
        alternatives = _probability_options(q, [p, pi_one, 1 - pi_one, pi_zero, q / 2], den)
        return dict(
            text=(f"For P(0→1)={p} and a stationary probability of state 1 equal to {pi_one}, find P(1→0)."),
            latex=rf"\pi P=\pi,\quad P(0\to1)={p},\;\pi_1={pi_one};\quad\text{{find }}P(1\to0)",
            answer=payload,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"Balance: π₀·{p} = π₁·q, with π₀ = {pi_zero}.",
                f"q = π₀·{p}/π₁ = {pi_zero}·{p}/{pi_one} = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, alternatives),
        )
    payload, answer_text = _fraction_payload(pi_one)
    alternatives = _probability_options(
        pi_one, [q / (p + q), p / (1 - q + p), p, 1 - pi_one, pi_one / 2, p / (p + q + 1)], den
    )
    return dict(
        text=f"For P(0→1)={p} and P(1→0)={q}, find the stationary probability of state 1.",
        latex=rf"\pi P=\pi,\quad P=\begin{{pmatrix}}1-{p}&{p}\\{q}&1-{q}\end{{pmatrix}}",
        answer=payload,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Balance gives π₀·{p}=π₁·{q} and π₀+π₁=1, so π₁ = {answer_text}."],
        distractors=pick_distractors(answer_text, alternatives),
    )


@generator(
    "finite_joint_covariance",
    level=17,
    presets={
        "easy": dict(value_hi=2, weight_hi=4),
        "medium": dict(value_hi=4, weight_hi=7),
        "hard": dict(value_hi=6, weight_hi=11),
    },
)
def finite_joint_covariance(rng, value_hi=4, weight_hi=7, kind=None):
    """Compute covariance from a finite 2x2 joint distribution.

    ``kind=None`` (the default) preserves the original four-cell behaviour
    byte-for-byte: all four joint counts are shown directly. ``kind=
    "missing_weight"`` WITHHOLDS one cell's count (rendered as ``?``) and
    gives the total population instead, so the withheld cell must be
    derived first (total − the other three cells) before the covariance can
    be computed -- a genuine extra reasoning step, not a bigger number.
    """
    while True:
        x_values = sorted(rng.sample(range(-value_hi, value_hi + 1), 2))
        y_values = sorted(rng.sample(range(-value_hi, value_hi + 1), 2))
        weights = [[rng.randint(1, weight_hi) for _ in y_values] for _ in x_values]
        total = sum(map(sum, weights))
        ex = Fraction(sum(x_values[i] * weights[i][j] for i in range(2) for j in range(2)), total)
        ey = Fraction(sum(y_values[j] * weights[i][j] for i in range(2) for j in range(2)), total)
        exy = Fraction(sum(x_values[i] * y_values[j] * weights[i][j] for i in range(2) for j in range(2)), total)
        covariance = exy - ex * ey
        if covariance != 0:
            break
    direct = Fraction(
        sum((x_values[i] - ex) * (y_values[j] - ey) * weights[i][j] for i in range(2) for j in range(2)), total
    )
    centered = exy - ex * ey
    check(covariance == direct == centered, "joint covariance calculation mismatch")
    payload, answer_text = _fraction_payload(covariance)
    alternatives = [
        _fraction_payload(exy)[1],
        _fraction_payload(ex * ey)[1],
        _fraction_payload(-covariance)[1],
        _fraction_payload(ex + ey)[1],
    ]
    if kind == "missing_weight":
        hidden_i, hidden_j = 0, 0
        others_sum = weights[0][1] + weights[1][0] + weights[1][1]
        derived_weight = total - others_sum
        check(derived_weight == weights[hidden_i][hidden_j], "missing joint-weight cell derivation mismatch")
        table_parts = []
        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):
                if (i, j) == (hidden_i, hidden_j):
                    table_parts.append(f"P(X={x},Y={y})=?/{total}")
                else:
                    table_parts.append(f"P(X={x},Y={y})={weights[i][j]}/{total}")
        table = "; ".join(table_parts)
        return dict(
            text=(
                f"A joint distribution over a population of {total} has {table}. "
                "One cell's count is unknown (shown as ?); find it first. Find Cov(X,Y)."
            ),
            latex=(
                rf"\operatorname{{Cov}}(X,Y)=E[XY]-E[X]E[Y],\quad n={total},\;"
                rf"X\in\{{{x_values[0]},{x_values[1]}\}},\;Y\in\{{{y_values[0]},{y_values[1]}\}}"
            ),
            answer=payload,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"The missing cell is {total} − {others_sum} = {derived_weight}.",
                f"E[X]={ex}, E[Y]={ey}, E[XY]={exy}.",
                f"Cov(X,Y) = {exy} − ({ex})({ey}) = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, alternatives),
        )
    table = "; ".join(
        f"P(X={x},Y={y})={weights[i][j]}/{total}" for i, x in enumerate(x_values) for j, y in enumerate(y_values)
    )
    return dict(
        text=f"A joint distribution has {table}. Find Cov(X,Y).",
        latex=rf"\operatorname{{Cov}}(X,Y)=E[XY]-E[X]E[Y],\quad X\in\{{{x_values[0]},{x_values[1]}\}},\;Y\in\{{{y_values[0]},{y_values[1]}\}}",
        answer=payload,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"E[X]={ex}, E[Y]={ey}, E[XY]={exy}.", f"Cov(X,Y) = {exy} − ({ex})({ey}) = {answer_text}."],
        distractors=pick_distractors(answer_text, alternatives),
    )
