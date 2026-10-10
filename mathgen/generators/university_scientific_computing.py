"""Small, exact numerical-analysis building blocks.

This module intentionally covers only one mechanically checkable update at a
time.  It does not claim to assess convergence proofs, stability arguments,
or floating-point implementation details.  Those require a richer assessment
contract than the current typed graders provide.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

from ..core import check, fmt_signed, generator, join_terms, latex_answer, pick_distractors, rational_text


def _fraction_payload(value: Fraction) -> tuple[dict, str]:
    value = Fraction(value)
    return {"p": value.numerator, "q": value.denominator}, rational_text(value)


def _pair_payload(left: Fraction, right: Fraction, labels=("x", "y")) -> tuple[dict, str]:
    left, right = Fraction(left), Fraction(right)
    text_left, text_right = rational_text(left), rational_text(right)
    return {labels[0]: left, labels[1]: right}, f"{labels[0]} = {text_left}, {labels[1]} = {text_right}"


@generator(
    "condition_number_diagonal",
    level=17,
    presets={
        "easy": dict(value_hi=5),
        "medium": dict(value_hi=10),
        "hard": dict(value_hi=18),
    },
)
def condition_number_diagonal(rng, value_hi=10, size=2):
    """Find the 2-norm condition number of a nonsingular diagonal matrix.

    For a positive diagonal matrix the 2-norm condition number is the ratio
    of its largest to smallest diagonal entries.  Keeping the matrix diagonal
    makes this a transparent first computational slice rather than a claim to
    cover general singular-value computation.

    ``size=2`` (default) preserves the original two-entry behaviour
    byte-for-byte. ``size=3`` draws three distinct positive diagonal
    entries, so the largest and smallest must be picked out of three values
    -- genuine ordering, not a bigger pair.
    """
    if size == 3:
        values: list[int] = []
        while len(values) < 3:
            candidate = rng.randint(1, value_hi)
            if candidate not in values:
                values.append(candidate)
        smallest, middle, largest = sorted(values)
        answer = Fraction(largest, smallest)
        direct = Fraction(largest, 1) * Fraction(1, smallest)
        check(answer == direct, "diagonal condition-number check failed (3x3)")
        payload, answer_text = _fraction_payload(answer)
        alternatives = [
            rational_text(Fraction(smallest, largest)),
            rational_text(Fraction(largest, middle)),
            rational_text(Fraction(middle, smallest)),
            rational_text(Fraction(largest * largest, smallest * smallest)),
        ]
        values_text = ", ".join(str(v) for v in values)
        return dict(
            text=(f"For A = diag({values_text}), find the 2-norm condition number κ₂(A) = ‖A‖₂‖A⁻¹‖₂."),
            latex=rf"A=\operatorname{{diag}}({values_text}),\quad \kappa_2(A)",
            answer=payload,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"‖A‖₂ = {largest} (largest |entry|) and ‖A⁻¹‖₂ = 1/{smallest} (smallest |entry|).",
                f"Therefore κ₂(A) = {answer_text}.",
            ],
            distractors=pick_distractors(answer_text, alternatives),
        )
    first = rng.randint(1, value_hi)
    second = rng.randint(1, value_hi)
    while second == first:
        second = rng.randint(1, value_hi)
    largest, smallest = max(first, second), min(first, second)
    answer = Fraction(largest, smallest)
    direct = Fraction(max(abs(first), abs(second)), min(abs(first), abs(second)))
    inverse_product = Fraction(largest, 1) * Fraction(1, smallest)
    check(answer == direct == inverse_product, "diagonal condition-number check failed")
    payload, answer_text = _fraction_payload(answer)
    alternatives = [
        rational_text(Fraction(smallest, largest)),
        rational_text(Fraction(first + second, 1)),
        rational_text(Fraction(largest - smallest, 1)),
        rational_text(Fraction(largest * largest, smallest * smallest)),
    ]
    return dict(
        text=(f"For A = diag({first}, {second}), find the 2-norm condition number κ₂(A) = ‖A‖₂‖A⁻¹‖₂."),
        latex=rf"A=\operatorname{{diag}}({first},{second}),\quad \kappa_2(A)",
        answer=payload,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"‖A‖₂ = {largest} and ‖A⁻¹‖₂ = 1/{smallest}.", f"Therefore κ₂(A) = {answer_text}."],
        distractors=pick_distractors(answer_text, alternatives),
    )


@generator(
    "jacobi_iteration_2x2",
    level=17,
    presets={
        "easy": dict(coefficient_hi=4, point_hi=3),
        "medium": dict(coefficient_hi=8, point_hi=5),
        "hard": dict(coefficient_hi=14, point_hi=8),
    },
)
def jacobi_iteration_2x2(rng, coefficient_hi=8, point_hi=5):
    """Perform one Jacobi update for a diagonally dominant 2x2 system."""
    diagonal_a = rng.randint(2, coefficient_hi)
    diagonal_d = rng.randint(2, coefficient_hi)
    off_b = rng.randint(-max(1, diagonal_a - 1), max(1, diagonal_a - 1))
    off_c = rng.randint(-max(1, diagonal_d - 1), max(1, diagonal_d - 1))
    x_true = rng.randint(-point_hi, point_hi)
    y_true = rng.randint(-point_hi, point_hi)
    x0 = rng.randint(-point_hi, point_hi)
    y0 = rng.randint(-point_hi, point_hi)
    rhs_one = diagonal_a * x_true + off_b * y_true
    rhs_two = off_c * x_true + diagonal_d * y_true

    x1 = Fraction(rhs_one - off_b * y0, diagonal_a)
    y1 = Fraction(rhs_two - off_c * x0, diagonal_d)
    # Independent checks: coordinate formula and a matrix residual using the
    # frozen (x0, y0) iterate in each row.
    check(diagonal_a * x1 + off_b * y0 == rhs_one, "Jacobi first-coordinate update mismatch")
    check(off_c * x0 + diagonal_d * y1 == rhs_two, "Jacobi second-coordinate update mismatch")
    check(x1 == Fraction(rhs_one, diagonal_a) - Fraction(off_b, diagonal_a) * y0, "Jacobi matrix-row check failed")
    answer, answer_text = _pair_payload(x1, y1)
    alternatives = [
        f"x = {rational_text(y1)}, y = {rational_text(x1)}",
        f"x = {rational_text(Fraction(rhs_one, diagonal_a))}, y = {rational_text(Fraction(rhs_two, diagonal_d))}",
        f"x = {rational_text(-x1)}, y = {rational_text(y1)}",
        f"x = {rational_text(x0)}, y = {rational_text(y0)}",
    ]
    return dict(
        text=(
            f"For {diagonal_a}x + ({off_b})y = {rhs_one} and "
            f"({off_c})x + {diagonal_d}y = {rhs_two}, perform one Jacobi "
            f"iteration from (x₀, y₀) = ({x0}, {y0}). Give (x₁, y₁)."
        ),
        latex=(
            rf"A=\begin{{pmatrix}}{diagonal_a}&{off_b}\\{off_c}&{diagonal_d}\end{{pmatrix}},"
            rf"\quad b=\begin{{pmatrix}}{rhs_one}\\{rhs_two}\end{{pmatrix}},"
            rf"\quad (x_0,y_0)=({x0},{y0})"
        ),
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"x₁ = ({rhs_one} − ({off_b})·{y0})/{diagonal_a} = {rational_text(x1)}.",
            f"y₁ = ({rhs_two} − ({off_c})·{x0})/{diagonal_d} = {rational_text(y1)}.",
        ],
        distractors=pick_distractors(answer_text, alternatives),
    )


@generator(
    "explicit_euler_linear_ode",
    level=16,
    presets={
        "easy": dict(coefficient_hi=3, step_denominator_hi=4, value_hi=5),
        "medium": dict(coefficient_hi=6, step_denominator_hi=7, value_hi=8),
        "hard": dict(coefficient_hi=10, step_denominator_hi=11, value_hi=12),
    },
)
def explicit_euler_linear_ode(rng, coefficient_hi=6, step_denominator_hi=7, value_hi=8, steps=1):
    """Take one (or two) exact explicit-Euler step(s) for y' = λy + c.

    ``steps=1`` (default) preserves the original one-step behaviour
    byte-for-byte. ``steps=2`` takes a SECOND Euler step from y1 using the
    same step size h, re-evaluating the slope at the new point rather than
    reusing the first step's slope -- a genuine second iteration, not a
    bigger step or a bigger coefficient.
    """
    lam = rng.randint(-coefficient_hi, coefficient_hi)
    forcing = rng.randint(-coefficient_hi, coefficient_hi)
    y0 = rng.randint(-value_hi, value_hi)
    denominator = rng.randint(2, step_denominator_hi)
    h = Fraction(1, denominator)
    slope0 = lam * y0 + forcing
    y1 = Fraction(y0) + h * slope0
    direct1 = Fraction(y0) + h * (Fraction(lam) * y0 + forcing)
    check(y1 == direct1, "explicit Euler update mismatch")
    check((y1 - y0) / h == slope0, "explicit Euler increment check failed")

    if steps == 2:
        slope1 = lam * y1 + forcing
        y2 = y1 + h * slope1
        direct2 = y1 + h * (Fraction(lam) * y1 + forcing)
        check(y2 == direct2, "second explicit Euler update mismatch")
        check((y2 - y1) / h == slope1, "second explicit Euler increment check failed")
        final = y2
    else:
        final = y1

    payload, answer_text = _fraction_payload(final)
    if steps == 2:
        alternatives = [
            rational_text(y1),
            rational_text(Fraction(y0) + 2 * h * slope0),
            rational_text(y1 - h * slope1),
            rational_text(final + h),
            rational_text(final - h),
            rational_text(final + 1),
            rational_text(final - 1),
        ]
        text = (
            f"Use two explicit Euler steps of size h = {rational_text(h)} for "
            f"y′ = ({lam})y + ({forcing}), starting from y(0) = {y0}. "
            "Find the approximation y(2h)."
        )
        latex = rf"y'=({lam})y+({forcing}),\quad y(0)={y0},\quad h={rational_text(h)},\quad y_2"
        working = [
            f"y₁ = y₀ + h·(({lam})y₀+({forcing})) = {y0} + ({rational_text(h)})({slope0}) = {rational_text(y1)}.",
            f"y₂ = y₁ + h·(({lam})y₁+({forcing})) = {rational_text(y1)} + ({rational_text(h)})({slope1}) = {answer_text}.",
        ]
    else:
        alternatives = [
            rational_text(Fraction(y0)),
            rational_text(Fraction(y0) + slope0),
            rational_text(Fraction(y0) - h * slope0),
            rational_text(h * slope0),
            rational_text(final + h),
            rational_text(final - h),
            rational_text(final + 1),
            rational_text(final - 1),
        ]
        text = (
            f"Use one explicit Euler step of size h = {rational_text(h)} for "
            f"y′ = ({lam})y + ({forcing}), starting from y(0) = {y0}. "
            "Find the approximation y(h)."
        )
        latex = rf"y'=({lam})y+({forcing}),\quad y(0)={y0},\quad h={rational_text(h)},\quad y_1"
        working = [
            f"The initial slope is ({lam})({y0}) + ({forcing}) = {slope0}.",
            f"y₁ = y₀ + h·slope = {y0} + ({rational_text(h)})({slope0}) = {answer_text}.",
        ]

    return dict(
        text=text,
        latex=latex,
        answer=payload,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, alternatives),
    )


@generator(
    "centered_difference_polynomial",
    level=16,
    presets={
        "easy": dict(coefficient_hi=3, point_hi=3, step_denominator_hi=3),
        "medium": dict(coefficient_hi=6, point_hi=5, step_denominator_hi=6),
        "hard": dict(coefficient_hi=10, point_hi=8, step_denominator_hi=10),
    },
)
def centered_difference_polynomial(rng, coefficient_hi=6, point_hi=5, step_denominator_hi=6):
    """Estimate a cubic derivative with a centred finite difference."""
    cubic = rng.randint(1, coefficient_hi)
    quadratic = rng.randint(-coefficient_hi, coefficient_hi)
    linear = rng.randint(-coefficient_hi, coefficient_hi)
    constant = rng.randint(-coefficient_hi, coefficient_hi)
    point = rng.randint(-point_hi, point_hi)
    h = Fraction(1, rng.randint(2, step_denominator_hi))

    def f(x):
        return cubic * x**3 + quadratic * x**2 + linear * x + constant

    answer = Fraction(f(Fraction(point) + h) - f(Fraction(point) - h), 2 * h)
    direct = (f(Fraction(point) + h) - f(Fraction(point) - h)) / (2 * h)
    # For a cubic, the exact centred result is f'(x) + a h².  This is an
    # independent polynomial identity check, not merely a repeated formula.
    identity = 3 * cubic * point**2 + 2 * quadratic * point + linear + cubic * h**2
    check(answer == direct == identity, "centred finite-difference identity failed")
    payload, answer_text = _fraction_payload(answer)
    exact_derivative = 3 * cubic * point**2 + 2 * quadratic * point + linear
    forward = (f(Fraction(point) + h) - f(Fraction(point))) / h
    alternatives = [
        rational_text(exact_derivative),
        rational_text(forward),
        rational_text(-answer),
        rational_text(answer + h),
    ]
    # Use the shared signed-term formatters so unit coefficients render as
    # ``x``/``-x`` rather than the malformed ``1x``/``-1x``.  This matters in
    # both the plain prompt and LaTeX, including the cubic's leading term.
    polynomial = join_terms([(cubic, "x³"), (quadratic, "x²"), (linear, "x"), (constant, "")])
    latex_terms = [fmt_signed(cubic, "x^3")]
    for coefficient, variable in ((quadratic, "x^2"), (linear, "x"), (constant, "")):
        if coefficient == 0:
            continue
        sign = "+" if coefficient > 0 else "-"
        latex_terms.append(f"{sign} {fmt_signed(abs(coefficient), variable)}")
    polynomial_latex = " ".join(latex_terms)
    return dict(
        text=(
            f"For f(x) = {polynomial}, use the centred finite difference with "
            f"h = {rational_text(h)} at x = {point}. Find "
            "[f(x+h) − f(x−h)]/(2h)."
        ),
        latex=rf"f(x)={polynomial_latex},\quad x={point},\ h={rational_text(h)}",
        answer=payload,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"f({point}+h) = {rational_text(f(Fraction(point) + h))} and f({point}−h) = {rational_text(f(Fraction(point) - h))}.",
            f"The centred difference is {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, alternatives),
    )


@generator(
    "bisection_interval_update",
    level=16,
    presets={
        "easy": dict(value_hi=12),
        "medium": dict(value_hi=30),
        "hard": dict(value_hi=80),
    },
)
def bisection_interval_update(rng, value_hi=30, reversed_sign=False):
    """Perform one bisection update for f(x)=x²−c (or its sign-reversed
    form f(x)=c−x²) on a positive bracket.

    ``reversed_sign=False`` (the default) preserves the original behaviour
    byte-for-byte. ``reversed_sign=True`` uses f(x) = c − x² instead, so the
    sign-to-half mapping the student must apply is the mirror image of the
    default convention: a positive f(midpoint) now means the root lies to
    the RIGHT of the midpoint rather than the left, exercising the update
    rule itself rather than a memorised sign pattern.
    """
    c = rng.randint(2, value_hi)
    while isqrt(c) ** 2 == c:
        c = rng.randint(2, value_hi)
    k = isqrt(c)
    left = 0
    right = k + 1
    if reversed_sign:
        # Randomise the bracket on the hard path only so the root can land in
        # either half; every left in [0, k] satisfies left**2 < c (c is a
        # non-square, so k**2 < c) and every right in [k+1, k+4] satisfies
        # right**2 > c (c < (k+1)**2 <= right**2 for any right >= k+1). The
        # default path (easy and medium) must stay byte-for-byte identical,
        # so this draw only happens here. Width of right's range sets the
        # analytic left-half split at the hard preset's value_hi=80:
        # [k+1, k+2] 17.1%, [k+1, k+3] 25.5%, [k+1, k+4] 33.4% --
        # the narrowest that meets the 30/70 target.
        left = rng.randint(0, k)
        right = rng.randint(k + 1, k + 4)
    midpoint = Fraction(left + right, 2)
    raw_midpoint = midpoint**2 - c
    f_midpoint = -raw_midpoint if reversed_sign else raw_midpoint
    if reversed_sign:
        # f(x) = c - x^2 is DEcreasing: f_midpoint>0 means the root is to
        # the right of the midpoint (mirror image of the default rule).
        if f_midpoint > 0:
            next_left, next_right = midpoint, Fraction(right)
        else:
            next_left, next_right = Fraction(left), midpoint
    else:
        if f_midpoint > 0:
            next_left, next_right = Fraction(left), midpoint
        else:
            next_left, next_right = midpoint, Fraction(right)
    if reversed_sign:
        check(
            c - Fraction(left) ** 2 > 0 > c - Fraction(right) ** 2,
            "initial bisection interval does not bracket the root",
        )
        check(c - next_left**2 >= 0 >= c - next_right**2, "bisection update lost the bracket")
    else:
        check(
            Fraction(left) ** 2 - c < 0 < Fraction(right) ** 2 - c,
            "initial bisection interval does not bracket the root",
        )
        check(next_left**2 - c <= 0 <= next_right**2 - c, "bisection update lost the bracket")
    check(next_right - next_left == Fraction(right - left, 2), "bisection width did not halve")
    answer, answer_text = _pair_payload(next_left, next_right, labels=("left", "right"))
    alternatives = [
        f"left = {rational_text(Fraction(left))}, right = {rational_text(Fraction(right))}",
        f"left = {rational_text(midpoint)}, right = {rational_text(midpoint)}",
        f"left = {rational_text(next_right)}, right = {rational_text(next_left)}",
        f"left = {rational_text(next_left - Fraction(1, 2))}, right = {rational_text(next_right + Fraction(1, 2))}",
    ]
    if reversed_sign:
        function_text = f"f(x) = {c} − x²"
        function_latex = rf"f(x)={c}-x^2"
    else:
        function_text = f"f(x) = x² − {c}"
        function_latex = rf"f(x)=x^2-{c}"
    return dict(
        text=(
            f"For {function_text}, the current positive bracket is "
            f"[{left}, {right}]. Perform one bisection update and give "
            "the new [left, right] interval."
        ),
        latex=rf"{function_latex},\quad [a,b]=[{left},{right}],\quad m=\frac{{a+b}}2",
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=[
            f"The midpoint is m = {rational_text(midpoint)} and f(m) = {rational_text(f_midpoint)}.",
            f"Keep the half containing √{c}: [{rational_text(next_left)}, {rational_text(next_right)}].",
        ],
        distractors=pick_distractors(answer_text, alternatives),
    )
