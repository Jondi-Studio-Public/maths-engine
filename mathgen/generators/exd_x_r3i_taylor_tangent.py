"""Exam-tier source for `ucv1-taylor-linear-approximation`: a tangent-line chain on a cubic.

The stem gives a cubic f and a base point a. The parts build on each other:

(a) the derivative f'(a)                                  number,   1 mark
(b) the linear approximation L(x0) near a                 number,   2 marks, f(a) + (x0 - a) * part_a
(c) the error f(x0) - L(x0) of that approximation         number,   2 marks, f(x0) - part_b
(d) where the tangent line meets the x-axis               fraction, 2 marks, a - f(a) / part_a

## Construction

The cubic has integer coefficients and a non-zero cubic term, so f'(a) needs the power rule on every term and the
error is a genuine remainder (a quadratic function would have f'' as the whole story). The offset x0 - a is a whole
number of tenths, so L(x0) is exact at 1 decimal place and the error f(x0) - L(x0) = (3 c3 a + c2) d^2 + c3 d^3 is
exact at 3 decimal places. Every quantity is computed in exact rationals; nothing is rounded except to display.
f'(a) and f(a) are non-zero, so the tangent has a single x-intercept different from a.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, join_terms, paren_negative, phrase, rational_text
from ._exd_x_r3_distinct import distinct_answers
from .exd_x_r3i_support import fraction_part, number_part


def _poly_text(c3: int, c2: int, c1: int, c0: int) -> str:
    return join_terms([(c3, "x^3"), (c2, "x^2"), (c1, "x"), (c0, "")])


@generator("exd_x_r3i_taylor_tangent", level=14, presets={"exam": dict(tier="exam")})
@distinct_answers
def exd_x_r3i_taylor_tangent(rng, tier: str = "exam") -> dict:
    """Derivative, linear approximation, its error and the tangent's x-intercept for a cubic."""
    check(tier == "exam", f"unknown tier {tier!r}")

    def f_at(c, x):
        return ((c[0] * x + c[1]) * x + c[2]) * x + c[3]

    for _ in attempts(1000, "a cubic with a usable tangent"):
        c3 = rng.choice([-2, -1, 1, 2])
        c2 = rng.randint(-4, 4)
        c1 = rng.randint(-6, 6)
        c0 = rng.randint(-9, 9)
        a = rng.choice([-3, -2, -1, 1, 2, 3])
        d10 = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        coeffs = (c3, c2, c1, c0)
        fa = f_at(coeffs, a)
        fpa = 3 * c3 * a * a + 2 * c2 * a + c1
        d_f = Fraction(d10, 10)
        fx0 = f_at(coeffs, a + d_f)
        approx = fa + fpa * d_f
        if fa != 0 and fpa != 0 and fx0 != approx:
            break
    x0 = a + d_f
    error = fx0 - approx
    intercept = a - Fraction(fa, fpa)
    check(approx * 10 == int(approx * 10), "L(x0) is not exact at one decimal place")

    # Independent routes: f'(a) from a symmetric difference ((f(a+1) - f(a-1)) / 2 = f'(a) + c3 for a cubic), and the
    # error as the Taylor remainder f''(a)/2 d^2 + f'''(a)/6 d^3.
    check(Fraction(f_at(coeffs, a + 1) - f_at(coeffs, a - 1), 2) - c3 == fpa, "f'(a) disagrees with the difference")
    check(error == (3 * c3 * a + c2) * d_f**2 + c3 * d_f**3, "error disagrees with the Taylor remainder")

    f_text = _poly_text(c3, c2, c1, c0)
    x0_text = f"{x0.numerator / x0.denominator:.1f}"
    stem = phrase(
        rng,
        (
            "The function f is defined by f(x) = {f}. The tangent line to the graph of f at x = {a} is used to "
            "approximate f near x = {a}.",
            "Let f(x) = {f}. Near x = {a}, the graph of f is approximated by its tangent line y = L(x).",
            "A smooth function is given by f(x) = {f}. Its linear approximation L(x) at x = {a} is used to estimate "
            "values of f close to x = {a}.",
        ),
        f=f_text,
        a=a,
    )

    f_a = f"f({a})"
    cubic_derivative = 3 * c3 * a**3 + 2 * c2 * a**2 + c1 * a  # exponents kept instead of lowered
    part_a = number_part(
        f"Find f'({a}).",
        Fraction(fpa),
        0,
        1,
        [f"f'(x) = {join_terms([(3 * c3, 'x^2'), (2 * c2, 'x'), (c1, '')])}, so f'({a}) = {fpa}."],
        [
            ("evaluated-function-for-derivative", Fraction(fa)),
            ("power-rule-exponent-not-lowered", Fraction(cubic_derivative)),
            ("constant-term-differentiated-as-itself", Fraction(fpa + c0)),
            ("differentiated-twice", Fraction(6 * c3 * a + 2 * c2)),
            ("derivative-of-cubic-term-only", Fraction(3 * c3 * a**2)),
            ("derivative-drops-cubic-term", Fraction(2 * c2 * a + c1)),
        ],
    )
    part_b = number_part(
        f"Using your answer to (a), find the linear approximation L({x0_text}) of f({x0_text}), correct to 1 decimal "
        "place.",
        approx,
        1,
        2,
        [
            f"{f_a} = {fa}.",
            f"L({x0_text}) = {fa} + {paren_negative(fpa)} × {paren_negative(d_f)} = {float(approx):.1f}.",
        ],
        [
            ("dropped-linear-correction", Fraction(fa)),
            ("evaluated-function-at-target", fx0),
            ("sign-slip-in-formula", fa - fpa * d_f),
            ("dropped-base-value", fpa * d_f),
        ],
        follow_through=f"{fa} + ({d10})*part_a/10",
    )
    part_c = number_part(
        f"Find the error f({x0_text}) - L({x0_text}) in this approximation, correct to 3 decimal places.",
        error,
        3,
        2,
        [
            f"f({x0_text}) = {fx0}.",
            f"f({x0_text}) - L({x0_text}) = {fx0} - {paren_negative(approx)} = {float(error):.3f}.",
        ],
        [
            ("sign-of-result-reversed", -error),
            ("change-in-f-for-error", fx0 - fa),
            ("reported-approximation-as-error", approx),
            ("error-uses-derivative-at-target", fx0 - fpa * d_f),
        ],
        follow_through=f"({fx0.numerator})/({fx0.denominator}) - part_b",
    )
    part_d = fraction_part(
        "The line y = L(x) meets the x-axis at x = r. Find r as a fraction in its simplest form.",
        intercept,
        2,
        [
            f"L(x) = {fa} + {paren_negative(fpa)}(x - {paren_negative(a)}) = 0, so x = {a} - {paren_negative(fa)}/{paren_negative(fpa)} = {rational_text(intercept)}."
        ],
        [
            ("sign-slip-in-formula", a + Fraction(fa, fpa)),
            ("tangent-intercept-ignores-base-point", -Fraction(fa, fpa)),
            ("tangent-intercept-ratio-inverted", a - Fraction(fpa, fa)),
            ("tangent-intercept-ignores-base-point", Fraction(fa, fpa)),
            ("sign-slip-in-formula", -a - Fraction(fa, fpa)),
        ],
        follow_through=f"{a} - ({fa})/part_a",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
