"""Exam-tier source for `ucv1-derivative-definition`: a cubic with an unknown coefficient, found from a gradient.

The practice tiers give a cubic with numeric coefficients and ask for f'(a). Here one coefficient is a constant
``k`` and the gradient at a point is given instead, so the learner works the limit definition on a function with a
parameter, solves for it, and carries it forward:

(a) the value of k from the stated gradient at x = a          fraction, 2 marks
(b) the gradient at a second point x = b                      fraction, 1 mark,  ``3b^2 + 2 part_a b + r``
(c) where the tangent at x = b meets the y-axis               fraction, 2 marks, ``f(b) - b f'(b)``

``f(x) = x^3 + k x^2 + r x + s`` with ``k = n / 2``, so the stated gradient ``3a^2 + n a + r`` is a whole number and
k is a half-integer in general. The derivative is found from the difference quotient by expanding with SymPy and
setting h = 0 (never `sympy.diff`), and k is then solved from the stated gradient independently.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Rational, expand, symbols

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    attempts,
    check,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    phrase,
    pick_distractors,
    rational_text,
)

_X, _H, _K = symbols("x h k")


def _fraction_part(prompt, value, marks, working, candidates, follow_through=None):
    """A `fraction` part whose answer is `value`; `candidates` are (misconception id or None, value) pairs."""
    text = rational_text(value)
    options = [Distractor(rational_text(v), tag, exact_value(v)) for tag, v in candidates if v != value]
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=pick_distractors(text, options, 3),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _draw(rng: random.Random) -> dict | None:
    a = rng.choice((-3, -2, -1, 1, 2, 3))
    b = rng.choice([v for v in range(-3, 4) if v not in (0, a)])
    n = rng.choice([v for v in range(-25, 26) if v])
    r = rng.choice([v for v in range(-9, 10) if v])
    s = rng.choice([v for v in range(-9, 10) if v])
    k = Fraction(n, 2)

    def f(t):
        return t**3 + k * t * t + r * t + s

    def f_prime(t):
        return 3 * t * t + 2 * k * t + r

    gradient_a = 3 * a * a + n * a + r  # the stated gradient, a whole number
    slope_b = f_prime(b)
    if slope_b == 0:
        return None
    intercept = f(b) - b * slope_b

    # Independent route: the difference quotient of the symbolic cubic, h set to zero, then solve for k.
    symbolic = _X**3 + _K * _X**2 + r * _X + s
    quotient = expand((symbolic.subs(_X, _X + _H) - symbolic) / _H)
    derivative = quotient.subs(_H, 0)
    equation = expand(derivative.subs(_X, a) - gradient_a)  # linear in k
    check(-equation.subs(_K, 0) / equation.coeff(_K) == Rational(n, 2), "k disagrees with the limit definition")
    check(Fraction(int(derivative.subs({_X: b, _K: Rational(n, 2)}))) == slope_b, "f'(b) disagrees")
    check(intercept == s - 2 * b**3 - k * b * b, "intercept disagrees with the closed form")

    f_text = join_terms([(1, "x³"), (1, "kx²"), (r, "x"), (s, "")])
    stem = phrase(
        rng,
        (
            "Let f(x) = {f}, where k is a constant. The tangent to the curve y = f(x) at x = {a} has gradient {m}.",
            "A curve has equation y = {f}, where k is a constant. At x = {a} the gradient of the curve is {m}.",
            "The function f(x) = {f} contains a constant k. The gradient of the tangent to y = f(x) where x = {a} "
            "is {m}.",
        ),
        f=f_text,
        a=a,
        m=gradient_a,
    )

    f_b = f(b)
    # k from the stated gradient under a slip in the derivative formula.
    wrong_k_flat = Fraction(gradient_a - 3 * a * a - r, a)  # derivative of k x^2 taken as k x
    wrong_k_value = Fraction(gradient_a - a**3 - r * a - s, a * a)  # f(a) set equal to the gradient
    wrong_k_sign = Fraction(gradient_a - 3 * a * a + r, 2 * a)  # sign slip on the linear coefficient
    wrong_k_square = Fraction(gradient_a - 3 * a - r, 2 * a)  # derivative of x^3 taken as 3x
    part_a = _fraction_part(
        f"Use the limit definition f'(a) = lim(h→0) [f(a+h) − f(a)] / h at x = {a} to find the value of k.",
        k,
        2,
        [
            f"The difference quotient simplifies to 3x² + 2kx + r + h(3x + k) + h², so f'(x) = 3x² + 2kx + {paren_negative(r)}.",
            f"f'({a}) = {join_terms([(3 * a * a, ''), (2 * a, 'k'), (r, '')])} = {gradient_a}, so k = {rational_text(k)}.",
        ],
        [
            ("dropped-factor-two-in-derivative", wrong_k_flat),
            ("evaluates-function-not-derivative", wrong_k_value),
            (None, wrong_k_sign),
            (None, wrong_k_square),
        ],
    )
    part_b = _fraction_part(
        f"Hence find the gradient of the curve at x = {b}.",
        slope_b,
        1,
        [f"f'({b}) = {join_terms([(3 * b * b, ''), (2 * b, 'k'), (r, '')])} = {rational_text(slope_b)}."],
        [
            ("evaluates-function-not-derivative", f_b),
            ("dropped-factor-two-in-derivative", 3 * b * b + k * b + r),
            (None, 3 * b * b + 2 * k * b),
            (None, 3 * b + 2 * k * b + r),
        ],
        follow_through=f"3*({b})**2 + 2*part_a*({b}) + ({r})",
    )
    part_c = _fraction_part(
        f"Hence find the y-intercept of the tangent to the curve at x = {b}.",
        intercept,
        2,
        [
            f"f({b}) = {rational_text(f_b)}, so the tangent is y − {paren_negative(f_b)} = {rational_text(slope_b)}(x − {paren_negative(b)}).",
            f"Putting x = 0 gives y = {rational_text(f_b)} − {paren_negative(b)} × {paren_negative(slope_b)} = {rational_text(intercept)}.",
        ],
        [
            (None, f_b),
            (None, f_b - slope_b),
            (None, f_b + b * slope_b),
            ("gave-x-intercept-for-y-intercept", b - f_b / slope_b),
        ],
        follow_through=f"({b})**3 + part_a*({b})**2 + ({r})*({b}) + ({s}) - ({b})*part_b",
    )
    if any(len(p["distractors"]) < 3 for p in (part_a, part_b, part_c)):
        return None
    return dict(text=stem, parts=[part_a, part_b, part_c])


@generator("exd_x_u5_derivative_definition", level=17, presets={"exam": dict(tier="exam")})
def exd_x_u5_derivative_definition(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: find a constant from a gradient via the limit definition, then use it for a tangent."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a cubic with three distinct distractors per part"):
        question = _draw(rng)
        if question is not None:
            return question
