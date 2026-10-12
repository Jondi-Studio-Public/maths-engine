"""Exam tier for "interpret a bounded optimum perturbation" of a convex quadratic.

The practice tiers pick, from four stated conclusions, the one justified by a perturbation. Here the learner
works the whole sensitivity analysis of f(x) = a x^2 + b x + c (a > 0) as a one-variable optimisation exam would
set it:

(a) the minimiser x* = -b / (2a);
(b) the minimum value V = c - b^2 / (4a);
(c) the minimum value after the coefficient b is changed by d, from (b);
(d) how far the first-order estimate V + d x* (the old minimiser used as the rate of change of V with b) is from
    the exact new minimum value, from (a), (b) and (c). The exact minus the estimate is -d^2 / (4a).

All answers are exact fractions.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator, join_terms, rational_text
from ._exd_x_r3_distinct import distinct_answers
from ._exd_x_r3g_kit import clashes, fraction_part

_STEMS = (
    "Let f(x) = {f}, defined for all real x. The coefficient of x in f is then changed from {b} to {b2}.",
    "A smooth cost model is f(x) = {f}. A revised estimate changes the linear coefficient from {b} to {b2}, "
    "leaving the other coefficients as they are.",
    "The function f(x) = {f} is to be minimised over the real numbers. Suppose the coefficient b of x is "
    "perturbed from {b} to {b2}.",
)


@generator("exd_x_r3g_quadratic_sensitivity", level=12, presets={"exam": dict(tier="exam")})
@distinct_answers
def exd_x_r3g_quadratic_sensitivity(rng: random.Random, tier: str = "exam") -> dict:
    """Minimiser, minimum value, the minimum after a perturbation of b, and the error of the linear estimate."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a quadratic whose minimum moves"):
        a = rng.randint(1, 6)
        b, c = rng.randint(-12, 12), rng.randint(-9, 9)
        d = rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
        if b == 0 or b + d == 0 or (b + d) == -b:
            continue
        xstar = Fraction(-b, 2 * a)
        vmin = c - Fraction(b * b, 4 * a)
        vnew = c - Fraction((b + d) ** 2, 4 * a)
        if vnew == vmin or xstar.denominator == 1 and vmin.denominator == 1:
            continue
        f = join_terms([(a, "x²"), (b, "x"), (c, "")])
        stem = rng.choice(_STEMS).format(f=f, b=b, b2=b + d)
        pd = (
            "The first-order estimate of the new minimum value is the old minimum value plus the change in "
            "the coefficient multiplied by the old minimiser. Find the exact new minimum value minus this estimate."
        )
        error = vnew - (vmin + d * xstar)
        if clashes([rational_text(v) for v in (xstar, vmin, vnew, error)], stem, pd):
            continue
        break

    check(error == -Fraction(d * d, 4 * a), "the estimate error is not -d^2 / (4a)")
    parts = [
        fraction_part(
            "Find the value of x at which f is minimised.",
            xstar,
            2,
            [f"f'(x) = {2 * a}x + ({b}) = 0, so x = {rational_text(xstar)}."],
            [
                (Fraction(-b, a), "dropped-factor-of-two"),
                (Fraction(b, 2 * a), "sign-slip-in-formula"),
                (Fraction(-b, 4 * a), "wrong-constant-in-formula"),
                (Fraction(-c, 2 * a), "minimiser-uses-the-constant-term"),
            ],
        ),
        fraction_part(
            "Find the minimum value of f.",
            vmin,
            2,
            [f"f({rational_text(xstar)}) = {rational_text(vmin)}."],
            [
                (c - Fraction(b * b, 2 * a), "dropped-factor-of-four"),
                (c + Fraction(b * b, 4 * a), "sign-slip-in-formula"),
                (xstar, "minimum-value-reports-the-minimiser"),
                (Fraction(c), "minimum-value-reports-the-constant-term"),
            ],
            follow_through=f"{a}*part_a**2 + ({b})*part_a + ({c})",
        ),
        fraction_part(
            "Find the minimum value of f after the change.",
            vnew,
            3,
            [
                f"New minimum = {c} - ({b + d})² / {4 * a} = {rational_text(vnew)}.",
            ],
            [
                (vmin, "new-minimum-ignores-the-change"),
                (vmin + d * xstar, "new-minimum-linearised"),
                (vmin - Fraction(d * d, 4 * a), "new-minimum-drops-the-cross-term"),
                (c - Fraction((b + d) ** 2, 2 * a), "dropped-factor-of-four"),
            ],
            follow_through=f"part_b - ({2 * b * d + d * d})/{4 * a}",
        ),
        fraction_part(
            pd,
            error,
            3,
            [
                f"Estimate = {rational_text(vmin)} + ({d})({rational_text(xstar)}) = {rational_text(vmin + d * xstar)}.",
                f"Exact - estimate = {rational_text(vnew)} - {rational_text(vmin + d * xstar)} = {rational_text(error)}.",
            ],
            [
                (Fraction(0), "sensitivity-estimate-assumed-exact"),
                (Fraction(d * d, 4 * a), "sign-of-result-reversed"),
                (-Fraction(d * d, 2 * a), "dropped-factor-of-four"),
                (-Fraction(d, 4 * a), "sensitivity-error-omits-the-square"),
            ],
            follow_through=f"part_c - part_b - ({d})*part_a",
        ),
    ]
    return dict(text=stem, parts=parts)
