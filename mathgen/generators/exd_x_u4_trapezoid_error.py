"""Exam-tier source for `composite-trapezoidal-polynomial`: a trapezoidal estimate set against the exact integral.

The practice tiers ask for the composite trapezoidal value of a polynomial. The exam question models the depth of a
drainage channel across its width, asks for the estimated cross-sectional area, then for the exact area by
integration, and finally for the error of the estimate: the learner combines the rule with an exact integral and reads
the sign of the difference.

1. the composite trapezoidal estimate with `n` equal strips,
2. the exact area by integrating the polynomial, and
3. the estimate minus the exact area (follow-through from (a) and (b)).

The depth is a quadratic or a cubic that stays positive across the channel, and the strips are chosen so the
estimate is not exact. The independent routes sum the trapezoids pair by pair and apply Simpson's rule, which is exact for these polynomials.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import check, generator, join_terms, rational_text

from ._u4_kit import fraction_distractors, fraction_payload, survives_nsimplify

_POWERS = {3: "x³", 2: "x²", 1: "x", 0: ""}


def _poly_value(coefs: dict[int, int], x: Fraction) -> Fraction:
    return sum((c * x**power for power, c in coefs.items()), Fraction(0))


def _antiderivative(coefs: dict[int, int], x: Fraction) -> Fraction:
    return sum((Fraction(c, power + 1) * x ** (power + 1) for power, c in coefs.items()), Fraction(0))


@generator("exd_x_u4_trapezoid_error", level=16, presets={"exam": dict(tier="exam")})
def exd_x_u4_trapezoid_error(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: trapezoidal estimate, exact integral, and the error, for a channel depth model."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(1000):
        cubic = rng.random() < 0.5
        coefs = {
            3: rng.randint(1, 2) if cubic else 0,
            2: rng.randint(-3, 3),
            1: rng.randint(-6, 6),
            0: rng.randint(1, 10),
        }
        if not cubic and coefs[2] == 0:
            continue
        left = rng.randint(0, 3)
        n = rng.randint(2, 4)
        width = n * rng.randint(1, 2) if rng.random() < 0.5 else rng.randint(2, 6)
        right = left + width
        if width == n:
            continue
        # depth stays positive across the channel (checked on a fine grid)
        if any(_poly_value(coefs, Fraction(left) + Fraction(i, 4)) <= 0 for i in range(4 * width + 1)):
            continue
        step = Fraction(width, n)
        xs = [Fraction(left) + i * step for i in range(n + 1)]
        ys = [_poly_value(coefs, x) for x in xs]
        estimate = step * (ys[0] / 2 + sum(ys[1:-1], Fraction(0)) + ys[-1] / 2)
        exact = _antiderivative(coefs, Fraction(right)) - _antiderivative(coefs, Fraction(left))
        error = estimate - exact
        if error == 0 or {left, right, n} & {estimate, exact, error}:
            continue
        if 1 in (estimate.denominator, exact.denominator) and error.denominator == 1:
            continue
        if not survives_nsimplify(estimate, exact, error):
            continue
        break
    else:
        check(False, "no usable depth model in 1000 tries")

    # Independent routes: pairwise trapezoids, and Simpson's rule, which is exact for polynomials up to degree 3.
    pairwise = sum((step * (ys[i] + ys[i + 1]) / 2 for i in range(n)), Fraction(0))
    check(pairwise == estimate, "pairwise trapezoids disagree with the weighted sum")
    mid = Fraction(left + right, 2)
    simpson = Fraction(width, 6) * (
        _poly_value(coefs, Fraction(left)) + 4 * _poly_value(coefs, mid) + _poly_value(coefs, Fraction(right))
    )
    check(simpson == exact, "Simpson's rule disagrees with the integral")

    terms = [(coefs[p], _POWERS[p]) for p in (3, 2, 1, 0) if coefs[p] != 0]
    depth = join_terms(terms)
    stem = (
        f"The depth of a drainage channel, in metres, at a point x metres across it is modelled by d(x) = {depth} "
        f"for {left} ≤ x ≤ {right}. The cross-sectional area of the channel is the area under this curve, in m². "
        "Give exact values as fractions."
    )
    a_text, b_text, c_text = (rational_text(v) for v in (estimate, exact, error))
    full_weights = step * sum(ys, Fraction(0))
    left_sum = step * sum(ys[:-1], Fraction(0))
    width_step = Fraction(width) * (ys[0] / 2 + sum(ys[1:-1], Fraction(0)) + ys[-1] / 2)
    a_candidates = [
        (full_weights, "trapezoid-full-endpoints"),
        (width_step, "trapezoid-wrong-step"),
        (left_sum, "trapezoid-left-sum"),
        (exact, None),
    ]
    no_division = sum(
        (Fraction(c) * Fraction(right) ** (p + 1) - c * Fraction(left) ** (p + 1) for p, c in coefs.items()),
        Fraction(0),
    )
    upper_only = _antiderivative(coefs, Fraction(right))
    b_candidates = [
        (no_division, "antiderivative-no-division"),
        (upper_only, "lower-limit-ignored"),
        (_antiderivative(coefs, Fraction(right)) + _antiderivative(coefs, Fraction(left)), "limits-added"),
        (estimate, None),
    ]
    c_candidates = [
        (exact - estimate, "gap-sign-reversed"),
        (estimate, "residual-of-estimate-not-value"),
        (exact, None),
    ]
    steps_text = f"{n} equal strips of width {rational_text(step)}"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Use the trapezoidal rule with {n} equal strips to estimate the cross-sectional area.",
                answer_kind="fraction",
                answer=fraction_payload(estimate)[0],
                answer_text=a_text,
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"{steps_text}; the depths at the strip edges are {', '.join(rational_text(y) for y in ys)}.",
                    f"Area ≈ h[½(first) + (middle) + ½(last)] = {a_text} m².",
                ],
                distractors=fraction_distractors(estimate, a_candidates),
            ),
            dict(
                prompt="Find the exact cross-sectional area by integration.",
                answer_kind="fraction",
                answer=fraction_payload(exact)[0],
                answer_text=b_text,
                marks=2,
                sensible_range=(0, None),
                working=[f"The exact area is the integral of d(x) from {left} to {right}, which is {b_text} m²."],
                distractors=fraction_distractors(exact, b_candidates),
            ),
            dict(
                prompt="Using your answers to (a) and (b), find the estimate minus the exact area. "
                "A positive result means the estimate is too large.",
                answer_kind="fraction",
                answer=fraction_payload(error)[0],
                answer_text=c_text,
                marks=1,
                follow_through="part_a - part_b",
                working=[f"{a_text} − {b_text} = {c_text}."],
                distractors=fraction_distractors(error, c_candidates),
            ),
        ],
    )
