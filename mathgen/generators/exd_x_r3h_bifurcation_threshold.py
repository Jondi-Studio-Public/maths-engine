"""Exam tier for detecting a one-parameter equilibrium bifurcation: a three-part chain on dx/dt = a x^2 + b x + p r + q.

The parameter r enters through the constant term only as p r + q, so the threshold is not simply b^2/(4a).

(a) the threshold r*, where the discriminant b^2 - 4a(p r* + q) is zero (2 decimal places);
(b) the single equilibrium x = -b/(2a) at the threshold;
(c) for a stated r1 on the two-equilibrium side, the distance between the equilibria, sqrt(-4 a p (r1 - r*)) / |a|,
    which follows through from the learner's r*.

Built backwards from integer equilibria e1 < e2 at r = r1 (an even gap, so the repeated root is a whole number).
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, join_terms, paren_negative
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3h_kit import distinct_slips, number_part


@generator("exd_x_r3h_bifurcation_threshold", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3h_bifurcation_threshold(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a family with integer equilibria at r1"):
        a = rng.choice((-3, -2, -1, 1, 2, 3))
        p = rng.choice((1, 2, 4, -1, -2, -4))
        r1 = rng.randint(-4, 6)
        e1 = rng.randint(-6, 4)
        e2 = e1 + 2 * rng.randint(1, 4)
        if e1 + e2 == 0 or abs(e2) > 7:
            continue
        b = -a * (e1 + e2)
        q = a * e1 * e2 - p * r1
        threshold = (Fraction(b * b, 4 * a) - q) / p
        bf, af = Fraction(b), Fraction(a)
        slips_a = [
            ((bf * bf / af - q) / p, "dropped-factor-of-four"),
            (bf * bf / (4 * af) - q, "forgot-to-divide-by-coefficient"),
            ((bf * bf / (4 * af) + q) / p, "sign-error-moving-term"),
            (bf * bf / (4 * af * p), "ignored-constant-term"),
        ]
        if distinct_slips(threshold, 2, slips_a) >= 3:
            break
    gap = Fraction(e2 - e1)
    repeated = Fraction(-b, 2 * a)
    check(threshold == r1 + Fraction(a * (e2 - e1) ** 2, 4 * p), "threshold from the equilibria at r1")
    check(b * b - 4 * a * (p * threshold + q) == 0, "discriminant at the threshold")
    check(b * b - 4 * a * (p * r1 + q) == a * a * gap * gap, "gap at r1")

    rhs = join_terms([(a, "x^2"), (b, "x"), (p, "r"), (q, "")])
    stem = (
        f"A quantity x changes over time according to dx/dt = {rhs}, where r is a parameter. "
        "The equilibria are the real values of x at which dx/dt = 0."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                **number_part(
                    "Find the value r* of r at which the number of real equilibria changes, "
                    "correct to 2 decimal places.",
                    threshold,
                    2,
                    3,
                    [
                        "The number of equilibria changes where the discriminant b^2 - 4ac is zero.",
                        f"{paren_negative(b)}^2 - 4({a})({join_terms([(p, 'r'), (q, '')])}) = 0, so r* = {threshold}.",
                    ],
                    slips_a,
                )
            ),
            dict(
                **number_part(
                    "At r = r*, there is exactly one equilibrium. Find its value of x.",
                    repeated,
                    0,
                    2,
                    [f"A repeated root is x = -b/(2a) = -({b})/(2({a})) = {repeated}."],
                    [
                        (Fraction(b, 2 * a), "sign-slip-in-formula"),
                        (Fraction(-b, a), "dropped-factor-of-two"),
                        (e2, "gave-root-at-other-parameter-value"),
                        (e1, "gave-root-at-other-parameter-value"),
                    ],
                )
            ),
            dict(
                **number_part(
                    f"For r = {r1} there are two equilibria. Find the distance between them, "
                    "correct to 2 decimal places.",
                    gap,
                    2,
                    3,
                    [
                        f"The discriminant at r = {r1} is {b * b - 4 * a * (p * r1 + q)}.",
                        f"The roots differ by sqrt(discriminant)/|{a}| = {gap}.",
                    ],
                    [
                        (gap * gap, "forgot-square-root"),
                        (gap / 2, "gave-half-the-distance-between-equilibria"),
                        (gap * abs(af), "forgot-to-divide-roots-by-leading-coefficient"),
                        (abs(threshold - r1), "gave-parameter-gap-for-equilibrium-gap"),
                    ],
                    follow_through=f"sqrt(-4*({a})*({p})*({r1} - part_a))/{abs(a)}",
                    sensible_range=(0, None),
                )
            ),
        ],
    )
