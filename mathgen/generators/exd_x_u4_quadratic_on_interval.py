"""Exam-tier source for `convex-quadratic-minimum`: the least value of a quadratic over a restricted range.

The practice tiers minimise a convex quadratic over all real inputs. The exam question models the fuel use of a ferry
as a quadratic in its speed and allows only speeds in a stated range. The vertex of the parabola may lie outside that
range, in which case the minimum is at the end of the range nearest the vertex, so the learner must

1. find the vertex of the parabola,
2. decide which allowed speed minimises the fuel use (the vertex if it is allowed, otherwise the nearer end), and
3. evaluate the fuel use at that speed.

Parts (b) and (c) carry follow-through. Built backwards from the vertex, so every coefficient is a whole number;
about two draws in three place the vertex outside the range. The independent route searches a grid of quarter steps across
the range and checks that no grid point beats the claimed minimum.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import check, generator, join_terms, rational_text

from ._u4_kit import fraction_distractors, fraction_payload, survives_nsimplify

_SETTINGS = [
    (
        "The fuel use F litres per hour of a ferry at a speed of x knots is modelled by F(x) = {poly}. "
        "The ferry's speed must stay between {lo} and {hi} knots, inclusive.",
        "knots",
    ),
    (
        "The running cost C dollars per hour of a pump delivering x kilolitres per hour is modelled by C(x) = {poly}. "
        "The pump can only be set between {lo} and {hi} kilolitres per hour, inclusive.",
        "kilolitres per hour",
    ),
    (
        "The energy use E kilojoules per minute of a treadmill at a speed of x kilometres per hour is modelled by "
        "E(x) = {poly}. The treadmill can only be set between {lo} and {hi} kilometres per hour, inclusive.",
        "kilometres per hour",
    ),
]


@generator("exd_x_u4_quadratic_on_interval", level=16, presets={"exam": dict(tier="exam")})
def exd_x_u4_quadratic_on_interval(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: vertex, constrained minimiser and constrained minimum of a convex quadratic."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(1000):
        a = rng.randint(1, 4)
        lo = rng.randint(2, 9)
        hi = lo + rng.randint(3, 8)
        where = rng.choice(["inside", "inside", "below", "above"])
        if where == "inside":
            vertex2 = rng.randint(2 * lo + 1, 2 * hi - 1)
        elif where == "below":
            vertex2 = rng.randint(2, 2 * lo - 1)
        else:
            vertex2 = rng.randint(2 * hi + 1, 2 * hi + 10)
        vertex = Fraction(vertex2, 2)  # twice the vertex is a whole number, so the vertex is whole or a half
        b = -a * vertex2
        c = -(-a * vertex2 * vertex2 // 4) + rng.randint(2, 30)  # the unrestricted minimum stays positive
        inside = lo < vertex < hi
        outside = not inside

        def value(x: Fraction) -> Fraction:
            return a * x * x + b * x + c

        best_x = min(max(vertex, Fraction(lo)), Fraction(hi))
        least = value(best_x)
        # the least value must stay positive over the range so the model is sensible
        if min(value(Fraction(lo)), value(Fraction(hi)), value(vertex) if inside else value(best_x)) <= 0:
            continue
        keys = (vertex, best_x, least)
        if {a, abs(b), c, lo, hi} & {vertex, least}:
            continue  # no answer may repeat a number the question shows
        if outside and best_x == vertex:
            continue
        if not survives_nsimplify(*keys):
            continue
        break
    else:
        check(False, "no usable quadratic and range in 1000 tries")

    # Independent route: the least value on a grid of quarter steps (which includes both ends) is never below the key.
    grid = [Fraction(lo) + Fraction(i, 4) for i in range(4 * (hi - lo) + 1)]
    check(least == min(value(x) for x in grid + [best_x]), "least value disagrees with a grid search")
    check(all(value(x) >= least for x in grid), "a grid point beats the claimed minimum")

    poly = join_terms([(a, "x²"), (b, "x"), (c, "")])
    template, _unit = rng.choice(_SETTINGS)
    stem = f"{template.format(poly=poly, lo=lo, hi=hi)} Give exact values as fractions or integers."
    other_end = Fraction(hi) if best_x == lo else Fraction(lo)
    a_candidates = [
        (Fraction(-b, a), "quadratic-factor-two-missing"),
        (-vertex, "vertex-sign-error"),
        (Fraction(b, 2 * a) * 2, None),
    ]
    if inside:
        b_candidates = [
            (Fraction(lo), "endpoint-taken-though-vertex-inside"),
            (Fraction(hi), "endpoint-taken-though-vertex-inside"),
            (Fraction(-b, a), "quadratic-factor-two-missing"),
        ]
        c_candidates = [
            (value(Fraction(lo)), "endpoint-taken-though-vertex-inside"),
            (value(Fraction(hi)), "endpoint-taken-though-vertex-inside"),
            (Fraction(c), "constant-term-as-minimum"),
        ]
    else:
        b_candidates = [
            (vertex, "vertex-outside-interval-ignored"),
            (other_end, "wrong-end-of-interval"),
            (Fraction(-b, a), "vertex-outside-interval-ignored"),
        ]
        c_candidates = [
            (value(vertex), "evaluates-at-vertex-not-minimiser"),
            (value(other_end), "wrong-end-of-interval"),
            (Fraction(c), "constant-term-as-minimum"),
        ]
    vt, bt, lt = (rational_text(k) for k in (vertex, best_x, least))
    reasoning = (
        f"{vt} is inside the range, so it is the minimiser."
        if inside
        else f"{vt} is outside {lo} to {hi}; the parabola opens upwards, so the nearest allowed value {bt} is best."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the value of x at which the model would be least if x were not restricted.",
                answer_kind="fraction",
                answer=fraction_payload(vertex)[0],
                answer_text=vt,
                marks=2,
                sensible_range=(0, None),
                working=[f"x = −b/(2a) = {-b}/{2 * a}" + (f" = {vt}." if f"{-b}/{2 * a}" != vt else ".")],
                distractors=fraction_distractors(vertex, a_candidates),
            ),
            dict(
                prompt="Using your answer to (a), find the allowed value of x at which the model is least.",
                answer_kind="fraction",
                answer=fraction_payload(best_x)[0],
                answer_text=bt,
                marks=1,
                sensible_range=(0, None),
                follow_through=f"Min(Max(part_a, {lo}), {hi})",
                working=[reasoning],
                distractors=fraction_distractors(best_x, b_candidates),
            ),
            dict(
                prompt="Using your answer to (b), find the least value of the model over the allowed range.",
                answer_kind="fraction",
                answer=fraction_payload(least)[0],
                answer_text=lt,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{a}*part_b**2 + ({b})*part_b + {c}",
                working=[f"At x = {bt} the model equals {lt}."],
                distractors=fraction_distractors(least, c_candidates),
            ),
        ],
    )
