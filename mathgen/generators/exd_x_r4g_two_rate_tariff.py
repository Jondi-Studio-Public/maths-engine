"""Exam chain for a two-rate tariff compared with a flat plan.

(a) the cost of a stated usage on the two-rate plan (Plan A)                   number, 2 marks
(b) the cost of the same usage on the flat plan (Plan B)                       number, 1 mark
(c) how much cheaper Plan B is at that usage (follow-through (a), (b))         number, 1 mark
(d) the usage at which the two plans cost the same                             number, 3 marks

The usage in (a) to (c) lies beyond the threshold and below the break-even, so Plan B is the cheaper plan there.
Every figure is checked by evaluating both plans; the break-even is found by searching whole units.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._r4g_kit import number_part, retry
from .r4g_two_rate_tariff_build import _break_even_draw, _plan_b_text, _usage_text, cents, money


@distinct_part_answers
def _build(rng):
    ctx, a, f_b, r_b, u_star = _break_even_draw(rng)
    u1 = rng.randint(a.t + 1, u_star - 1)
    ca, cb = a.cost(u1), f_b + r_b * u1
    saving = ca - cb
    check(saving > 0, "Plan B is not cheaper at the chosen usage")
    check([k for k in range(0, 4 * u_star) if a.cost(k) == f_b + r_b * k] == [u_star], "break-even not unique")
    text = f"Plan A: {a.text(ctx)} {_plan_b_text(ctx, f_b, r_b)}"
    parts = [
        number_part(
            f"Find the cost of {_usage_text(ctx, u1)} on Plan A, in dollars.",
            Fraction(ca, 100),
            2,
            2,
            [f"{cents(a.f)} + {a.t} x {cents(a.r1)} + {u1 - a.t} x {cents(a.r2)} = {money(ca)}."],
            [
                (Fraction(a.f + a.r1 * u1, 100), "r4g-first-rate-used-for-all-units"),
                (Fraction(a.f + a.r2 * u1, 100), "r4g-second-rate-used-for-all-units"),
                (Fraction(a.f + a.r1 * a.t + a.r2 * u1, 100), "r4g-threshold-not-subtracted"),
                (Fraction(a.r1 * a.t + a.r2 * (u1 - a.t), 100), "r4g-fixed-charge-left-out"),
            ],
        ),
        number_part(
            f"Find the cost of the same use on Plan B, in dollars.",
            Fraction(cb, 100),
            2,
            1,
            [f"{cents(f_b)} + {u1} x {cents(r_b)} = {money(cb)}."],
            [
                (Fraction(r_b * u1, 100), "r4g-fixed-charge-left-out"),
                (Fraction(f_b + r_b * a.t, 100), "r4g-flat-rate-applied-to-threshold-only"),
                (Fraction(f_b + r_b * (u1 - a.t), 100), "r4g-threshold-subtracted-from-flat-plan"),
                (Fraction(f_b + a.r2 * u1, 100), "r4g-wrong-plans-rate-used"),
            ],
        ),
        number_part(
            "How much cheaper is Plan B than Plan A for this use? Use your costs from (a) and (b).",
            Fraction(saving, 100),
            2,
            1,
            [f"{money(ca)} - {money(cb)} = {money(saving)}."],
            [
                (Fraction(ca + cb, 100), "r4g-costs-added-not-subtracted"),
                (Fraction(ca, 100), "r4g-one-cost-not-the-difference"),
                (Fraction(abs(a.f - f_b), 100), "r4g-fixed-charges-compared-only"),
                (Fraction(abs(a.r2 - r_b) * u1, 100), "r4g-rates-compared-only"),
            ],
            follow_through="part_a - part_b",
        ),
        number_part(
            f"For how many {ctx['units']} of use do the two plans cost exactly the same?",
            u_star,
            0,
            3,
            [
                f"Beyond {a.t} {ctx['units']}: {cents(a.f)} + {a.t} x {cents(a.r1)} + (u - {a.t}) x {cents(a.r2)} = {cents(f_b)} + u x {cents(r_b)}.",
                f"Solving gives u = {u_star}; both plans then cost {money(a.cost(u_star))}.",
            ],
            [
                (Fraction(a.f - f_b, r_b - a.r1) if r_b != a.r1 else u_star + 3, "r4g-break-even-in-first-section"),
                (Fraction(a.f - f_b, r_b - a.r2), "r4g-threshold-not-subtracted"),
                (Fraction(a.cost(u_star), r_b), "r4g-fixed-charges-left-out"),
                (u_star + a.t, "r4g-threshold-added-to-break-even"),
            ],
        ),
    ]
    return dict(text=text, parts=parts)


@generator("exd_x_r4g_two_rate_tariff", level=11, presets={"exam": {}})
def exd_x_r4g_two_rate_tariff(rng):
    return retry(_build, rng)
