"""Exam source: a fortnightly allowance with an income test, followed through to the cut-out income.

One hypothetical scheme: a base rate plus a supplement for each dependent child, reduced by a fixed amount for
every complete block of income above a free threshold, and never below zero.

(a) the full fortnightly rate before any income test;
(b) the reduction for the stated income (a count of COMPLETE blocks, then times the reduction);
(c) the fortnightly payment, from (a) and (b);
(d) the lowest fortnightly income at which the payment falls to nothing, from (a): the learner must see that
    the number of complete blocks needed is a ceiling, not a rounded quotient.

All amounts are whole dollars, so every key is exact. A draw is kept only when the payment in (c) is positive and
(a) divided by the reduction is not a whole number (so the ceiling matters).
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase
from mathgen.generators.exd_x_n1_support import number_part

_STEM = (
    "A hypothetical government support payment pays a base rate of ${base} per fortnight, plus ${kid} for each "
    "dependent child. The payment is reduced by ${cut} for every complete ${block} of fortnightly income above "
    "an income-free threshold of ${free}. The payment cannot fall below $0. "
    "A person with {kids} dependent {child_word} earns ${income} per fortnight.",
    "A hypothetical allowance scheme pays ${base} per fortnight, with an extra ${kid} per fortnight for each "
    "dependent child. For every complete ${block} that a person's fortnightly income is above ${free}, the "
    "payment is reduced by ${cut}. The payment cannot go below $0. "
    "A person with {kids} dependent {child_word} has a fortnightly income of ${income}.",
)


@generator("exd_x_n1_allowance_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_n1_allowance_chain(rng, tier="exam"):
    for _ in attempts(1000, "an allowance draw with a positive payment and a non-integer cut-out ratio"):
        base = rng.randint(300, 520)
        kid = rng.randint(28, 64)
        kids = rng.randint(1, 3)
        cut = rng.randint(12, 34)
        block = rng.randint(40, 120)
        free = rng.randint(250, 520)
        full = base + kid * kids
        steps_needed = -(-full // cut)  # ceiling
        if full % cut == 0 or steps_needed < 4:
            continue
        steps = rng.randint(1, steps_needed - 2)
        income = free + steps * block + rng.randint(0, block - 1)
        reduction = steps * cut
        payment = full - reduction
        if payment <= 0:
            continue
        cutout = free + steps_needed * block
        break

    # Independent routes
    check_steps = (income - free) // block
    check(check_steps == steps and full - check_steps * cut == payment, "payment routes disagree")
    cut_blocks = (cutout - free) // block
    check(cut_blocks * cut >= full > (cut_blocks - 1) * cut, "cut-out income is not the lowest")

    stem = phrase(
        rng,
        _STEM,
        base=base,
        kid=kid,
        cut=cut,
        block=block,
        free=free,
        kids=kids,
        child_word="child" if kids == 1 else "children",
        income=income,
    )
    f = Fraction
    part_a = number_part(
        "Find the full fortnightly rate for this person before the income test is applied, in dollars.",
        f(full),
        0,
        1,
        [f"Full rate = {base} + {kids} × {kid} = {full}."],
        [(f(base), None), (f(base + kid), None), (f(base + 2 * kid * kids), None)],
        sensible_range=(0, None),
    )
    part_b = number_part(
        "Find the reduction to the payment because of the person's income, in dollars.",
        f(reduction),
        0,
        2,
        [
            f"Income above the threshold = {income} - {free} = {income - free}.",
            f"Complete blocks of {block} in {income - free}: {steps}.",
            f"Reduction = {steps} × {cut} = {reduction}.",
        ],
        [
            (f(income - free) / block * cut, None),
            ((steps + 1) * f(cut), None),
            (f(income - free) / block, None),
            (f(cut), None),
        ],
        sensible_range=(0, None),
    )
    part_c = number_part(
        "Using your answers to (a) and (b), find the person's fortnightly payment, in dollars.",
        f(payment),
        0,
        1,
        [f"Payment = {full} - {reduction} = {payment}."],
        [(f(full + reduction), None), (f(reduction), None), (f(full), None)],
        follow_through="part_a - part_b",
        sensible_range=(0, None),
    )
    part_d = number_part(
        "Find the lowest fortnightly income at which this person would receive no payment, in dollars.",
        f(cutout),
        0,
        3,
        [
            f"The reduction must reach {full}: {full} ÷ {cut} = {full / cut:.2f}, so {steps_needed} complete blocks "
            "are needed (not a part block).",
            f"Income = {free} + {steps_needed} × {block} = {cutout}.",
        ],
        [
            (free + f(full, cut) * block, None),
            (free + (steps_needed - 1) * f(block), None),
            (f(full) * block / cut, None),
            (free + f(steps_needed) * cut, None),
        ],
        follow_through=f"{free} + ceiling(part_a/{cut})*{block}",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
