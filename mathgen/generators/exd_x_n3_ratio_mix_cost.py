"""Exam-tier source for `ratio-rate-proportion`: a three-part mix, then its cost, then its cost per kilogram.

The practice tiers simplify a ratio or convert units inside one. A national-assessment problem sets a ratio and a
rate in the same context: a batch of mixed ingredients shared in a stated ratio (so the learner must find the
mass of one part), priced by the kilogram (a rate applied to each share), and then summarised as a single rate
for the whole batch. Parts (b) and (c) follow through, so a wrong share still earns the cost and the unit rate
that follow from it.

Built backwards: the mass of one ratio part is drawn first (a multiple of 250 g), so the shares are exact; the
prices are whole multiples of 5 cents. The total cost is an exact number of cents and the rate per kilogram is
kept clear of a rounding boundary, so the exact route and a learner's route (using their own rounded cost)
agree.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import (
    GRADER_TOLERANCE,
    check,
    draw_until,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_MIXES = [
    ("A batch of trail mix", ("roasted nuts", "dried fruit", "chocolate chips")),
    ("A batch of muesli", ("rolled oats", "almonds", "sultanas")),
    ("A bag of birdseed", ("sunflower seeds", "millet", "cracked corn")),
    ("A batch of garden fertiliser", ("blood and bone", "seaweed meal", "potash")),
]
_RATIOS = [(a, b, c) for a in range(1, 8) for b in range(1, 8) for c in range(1, 8) if len({a, b, c}) >= 2]
_UNIT_KG = [Fraction(n, 4) for n in (1, 2, 3, 4, 5, 6, 8, 10)]  # mass of one ratio part, kg


def _dollars(cents: Fraction) -> str:
    return stated_precision_string(cents / 100, 2)


def _dd(cents: Fraction) -> str:
    """A distractor amount in dollars, rounded to the nearest cent."""
    return _dollars(round_half_up(cents, 0))


def _kg(x: Fraction) -> str:
    return exact_decimal_string(x, 4)


def _kd(x: Fraction) -> str:
    """A distractor mass at 2 decimal places, like the answer."""
    return stated_precision_string(round_half_up(x, 2), 2)


@generator("exd_x_n3_ratio_mix_cost", level=7, presets={"exam": dict(tier="exam")})
def exd_x_n3_ratio_mix_cost(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: share by a three-part ratio, price each share, then find the mix's rate per kilogram."""
    check(tier == "exam", f"unknown tier {tier!r}")

    def draw(r):
        ratio = r.choice(_RATIOS)
        k = r.choice(_UNIT_KG)
        prices = [5 * r.randint(60, 560) for _ in range(3)]  # cents per kg
        pick = r.randrange(3)
        return ratio, k, prices, pick

    def accept(t):
        ratio, k, prices, pick = t
        s = sum(ratio)
        if s > 16 or len(set(prices)) < 3:
            return False
        total_c = k * sum(rr * p for rr, p in zip(ratio, prices))
        if total_c.denominator != 1:
            return False
        rate = total_c / (k * s)  # cents per kg
        frac = (rate % 1) - Fraction(1, 2)
        return abs(frac) > Fraction(1, 50) and total_c / 100 < 2000

    ratio, k, prices, j = draw_until(rng, draw, accept, what="a ratio, a unit mass and prices")
    s = sum(ratio)
    mass_total = k * s
    share = k * ratio[j]
    total_c = k * sum(rr * p for rr, p in zip(ratio, prices))
    rate_c = round_half_up(total_c / mass_total, 0)  # cents per kg, to the nearest cent
    check(Fraction(sum(rr * p for rr, p in zip(ratio, prices)), s) == total_c / mass_total, "rate disagrees")

    lead, names = rng.choice(_MIXES)
    ratio_text = " : ".join(map(str, ratio))
    price_text = ", ".join(f"{n} ${_dollars(Fraction(p))} per kg" for n, p in zip(names, prices))
    stem = (
        f"{lead} is made from {names[0]}, {names[1]} and {names[2]} in the ratio {ratio_text} by mass. "
        f"The batch has a total mass of {_kg(mass_total)} kg. The ingredients cost {price_text}."
    )

    a_text = stated_precision_string(share, 2)
    b_text = _dollars(total_c)
    c_text = _dollars(rate_c)
    simple_avg = Fraction(sum(prices), 3)
    a_cands = [
        _kd(k),  # the mass of one part, not of the part asked for
        _kd(mass_total * ratio[j] / (s - ratio[j])) if s != ratio[j] else None,  # parts of the other ingredients
        _kd(mass_total / 3),  # three equal shares
        _kd(mass_total - share),  # the rest of the batch
    ]
    b_cands = [
        _dd(mass_total * prices[j]),  # the whole batch at one ingredient's price
        _dd(mass_total * simple_avg),  # equal shares of each ingredient
        _dd(share * prices[j]),  # only the one share costed
        _dd(Fraction(sum(rr * p for rr, p in zip(ratio, prices)))),  # one part of each, not the batch
    ]
    c_cands = [
        _dd(simple_avg),  # unweighted average of the three prices
        _dd(total_c / s),  # cost per ratio part
        _dd(total_c / 3),  # cost divided by the number of ingredients
        _dd(total_c / mass_total / 2),
    ]
    a_dist = pick_distractors(a_text, [c for c in a_cands if c], 3, tolerance=GRADER_TOLERANCE)
    b_dist = pick_distractors(b_text, b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_dist = pick_distractors(c_text, c_cands, 3, tolerance=GRADER_TOLERANCE)
    check(min(len(a_dist), len(b_dist), len(c_dist)) >= 2, "too few distractors")

    weights = sum(rr * p for rr, p in zip(ratio, prices))
    mass_expr = f"({mass_total.numerator}/{mass_total.denominator})"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the mass of {names[j]} in the batch, in kilograms, correct to 2 decimal places.",
                answer_kind="number",
                answer=numeric_answer(share),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"The ratio has {ratio_text.replace(' : ', ' + ')} = {s} parts, so one part is {_kg(mass_total)} ÷ {s} = {_kg(k)} kg.",
                    f"{names[j]} is {ratio[j]} part{'' if ratio[j] == 1 else 's'}: {ratio[j]} × {_kg(k)} = {a_text} kg.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt="Find the total cost of the ingredients in the batch, in dollars.",
                answer_kind="number",
                answer=numeric_answer(total_c / 100),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*{weights}/({100 * ratio[j]})",
                working=[
                    f"Cost = ({' + '.join(f'{rr} × {_dollars(Fraction(p))}' for rr, p in zip(ratio, prices))}) × {_kg(k)} = ${b_text}."
                ],
                distractors=b_dist,
            ),
            dict(
                prompt="Find the average cost of the mixture per kilogram, in dollars, correct to 2 decimal places.",
                answer_kind="number",
                answer=numeric_answer(rate_c / 100),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_b/{mass_expr}",
                working=[f"${b_text} ÷ {_kg(mass_total)} kg = ${c_text} per kg (2 d.p.)."],
                distractors=c_dist,
            ),
        ],
    )
