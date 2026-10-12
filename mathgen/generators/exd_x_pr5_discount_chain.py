"""Exam tier for `f10-fraction-percent-quantity-strategy`: two successive discounts and the single percentage.

A shop takes a familiar percentage off a price, then a second percentage off the REDUCED price. The learner finds
the first sale price, applies the second discount to the reduced price, totals the saving, and then finds the one
percentage that matches: the "add the two percentages" shortcut is exactly what the last part exposes.

* (a) the price after the first discount (2 marks);
* (b) the price after the second discount, taken off the answer to (a) (2 marks, follow-through from (a));
* (c) the total saving in dollars (1 mark, follow-through from (b));
* (d) the single percentage discount that gives the same final price, to 1 decimal place (2 marks,
  follow-through from (c)).

Built backwards from whole-dollar prices: the original price, the first sale price and the final price are all
whole dollars, and the percentage in (d) is exact to one decimal place.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, numeric_answer, pick_distractors, round_half_up
from mathgen.core import stated_precision_string

_FIRST = [Fraction(v) for v in (10, 15, 20, 25, 30, 40, 50, 60)] + [Fraction(25, 2), Fraction(75, 2), Fraction(125, 2)]
_SECOND = [Fraction(v) for v in (5, 10, 15, 20, 25, 30, 40, 50)]
# (item, shop and place, lowest and highest plausible marked price in dollars)
_ITEMS = [
    ("a pair of running shoes", "at a sports store in Parramatta", 60, 240),
    ("a school backpack", "at a department store in Hobart", 40, 160),
    ("a surfboard", "at a surf shop on the Gold Coast", 300, 900),
    ("a cricket bat", "at a sports shop in Adelaide", 80, 400),
    ("a bicycle helmet", "at a bike shop in Canberra", 40, 200),
    ("a tent", "at an outdoors store in Darwin", 120, 640),
    ("a guitar", "at a music shop in Fremantle", 200, 900),
]


def _pct_text(p: Fraction) -> str:
    return f"{float(p):g}%"


def _sensible(candidates):
    """Only candidates a learner could actually reach: a positive value."""
    return [c for c in candidates if Fraction(c.value) > 0]


def _money(value: Fraction) -> str:
    return str(int(value))


class _Retry(Exception):
    pass


def _attempt(rng: random.Random) -> dict:
    for _ in attempts(1000, "whole-dollar prices for two successive discounts"):
        item, place, low, high = rng.choice(_ITEMS)
        p1, p2 = rng.choice(_FIRST), rng.choice(_SECOND)
        if p1 + p2 >= 100:
            continue
        # The smallest price at which both the sale price and the final price are whole dollars.
        step = math.lcm(((100 - p1) / 100).denominator, (((100 - p1) * (100 - p2)) / 10000).denominator)
        price = Fraction(step * rng.randint(math.ceil(low / step), max(high // step, math.ceil(low / step))))
        sale = price * (1 - p1 / 100)
        final = sale * (1 - p2 / 100)
        share = final / price * 100
        if sale.denominator != 1 or final.denominator != 1:
            continue
        if (share * 10).denominator != 1 or share in (p1, p2):
            continue
        shown = {int(price), int(p1) if p1.denominator == 1 else -1, int(p2)}
        if (
            {int(sale), int(final), int(price - final)} & shown
            or len({sale, final, price - final}) < 3
            or share.denominator != 1
            and share * 10 % 1
        ):
            continue
        break

    stem = (
        f"{item[0].upper()}{item[1:]} {place} is marked at ${int(price)}. This week the shop takes {_pct_text(p1)} "
        f"off the marked price. A member then takes a further {_pct_text(p2)} off the reduced price."
    )

    off1 = price * p1 / 100
    off2 = sale * p2 / 100
    d_a = pick_distractors(
        _money(sale),
        _sensible(
            [
                Distractor(_money(off1), "reports-discount-not-sale-price", off1),
                Distractor(f"{float(price - p1):g}", "percent-as-absolute", price - p1),
                Distractor(
                    _money(price * (1 - p1 / 100 - p2 / 100)),
                    "adds-successive-percentages",
                    price * (1 - p1 / 100 - p2 / 100),
                ),
            ]
        ),
        3,
    )
    off_both = price * (1 - (p1 + p2) / 100)
    d_b = pick_distractors(
        _money(final),
        _sensible(
            [
                Distractor(_money(off2), "reports-discount-not-sale-price", off2),
                Distractor(f"{float(off_both):g}", "adds-successive-percentages", off_both),
                Distractor(
                    _money(price * (1 - p2 / 100)), "applies-fraction-to-original-whole", price * (1 - p2 / 100)
                ),
                Distractor(_money(sale - p2), "percent-as-absolute", sale - p2),
            ]
        ),
        3,
    )
    saving = price - final
    d_c = pick_distractors(
        _money(saving),
        _sensible(
            [
                Distractor(_money(price * (p1 + p2) / 100), "adds-successive-percentages", price * (p1 + p2) / 100),
                Distractor(_money(off2), "omitted-a-term", off2),
                Distractor(_money(off1), "omitted-a-term", off1),
                Distractor(_money(final), "answers-amount-used-not-remaining", final),
            ]
        ),
        3,
    )
    one_off = 100 - share
    one_off_text = stated_precision_string(round_half_up(one_off, 1), 1)
    d_d = pick_distractors(
        one_off_text,
        _sensible(
            [
                Distractor(stated_precision_string(p1 + p2, 1), "adds-successive-percentages", p1 + p2),
                Distractor(
                    stated_precision_string(round_half_up(share, 1), 1), "gave-discount-percent-not-remaining", share
                ),
                Distractor(
                    stated_precision_string(round_half_up(sale / price * 100, 1), 1),
                    "omitted-a-term",
                    sale / price * 100,
                ),
                Distractor(stated_precision_string(p2, 1), "omitted-a-term", p2),
            ]
        ),
        3,
        tolerance=Fraction(1, 20),
    )
    if min(len(d_a), len(d_b), len(d_c), len(d_d)) < 3:
        raise _Retry
    check(Fraction(final) == price * (1 - p1 / 100) * (1 - p2 / 100), "final price disagrees")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="What is the price after the first discount? Give your answer in dollars.",
                answer_kind="number",
                answer=numeric_answer(sale),
                answer_text=_money(sale),
                answer_latex=latex_answer(_money(sale)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"{_pct_text(p1)} of ${int(price)} = ${float(off1):g}.",
                    f"${int(price)} - ${float(off1):g} = ${int(sale)}.",
                ],
                distractors=d_a,
            ),
            dict(
                prompt="What is the price after the member's further discount? Give your answer in dollars.",
                answer_kind="number",
                answer=numeric_answer(final),
                answer_text=_money(final),
                answer_latex=latex_answer(_money(final)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*(100 - {p2.numerator}/{p2.denominator})/100",
                working=[
                    f"The second discount is taken off ${int(sale)}: {_pct_text(p2)} of ${int(sale)} = ${float(off2):g}.",
                    f"${int(sale)} - ${float(off2):g} = ${int(final)}.",
                ],
                distractors=d_b,
            ),
            dict(
                prompt="How much money does the member save altogether, compared with the marked price? "
                "Give your answer in dollars.",
                answer_kind="number",
                answer=numeric_answer(saving),
                answer_text=_money(saving),
                answer_latex=latex_answer(_money(saving)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{int(price)} - part_b",
                working=[f"${int(price)} - ${int(final)} = ${int(saving)}."],
                distractors=d_c,
            ),
            dict(
                prompt="What single percentage discount on the marked price would give the same final price? "
                "Give your answer correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(round_half_up(one_off, 1)),
                answer_text=one_off_text,
                answer_latex=latex_answer(one_off_text),
                marks=2,
                sensible_range=(0, 100),
                follow_through=f"part_c*100/{int(price)}",
                working=[f"${int(saving)} ÷ ${int(price)} × 100 = {one_off_text}%."],
                distractors=d_d,
            ),
        ],
    )


def _build(rng: random.Random) -> dict:
    for _ in attempts(200, "a draw with three distinct distractors per part"):
        try:
            return _attempt(rng)
        except _Retry:
            continue


@generator("exd_x_pr5_discount_chain", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr5_discount_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Two successive percentage discounts, then the final price as a percentage of the marked price."""
    check(tier == "exam", "exd_x_pr5_discount_chain: only the exam tier exists")
    return _build(rng)
