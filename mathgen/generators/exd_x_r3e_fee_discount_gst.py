"""Exam-tier source for `f10-rational-percentage-financial-model`: a fee that is a fraction of the price, a percentage
discount, then GST.

A price B has a fee of q (a fraction such as 3/8) of B added, a discount of p% of B taken off, and GST of 10% added to
the result. The chain:

(a) the fee, B q                                   number, 1 mark
(b) the amount after the fee and discount          number, 2 marks    B + part_a - B p / 100
(c) the GST on that amount                         number, 1 mark     part_b / 10
(d) the total payable                              number, 1 mark     part_b + part_c

B is a multiple of 40 dollars and p a multiple of 5, so the fee and the discount are whole dollars and the GST is a
whole number of ten-cent steps: every answer is exact to the cent. Answers are written to 2 decimal places.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor
from ..core import (
    check,
    generator,
    numeric_answer,
    pick_distractors,
    rational_text,
    round_half_up,
    stated_precision_string,
)
from ._exd_distinct_parts import distinct_part_answers

_FRACTIONS = [
    Fraction(1, 8),
    Fraction(1, 5),
    Fraction(1, 4),
    Fraction(3, 10),
    Fraction(3, 8),
    Fraction(2, 5),
    Fraction(1, 10),
    Fraction(3, 20),
]
_DISCOUNTS = [5, 10, 15, 20, 25]
_SETTINGS = (
    dict(
        item="the hire cost",
        stem="A community hall costs ${B} to hire for a day. A cleaning fee equal to {q} of the hire cost is added. "
        "For a regular booking, a discount of {p}% of the hire cost is taken off. GST of 10% is then added to the "
        "amount left after the fee and the discount.",
        fee="cleaning fee",
    ),
    dict(
        item="the site cost",
        stem="A caravan park charges ${B} for a week's site hire. A linen fee equal to {q} of the site cost is added, "
        "and an early-booking discount of {p}% of the site cost is taken off. GST of 10% is then added to the amount "
        "left after the fee and the discount.",
        fee="linen fee",
    ),
    dict(
        item="the quote",
        stem="A caterer quotes ${B} for a school fete. A delivery fee equal to {q} of the quote is added and a school "
        "discount of {p}% of the quote is taken off. GST of 10% is then added to the amount left after the fee and "
        "the discount.",
        fee="delivery fee",
    ),
    dict(
        item="the labour charge",
        stem="An electrician charges ${B} for labour on a job. A call-out fee equal to {q} of the labour charge is "
        "added, and a pensioner discount of {p}% of the labour charge is taken off. GST of 10% is then added to the "
        "amount left after the fee and the discount.",
        fee="call-out fee",
    ),
)


def _money(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 2), 2)


def _options(key: Fraction, candidates, n: int = 3):
    key_text = _money(key)
    out, seen = [], [round_half_up(key, 2)]
    for tag, value in candidates:
        value = round_half_up(Fraction(value), 2)
        if value <= 0 or any(abs(value - s) < Fraction(1, 50) for s in seen):
            continue
        seen.append(value)
        out.append(Distractor(_money(value), tag, value=value))
    return pick_distractors(key_text, out, n, tolerance=Fraction(1, 100))


@generator("exd_x_r3e_fee_discount_gst", level=10, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_fee_discount_gst(rng, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    setting = rng.choice(_SETTINGS)
    for _ in range(2000):
        B = 40 * rng.randint(10, 50)
        q = rng.choice(_FRACTIONS)
        p = rng.choice(_DISCOUNTS)
        fee = B * q
        discount = Fraction(B * p, 100)
        subtotal = B + fee - discount
        gst = subtotal / 10
        total = subtotal + gst
        if fee.denominator != 1 or discount.denominator != 1 or (gst * 100).denominator != 1:
            continue
        rounded_q = round_half_up(q, 2)
        a_opts = _options(
            fee,
            [
                ("fraction-numerator-ignored", Fraction(B, q.denominator)),
                ("fraction-rounded-to-two-places", B * rounded_q if rounded_q != q else 0),
                ("percent-as-absolute", Fraction(q.numerator * 100, q.denominator)),
                ("fraction-inverted-in-part-of", B / q),
            ],
        )
        b_opts = _options(
            subtotal,
            [
                ("discount-left-out", B + fee),
                ("discount-added-not-subtracted", B + fee + discount),
                ("surcharge-left-out", B - discount),
                ("percentage-of-wrong-base", (B + fee) * (1 - Fraction(p, 100))),
            ],
        )
        c_opts = _options(
            gst,
            [
                ("percentage-of-wrong-base", Fraction(B, 10)),
                ("gst-treated-as-inclusive", subtotal / 11),
                ("percent-as-absolute", Fraction(10)),
                ("percentage-decimal-point-slip", subtotal / 100),
            ],
        )
        d_opts = _options(
            total,
            [
                ("gst-subtracted-not-added", subtotal - gst),
                ("gst-left-out-of-total", subtotal),
                ("percentage-of-wrong-base", subtotal + Fraction(B, 10)),
                ("percent-as-absolute", subtotal + 10),
            ],
        )
        if min(len(a_opts), len(b_opts), len(c_opts), len(d_opts)) >= 3:
            break
    else:
        check(False, "no price with enough distractors")

    # Independent check in whole cents.
    cents = lambda v: int(v * 100)  # noqa: E731
    check(cents(subtotal) == 100 * B + cents(fee) - cents(discount), "subtotal disagrees in cents")
    check(cents(total) == cents(subtotal) + cents(subtotal) // 10, "total disagrees in cents")

    stem = setting["stem"].format(B=B, q=rational_text(q), p=p)
    disc_text = int(discount)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the {setting['fee']}, in dollars.",
                answer_kind="number",
                answer=numeric_answer(fee),
                answer_text=_money(fee),
                marks=1,
                working=[f"{rational_text(q)} of ${B} is ${_money(fee)}."],
                distractors=a_opts,
            ),
            dict(
                prompt="Hence find the amount left after the fee is added and the discount is taken off, in dollars.",
                answer_kind="number",
                answer=numeric_answer(subtotal),
                answer_text=_money(subtotal),
                marks=2,
                working=[
                    f"The discount is {p}% of ${B} = ${disc_text}.00.",
                    f"${B} + ${_money(fee)} - ${disc_text}.00 = ${_money(subtotal)}.",
                ],
                distractors=b_opts,
                follow_through=f"{B} + part_a - {disc_text}",
            ),
            dict(
                prompt="Find the GST, in dollars.",
                answer_kind="number",
                answer=numeric_answer(gst),
                answer_text=_money(gst),
                marks=1,
                working=[f"10% of ${_money(subtotal)} = ${_money(gst)}."],
                distractors=c_opts,
                follow_through="part_b/10",
            ),
            dict(
                prompt="Hence find the total amount payable, in dollars.",
                answer_kind="number",
                answer=numeric_answer(total),
                answer_text=_money(total),
                marks=1,
                working=[f"${_money(subtotal)} + ${_money(gst)} = ${_money(total)}."],
                distractors=d_opts,
                follow_through="part_b + part_c",
            ),
        ],
    )
