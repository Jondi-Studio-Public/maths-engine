"""Exam-tier source for `f10-exact-approximate-repeated-calculation`: a unit price rounded once, then scaled up.

The practice tiers compare the exact and the step-by-step rounded results of a stated chain of operations. An
exam question puts the same idea in a buying problem: a pack price is shared among a stated number of items and
rounded to the nearest cent, that rounded price is multiplied up to an order, and the learner measures how far
the order's cost has drifted from the exact one. The rounding error of one item is under half a cent; the error
of the order is that error multiplied by the number of items, which is the point of the question.

Parts: (a) the rounded price of one item; (b) the cost of the order at that price; (c) the difference between
(b) and the exact cost of the order. Parts (b) and (c) follow through from the learner's own earlier answers.

Built from exact fractions of a cent. Every draw is kept clear of a rounding boundary in the unit price and in
the final difference, so the exact route and a learner's route (using their rounded unit price) agree.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import (
    check,
    draw_until,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_ITEMS = [
    ("exercise books", "exercise book"),
    ("water bottles", "water bottle"),
    ("sunscreen sachets", "sunscreen sachet"),
    ("tennis balls", "tennis ball"),
    ("pencil cases", "pencil case"),
    ("lunch boxes", "lunch box"),
    ("show bags", "show bag"),
    ("sausage rolls", "sausage roll"),
]
_PACK_SIZES = [3, 6, 7, 9, 11, 12, 13, 14, 15]


def _dollars(cents: Fraction) -> str:
    """`cents` (already a whole number of cents) as dollars at 2 decimal places."""
    return stated_precision_string(Fraction(cents) / 100, 2)


def _dd(cents: Fraction) -> str:
    """A distractor in dollars: `cents` rounded to the nearest cent."""
    return _dollars(round_half_up(Fraction(cents), 0))


@generator("exd_x_n3_unit_price_rounding", level=10, presets={"exam": dict(tier="exam")})
def exd_x_n3_unit_price_rounding(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: round a unit price, scale it to an order, then measure the drift from the exact cost."""
    check(tier == "exam", f"unknown tier {tier!r}")

    def draw(r):
        return r.choice(_PACK_SIZES), r.randint(600, 9500), r.randint(20, 95), r.randrange(len(_ITEMS))

    def accept(t):
        n, pack, m, _ = t
        unit = Fraction(pack, n)
        if unit.denominator == 1:
            return False
        if abs(unit % 1 - Fraction(1, 2)) < Fraction(1, 10):
            return False
        err = abs(round_half_up(unit, 0) - unit)
        drift = m * err  # cents
        return (
            err >= Fraction(1, 5)
            and drift >= 2
            and drift.denominator != 1
            and abs(drift % 1 - Fraction(1, 2)) >= Fraction(1, 20)
        )

    n, pack, m, idx = draw_until(rng, draw, accept, what="a pack size, price and order")
    plural, single = _ITEMS[idx]
    unit = Fraction(pack, n)
    unit_r = round_half_up(unit, 0)  # cents
    cost_r = m * unit_r  # cents, exact
    cost_x = Fraction(m * pack, n)  # cents, exact
    drift = abs(cost_r - cost_x)
    drift_r = round_half_up(drift, 0)
    check(drift_r >= 1 and drift != drift_r, "drift is not a rounding-sensitive amount")

    a_text, b_text, c_text = _dollars(unit_r), _dollars(cost_r), _dollars(drift_r)
    truncated = Fraction(pack // n)
    err = abs(unit_r - unit)
    a_cands = [
        _dollars(truncated),  # rounds down instead of to the nearest cent
        _dd(Fraction(pack, n + 1)),  # shares the pack among one item too many
        _dd(Fraction(pack, n - 1)),  # shares the pack among one item too few
        _dollars(Fraction(pack)),  # the pack price itself
    ]
    b_cands = [
        _dd(cost_x),  # the exact cost, not the cost at the rounded price
        _dollars(Fraction(m * pack)),  # an order of packs: forgets to divide by the pack size
        _dollars(m * truncated),  # truncated unit price
        _dd(Fraction(m * pack, n + 1)),
    ]
    c_cands = [
        _dd(err),  # the rounding error of one item, not of the order
        _dd(cost_x),  # the exact cost
        b_text,  # the cost at the rounded price
        _dd(drift / m) if drift_r != round_half_up(drift / m, 0) else None,
        _dd(cost_r + cost_x),
    ]
    a_dist = pick_distractors(a_text, a_cands, 3, tolerance=stated_precision_tolerance(2))
    b_dist = pick_distractors(b_text, b_cands, 3, tolerance=stated_precision_tolerance(2))
    c_dist = pick_distractors(c_text, [c for c in c_cands if c], 3, tolerance=stated_precision_tolerance(2))
    check(
        min(len(a_dist), len(b_dist), len(c_dist)) >= 2,
        f"too few distractors {a_dist} {b_dist} {c_dist} {n} {pack} {m}",
    )

    stem = (
        f"A pack of {n} {plural} costs ${_dollars(Fraction(pack))}. A school canteen prices each {single} at "
        f"the pack price divided by {n}, rounded to the nearest cent, and sells {m} {plural} in one day."
    )
    exact_c = f"{m * pack}/({100 * n})"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the canteen's price for one {single}, in dollars.",
                answer_kind="number",
                answer=numeric_answer(unit_r / 100),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=1,
                sensible_range=(0, None),
                working=[
                    f"${_dollars(Fraction(pack))} ÷ {n} = {float(unit / 100):.4f}..., so ${a_text} to the nearest cent."
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=f"Find the amount taken for the {m} {plural} at that price, in dollars.",
                answer_kind="number",
                answer=numeric_answer(cost_r / 100),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a*{m}",
                working=[f"{m} × ${a_text} = ${b_text}."],
                distractors=b_dist,
            ),
            dict(
                prompt=f"By how much does this differ from the exact cost of {m} {plural} at the pack price "
                f"divided by {n}, in dollars correct to 2 decimal places?",
                answer_kind="number",
                answer=numeric_answer(drift_r / 100),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"Abs(part_b - {exact_c})",
                working=[
                    f"Exact cost = {m} × ${_dollars(Fraction(pack))} ÷ {n} = ${float(cost_x / 100):.4f}.",
                    f"Difference = |${b_text} − ${float(cost_x / 100):.4f}| = ${c_text} (2 d.p.).",
                ],
                distractors=c_dist,
            ),
        ],
    )
