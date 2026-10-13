"""Exam question on estimating with decimals: an estimate, the exact amount, and how far apart they are.

A shopper buys three items and uses a voucher. Part (a) asks for the estimate from rounding each price (and the
voucher) to the nearest dollar, part (b) the exact amount paid, and part (c) how far the estimate is from the
exact amount, which depends on whether the individual roundings pulled the same way or cancelled.

Prices are drawn in cents so every figure is exact; no price ends in exactly 50 cents (a rounding tie) and the
estimate always differs from the exact amount.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    currency_from_cents,
    generator,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_SHOPS = (
    (
        "At a sports store in Newcastle",
        "{name} buys a cricket bat grip for ${p1}, a water bottle for ${p2} and a "
        "pair of socks for ${p3}, then uses a ${v} voucher.",
    ),
    (
        "At a bookshop in Fremantle",
        "{name} buys a novel for ${p1}, a diary for ${p2} and a pack of pens for ${p3}, "
        "then takes ${v} off with a gift card.",
    ),
    (
        "At a garden centre in Ballarat",
        "{name} buys a potted herb for ${p1}, a bag of potting mix for ${p2} and a "
        "trowel for ${p3}, then uses a ${v} discount coupon.",
    ),
    (
        "At a school canteen in Cairns",
        "{name} buys a sandwich for ${p1}, a juice for ${p2} and a fruit cup for ${p3}, "
        "then uses a ${v} canteen credit.",
    ),
)
_NAMES = ("Mei", "Tom", "Aroha", "Sam", "Priya", "Liam", "Zara", "Noah")


def _dollars(cents: int) -> str:
    return currency_from_cents(cents, "", allow_negative=False)


def _round_dollar(cents: int) -> int:
    return int(round_half_up(Fraction(cents, 100), 0))


def _two_dp(cents: Fraction | int) -> str:
    return stated_precision_string(Fraction(cents, 100), 2)


@generator("exd_x_pr4_decimal_estimate_budget", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr4_decimal_estimate_budget(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", "exd_x_pr4_decimal_estimate_budget: only the exam tier exists")
    for _ in attempts(1000, "prices with a clear estimate"):
        prices = [rng.randint(210, 2995) for _ in range(3)]
        voucher = rng.randrange(150, 900, 5)
        if any(c % 100 == 50 for c in (*prices, voucher)):
            continue
        exact = sum(prices) - voucher
        estimate = sum(_round_dollar(c) for c in prices) - _round_dollar(voucher)
        errors = [100 * _round_dollar(c) - c for c in (*prices, voucher)]
        gap = abs(100 * estimate - exact)
        if gap < 10 or exact < 500:
            continue

        # Independent route: the gap is the signed sum of the four rounding changes.
        check(100 * estimate - exact == errors[0] + errors[1] + errors[2] - errors[3], "gap disagrees")

        est_text, exact_text, gap_text = str(estimate), _two_dp(exact), _two_dp(gap)
        rounds_first_only = _round_dollar(prices[0]) * 100 + prices[1] + prices[2] - voucher
        truncated = sum(c // 100 for c in prices) - voucher // 100
        trunc_gap = abs(100 * truncated - exact)
        sum_of_sizes = sum(abs(e) for e in errors)
        # The estimate is a whole number of dollars, so every option for (a) is one too.
        rounded_sum = sum(_round_dollar(c) for c in prices)
        a_candidates = [
            Distractor(
                str(_round_dollar(rounds_first_only)), "rounds-only-one-operand", _round_dollar(rounds_first_only)
            ),
            Distractor(str(_round_dollar(exact)), "computes-exact-result-instead-of-estimating", _round_dollar(exact)),
            Distractor(str(truncated), "rounded-down-not-to-nearest", truncated),
            Distractor(
                str(rounded_sum + _round_dollar(voucher)), None, rounded_sum + _round_dollar(voucher)
            ),  # voucher added
            Distractor(str(rounded_sum), None, rounded_sum),  # the voucher left out
        ]
        b_candidates = [
            Distractor(_two_dp(sum(prices)), None, Fraction(sum(prices), 100)),  # the voucher left out
            Distractor(_two_dp(sum(prices) + voucher), None, Fraction(sum(prices) + voucher, 100)),  # voucher added on
        ]
        c_candidates = [
            Distractor(_two_dp(trunc_gap), "rounded-down-not-to-nearest", Fraction(trunc_gap, 100)),
            Distractor(_two_dp(sum_of_sizes), "adds-rounding-errors-ignoring-direction", Fraction(sum_of_sizes, 100)),
            Distractor(
                _two_dp(abs(rounds_first_only - exact)),
                "rounds-only-one-operand",
                Fraction(abs(rounds_first_only - exact), 100),
            ),
        ]
        a_d = pick_distractors(est_text, a_candidates, 3, tolerance=Fraction(1, 100))
        b_d = pick_distractors(exact_text, b_candidates, 3, tolerance=Fraction(1, 200))
        c_d = pick_distractors(gap_text, c_candidates, 3, tolerance=Fraction(1, 200))
        if min(len(a_d), len(b_d), len(c_d)) >= 2:
            break

    place, template = rng.choice(_SHOPS)
    stem = f"{place}, " + template.format(
        name=rng.choice(_NAMES),
        p1=_dollars(prices[0]),
        p2=_dollars(prices[1]),
        p3=_dollars(prices[2]),
        v=_dollars(voucher),
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Estimate the amount paid by rounding each price and the voucher to the nearest dollar first. "
                "Give your estimate in dollars.",
                answer_kind="number",
                answer=estimate,
                answer_text=est_text,
                marks=2,
                sensible_range=(0, None),
                working=[
                    "Rounded prices: "
                    + ", ".join(f"${_round_dollar(c)}" for c in prices)
                    + f"; voucher ${_round_dollar(voucher)}.",
                    f"Estimate = ${est_text}.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="Find the exact amount paid, in dollars.",
                answer_kind="number",
                answer=numeric_answer(Fraction(exact, 100)),
                answer_text=exact_text,
                marks=1,
                sensible_range=(0, None),
                working=[f"Exact amount = {_two_dp(sum(prices))} - {_dollars(voucher)} = ${exact_text}."],
                distractors=b_d,
            ),
            dict(
                prompt="By how many dollars does your estimate differ from the exact amount?",
                answer_kind="number",
                answer=numeric_answer(Fraction(gap, 100)),
                answer_text=gap_text,
                marks=1,
                sensible_range=(0, None),
                follow_through="Abs(part_a - part_b)",
                working=[f"|{est_text} - {exact_text}| = {gap_text}."],
                distractors=c_d,
            ),
        ],
    )
