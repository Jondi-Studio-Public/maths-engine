"""Exam tier for "multi-step problems with signed integers and rational numbers": an overdrawn account.

The practice tiers evaluate a signed expression step by step. Here a story supplies the steps in three connected
parts, so the learner has to decide what each operation does to a negative balance:

- (a) a monthly fee is charged several times to an account that is already overdrawn (decimals, repeated, negative),
- (b) the account holder pays off a fraction of the amount owing (the new balance is a fraction of a negative),
- (c) a deposit is needed to reach a positive target balance (subtracting a negative).

Parts (b) and (c) use the earlier answers, so they carry `follow_through`. Built backwards from the debt left
after (b) so that every amount is a whole number of cents.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    attempts,
    check,
    currency_from_cents,
    fixed_point_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    stated_precision_tolerance,
)

_NAMES = ["Mia", "Liam", "Priya", "Jack", "Chloe", "Noah", "Zara", "Ethan"]
_TOWNS = ["Hobart", "Darwin", "Cairns", "Adelaide", "Geelong", "Newcastle", "Fremantle", "Wollongong"]
_PAYOFFS = [(1, 4), (3, 4), (1, 3), (2, 3), (2, 5), (3, 5)]
_TARGETS = [50, 100, 150, 200, 250]
_TOL = stated_precision_tolerance(2)


def _amount(cents: int) -> str:
    return fixed_point_string(int(cents), 2)


def _tag(cents: int, tag: str) -> Distractor:
    return Distractor(_amount(cents), tag, exact_value(Fraction(int(cents), 100)))


@generator("exd_x_pr7_overdrawn_account", level=8, presets={"exam": {"tier": "exam"}})
def exd_x_pr7_overdrawn_account(rng: random.Random, tier: str = "exam") -> dict:
    """Repeated fees on an overdrawn balance, a fraction paid off, then a deposit to a target. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "an opening debt whose pay-off leaves a whole number of cents"):
        opening = 5 * rng.randint(400, 3000)  # cents overdrawn at the start: $20 to $150
        months = rng.randint(3, 6)
        fee = 5 * rng.randint(80, 300)  # cents, a multiple of 5c: $4.00 to $15.00
        owing = opening + months * fee
        payoffs = [(p, q) for p, q in _PAYOFFS if owing * (q - p) % q == 0]
        if payoffs:
            p, q = rng.choice(payoffs)
            left = owing * (q - p) // q  # cents still owing after the pay-off
            break
    target = rng.choice(_TARGETS)

    after_fees = -owing
    after_payoff = -left
    deposit = target * 100 + left
    # Second route: the pay-off as the fraction of the debt that is paid, rebuilt from the amount paid.
    paid = owing * p // q
    check(owing * p % q == 0 and owing - paid == left, "pay-off does not leave the debt it was built from")
    check(opening + months * fee == owing and deposit - target * 100 == left)

    name = rng.choice(_NAMES)
    stem = (
        f"{name}'s bank account in {rng.choice(_TOWNS)} is overdrawn: its balance is {currency_from_cents(-opening)}. "
        f"A fee of {currency_from_cents(fee, allow_negative=False)} is taken from the account at the end of each of the next "
        f"{months} months, with no other money paid in or out. Then {name} pays off {p}/{q} of the amount owing. "
        "Give each answer in dollars, to the nearest cent, as a negative number if the account is still overdrawn."
    )

    a_options = pick_distractors(
        _amount(after_fees),
        [
            _tag(-opening + months * fee, "fee-added-to-balance"),
            _tag(-opening - fee, "forgot-to-multiply-by-count"),
            _tag(owing, "sign-dropped-on-negative"),
        ],
        3,
        tolerance=_TOL,
    )
    b_options = pick_distractors(
        _amount(after_payoff),
        [
            _tag(-paid, "paid-fraction-used-as-remaining"),
            _tag(left, "sign-dropped-on-negative"),
        ],
        3,
        tolerance=_TOL,
    )
    c_options = pick_distractors(
        _amount(deposit),
        [
            _tag(target * 100 - left, "negative-subtracted-as-positive"),
            _tag(left, "ignored-target-value"),
            _tag(target * 100, "omitted-a-term"),
        ],
        3,
        tolerance=_TOL,
    )
    check(min(len(a_options), len(b_options), len(c_options)) >= 2, "too few distractors")

    def part(prompt, cents, marks, options, working, sensible, follow_through=None):
        value = Fraction(cents, 100)
        return dict(
            prompt=prompt,
            answer_kind="number",
            answer=numeric_answer(value),
            answer_text=_amount(cents),
            answer_latex=latex_answer(_amount(cents)),
            marks=marks,
            sensible_range=sensible,
            working=working,
            distractors=options,
            **({"follow_through": follow_through} if follow_through else {}),
        )

    return dict(
        text=stem,
        parts=[
            part(
                "What is the balance after the last fee has been taken?",
                after_fees,
                2,
                a_options,
                [
                    f"{currency_from_cents(-opening)} - {months} x {currency_from_cents(fee)} = {currency_from_cents(after_fees)}."
                ],
                (None, 0),
            ),
            part(
                "What is the balance after the payment?",
                after_payoff,
                2,
                b_options,
                [
                    f"The amount still owing is {q - p}/{q} of {currency_from_cents(owing)}.",
                    f"The balance is {currency_from_cents(after_payoff)}.",
                ],
                (None, 0),
                f"part_a * (1 - {p}/{q})",
            ),
            part(
                f"How much more must be paid in so that the balance is ${target}?",
                deposit,
                1,
                c_options,
                [f"${target} - ({currency_from_cents(after_payoff)}) = {currency_from_cents(deposit)}."],
                (0, None),
                f"{target} - part_b",
            ),
        ],
    )
