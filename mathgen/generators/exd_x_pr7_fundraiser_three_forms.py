"""Exam tier for "combine a fraction, a decimal and a percentage": a fundraiser's money, in three connected parts.

The practice tiers evaluate a bare expression such as 3/25 + 0.36 + 50% of 16. Here the same three
representations arrive in a story where each share is taken from a different base:

- (a) a fraction of the money raised is spent on equipment,
- (b) a percentage of the money LEFT AFTER (a) is spent on prizes,
- (c) a decimal share of the ORIGINAL total is donated, and the learner finds what remains.

The learner has to decide which amount each share applies to, and (b) and (c) use the earlier answers, so they
carry `follow_through`. Every amount is a whole number of cents by construction: the total is a multiple of 40,
which every denominator used divides.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_EVENTS = [
    ("school fete", "Ballarat"),
    ("surf life saving club fundraiser", "Torquay"),
    ("fun run", "Launceston"),
    ("community garage sale", "Bunbury"),
    ("book fair", "Toowoomba"),
    ("charity barbecue", "Mildura"),
]
_FRACTIONS = [(1, 4), (3, 8), (2, 5), (3, 10), (1, 5), (1, 8), (3, 5), (1, 2)]
_PERCENTS = [15, 20, 25, 30, 35, 40, 45, 60]
_DECIMALS = [5, 10, 15, 20, 25]  # hundredths
_TOL = stated_precision_tolerance(2)


def _money(v: Fraction) -> str:
    return stated_precision_string(v, 2)


def _dist(value: Fraction, tag: str | None) -> Distractor | str:
    value = round_half_up(value, 2)  # a slip's result is shown to the cent
    text = _money(value)
    return Distractor(text, tag, exact_value(value)) if tag else text


@generator("exd_x_pr7_fundraiser_three_forms", level=10, presets={"exam": {"tier": "exam"}})
def exd_x_pr7_fundraiser_three_forms(rng: random.Random, tier: str = "exam") -> dict:
    """Fraction of the total, percentage of the remainder, decimal of the total. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "shares that leave some money over"):
        total = 40 * rng.randint(20, 200)
        num, den = rng.choice(_FRACTIONS)
        pct = rng.choice(_PERCENTS)
        hundredths = rng.choice(_DECIMALS)
        a = Fraction(total * num, den)
        left = total - a
        b = left * pct / 100
        donation = Fraction(total * hundredths, 100)
        final = total - a - b - donation
        if final > Fraction(total, 20):
            break
    check(a.denominator == 1 and (b * 100).denominator == 1, "an amount is not a whole number of cents")
    # Second route: shares as fractions of the total, summed.
    shares = Fraction(num, den) + (1 - Fraction(num, den)) * Fraction(pct, 100) + Fraction(hundredths, 100)
    check(final == total * (1 - shares), "remaining money disagrees with the shares")

    event, town = rng.choice(_EVENTS)
    decimal_text = f"0.{hundredths:02d}" if hundredths % 10 else f"0.{hundredths // 10}"
    stem = (
        f"A {event} in {town} raised ${total}. The organisers spent {num}/{den} of the money on equipment. "
        f"They then spent {pct}% of the money that was left on prizes. Finally they donated {decimal_text} "
        "of the original total to a local community group. Give each answer in dollars, to the nearest cent."
    )

    t = Fraction(total)
    a_options = pick_distractors(
        _money(a),
        [
            _dist(t / den, "ignores-numerator"),
            _dist(t * den / num, "inverted-fraction"),
            _dist(t * Fraction(int(f"{num}{den}"), 10 ** len(f"{num}{den}")), "fraction-digits-as-decimal"),
            _dist(left, None),
        ],
        3,
        tolerance=_TOL,
    )
    b_options = pick_distractors(
        _money(b),
        [
            _dist(t * pct / 100, "applies-fraction-to-original-whole"),
            _dist(left * pct / 1000, "decimal-place-shift"),
            _dist(left - b, None),
        ],
        3,
        tolerance=_TOL,
    )
    c_options = pick_distractors(
        _money(final),
        [
            _dist(t - a - b, "omitted-a-term"),
            _dist(a + b + donation, "answers-amount-used-not-remaining"),
            _dist(t - a - b - t * hundredths / 10000, "decimal-as-percent"),
            _dist(t - a - b - (t - a - b) * hundredths / 100, "decimal-of-remainder-not-total"),
        ],
        3,
        tolerance=_TOL,
    )
    check(min(len(a_options), len(b_options), len(c_options)) >= 2, "too few distractors")

    def part(prompt, value, marks, options, working, follow_through=None):
        return dict(
            prompt=prompt,
            answer_kind="number",
            answer=numeric_answer(value),
            answer_text=_money(value),
            answer_latex=latex_answer(_money(value)),
            marks=marks,
            sensible_range=(0, None),
            working=working,
            distractors=options,
            **({"follow_through": follow_through} if follow_through else {}),
        )

    return dict(
        text=stem,
        parts=[
            part(
                "How much was spent on equipment?",
                a,
                2,
                a_options,
                [f"{num}/{den} of ${total} = ${_money(a)}."],
            ),
            part(
                "How much was spent on prizes?",
                b,
                2,
                b_options,
                [
                    f"Money left = ${total} - ${_money(a)} = ${_money(left)}.",
                    f"{pct}% of ${_money(left)} = ${_money(b)}.",
                ],
                f"({total} - part_a) * {pct} / 100",
            ),
            part(
                "How much money is left after the donation?",
                final,
                2,
                c_options,
                [f"Donation = {decimal_text} of ${total} = ${_money(donation)}.", f"Left = ${_money(final)}."],
                f"{total} - part_a - part_b - {total} * {hundredths} / 100",
            ),
        ],
    )
