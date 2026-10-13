"""Exam source: a casual worker's week, from the loaded hourly rate to take-home pay.

One scenario: a base hourly rate with a percentage casual loading, some weekday hours at the casual rate, some
Saturday hours paid at a percentage of the casual rate, and tax withheld as a percentage of gross pay.

(a) the casual hourly rate (base plus loading);
(b) the gross pay for the week, from (a): weekday hours at the casual rate and Saturday hours at the higher
    percentage of it;
(c) the pay after tax, from (b).

Rates and gross pay are drawn so they are an exact number of cents; only (c) is rounded (to the nearest cent), and
a draw is kept only when (c) is clear of a half-cent boundary.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, generator, phrase
from mathgen.generators.exd_x_n1_support import clear_of_boundary, number_part, show

_STEM = (
    "A casual worker's base rate of pay is ${base} an hour. A casual loading of {load}% is added to the base rate. "
    "In one week the worker is on duty for {h1} hours on weekdays, paid at the casual rate, and {h2} hours on "
    "Saturday, paid at {pen}% of the casual rate. Tax of {tax}% of gross pay is withheld.",
    "A hospitality worker is paid a base rate of ${base} an hour plus a {load}% casual loading. One week they work "
    "{h1} hours from Monday to Friday at the casual rate and {h2} hours on Saturday at {pen}% of the casual rate. "
    "{tax}% of the gross pay is withheld as tax.",
    "A café employee on a casual contract earns the base rate of ${base} an hour, and a loading of {load}% is "
    "added to it. This week the employee works {h1} hours on weekdays at the casual rate and {h2} hours on "
    "Saturday, when the pay is {pen}% of the casual rate. The employer withholds {tax}% of the gross pay as tax.",
)


def _cents_exact(value: Fraction) -> bool:
    return (value * 100).denominator == 1


@generator("exd_x_n1_casual_pay_week", level=11, presets={"exam": dict(tier="exam")})
def exd_x_n1_casual_pay_week(rng, tier="exam"):
    for _ in attempts(1000, "a pay draw with exact cents and a stable rounding"):
        base = Fraction(rng.randint(2000, 3800), 100)
        load = rng.choice((20, 25))
        h1 = rng.randint(10, 32)
        h2 = rng.randint(3, 9)
        pen = rng.choice((125, 150, 175, 200))
        tax = rng.randint(12, 32)
        casual = base * (100 + load) / 100
        gross = casual * h1 + casual * h2 * pen / 100
        if not (_cents_exact(casual) and _cents_exact(gross)):
            continue
        net = gross * (100 - tax) / 100
        if not clear_of_boundary(net, 2):
            continue
        break

    stem = phrase(rng, _STEM, base=show(base, 2), load=load, h1=h1, h2=h2, pen=pen, tax=tax)
    f = Fraction
    part_a = number_part(
        "Find the casual hourly rate, in dollars.",
        casual,
        2,
        2,
        [f"Casual rate = {show(base, 2)} × {1 + f(load, 100)} = ${show(casual, 2)} per hour."],
        [
            (base * load / 100, None),
            (base + load, "percent-as-absolute"),
            (base * (100 - load) / 100, None),
        ],
        sensible_range=(0, None),
    )
    part_b = number_part(
        "Using your answer to (a), find the worker's gross pay for the week, in dollars.",
        gross,
        2,
        3,
        [
            f"Weekdays: {h1} × {show(casual, 2)} = ${show(casual * h1, 2)}.",
            f"Saturday rate = {pen}% × {show(casual, 2)} = ${show(casual * pen / 100, 2)}; {h2} hours = ${show(casual * h2 * pen / 100, 2)}.",
            f"Gross = ${show(gross, 2)}.",
        ],
        [
            (casual * (h1 + h2), None),
            (casual * h1 + base * h2 * pen / 100, None),
            (casual * h1 + casual * h2 * (pen - 100) / 100, None),
            (casual * h1 + casual * h2 + casual * pen / 100, None),
        ],
        follow_through=f"part_a*({h1} + {h2}*{pen}/100)",
        sensible_range=(0, None),
    )
    part_c = number_part(
        "Using your answer to (b), find the worker's pay after tax, correct to the nearest cent.",
        net,
        2,
        2,
        [
            f"Tax = {tax}% × {show(gross, 2)} = ${show(gross * tax / 100, 2)}.",
            f"Pay after tax = {show(gross, 2)} - {show(gross * tax / 100, 2)} = ${show(net, 2)}.",
        ],
        [
            (gross * tax / 100, None),
            (gross * (100 + tax) / 100, None),
            (gross - tax, "percent-as-absolute"),
            (gross, None),
        ],
        follow_through=f"part_b*(100 - {tax})/100",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])
