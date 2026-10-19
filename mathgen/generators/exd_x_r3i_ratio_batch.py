"""Exam-tier source for `senior-demonstrate-an-understanding-of-the-elementary-ideas-and` (ratio): from a test batch
measured in decimals to a full batch.

A test batch of two components is measured to one decimal place. The parts take the ratio through to quantities:

(a) the ratio of the first to the second in simplest whole-number form     fraction (written a:b), 2 marks
(b) the fraction of the mixture that is the first component                fraction, 1 mark,  part_a / (1 + part_a)
(c) the amount of the first component in a larger batch, same ratio        number, 2 marks,  total * part_b
(d) the amount of the second component in that batch                       number, 1 mark,   total - part_c

## Construction

The ratio p:q is drawn in lowest terms (p and q different, at most 9) and the test batch is built as p*s and q*s with
s a whole number of tenths, so at least one measured amount has a decimal point and clearing it is needed. The larger
batch is a whole multiple of p + q, so (c) and (d) are whole numbers.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator, phrase
from .exd_x_r3i_support import fraction_part, number_part

#: (first component, second component, the mixture, unit, the unit's plural for amounts)
_MIXES = (
    ("blue paint", "white paint", "paint mix", "litres"),
    ("concentrate", "water", "cordial", "litres"),
    ("red paint", "yellow paint", "orange paint mix", "litres"),
    ("compost", "sand", "potting mix", "litres"),
    ("weedkiller concentrate", "water", "spray mix", "litres"),
    ("orange juice", "sparkling water", "punch", "litres"),
)


def _ratio_text(value: Fraction) -> str:
    return f"{value.numerator}:{value.denominator}"


def _amount(value: Fraction, unit: str) -> str:
    """The amount with its unit, singular for exactly one ("1 litre")."""
    return f"1 {unit[:-1]}" if value == 1 else f"{_dec(value)} {unit}"


def _dec(value: Fraction) -> str:
    return str(int(value)) if value.denominator == 1 else f"{float(value):.1f}"


@generator("exd_x_r3i_ratio_batch", level=8, presets={"exam": dict(tier="exam")})
def exd_x_r3i_ratio_batch(rng, tier: str = "exam") -> dict:
    """Simplest ratio of two decimal amounts, the first component's fraction, and the amounts in a bigger batch."""
    check(tier == "exam", f"unknown tier {tier!r}")
    first, second, mix, unit = rng.choice(_MIXES)
    for _ in attempts(1000, "a ratio with decimal test amounts"):
        p, q = rng.randint(1, 9), rng.randint(1, 9)
        s10 = rng.randint(2, 45)  # scale in tenths
        mult = rng.randint(2, 20)
        x, y = Fraction(p * s10, 10), Fraction(q * s10, 10)
        total = mult * (p + q)
        if p == q or math.gcd(p, q) != 1 or (x.denominator == 1 and y.denominator == 1):
            continue
        if not Fraction(1, 4) <= Fraction(p, q) <= 4:
            continue
        if x < Fraction(2, 5) or y < Fraction(2, 5) or not 20 <= total <= 240 or total % 5:
            continue
        break
    check(x / y == Fraction(p, q), "test amounts disagree with the ratio")
    ratio = Fraction(p, q)
    share = Fraction(p, p + q)
    amount_first = Fraction(mult * p)
    amount_second = Fraction(mult * q)
    check(amount_first + amount_second == total and share * total == amount_first, "batch amounts disagree")

    stem = phrase(
        rng,
        (
            "A test batch of {mix} uses {x} of {first} and {y} of {second}.",
            "To test a recipe, {x} of {first} are mixed with {y} of {second} to make {mix}.",
            "A trial batch of {mix} is made from {x} of {first} and {y} of {second}.",
        ),
        mix=mix,
        first=first,
        second=second,
        x=_amount(x, unit),
        y=_amount(y, unit),
    )
    part_a = fraction_part(
        f"Write the ratio of {first} to {second} in simplest whole-number form.",
        ratio,
        2,
        [f"{_dec(x)} : {_dec(y)} = {int(x * 10)} : {int(y * 10)} = {p}:{q}."],
        [
            ("ratio-terms-reversed", Fraction(q, p)),
            ("part-to-part-and-part-to-whole-confused", Fraction(p, p + q)),
            ("ratio-decimal-cleared-by-different-powers-of-ten", Fraction(p, 10 * q)),
            ("ratio-of-difference-to-first-term", Fraction(q - p, p) if q != p else None),
        ],
        text_of=_ratio_text,
    )
    part_b = fraction_part(
        f"What fraction of the {mix} is {first}?",
        share,
        1,
        [f"Parts in total: {p} + {q} = {p + q}, so the fraction is {p}/{p + q}."],
        [
            ("other-component-given", Fraction(q, p + q)),
            ("part-to-part-and-part-to-whole-confused", Fraction(p, q)),
            ("whole-over-part-fraction-inverted", Fraction(p + q, p)),
            ("fraction-ignores-the-second-term", Fraction(1, p) if p > 1 else Fraction(1, p + 1)),
        ],
        follow_through="part_a/(1 + part_a)",
        sensible_range=(0, 1),
    )
    part_c = number_part(
        f"A full batch of {total} {unit} of {mix} is made in the same ratio. How many {unit} of {first} does it need?",
        amount_first,
        0,
        2,
        [f"{total} × {p}/{p + q} = {int(amount_first)} {unit}."],
        [
            ("other-component-given", amount_second),
            ("half-of-the-batch-for-each-component", Fraction(total, 2)),
            ("batch-times-ratio-instead-of-share", Fraction(total * p, q)),
            ("batch-divided-by-one-term", Fraction(total, p)),
            ("multiplier-reported-as-amount", Fraction(mult)),
            ("ratio-term-subtracted-from-batch", Fraction(total - p)),
        ],
        follow_through=f"{total}*part_b",
        sensible_range=(0, None),
    )
    part_d = number_part(
        f"How many {unit} of {second} does the full batch need?",
        amount_second,
        0,
        1,
        [f"{total} - {int(amount_first)} = {int(amount_second)} {unit}."],
        [
            ("other-component-given", amount_first),
            ("half-of-the-batch-for-each-component", Fraction(total, 2)),
            ("batch-times-ratio-instead-of-share", Fraction(total * q, p)),
            ("batch-divided-by-one-term", Fraction(total, q)),
            ("multiplier-reported-as-amount", Fraction(mult)),
            ("ratio-term-subtracted-from-batch", Fraction(total - q)),
        ],
        follow_through=f"{total} - part_c",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
