"""Exam tier for `algebraic-notation`: two readings of nearly the same notation, then a comparison.

The hard tier evaluates one piece of notation at a time. Here the same variables appear in two expressions that
differ only in where the brackets are, such as 3ab² and (3ab)², or a² + b² and (a + b)². Part (a) and part (b)
evaluate each (negative values make the power rules matter), and part (c) compares them (a ratio or a difference),
which only works if both readings were right. Part (c) carries `follow_through` from (a) and (b).

Built backwards from whole-number variable values, so every result is a whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator
from .exd_x_sc3_support import number_part, redraw, tagged

F = Fraction
_LEADS = ["Let a = {a} and b = {b}.", "Take a = {a} and b = {b}.", "Suppose a = {a} and b = {b}."]


def _nonzero(rng: random.Random, lo: int, hi: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def _product_power(rng: random.Random) -> dict:
    k = rng.randint(2, 4)
    a, b = rng.randint(1, 5), _nonzero(rng, -5, 5)  # a > 0 so the ratio of the two readings is positive
    first, second = k * a * b * b, (k * a * b) ** 2
    ratio = F(second, first)
    check(ratio == k * a and ratio.denominator == 1, "ratio of the two readings")
    kab = k * a * b
    slips_first = [(F(-first), "sign-on-power")] if b < 0 else []
    slips_first += [
        (F(second), "squares-more-than-the-letter"),
        (F(2 * kab), "squared-as-doubled"),
        (F(k + a + b * b), "juxtaposition-as-addition"),
    ]
    slips_second = [(F(-second), "sign-on-power")] if kab < 0 else []
    slips_second += [
        (F(first), "squares-only-part-of-the-bracket"),
        (F(2 * kab), "squared-as-doubled"),
        (F(k * a * a * b * b), "squares-only-part-of-the-bracket"),
    ]
    return dict(
        stem=rng.choice(_LEADS).format(a=a, b=b),
        ask_first=f"Find the value of {k}ab².",
        ask_second=f"Find the value of ({k}ab)².",
        first=F(first),
        second=F(second),
        slips_first=slips_first,
        slips_second=slips_second,
        ask_third=f"How many times as large as {k}ab² is ({k}ab)²?",
        third=ratio,
        slips_third=[
            (F(second - first), "uses-subtraction-for-how-many-times"),
            (F(kab), "ratio-cancelling-slip"),
            (F(k), "ratio-cancelling-slip"),
            (F(a), "ratio-cancelling-slip"),
        ],
        follow_through="part_b/part_a",
        working=(
            f"{k}ab² = {k} × {a} × ({b})² = {first}.",
            f"({k}ab)² = ({k} × {a} × {b})² = ({kab})² = {second}.",
            f"{second} ÷ {first} = {ratio}.",
        ),
    )


def _sum_square(rng: random.Random) -> dict:
    for _ in attempts(1000, "two values whose sum is not zero"):
        a, b = _nonzero(rng, -9, 9), _nonzero(rng, -9, 9)
        if a + b != 0:
            break
    first, second, diff = a * a + b * b, (a + b) ** 2, 2 * a * b
    check(second - first == diff, "difference of the two readings")
    sop = (-(a * a) if a < 0 else a * a) + (-(b * b) if b < 0 else b * b)
    slips_first = [(F(sop), "sign-on-power")] if sop != first else []
    slips_first += [(F(second), "confuses-square-of-sum-with-sum-of-squares"), (F(2 * a + 2 * b), "squared-as-doubled")]
    slips_second = [(F(-second), "sign-on-power")] if a + b < 0 else []
    slips_second += [
        (F(first), "confuses-square-of-sum-with-sum-of-squares"),
        (F(2 * (a + b)), "squared-as-doubled"),
        (F(a * a + b), "squares-only-part-of-the-bracket"),
    ]
    return dict(
        stem=rng.choice(_LEADS).format(a=a, b=b),
        ask_first="Find the value of a² + b².",
        ask_second="Find the value of (a + b)².",
        first=F(first),
        second=F(second),
        slips_first=slips_first,
        slips_second=slips_second,
        ask_third="Hence find the value of (a + b)² - (a² + b²).",
        third=F(diff),
        slips_third=[
            (F(a * b), "drops-the-factor-of-two"),
            (F(-diff), "subtracts-in-the-wrong-order"),
            (F(2 * (a + b)), "squared-as-doubled"),
        ],
        follow_through="part_b - part_a",
        working=(
            f"a² + b² = ({a})² + ({b})² = {first}.",
            f"(a + b)² = ({a + b})² = {second}.",
            f"{second} - {first} = {diff}.",
        ),
    )


def _constant_minus_square(rng: random.Random) -> dict:
    k = rng.randint(2, 9)
    a = _nonzero(rng, -7, 7)
    if a == k:
        a = -a
    first, second = k - a * a, (k - a) ** 2
    diff = second - first
    check(diff != 0, "the two readings agree")
    slips_first = [(F(k + a * a), "sign-on-power")] if a < 0 else []
    slips_first += [(F(second), "squares-more-than-the-letter"), (F(k - 2 * a), "squared-as-doubled")]
    slips_second = [(F(first), "squares-only-part-of-the-bracket"), (F(k * k - a * a), "squares-each-term-separately")]
    slips_second += [(F(2 * (k - a)), "squared-as-doubled")]
    return dict(
        stem=rng.choice(["Let a = {a}.", "Take a = {a}.", "Suppose a = {a}."]).format(a=a),
        ask_first=f"Find the value of {k} - a².",
        ask_second=f"Find the value of ({k} - a)².",
        first=F(first),
        second=F(second),
        slips_first=slips_first,
        slips_second=slips_second,
        ask_third=f"Hence find the value of ({k} - a)² - ({k} - a²).",
        third=F(diff),
        slips_third=[(F(-diff), "subtracts-in-the-wrong-order"), (F(first + second), "adds-instead-of-subtracting")],
        follow_through="part_b - part_a",
        working=(
            f"{k} - a² = {k} - ({a})² = {first}.",
            f"({k} - a)² = ({k - a})² = {second}.",
            f"{second} - ({first}) = {diff}.",
        ),
    )


_BUILDERS = [_product_power, _sum_square, _constant_minus_square]


@generator("exd_x_sc3_notation_chain", level=9, presets={"exam": dict(tier="exam")})
def exd_x_sc3_notation_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Evaluate two expressions that differ only in their brackets, then compare them."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return redraw(_build, rng, "an expression pair with usable wrong answers")


def _build(rng: random.Random) -> dict:
    s = rng.choice(_BUILDERS)(rng)
    check(s["first"] != s["second"], "the two readings must differ")
    parts = [
        number_part(s["ask_first"], s["first"], 0, 1, [s["working"][0]], tagged(s["first"], s["slips_first"], 0, n=3)),
        number_part(
            s["ask_second"], s["second"], 0, 2, [s["working"][1]], tagged(s["second"], s["slips_second"], 0, n=3)
        ),
        number_part(
            s["ask_third"],
            s["third"],
            0,
            2,
            [s["working"][2]],
            tagged(s["third"], s["slips_third"], 0, n=3),
            follow_through=s["follow_through"],
        ),
    ]
    return dict(text=s["stem"], parts=parts)
