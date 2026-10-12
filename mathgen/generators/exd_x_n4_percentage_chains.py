"""Exam-tier sources for `percentage-change` and `reverse-percentage`: two successive percentage changes.

A single percentage change is a practice question. An exam question chains two of them, and the learner has to see
that the second percentage is taken of the amount left by the first, not of the starting amount.

`exd_x_n4_percentage_change` runs forward from a starting price:

    (a) the price after the first change, (b) the price after the second change, (c) the overall percentage change.

`exd_x_n4_reverse_percentage` runs backward from the price paid, undoing the changes in reverse order:

    (a) the price before the second change, (b) the original price, (c) how much more or less was paid overall.

Both are built from whole-dollar starting prices and percentages that are multiples of 5, so every dollar amount is
exact to the cent and the overall percentage is exact to one decimal place. Each later part follows through from
the earlier ones, so a wrong (a) used correctly in (b) still earns (b).
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_RISES = tuple(range(5, 65, 5))
_FALLS = tuple(range(5, 55, 5))

_OPENERS = (
    "A hardware store in Perth sells a cordless drill for ${price}.",
    "A bike shop in Hobart sells a mountain bike for ${price}.",
    "A cafe in Fremantle buys a sack of coffee beans for ${price}.",
    "A sports store in Adelaide sells a cricket bat for ${price}.",
    "A garden centre in Toowoomba sells a rainwater tank for ${price}.",
    "A camping shop in Alice Springs sells a swag for ${price}.",
)
_REVERSE_OPENERS = (
    "A hardware store in Perth sold a cordless drill.",
    "A bike shop in Hobart sold a mountain bike.",
    "A sports store in Adelaide sold a cricket bat.",
    "A garden centre in Toowoomba sold a rainwater tank.",
    "A camping shop in Alice Springs sold a swag.",
    "A music shop in Newcastle sold a ukulele.",
)
_RISE_WORDS = ("increased", "marked up", "raised")
_FALL_WORDS = ("reduced", "discounted", "cut")


def _change_text(rng, sign: int, pct: int) -> str:
    words = _RISE_WORDS if sign > 0 else _FALL_WORDS
    return f"{rng.choice(words)} by {pct}%"


def _money(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 2), 2)


def _percent(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 1), 1)


def _dist(fmt, value: Fraction, tag: str, dp: int) -> Distractor:
    """A tagged distractor whose value is exactly what its text displays."""
    text = fmt(value)
    return Distractor(text, tag, Fraction(text))


def _collides_with_stem(values, *texts: str) -> bool:
    """Whether any answer (or its absolute value) is one of the numbers written in the question."""
    shown = {Fraction(t) for text in texts for t in re.findall(r"\d+(?:\.\d+)?", text)}
    return any(abs(Fraction(v)) in shown for v in values)


def _step_factors(s1: int, p1: int, s2: int, p2: int) -> tuple[int, int]:
    """The two scale factors as twentieths: (100 + s*p) / 5, a whole number for a multiple of 5."""
    return (100 + s1 * p1) // 5, (100 + s2 * p2) // 5


@generator("exd_x_n4_percentage_change", level=9, presets={"exam": {"tier": "exam"}})
def exd_x_n4_percentage_change(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "two successive changes with exact answers"):
        k = rng.randint(3, 40)
        price = 20 * k
        s1, s2 = rng.choice((1, -1)), rng.choice((1, -1))
        p1 = rng.choice(_RISES if s1 > 0 else _FALLS)
        p2 = rng.choice(_RISES if s2 > 0 else _FALLS)
        f1, f2 = _step_factors(s1, p1, s2, p2)
        if (f1 * f2) % 2 or f1 * f2 == 400:
            continue
        after_first = Fraction(k * f1)
        after_second = Fraction(k * f1 * f2, 20)
        overall = Fraction(f1 * f2, 4) - 100
        change1 = _change_text(rng, s1, p1)
        change2 = _change_text(rng, s2, p2)
        stem = f"{rng.choice(_OPENERS).format(price=price)} " + phrase(
            rng,
            (
                "At the start of the season the price is {c1}. At the end of the season the new price is {c2}.",
                "The price is first {c1}, and later the new price is {c2}.",
                "First the price is {c1}. After that, the new price is {c2}.",
            ),
            c1=change1,
            c2=change2,
        )
        prompts = (
            "Find the price after the first change, correct to the nearest cent.",
            "Hence find the price after the second change, correct to the nearest cent.",
            "Find the overall percentage change from the original price, correct to 1 decimal place. "
            "Use a negative number for a decrease.",
        )
        if _collides_with_stem((after_first, after_second, overall), stem, *prompts):
            continue
        first_amount = Fraction(price * p1, 100)
        second_amount = Fraction(after_first * p2, 100)
        cands_a = [
            _dist(_money, first_amount, "change-amount-not-new-value", 2),
            _dist(_money, price + s1 * p1, "percent-as-absolute", 2),
        ]
        cands_b = [
            _dist(_money, Fraction(price * (100 + s2 * p2), 100), "second-percent-on-original", 2),
            _dist(_money, second_amount, "change-amount-not-new-value", 2),
            _dist(_money, after_first + s2 * p2, "percent-as-absolute", 2),
        ]
        cands_c = [
            _dist(_percent, Fraction(s1 * p1 + s2 * p2), "successive-percent-added", 1),
            _dist(
                _percent,
                Fraction((after_second - price) * 100, after_second),
                "divides-by-final",
                1,
            ),
            _dist(_percent, Fraction(s2 * p2), "reported-last-step-only", 1),
            _dist(_money, after_second - price, "gave-change-not-percent", 2),
        ]
        a_text, b_text, c_text = _money(after_first), _money(after_second), _percent(overall)
        tol2, tol1 = Fraction(1, 200), Fraction(1, 20)
        d_a = pick_distractors(a_text, cands_a, 3, tolerance=tol2)
        d_b = pick_distractors(b_text, cands_b, 3, tolerance=tol2)
        d_c = pick_distractors(c_text, cands_c, 3, tolerance=tol1)
        if min(len(d_a), len(d_b), len(d_c)) < 2:
            continue

        break

    check(after_first == Fraction(price * (100 + s1 * p1), 100), "first price disagrees with the direct calculation")
    check(
        after_second == Fraction(price * (100 + s1 * p1) * (100 + s2 * p2), 10000),
        "second price disagrees with the direct calculation",
    )
    check(overall == (after_second - price) * 100 / price, "overall change disagrees with change over original")
    check(overall.denominator in (1, 2), "overall change is not exact to one decimal place")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=numeric_answer(after_first),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=1,
                sensible_range=(0, None),
                working=[
                    f"The first change multiplies the price by {100 + s1 * p1}/100.",
                    f"${price} × {Fraction(100 + s1 * p1, 100)} = ${a_text}.",
                ],
                distractors=d_a,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=numeric_answer(after_second),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a*{100 + s2 * p2}/100",
                working=[
                    f"The second change is taken of ${a_text}, not of the original price.",
                    f"${a_text} × {Fraction(100 + s2 * p2, 100)} = ${b_text}.",
                ],
                distractors=d_b,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=numeric_answer(overall),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                follow_through=f"(part_b-{price})/{price}*100",
                working=[
                    f"Change = ${b_text} − ${price} = ${_money(after_second - price)}.",
                    f"Percentage change = change ÷ original × 100 = {c_text}%.",
                ],
                distractors=d_c,
            ),
        ],
    )


@generator("exd_x_n4_reverse_percentage", level=9, presets={"exam": {"tier": "exam"}})
def exd_x_n4_reverse_percentage(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "two successive changes to undo with exact answers"):
        k = rng.randint(3, 40)
        original = 20 * k
        s1 = rng.choice((1, -1))
        p1 = rng.choice(_RISES if s1 > 0 else _FALLS)
        gst = rng.random() < 0.4
        if gst:
            s2, p2 = 1, 10
        else:
            s2 = rng.choice((1, -1))
            p2 = rng.choice(_RISES if s2 > 0 else _FALLS)
        f1, f2 = _step_factors(s1, p1, s2, p2)
        if f1 * f2 == 400:
            continue
        middle = Fraction(k * f1)
        final = Fraction(k * f1 * f2, 20)
        first = _change_text(rng, s1, p1)
        second = (
            "10% GST was then added to the new price" if gst else f"the new price was then {_change_text(rng, s2, p2)}"
        )
        stem = (
            f"{rng.choice(_REVERSE_OPENERS)} "
            + phrase(
                rng,
                (
                    "The original price was first {first}, and {second}.",
                    "Its price was first {first}; {second}.",
                ),
                first=first,
                second=second,
            )
            + f" The customer paid ${_money(final)}."
        )
        more = final > original
        prompts = (
            "Find the price after the first change, before the second change, correct to the nearest cent.",
            "Hence find the original price, before either change, correct to the nearest cent.",
            f"How much {'more' if more else 'less'} than the original price did the customer pay? "
            "Give your answer correct to the nearest cent.",
        )
        gap = abs(final - original)
        if _collides_with_stem((middle, original, gap), stem, *prompts):
            continue
        cands_a = [
            _dist(
                _money,
                final * Fraction(100 - p2, 100) if s2 > 0 else final * Fraction(100 + p2, 100),
                "subtracts-percent" if s2 > 0 else "adds-percent-of-final",
                2,
            ),
            _dist(_money, final - s2 * p2, "percent-as-absolute", 2),
        ]
        cands_b = [
            _dist(
                _money,
                middle * Fraction(100 - p1, 100) if s1 > 0 else middle * Fraction(100 + p1, 100),
                "subtracts-percent" if s1 > 0 else "adds-percent-of-final",
                2,
            ),
            _dist(_money, middle - s1 * p1, "percent-as-absolute", 2),
        ]
        cands_c = [
            _dist(_money, abs(final - middle), "reported-last-step-only", 2),
            _dist(_money, final * abs(Fraction(f1 * f2, 400) - 1), "net-percent-of-final", 2),
        ]
        a_text, b_text, c_text = _money(middle), _money(original), _money(gap)
        tol = Fraction(1, 200)
        d_a = pick_distractors(a_text, cands_a, 3, tolerance=tol)
        d_b = pick_distractors(b_text, cands_b, 3, tolerance=tol)
        d_c = pick_distractors(c_text, cands_c, 3, tolerance=tol)
        if min(len(d_a), len(d_b), len(d_c)) < 2:
            continue
        break

    check(middle * Fraction(100 + s2 * p2, 100) == final, "price before the second change does not lead to the final")
    check(original * Fraction(100 + s1 * p1, 100) == middle, "original price does not lead to the middle price")
    check(final != original, "the two changes cancel")

    final_literal = f"{final.numerator}/{final.denominator}"

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=numeric_answer(middle),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"The second change multiplied the price by {100 + s2 * p2}/100.",
                    f"Divide to undo it: ${_money(final)} ÷ {Fraction(100 + s2 * p2, 100)} = ${a_text}.",
                ],
                distractors=d_a,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=numeric_answer(original),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a/({100 + s1 * p1}/100)",
                working=[
                    f"The first change multiplied the price by {100 + s1 * p1}/100.",
                    f"Divide to undo it: ${a_text} ÷ {Fraction(100 + s1 * p1, 100)} = ${b_text}.",
                ],
                distractors=d_b,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=numeric_answer(gap),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=(f"{final_literal}-part_b" if more else f"part_b-{final_literal}"),
                working=[
                    f"Difference = ${_money(max(final, original))} − ${_money(min(final, original))} = ${c_text}."
                ],
                distractors=d_c,
            ),
        ],
    )
