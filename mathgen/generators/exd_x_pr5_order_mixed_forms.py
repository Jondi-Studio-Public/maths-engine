"""Exam tier for `f10-fraction-equivalence-numberline`: order a fraction, decimals and percentages together.

The skill's other tiers order halves, thirds and quarters, all written as fractions. Here four values are close
together and each is written differently (a fraction, a decimal or a percentage), so none can be compared as printed:
every one must first be turned into a common form. The values are then read as distances along a ride of known
length.

* (a) the four values in order, smallest to largest, each written as it appears in the question (2 marks);
* (b) how far ahead the front friend is of the friend at the back, in the units of the event (1 mark).

Built backwards from percentages that are multiples of 2.5, so each has a terminating fraction and decimal, and the
total is a multiple of 40 units, so every distance is a whole number of units.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    rational_text,
    reversed_and_adjacent_swap,
)

_NAMES = ["Ayesha", "Ben", "Chloe", "Dylan", "Eli", "Freya", "Gus", "Hana", "Isla", "Jack", "Kai", "Lena"]
# (sentence, verb, noun, unit, possible totals). Every total is a multiple of 40, so 2.5% is a whole number of units.
_EVENTS = [
    (
        "A charity bike ride from Lilydale to Warburton is {total} kilometres long.",
        "ridden",
        "ride",
        "kilometres",
        [40, 80, 120],
    ),
    ("A school hike on the Overland Track is {total} kilometres long.", "walked", "hike", "kilometres", [40, 80]),
    ("An ocean swim at Bondi is {total} metres long.", "swum", "swim", "metres", [2000, 3000, 4000]),
]


class _Retry(Exception):
    pass


def _show(value: Fraction, form: str) -> str:
    """`value` written as a fraction, a decimal or a percentage."""
    if form == "fraction":
        return rational_text(value)
    if form == "decimal":
        text = f"{float(value):.3f}".rstrip("0").rstrip(".")
        return text
    return f"{float(value * 100):g}%"


def _printed_number(text: str) -> Fraction:
    """The number a learner who ignores the form would read: the numerator, the decimal, or the percentage."""
    if text.endswith("%"):
        return Fraction(text[:-1])
    if "/" in text:
        return Fraction(text.split("/")[0])
    return Fraction(text)


def _attempt(rng: random.Random) -> dict:
    for _ in attempts(1000, "four close values in at least three different forms"):
        base = rng.randrange(100, 700, 25)  # tenths of a percent: 10.0% .. 70.0%
        steps = sorted(rng.sample(range(0, 13), 4))
        pcts = [Fraction(base, 10) + Fraction(5, 2) * k for k in steps]
        if pcts[-1] >= 95:
            continue
        forms = [rng.choice(["fraction", "decimal", "percent"]) for _ in pcts]
        if len(set(forms)) < 3:
            continue
        values = [p / 100 for p in pcts]
        if any(f == "fraction" and v.denominator > 40 for f, v in zip(forms, values)):
            continue
        texts = [_show(v, f) for v, f in zip(values, forms)]
        if len(set(texts)) < 4:
            continue
        order = list(range(4))
        rng.shuffle(order)  # display order
        if order == sorted(order):
            continue
        # Ordering by the numbers as printed must differ from the true order, or that distractor is the key.
        printed = sorted(range(4), key=lambda i: _printed_number(texts[i]))
        if printed == list(range(4)):
            continue
        break

    event, verb, noun, unit, totals = rng.choice(_EVENTS)
    km = rng.choice(totals)
    names = rng.sample(_NAMES, 4)
    shown = ", ".join(f"{names[j]} {texts[order[j]]}" for j in range(4))
    stem = f"{event.format(total=km)} So far, four friends have {verb} these parts of the {noun}: {shown}."
    answer_text = ", ".join(texts)
    reversed_texts, swapped_texts = reversed_and_adjacent_swap(texts)
    printed_texts = [texts[i] for i in printed]
    d_a = pick_distractors(
        answer_text,
        [
            Distractor(", ".join(printed_texts), "compares-numbers-ignoring-form", printed_texts),
            Distractor(", ".join(reversed_texts), "reversed-order", reversed_texts),
            Distractor(", ".join(swapped_texts), "adjacent-swap", swapped_texts),
        ],
        3,
    )

    def km_of(p: Fraction) -> Fraction:
        return p * km / 100

    diff = km_of(pcts[-1] - pcts[0])
    check(diff.denominator == 1, "the gap is not a whole number of units")
    d_b = pick_distractors(
        str(int(diff)),
        [
            Distractor(str(int(km_of(pcts[-1] - pcts[1]))), "compared-wrong-pair-of-values", km_of(pcts[-1] - pcts[1])),
            Distractor(str(int(km_of(pcts[-2] - pcts[0]))), "compared-wrong-pair-of-values", km_of(pcts[-2] - pcts[0])),
            Distractor(str(int(km_of(pcts[-1]))), "omitted-a-term", km_of(pcts[-1])),
            Distractor(str(int(km_of(pcts[0]))), "omitted-a-term", km_of(pcts[0])),
        ],
        3,
    )
    if len(d_a) < 2 or len(d_b) < 3:
        raise _Retry
    check(all(Fraction(rational_text(v)) == v for v in values), "value lost in conversion")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Write the four values in order from the least distance to the greatest, "
                "each written the way it is shown above.",
                answer_kind="ordered_list",
                answer=list(texts),
                answer_text=answer_text,
                answer_latex=latex_answer(answer_text),
                marks=2,
                working=[
                    "Write each value as a percentage: "
                    + ", ".join(f"{t} = {float(p):g}%" for t, p in zip(texts, pcts))
                    + ".",
                    f"In order: {answer_text}.",
                ],
                distractors=d_a,
            ),
            dict(
                prompt=f"How many {unit} further has the front friend gone than the friend at the back?",
                answer_kind="number",
                answer=numeric_answer(diff),
                answer_text=str(int(diff)),
                answer_latex=latex_answer(str(int(diff))),
                marks=1,
                sensible_range=(0, None),
                working=[f"({float(pcts[-1]):g}% - {float(pcts[0]):g}%) of {km} {unit} = {int(diff)} {unit}."],
                distractors=d_b,
            ),
        ],
    )


def _build(rng: random.Random) -> dict:
    for _ in attempts(200, "a draw with enough distinct distractors"):
        try:
            return _attempt(rng)
        except _Retry:
            continue


@generator("exd_x_pr5_order_mixed_forms", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr5_order_mixed_forms(rng: random.Random, tier: str = "exam") -> dict:
    """Four close values, written as a fraction, decimals and percentages, ordered and then turned into a distance."""
    check(tier == "exam", "exd_x_pr5_order_mixed_forms: only the exam tier exists")
    return _build(rng)
