"""Exam question on decimal notation: tenths and hundredths named in words, then combined and converted.

Two lengths are each given as a count of tenths and a count of hundredths of a unit, where the hundredths count
is often ten or more (so it carries into the tenths). Parts (a) and (b) write the lengths as decimals, and
part (c) turns their total into the smaller unit, which needs the correct power of ten.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    exact_decimal_string,
    generator,
    numeric_answer,
    pick_distractors,
    stated_precision_string,
)

# (opening, first piece, second piece, unit word, small unit word, small units per unit)
_SETTINGS = (
    (
        "A walking track near Launceston has two sections.",
        "Section 1 is {t} {tenths} and {h} {hundredths} of a kilometre long.",
        "Section 2 is {h} {hundredths} and {t} {tenths} of a kilometre long.",
        "kilometres",
        "metres",
        1000,
    ),
    (
        "A craft teacher in Wollongong cuts two lengths of ribbon for medals.",
        "The first length is {t} {tenths} and {h} {hundredths} of a metre long.",
        "The second length is {h} {hundredths} and {t} {tenths} of a metre long.",
        "metres",
        "centimetres",
        100,
    ),
    (
        "A science teacher in Ballarat fills a bucket from two jugs.",
        "The first jug holds {t} {tenths} and {h} {hundredths} of a litre.",
        "The second jug holds {h} {hundredths} and {t} {tenths} of a litre.",
        "litres",
        "millilitres",
        1000,
    ),
)


def _decimal(tenths: int, hundredths: int) -> Fraction:
    return Fraction(tenths, 10) + Fraction(hundredths, 100)


@generator("exd_x_pr4_decimal_notation_lengths", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr4_decimal_notation_lengths(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", "exd_x_pr4_decimal_notation_lengths: only the exam tier exists")
    opening, first, second, unit, small, factor = rng.choice(_SETTINGS)
    for _ in attempts(1000, "two different amounts that need regrouping"):
        t1, t2 = rng.randint(1, 9), rng.randint(1, 9)
        h1, h2 = rng.randint(3, 98), rng.randint(3, 98)
        if h1 % 10 and h2 % 10 and max(h1, h2) >= 10 and (t1, h1) != (t2, h2) and _decimal(t1, h1) != _decimal(t2, h2):
            break
    v1, v2 = _decimal(t1, h1), _decimal(t2, h2)
    total = (v1 + v2) * factor

    # Independent route: everything in hundredths.
    check((t1 * 10 + h1 + t2 * 10 + h2) * factor == total * 100, "total disagrees")
    check(total.denominator == 1, "total in small units must be a whole number")

    a_text, b_text = stated_precision_string(v1, 2), stated_precision_string(v2, 2)
    c_text = str(int(total))

    def txt(v: Fraction) -> str:
        return exact_decimal_string(v, 4)

    def joined(t: int, h: int) -> Fraction:
        return Fraction(f"{t}.{h}")

    def tagged(v: Fraction, misconception: str | None) -> Distractor:
        return Distractor(txt(v), misconception, v)

    a_candidates = [
        tagged(joined(t1, h1), "joins-digits-as-decimal"),
        tagged(Fraction(t1 + h1, 100), None),  # every count taken as hundredths
        tagged(Fraction(t1 + h1, 10), "hundredths-read-as-tenths"),
    ]
    b_candidates = [
        tagged(joined(t2, h2), "joins-digits-as-decimal"),
        tagged(Fraction(t2 + h2, 100), None),
        tagged(Fraction(t2 + h2, 10), "hundredths-read-as-tenths"),
    ]
    c_candidates = [
        tagged((v1 + v2) * (factor // 10), "converts-with-wrong-power-of-ten"),
        tagged((v1 + v2) * factor * 10, "converts-with-wrong-power-of-ten"),
        tagged((joined(t1, h1) + joined(t2, h2)) * factor, "joins-digits-as-decimal"),
    ]
    a_d = pick_distractors(a_text, a_candidates, 3, tolerance=Fraction(1, 200))
    b_d = pick_distractors(b_text, b_candidates, 3, tolerance=Fraction(1, 200))
    c_d = pick_distractors(c_text, c_candidates, 3, tolerance=Fraction(1, 2))
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "too few distractors")

    def words(t: int, h: int) -> dict:
        return dict(
            t=t, h=h, tenths="tenth" if t == 1 else "tenths", hundredths="hundredth" if h == 1 else "hundredths"
        )

    stem = f"{opening} {first.format(**words(t1, h1))} {second.format(**words(t2, h2))}"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Write the first amount as a decimal number of {unit}.",
                answer_kind="number",
                answer=numeric_answer(v1),
                answer_text=a_text,
                marks=1,
                sensible_range=(0, None),
                working=[
                    f"{t1} tenths = {Fraction(t1, 10)} and {h1} hundredths = {Fraction(h1, 100)}.",
                    f"Total = {a_text} {unit}.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=f"Write the second amount as a decimal number of {unit}.",
                answer_kind="number",
                answer=numeric_answer(v2),
                answer_text=b_text,
                marks=1,
                sensible_range=(0, None),
                working=[f"{h2} hundredths + {t2} tenths = {b_text} {unit}."],
                distractors=b_d,
            ),
            dict(
                prompt=f"Find the total of the two amounts in {small}.",
                answer_kind="number",
                answer=int(total),
                answer_text=c_text,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{factor}*(part_a + part_b)",
                working=[f"({a_text} + {b_text}) × {factor} = {c_text} {small}."],
                distractors=c_d,
            ),
        ],
    )
