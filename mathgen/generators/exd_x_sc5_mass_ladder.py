"""Exam-tier source for `senior-eview-metric-units-of-mass-and-weight-their`: moving down the whole mass ladder.

A batch of an active ingredient is measured in kilograms and made into tablets each holding a stated number of
milligrams. The learner has to cross the whole ladder (kg, g, mg) in the right direction, twice:

* (a) the mass of the batch in milligrams (2 marks);
* (b) the number of tablets the batch makes (2 marks, follow-through from (a));
* (c) the mass of the ingredient in one pack of tablets, in grams (1 mark, independent).

The batch and tablet strength are drawn so that the number of tablets is a whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    check,
    draw_until,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
)

# (ingredient, name of the product, tablet strengths in mg, pack sizes)
_PRODUCTS = [
    ("paracetamol", "paracetamol tablets", "tablets", [200, 250, 400, 500], [20, 24, 50, 100]),
    ("vitamin C powder", "vitamin C tablets", "tablets", [100, 125, 200, 250, 500], [30, 60, 100]),
    ("calcium carbonate", "calcium tablets", "tablets", [125, 250, 500, 800], [30, 60, 120]),
    ("ibuprofen", "ibuprofen capsules", "capsules", [200, 400], [20, 24, 48]),
    ("zinc powder", "zinc tablets", "tablets", [25, 50, 100], [30, 60, 100]),
]
_OPENINGS = [
    "A pharmacy supplier uses {k} kg of {ingredient} to make a batch of {product}. Each one contains {m} mg of {ingredient}.",
    "Each of a company's {product} contains {m} mg of {ingredient}. One batch is made from {k} kg of {ingredient}.",
    "A laboratory makes a batch of {product} from {k} kg of {ingredient}, with {m} mg of {ingredient} in each.",
]
_PACKS = [
    "The {noun} are sold in packs of {p}.",
    "A pack holds {p} of the {noun}.",
]


def _cands(*pairs) -> list[Distractor]:
    """Tagged distractors for `(value, misconception)` pairs, skipping any value that is not a short decimal."""
    out = []
    for value, tag in pairs:
        value = Fraction(value)
        try:
            text = exact_decimal_string(value, 3)
        except ValueError:
            continue
        out.append(Distractor(text, tag, exact_value(value)))
    return out


@generator("exd_x_sc5_mass_ladder", level=9, presets={"exam": dict(tier="exam")})
def exd_x_sc5_mass_ladder(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: kg -> mg, divide by a tablet strength, then mg -> g for one pack."""
    check(tier == "exam", f"unknown tier {tier!r}")
    ingredient, product, noun, strengths, packs = rng.choice(_PRODUCTS)
    m = rng.choice(strengths)
    p = rng.choice(packs)

    def draw(r: random.Random):
        return r.randint(2, 15)

    k = draw_until(rng, draw, lambda v: (v * 1_000_000) % m == 0, what="a batch with a whole number of tablets")
    batch_mg = k * 1_000_000
    tablets = batch_mg // m
    pack_g = Fraction(p * m, 1000)
    check(batch_mg == k * 1000 * 1000, "ladder: kg -> g -> mg")
    check(tablets * m == batch_mg, "tablet count is not exact")

    a_cands = _cands(
        (k * 1000, "conversion-power-of-ten-slip"),
        (k * 100_000, "conversion-power-of-ten-slip"),
        (k * 10_000_000, "conversion-power-of-ten-slip"),
        (k, "forgot-to-convert-units"),
    )
    b_cands = _cands(
        (Fraction(k * 1000, m), "conversion-power-of-ten-slip"),
        (Fraction(k * 100_000, m), "conversion-power-of-ten-slip"),
        (Fraction(k * 10_000_000, m), "conversion-power-of-ten-slip"),
        (k * 1_000_000 * m, "multiplied-instead-of-divided"),
    )
    c_cands = _cands(
        (p * m, "forgot-to-convert-units"),
        (Fraction(p * m, 100), "conversion-power-of-ten-slip"),
        (Fraction(p * m, 10000), "conversion-power-of-ten-slip"),
        (Fraction(m, 1000), "gave-amount-for-one-item"),
    )
    a_text, b_text, c_text = str(batch_mg), str(tablets), exact_decimal_string(pack_g, 3)
    a_d = pick_distractors(a_text, a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(b_text, b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(c_text, c_cands, 3, tolerance=GRADER_TOLERANCE)
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "too few distractors survive")

    stem = rng.choice(_OPENINGS).format(k=k, ingredient=ingredient, product=product, m=m)
    pack_sentence = rng.choice(_PACKS).format(p=p, noun=noun)
    return dict(
        text=f"{stem} {pack_sentence}",
        parts=[
            dict(
                prompt="Write the mass of the batch of ingredient in milligrams.",
                answer_kind="number",
                answer=batch_mg,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"{k} kg = {k} × 1000 g = {k * 1000} g.",
                    f"{k * 1000} g = {k * 1000} × 1000 mg = {a_text} mg.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=f"How many {noun} does the batch make?",
                answer_kind="number",
                answer=tablets,
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a/{m}",
                working=[f"Number made = {a_text} / {m} = {b_text}."],
                distractors=b_d,
            ),
            dict(
                prompt="How many grams of the ingredient are in one pack?",
                answer_kind="number",
                answer=numeric_answer(pack_g),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=1,
                sensible_range=(0, None),
                working=[f"One pack: {p} × {m} = {p * m} mg.", f"{p * m} mg / 1000 = {c_text} g."],
                distractors=c_d,
            ),
        ],
    )
