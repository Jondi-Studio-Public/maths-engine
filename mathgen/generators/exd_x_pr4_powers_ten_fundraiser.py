"""Exam question on multiplying and dividing decimals by powers of ten, in a packing context.

A fundraiser packs 100, 1000 or 10 000 identical items (each a decimal mass in kilograms) into 10 or 100
boxes. Part (a) finds the total mass (multiply by a power of ten), part (b) the mass in each box (divide by a
smaller power of ten), and part (c) that mass in grams (multiply by 1000). The learner must keep three
different shifts of the decimal point straight and choose each direction.

Each item's mass is drawn as a whole number of tenths of a gram (so it has 4 decimal places in kilograms) and
every figure is an exact `Fraction`.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    check,
    draw_until,
    exact_decimal_string,
    generator,
    numeric_answer,
    pick_distractors,
    stated_precision_string,
    stated_precision_tolerance,
)

# (singular, plural, range of the mass in units of 0.0001 kg, the place that sells or packs them, largest power of ten
# that is a believable count of them)
_ITEMS = (
    ("mandarin", "mandarins", (600, 950), "A fruit stall in Griffith", 3),
    ("egg", "eggs", (520, 680), "A free-range farm near Bendigo", 3),
    ("lamington", "lamingtons", (400, 700), "A school fundraiser in Rockhampton", 2),
    ("tennis ball", "tennis balls", (560, 590), "A tennis club in Toowoomba", 3),
    ("macadamia nut", "macadamia nuts", (60, 95), "A processing shed near Bundaberg", 4),
    ("apple", "apples", (1300, 2000), "An orchard in the Huon Valley", 3),
)


def _text(value: Fraction) -> str:
    return exact_decimal_string(value, 8)


@generator("exd_x_pr4_powers_ten_fundraiser", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr4_powers_ten_fundraiser(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", "exd_x_pr4_powers_ten_fundraiser: only the exam tier exists")
    singular, plural, (lo, hi), place, max_power = rng.choice(_ITEMS)
    n = draw_until(rng, lambda r: r.randint(lo, hi), lambda v: v % 10 != 0, what="a mass in tenths of a gram")
    p = rng.randint(2, max_power)
    q = rng.randint(1, p - 1)
    each = Fraction(n, 10_000)
    count, boxes = 10**p, 10**q
    total = each * count
    per_box = total / boxes
    grams = per_box * 1000

    # Independent route: digit strings. n * 10^p / 10^4 and n * 10^(p - q - 1) grams.
    check(grams == n * 10 ** (p - q - 1) and grams.denominator == 1, "grams disagree")
    check(per_box == Fraction(n, 10_000) * 10 ** (p - q), "per box disagrees")

    a_text, b_text, c_text = _text(total), _text(per_box), _text(grams)

    def candidates(key_text, options):
        """Each `(value, misconception)` spelled at the key's own number of decimal places. A slip that needs more
        places than the key has cannot be spelled like it, so it is not offered; the point shifted the other way
        (a longer number, padded with a zero) is."""
        places = len(key_text.partition(".")[2])
        out = []
        for v, misconception in options:
            if (v * 10**places).denominator == 1:
                out.append(Distractor(stated_precision_string(v, places) if places else str(int(v)), misconception, v))
        return out

    ten = Fraction(10)
    a_candidates = candidates(
        a_text,
        [
            (each / count, "decimal-point-moved-wrong-way"),
            (total * ten, "decimal-point-moved-wrong-distance"),
            (total / ten, "decimal-point-moved-wrong-distance"),
            (total * ten**2, "decimal-point-moved-wrong-distance"),
            (total * ten**3, "decimal-point-moved-wrong-distance"),
            (total / ten**2, "decimal-point-moved-wrong-distance"),
        ],
    )
    b_candidates = candidates(
        b_text,
        [
            (total * boxes, "decimal-point-moved-wrong-way"),
            (per_box / ten, "decimal-point-moved-wrong-distance"),
            (per_box * ten, "decimal-point-moved-wrong-distance"),
            (per_box * ten**2, "decimal-point-moved-wrong-distance"),
            (per_box * ten**3, "decimal-point-moved-wrong-distance"),
            (per_box / ten**2, "decimal-point-moved-wrong-distance"),
        ],
    )
    c_candidates = candidates(
        c_text,
        [
            (per_box * 100, "converts-with-wrong-power-of-ten"),
            (per_box / 1000, "decimal-point-moved-wrong-way"),
            (per_box * 10_000, "converts-with-wrong-power-of-ten"),
            (grams * ten, "converts-with-wrong-power-of-ten"),
            (grams * ten**2, "converts-with-wrong-power-of-ten"),
            (grams / ten, "converts-with-wrong-power-of-ten"),
            (grams / ten**2, "converts-with-wrong-power-of-ten"),
        ],
    )
    a_d = pick_distractors(a_text, a_candidates, 3, tolerance=stated_precision_tolerance(8))
    b_d = pick_distractors(b_text, b_candidates, 3, tolerance=stated_precision_tolerance(8))
    c_d = pick_distractors(c_text, c_candidates, 3, tolerance=stated_precision_tolerance(8))
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "too few distractors")

    stem = f"{place} packs {count:,} {plural}, each with a mass of {_text(each)} kg, equally into {boxes} boxes."
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"What is the total mass of the {plural}, in kilograms?",
                answer_kind="number",
                answer=numeric_answer(total),
                answer_text=a_text,
                marks=1,
                sensible_range=(0, None),
                working=[f"{_text(each)} × {count} = {a_text} kg."],
                distractors=a_d,
            ),
            dict(
                prompt="What is the mass in each box, in kilograms?",
                answer_kind="number",
                answer=numeric_answer(per_box),
                answer_text=b_text,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a/{boxes}",
                working=[f"{a_text} ÷ {boxes} = {b_text} kg."],
                distractors=b_d,
            ),
            dict(
                prompt="What is the mass in each box, in grams?",
                answer_kind="number",
                answer=numeric_answer(grams),
                answer_text=c_text,
                marks=1,
                sensible_range=(0, None),
                follow_through="1000*part_b",
                working=[f"{b_text} × 1000 = {c_text} g."],
                distractors=c_d,
            ),
        ],
    )
