"""Exam-tier source for `senior-estimate-volume-and-capacity-of-various-objects`: the capacity of a container.

A rectangular container is described by its inside dimensions in centimetres, each as a measured value such as 83 cm.
The learner rounds every measurement to the nearest 10 cm, so this is an estimate, and then turns the estimated
volume in cubic centimetres into a capacity in litres, before working with a partial fill and a top-up in millilitres:

* (a) the estimated capacity in litres, from the rounded measurements (2 marks);
* (b) the litres held, by the same estimate, when filled to a stated depth (2 marks, follow-through from (a));
* (c) the millilitres that must still be added when the container already holds a stated number of litres
  (1 mark, follow-through from (b)).

Only capacity is asked, never the volume of the container's material, as the skill's description requires.
The rounded dimensions (multiples of 10 cm) are drawn first, so every estimate is a whole number of litres; each is
then shown as a measured value up to 4 cm either side, never 5, so the rounding is unambiguous.
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
    pick_distractors,
    round_half_up,
)

# (description with slots, length range, width range, height range, contents, deepest fill allowed as a share)
_CONTAINERS = [
    (
        "A rectangular aquarium measures {l} cm long, {w} cm wide and {h} cm high on the inside.",
        (60, 120),
        (30, 45),
        (30, 50),
        "water",
    ),
    ("The inside of an esky is {l} cm long, {w} cm wide and {h} cm deep.", (50, 80), (30, 40), (30, 45), "water"),
    (
        "A stock water trough is {l} cm long, {w} cm wide and {h} cm deep on the inside.",
        (100, 200),
        (40, 60),
        (40, 60),
        "water",
    ),
    (
        "A raised garden bed is {l} cm long, {w} cm wide and {h} cm deep.",
        (120, 240),
        (60, 120),
        (30, 45),
        "potting mix",
    ),
]
_DEPTH = [
    "It is filled with {stuff} to a depth of {d} cm.",
    "{stuff_cap} is added until it is {d} cm deep.",
]
_TOPUP = [
    "Before filling, it already holds {w} L of {stuff}. How many millilitres of {stuff} must be added to reach this depth?",
    "It already holds {w} L of {stuff}. How many millilitres more are needed to reach this depth?",
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


@generator("exd_x_sc5_tank_capacity", level=9, presets={"exam": dict(tier="exam")})
def exd_x_sc5_tank_capacity(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: capacity from cm dimensions, a partial fill, and a top-up in millilitres."""
    check(tier == "exam", f"unknown tier {tier!r}")
    template, l_range, w_range, h_range, stuff = rng.choice(_CONTAINERS)

    def draw(r: random.Random):
        l = r.randrange(l_range[0], l_range[1] + 1, 10)
        w = r.randrange(w_range[0], w_range[1] + 1, 10)
        h = r.randrange(h_range[0], h_range[1] + 1, 10)
        d = r.randrange(10, h, 10)
        return l, w, h, d

    def acceptable(v) -> bool:
        l, w, h, d = v
        return (l * w * h) % 1000 == 0 and (l * w * d) % 1000 == 0 and d * 4 >= h and d < h

    l, w, h, d = draw_until(rng, draw, acceptable, what="whole-litre dimensions")
    capacity = l * w * h // 1000
    held = l * w * d // 1000
    check(held * h == capacity * d, "depth fraction disagrees with the direct volume")
    start = rng.randint(max(1, held // 5), held - max(1, held // 5))
    topup_ml = (held - start) * 1000
    check(topup_ml > 0, "nothing to top up")

    # The measured values the learner sees; each rounds to the drawn multiple of 10 (never a half).
    ml, mw, mh, md = (v + rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for v in (l, w, h, d))
    check(all(round_half_up(Fraction(m, 10), 0) * 10 == v for m, v in ((ml, l), (mw, w), (mh, h), (md, d))), "rounding")
    stem = (
        template.format(l=ml, w=mw, h=mh)
        + " "
        + rng.choice(_DEPTH).format(stuff=stuff, stuff_cap=stuff[0].upper() + stuff[1:], d=md)
        + " To estimate, round every measurement to the nearest 10 cm."
    )
    exact_l = Fraction(ml * mw * mh, 1000)
    a_text, b_text, c_text = str(capacity), str(held), str(topup_ml)

    a_cands = _cands(
        (exact_l, "used-exact-values-not-rounded"),
        (l * w * h, "forgot-to-convert-units"),
        (Fraction(l * w * h, 100), "conversion-power-of-ten-slip"),
        (Fraction(l * w * h, 10000), "conversion-power-of-ten-slip"),
    )
    b_cands = _cands(
        (capacity, "ignored-fill-depth"),
        (l * w * d, "forgot-to-convert-units"),
        (Fraction(capacity * h, d), "inverted-the-fraction"),
        (Fraction(capacity * (h - d), h), "used-the-empty-part"),
    )
    c_cands = _cands(
        (held - start, "forgot-to-convert-units"),
        ((held - start) * 100, "conversion-power-of-ten-slip"),
        ((capacity - start) * 1000, "ignored-fill-depth"),
        ((held + start) * 1000, "added-instead-of-subtracted"),
    )
    a_d = pick_distractors(a_text, a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(b_text, b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(c_text, c_cands, 3, tolerance=GRADER_TOLERANCE)
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "too few distractors survive")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Estimate the capacity of the container when it is full, in litres, using the rounded measurements. (1 L = 1000 cm³.)",
                answer_kind="number",
                answer=capacity,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Rounded: {l} cm, {w} cm and {h} cm.",
                    f"Volume = {l} x {w} x {h} = {l * w * h} cm³.",
                    f"{l * w * h} cm³ / 1000 = {a_text} L.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="Using the same estimate, how many litres does it hold at the stated depth?",
                answer_kind="number",
                answer=held,
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*{d}/{h}",
                working=[
                    f"The rounded depth is {d} cm of {h} cm, which is {d}/{h} of the height, so the amount is {d}/{h} x {a_text} = {b_text} L."
                ],
                distractors=b_d,
            ),
            dict(
                prompt=rng.choice(_TOPUP).format(w=start, stuff=stuff),
                answer_kind="number",
                answer=topup_ml,
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"(part_b - {start})*1000",
                working=[f"{b_text} - {start} = {held - start} L.", f"{held - start} L x 1000 = {c_text} mL."],
                distractors=c_d,
            ),
        ],
    )
