"""Exam-tier source for `hookes-law`: a spring measured under two loads, natural length unknown.

The stem gives the length of a spring under two different forces. The natural length is not given, so the
learner cannot use F = k x directly: the extension is not the length.

* (a) the spring constant from the change in force over the change in length, k = (F2 - F1) / (l2 - l1);
* (b) the natural length, l0 = l1 - F1 / k, followed through from (a);
* (c) the force needed to hold the spring at a third length, F3 = k (l3 - l0), followed through from (a) and (b).

Built backwards from a whole-number spring constant, a natural length and three extensions in whole centimetres,
so every key is exact: (a) is an integer, (b) is exact to 2 decimal places and (c) to 1 decimal place. The draw
keeps the two measured forces different, the third length different from both measured lengths, and every wrong
route (the length used as the extension, the extension from the wrong measured length, the natural length added
instead of subtracted) different from its key.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import (
    check,
    generator,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_SETUPS = [
    "A spring obeys Hooke's law. Its length is {l1} m when a force of {f1} N is applied to it and {l2} m when the "
    "force is {f2} N.",
    "In a laboratory, a spring that obeys Hooke's law is {l1} m long under a load of {f1} N and {l2} m long "
    "under a load of {f2} N.",
    "A spring balance spring obeys Hooke's law. It is {l1} m long when stretched by a force of {f1} N, and {l2} m "
    "long when the force is increased to {f2} N.",
]
_A_PROMPTS = ["Find the spring constant, in N/m.", "Calculate the stiffness k of the spring, in N/m."]
_B_PROMPTS = [
    "Find the natural length of the spring, in metres.",
    "Hence find the natural (unstretched) length of the spring, in metres.",
]
_C_PROMPTS = [
    "Find the force needed to hold the spring at a length of {l3} m, in newtons.",
    "What force would stretch the spring to a length of {l3} m? Give your answer in newtons.",
]


def _fx(value: Fraction, places: int) -> str:
    return stated_precision_string(round_half_up(value, places), places)


def _rs(value: Fraction) -> str:
    return f"({value.numerator}/{value.denominator})"


def _draw(rng: random.Random):
    for _ in range(2000):
        k = rng.choice(range(20, 301, 10))
        l0 = Fraction(rng.randint(20, 80), 100)
        e1 = Fraction(rng.randint(3, 20), 100)
        e2 = e1 + Fraction(rng.randint(3, 20), 100)
        e3 = Fraction(rng.randint(2, 40), 100)
        if e3 in (e1, e2):
            continue
        f1, f2, f3 = k * e1, k * e2, k * e3
        if any((f * 10).denominator != 1 for f in (f1, f2, f3)):
            continue
        return k, l0, e1, e2, e3, f1, f2, f3, l0 + e1, l0 + e2, l0 + e3
    raise RuntimeError("no suitable spring in 2000 draws")


@generator("exd_x_u1_spring_two_loads", level=19, presets={"exam": dict(tier="exam")})
def exd_x_u1_spring_two_loads(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: spring constant, natural length and a third force from two measured lengths."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(200):
        question = _build(rng)
        if question is not None:
            return question
    raise RuntimeError("no spring question with enough distractors in 200 builds")


def _build(rng: random.Random) -> dict | None:
    k, l0, e1, e2, e3, f1, f2, f3, l1, l2, l3 = _draw(rng)

    # Independent route: the line F = k (l - l0) through the two measured points, read at the third length.
    slope = (f2 - f1) / (l2 - l1)
    check(slope == k, "spring constant disagrees with the slope between the two measurements")
    check(l1 - f1 / slope == l0 and l2 - f2 / slope == l0, "natural length disagrees with the two measurements")
    check(f1 + slope * (l3 - l1) == f3, "force at the third length disagrees with the line")

    def dist(value: Fraction, tag, places: int):
        rounded = round_half_up(value, places)
        return Distractor(_fx(rounded, places), tag, exact_value(rounded))

    def whole(value: Fraction, tag):
        # The key k is a whole number, so a wrong route is shown to the nearest whole number as well.
        rounded = round_half_up(value, 0)
        return Distractor(str(int(rounded)), tag, exact_value(rounded))

    a_cands = [
        whole(f2 / l2, "length-used-as-extension"),
        whole(f1 / l1, "length-used-as-extension"),
        whole((f2 - f1) / l2, "length-used-as-extension"),
        whole((f1 + f2) / (l1 + l2), None),
    ]
    b_cands = [
        dist(l1 + f1 / k, "natural-length-sign-reversed", 2),
        dist(f1 / k, "extension-given-for-natural-length", 2),
        dist(l1, "natural-length-taken-as-measured-length", 2),
        dist(l2 - f1 / k, "force-and-length-from-different-measurements", 2),
    ]
    c_cands = [
        dist(k * l3, "length-used-as-extension", 1),
        dist(k * (l3 - l1), "extension-from-wrong-length", 1),
        dist(f2 * l3 / l2, "force-proportional-to-length", 1),
        dist(k * (l3 - l2), "extension-from-wrong-length", 1),
    ]
    key_b = _fx(l0, 2)
    key_c = _fx(f3, 1)
    a_dist = pick_distractors(str(k), a_cands, 3)
    b_dist = pick_distractors(key_b, b_cands, 3, tolerance=stated_precision_tolerance(2) * 2)
    c_dist = pick_distractors(key_c, c_cands, 3, tolerance=stated_precision_tolerance(1) * 2)
    if min(len(a_dist), len(b_dist), len(c_dist)) < 2:
        return None

    fmt = dict(l1=_fx(l1, 2), l2=_fx(l2, 2), f1=_fx(f1, 1), f2=_fx(f2, 1), l3=_fx(l3, 2))
    return dict(
        text=rng.choice(_SETUPS).format(**fmt),
        parts=[
            dict(
                prompt=rng.choice(_A_PROMPTS),
                answer_kind="number",
                answer=k,
                answer_text=str(k),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"The force changes by {_fx(f2 - f1, 1)} N while the length changes by {_fx(l2 - l1, 2)} m.",
                    f"k = {_fx(f2 - f1, 1)} ÷ {_fx(l2 - l1, 2)} = {k} N/m.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=rng.choice(_B_PROMPTS),
                answer_kind="number",
                answer=numeric_answer(l0),
                answer_text=key_b,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{_rs(l1)} - {_rs(f1)}/part_a",
                working=[
                    f"At {fmt['f1']} N the extension is {fmt['f1']} ÷ {k} = {_fx(e1, 2)} m.",
                    f"Natural length = {fmt['l1']} - {_fx(e1, 2)} = {key_b} m.",
                ],
                distractors=b_dist,
            ),
            dict(
                prompt=rng.choice(_C_PROMPTS).format(**fmt),
                answer_kind="number",
                answer=numeric_answer(round_half_up(f3, 1)),
                answer_text=key_c,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*({_rs(l3)} - part_b)",
                working=[
                    f"Extension = {fmt['l3']} - {key_b} = {_fx(e3, 2)} m.",
                    f"F = k x = {k} × {_fx(e3, 2)} = {key_c} N.",
                ],
                distractors=c_dist,
            ),
        ],
    )
