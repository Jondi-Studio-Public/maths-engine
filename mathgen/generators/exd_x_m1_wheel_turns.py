"""Exam source of `f10-circle-measures`: a wheel's circumference, the distance for a number of turns, and the number
of turns for a distance.

The practice tiers ask for one circumference or one area from a diameter. An exam question puts the circumference to
work: (a) circumference from a diameter or radius, (b) the distance rolled in a stated number of turns (a product and
a change of unit), (c) the number of whole turns needed to cover a longer distance in kilometres (a quotient across
two unit changes, with the count rounded down because only complete turns count).

## Construction

The prompt tells the learner to use pi = 3.14, so every value is exact decimal arithmetic. The diameter is a
multiple of 5 cm, which makes the circumference exact to one decimal place (3.14 x 5 = 15.7): the learner's (a) is
the exact figure, with no rounding to carry into (b) and (c). Distance values are kept off an exact rounding tie.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

PI = Fraction(314, 100)  # the value of pi the prompt tells the learner to use

# object, diameters (cm), wording of a ride
_WHEELS = [
    ("bicycle wheel", [50, 55, 60, 65, 70], "ride"),
    ("scooter wheel", [10, 15, 20, 25], "scoot"),
    ("wheelchair wheel", [55, 60, 65], "roll"),
    ("tractor tyre", [100, 110, 120, 130, 140, 150], "drive"),
    ("go-kart wheel", [25, 30, 35, 40], "drive"),
    ("pram wheel", [20, 25, 30], "walk"),
]
_RIDES_KM = [Fraction(n, 10) for n in range(11, 100) if n % 10]  # 1.1 .. 9.9 km, not whole


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _tenths(value: Fraction) -> Fraction:
    return round_half_up(value, 1)


@generator("exd_x_m1_wheel_turns", level=8, presets={"exam": {"tier": "exam"}})
def exd_x_m1_wheel_turns(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a wheel, a number of turns and a ride clear of every tie and repeat"):
        thing, diameters, verb = rng.choice(_WHEELS)
        d = Fraction(rng.choice(diameters))
        given_radius = d % 10 == 0 and rng.random() < 0.5
        turns = rng.randint(12, 240)
        km = rng.choice(_RIDES_KM)
        circumference = PI * d
        metres = circumference * turns / 100
        if (metres * 100) % 10 == 5:  # an exact tie at the 1 d.p. boundary
            continue
        whole_turns = math.floor(km * 100000 / circumference)
        size = f"a radius of {int(d / 2)} cm" if given_radius else f"a diameter of {int(d)} cm"
        stem = f"A {thing} has {size}. Use π ≈ 3.14."
        p_a = "Find the circumference of the wheel, in centimetres, correct to 1 decimal place."
        p_b = (
            f"The wheel makes {turns} complete turns. Using your answer to (a), "
            "how far does it roll, in metres, correct to 1 decimal place?"
        )
        km_text = stated_precision_string(km, 1)
        p_c = f"Another trip is {km_text} km long. How many complete turns does the wheel make over the trip?"
        c_text, m_text = stated_precision_string(circumference, 1), stated_precision_string(_tenths(metres), 1)
        if {c_text, m_text, str(whole_turns)} & _numbers(" ".join([stem, p_a, p_b, p_c])):
            continue
        break
    check(circumference == _tenths(circumference), "the circumference must be exact to 1 d.p.")

    ride_turns = km * 100000 / circumference

    def tagged(value: Fraction, tag: str | None = None):
        text = stated_precision_string(_tenths(value), 1)
        return Distractor(text, tag, Fraction(text)) if tag else text

    a_cands = [
        tagged(PI * (d / 2) ** 2, "area-formula-for-circumference"),
        tagged(PI * d * d, "area-formula-for-circumference"),
        tagged(PI * d / 2),  # half the circumference
    ]
    if not given_radius:
        a_cands.insert(2, tagged(PI * d * 2, "diameter-for-radius"))  # a diameter used as if it were the radius
    b_cands = [
        tagged(circumference * turns, "unit-factor-wrong-power"),
        tagged(circumference * turns / 1000, "unit-factor-wrong-power"),
        tagged(d * turns / 100, "diameter-for-circumference"),
        tagged(circumference * turns / 10, "unit-factor-wrong-power"),
    ]
    c_cands = [
        Distractor(str(math.ceil(ride_turns)), "rounded-count-up", math.ceil(ride_turns)),
        Distractor(
            str(math.floor(km * 1000 / circumference)), "unit-factor-wrong-power", math.floor(km * 1000 / circumference)
        ),
        Distractor(str(math.floor(km * 100000 / d)), "diameter-for-circumference", math.floor(km * 100000 / d)),
        Distractor(
            str(math.floor(km * 100 / circumference)), "unit-factor-wrong-power", math.floor(km * 100 / circumference)
        ),
    ]
    a_d = pick_distractors(c_text, a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(m_text, b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(str(whole_turns), c_cands, 3)
    check(len(a_d) >= 2 and len(b_d) >= 3 and len(c_d) >= 2, "too few distractors survive")
    check(all(abs(Fraction(str(x)) - circumference) > GRADER_TOLERANCE for x in a_d), "an (a) distractor is accepted")
    check(all(abs(Fraction(str(x)) - _tenths(metres)) > GRADER_TOLERANCE for x in b_d), "a (b) distractor is accepted")
    check(all(int(str(x)) != whole_turns for x in c_d), "a (c) distractor equals the key")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=p_a,
                answer_kind="number",
                answer=numeric_answer(circumference),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2 if given_radius else 1,
                sensible_range=(0, None),
                working=[
                    "C = πd = 2πr.",
                    f"C = {'2 × π × ' + str(int(d / 2)) if given_radius else 'π × ' + str(int(d))} = {c_text} cm.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=p_b,
                answer_kind="number",
                answer=numeric_answer(_tenths(metres)),
                answer_text=m_text,
                answer_latex=latex_answer(m_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*{turns}/100",
                working=[
                    f"{turns} × {c_text} cm = {float(turns * circumference):.1f} cm.",
                    f"÷ 100 gives {m_text} m (1 d.p.).",
                ],
                distractors=b_d,
            ),
            dict(
                prompt=p_c,
                answer_kind="number",
                answer=whole_turns,
                answer_text=str(whole_turns),
                answer_latex=latex_answer(str(whole_turns)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"floor(100000*{km.numerator}/{km.denominator}/part_a)",
                working=[
                    f"{km_text} km = {km * 100000} cm.",
                    f"{km * 100000} ÷ {c_text} = {float(ride_turns):.2f}; only complete turns count, so {whole_turns}.",
                ],
                distractors=c_d,
            ),
        ],
    )
