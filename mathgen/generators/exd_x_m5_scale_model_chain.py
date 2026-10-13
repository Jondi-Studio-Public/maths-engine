"""Exam-tier source for scale factors of similar solids: a scale model and the full-size object.

A model is built at a stated height in centimetres and the full-size object is a stated height in metres, so
finding the scale factor already needs a unit conversion. The learner must then

(a) obtain the linear scale factor, mixing centimetres and metres,
(b) scale the model's surface area (in cm²) to the real object's, in m², and
(c) scale the model's capacity (in mL) to the real object's, in litres.

Part (b) needs the square of the factor and the 10 000 cm² in a m²; part (c) needs the cube and the 1000 mL in
a litre. Built backwards from a whole-number scale factor.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    VerificationError,
    attempts,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
)

from .exd_x_m5_support import clear_of_boundary, display, keys_absent, rounded, tagged_distractors

_OBJECTS = (
    ("a rainwater tank", "tank", "painted", "holds"),
    ("a grain silo", "silo", "painted", "holds"),
    ("a water tower", "tower", "painted", "holds"),
    ("a fuel storage tank", "tank", "coated", "holds"),
)

_FIRST = (
    "A scale model of {thing} is {h1} cm tall. The real {short} is {h2} m tall, and the model is "
    "geometrically similar to it. The outside surface of the model has an area of {area} cm² and the model "
    "{holds} {vol} mL.",
    "An engineer makes a model of {thing}. The model is {h1} cm tall and the real {short} is {h2} m tall. "
    "The two are similar solids. The model's outside surface area is {area} cm² and the model {holds} "
    "{vol} mL.",
    "{Thing} is {h2} m tall. A similar model of it, {h1} cm tall, has an outside surface area of {area} cm² "
    "and {holds} {vol} mL.",
)
_A_PROMPT = (
    "Find the scale factor from the model to the real {short}, as a whole number.",
    "What is the linear scale factor that enlarges the model to the real {short}? Give a whole number.",
)
_B_PROMPT = (
    "The outside of the real {short} is to be {verb}. Using your scale factor from (a), find its outside "
    "surface area in square metres, correct to 1 decimal place.",
    "Using your scale factor from (a), find the outside surface area of the real {short} in square metres, "
    "correct to 1 decimal place. (1 m² = 10 000 cm²)",
)
_C_PROMPT = (
    "Using your scale factor from (a), find how many litres the real {short} holds, to the nearest litre. "
    "(1 L = 1000 mL)",
    "Find the capacity of the real {short} in litres, to the nearest litre, using your scale factor from (a). "
    "(1 L = 1000 mL)",
)


@generator("exd_x_m5_scale_model_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m5_scale_model_chain(rng, tier: str = "exam") -> dict:
    """Exam tier: linear scale factor across cm and m, then the real surface area and capacity."""
    if tier != "exam":
        raise VerificationError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a model whose answers are clear of rounding boundaries"):
        k = rng.choice([8, 12, 15, 16, 18, 20, 24, 25, 30, 32, 36, 40, 45, 48, 50, 60, 75])
        h1 = rng.randint(6, 40)
        area = rng.randint(120, 900)
        vol = rng.randint(120, 950)
        h2 = Fraction(k * h1, 100)
        if h2.denominator not in (1, 2, 4, 5, 10, 20, 25, 50, 100):
            continue
        real_area = Fraction(area * k * k, 10_000)
        real_litres = Fraction(vol * k**3, 1000)
        if not clear_of_boundary(real_area, 1) or not clear_of_boundary(real_litres, 0):
            continue
        a_text, b_text, c_text = str(k), display(real_area, 1), display(real_litres, 0)
        thing, short, verb, holds = rng.choice(_OBJECTS)
        h2_text = display(h2, 2).rstrip("0").rstrip(".")
        first = rng.choice(_FIRST).format(
            thing=thing,
            Thing=thing[0].upper() + thing[1:],
            short=short,
            h1=h1,
            h2=h2_text,
            area=area,
            vol=vol,
            holds=holds,
        )
        prompts = [
            rng.choice(_A_PROMPT).format(short=short),
            rng.choice(_B_PROMPT).format(short=short, verb=verb),
            rng.choice(_C_PROMPT).format(short=short),
        ]
        if keys_absent(first, prompts, [a_text, b_text, c_text]):
            break

    a_dis = pick_distractors(
        a_text,
        [
            Distractor(str(k * 10), None, k * 10),  # took 1 m as 1000 cm
            Distractor(str(k * 100), None, k * 100),  # converted m to cm twice
            Distractor(str(k * h1 - h1), None, k * h1 - h1),  # subtracted the heights instead of dividing
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    b_dis = tagged_distractors(
        b_text,
        1,
        [
            (Fraction(area * k, 10_000), None),  # scaled the area by k, not k squared
            (Fraction(area * k**3, 10_000), None),  # used the volume exponent
            (Fraction(area * k * k, 100), None),  # 100 cm² in a m² instead of 10 000
            (Fraction(area * k * k), None),  # left in cm²
        ],
    )
    c_dis = tagged_distractors(
        c_text,
        0,
        [
            (Fraction(vol * k * k, 1000), None),  # used the area exponent
            (Fraction(vol * k, 1000), None),
            (Fraction(vol * k**3), None),  # left in mL
            (Fraction(vol * k**3, 100), None),  # 100 mL in a litre
        ],
    )
    return dict(
        text=first,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=k,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[f"Real height = {h2_text} m = {h2 * 100} cm; scale factor = {h2 * 100} ÷ {h1} = {k}."],
                distractors=a_dis,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=numeric_answer(rounded(real_area, 1)),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{area}*part_a**2/10000",
                working=[f"Areas scale by k²: {area} × {k}² = {area * k * k} cm² = {b_text} m² (1 d.p.)."],
                distractors=b_dis,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=numeric_answer(rounded(real_litres, 0)),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{vol}*part_a**3/1000",
                working=[f"Volumes scale by k³: {vol} × {k}³ = {vol * k**3} mL = {c_text} L."],
                distractors=c_dis,
            ),
        ],
    )
