"""Exam-tier source for volume and capacity: a cylindrical tank, filling it, and a rectangular replacement.

The stem gives a cylindrical tank's diameter and height in metres, the rate a pump fills it in litres per
minute, and the square base of a second, rectangular tank. The learner must

(a) find the capacity of the cylinder in litres (a volume in cubic metres, then 1000 L per cubic metre),
(b) find how long the pump takes to fill it to a stated percentage, and
(c) find the height a square-based tank needs to hold the same amount.

Built backwards from whole-tenth dimensions. A draw is kept only when the answers read the same whether the
learner rounds the capacity to the nearest litre before (b) and (c) or carries the exact value.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import VerificationError, attempts, generator, latex_answer, numeric_answer

from .exd_x_m5_support import PI, clear_of_boundary, display, keys_absent, rounded, tagged_distractors

_OBJECTS = (
    ("A rainwater tank on a farm near Dubbo", "tank"),
    ("A cylindrical water tank at a camping ground", "tank"),
    ("A storage tank at a winery in the Barossa Valley", "tank"),
    ("A cylindrical tank at a school in Cairns", "tank"),
    ("A cylindrical water tank behind a Perth home", "tank"),
)
_FIRST = (
    "{obj} is a cylinder with a diameter of {d} m and a height of {h} m. A pump fills it from empty at "
    "{rate} litres per minute. A replacement tank is to have a square base of side {s} m.",
    "{obj} has the shape of a cylinder {d} m across and {h} m high. It is filled from empty by a pump "
    "delivering {rate} L every minute. The replacement tank will have a square base with sides of {s} m.",
)
_A_PROMPT = (
    "Find the capacity of the tank in litres, to the nearest litre. (1 m³ = 1000 L)",
    "Calculate how many litres the tank holds when full, to the nearest litre. (1 m³ = 1000 L)",
)
_B_PROMPT = (
    "Using your answer to (a), find how many minutes the pump takes to fill the tank to {p}% of its capacity, "
    "to the nearest minute.",
    "Use your capacity from (a) to find the time, to the nearest minute, for the pump to fill the tank "
    "to {p}% of its capacity.",
)
_C_PROMPT = (
    "The replacement rectangular tank must hold the same capacity as the cylinder. Using your answer to (a), "
    "find its height in metres, correct to 1 decimal place.",
    "Using your capacity from (a), find the height, in metres and correct to 1 decimal place, that the "
    "square-based replacement tank needs to hold the same amount.",
)


@generator("exd_x_m5_tank_capacity_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m5_tank_capacity_chain(rng, tier: str = "exam") -> dict:
    """Exam tier: capacity of a cylinder, time to part-fill it, and the height of a square-based equivalent."""
    if tier != "exam":
        raise VerificationError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a tank whose answers agree along every rounding route"):
        d = Fraction(rng.randint(16, 40), 10)
        h = Fraction(rng.randint(12, 36), 10)
        s = Fraction(rng.randint(12, 30), 10)
        rate = 10 * rng.randint(5, 30)
        p = rng.choice([60, 70, 75, 80, 85, 90])
        litres = PI * (d / 2) ** 2 * h * 1000
        cap = rounded(litres, 0)
        minutes = Fraction(p, 100) * litres / rate
        height = litres / 1000 / (s * s)
        if not (
            clear_of_boundary(litres, 0, Fraction(1, 20))
            and clear_of_boundary(minutes, 0, Fraction(1, 20))
            and clear_of_boundary(height, 1, Fraction(1, 20))
            and rounded(minutes, 0) == rounded(Fraction(p, 100) * cap / rate, 0)
            and rounded(height, 1) == rounded(cap / 1000 / (s * s), 1)
        ):
            continue
        a_text, b_text, c_text = display(cap, 0), display(minutes, 0), display(height, 1)
        obj, _ = rng.choice(_OBJECTS)
        first = rng.choice(_FIRST).format(obj=obj, d=display(d, 1), h=display(h, 1), rate=rate, s=display(s, 1))
        prompts = [rng.choice(_A_PROMPT), rng.choice(_B_PROMPT).format(p=p), rng.choice(_C_PROMPT)]
        if keys_absent(first, prompts, [a_text, b_text, c_text]):
            break

    r = d / 2
    a_dis = [
        *tagged_distractors(
            a_text,
            0,
            [
                (litres / 10, None),  # took 1 m³ as 100 L
                (PI * d * d * h * 1000, None),  # used the diameter as the radius
                (PI * r * r * h * 1000 / 3, None),  # used the cone formula
                (2 * PI * r * h * 1000, None),  # used the curved surface area
            ],
            3,
        ),
    ]
    b_dis = tagged_distractors(
        b_text,
        0,
        [
            (litres / rate, None),  # time to fill the whole tank
            (Fraction(100 - p, 100) * litres / rate, None),  # used the percentage left empty
            (Fraction(p) * litres / rate, None),  # used p instead of p/100
        ],
    )
    c_dis = tagged_distractors(
        c_text,
        1,
        [
            (litres / (s * s), None),  # left the capacity in litres
            (litres / 1000 / s, None),  # divided by the side instead of its square
            (litres / 1000 / (s * s) * 2, None),
            (litres / 1000 / (s * 4), None),  # divided by the perimeter of the base
        ],
    )
    return dict(
        text=first,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=numeric_answer(cap),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"r = {display(d, 1)} ÷ 2 = {display(r, 2)} m, so V = π × {display(r, 2)}² × {display(h, 1)} cubic metres.",
                    f"= {display(litres / 1000, 3)} m³ = {a_text} L (nearest litre).",
                ],
                distractors=a_dis,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=numeric_answer(rounded(minutes, 0)),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{p}/100*part_a/{rate}",
                working=[
                    f"{p}% of {a_text} L = {display(Fraction(p, 100) * cap, 1)} L; {display(Fraction(p, 100) * cap, 1)} L ÷ {rate} L/min = {b_text} min."
                ],
                distractors=b_dis,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=numeric_answer(rounded(height, 1)),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a/1000/({display(s, 1)}**2)",
                working=[
                    f"{a_text} L = {display(cap / 1000, 3)} m³; height = volume ÷ base area = {display(cap / 1000, 3)} m³ ÷ {display(s * s, 2)} m² = {c_text} m."
                ],
                distractors=c_dis,
            ),
        ],
    )
