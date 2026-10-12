"""Exam-tier source for the parallelogram law: one diagonal, then the angle and the area.

A parallelogram ABCD has stated side lengths AB and AD and one stated diagonal. The learner must

(a) use ``d1^2 + d2^2 = 2(a^2 + b^2)`` to find the other diagonal BD,
(b) use BD in the cosine rule on triangle ABD to find the angle DAB, and
(c) use that angle for the area of the parallelogram, ``a * b * sin(A)``.

Built backwards from side lengths and diagonals that are all whole numbers, so (a) is exact and the angle is
the only rounded quantity carried forward.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.core import VerificationError, attempts, generator, latex_answer, numeric_answer

from .exd_x_m5_support import clear_of_boundary, display, keys_absent, rounded, tagged_distractors

# (a, b, d1, d2) with a < b, d1 < d2 and 2(a^2 + b^2) = d1^2 + d2^2, all whole numbers.
_TABLE = tuple(
    (a, b, d1, math.isqrt(2 * (a * a + b * b) - d1 * d1))
    for a in range(4, 41)
    for b in range(a + 1, 41)
    for d1 in range(b - a + 1, a + b)
    if math.isqrt(2 * (a * a + b * b) - d1 * d1) ** 2 == 2 * (a * a + b * b) - d1 * d1
    and d1 < math.isqrt(2 * (a * a + b * b) - d1 * d1) < a + b
)

_OBJECTS = (
    ("A parallelogram-shaped garden bed", "m"),
    ("A parallelogram-shaped paving slab", "cm"),
    ("A parallelogram-shaped shade sail", "m"),
    ("A parallelogram-shaped sign", "cm"),
    ("A parallelogram-shaped vegetable plot", "m"),
)
_FIRST = (
    "{obj} ABCD has AB = {a} {u} and AD = {b} {u}. Its diagonal {given_name} measures {given} {u}.",
    "{obj} is drawn as ABCD with sides AB = {a} {u} and AD = {b} {u}. The diagonal {given_name} is {given} {u} long.",
    "In {obj_lower} ABCD, side AB is {a} {u} and side AD is {b} {u}, and the diagonal {given_name} is {given} {u}.",
)
_A_PROMPT = (
    "Find the length of the other diagonal, {other_name}, in {u}.",
    "Use the parallelogram law to find the length of diagonal {other_name} in {u}.",
)
_B_PROMPT = (
    "Using your answer to (a) in triangle ABD, find the size of angle DAB in degrees, correct to 1 decimal place.",
    "Use your length for {other_name} and the cosine rule in triangle ABD to find angle DAB, "
    "correct to 1 decimal place.",
)
_C_PROMPT = (
    "Using your angle from (b), find the area of the parallelogram in square {u}, to the nearest whole number.",
    "Hence find the area of ABCD in square {u}, correct to the nearest whole number. Use your angle from (b).",
)


@generator("exd_x_m5_parallelogram_diagonal_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m5_parallelogram_diagonal_chain(rng, tier: str = "exam") -> dict:
    """Exam tier: the second diagonal from the parallelogram law, then angle DAB, then the area."""
    if tier != "exam":
        raise VerificationError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a parallelogram whose answers are clear of rounding boundaries"):
        a, b, d1, d2 = rng.choice(_TABLE)
        if rng.random() < 0.5:
            a, b = b, a
        ac, bd = (d1, d2) if rng.random() < 0.5 else (d2, d1)
        given, other = ac, bd
        given_name, other_name = "AC", "BD"
        # Angle DAB sits opposite BD in triangle ABD.
        cos_a = Fraction(a * a + b * b - bd * bd, 2 * a * b)
        angle = math.degrees(math.acos(cos_a))
        angle_shown = rounded(Fraction(angle), 1)
        area_exact = a * b * math.sin(math.radians(angle))
        area_from_shown = a * b * math.sin(math.radians(float(angle_shown)))
        if not (
            clear_of_boundary(Fraction(angle), 1)
            and clear_of_boundary(Fraction(area_exact), 0, Fraction(1, 50))
            and rounded(Fraction(area_exact), 0) == rounded(Fraction(area_from_shown), 0)
        ):
            continue
        obj, unit = rng.choice(_OBJECTS)
        a_text, b_text, c_text = str(other), display(angle_shown, 1), display(Fraction(area_exact), 0)
        first = rng.choice(_FIRST).format(
            obj=obj, obj_lower=obj[0].lower() + obj[1:], a=a, b=b, u=unit, given=given, given_name=given_name
        )
        prompts = [
            rng.choice(_A_PROMPT).format(other_name=other_name, u=unit),
            rng.choice(_B_PROMPT).format(other_name=other_name),
            rng.choice(_C_PROMPT).format(u=unit),
        ]
        if keys_absent(first, prompts, [a_text, b_text, c_text]):
            break

    def root(n: int) -> Fraction:
        return Fraction(math.sqrt(n)) if n > 0 else Fraction(0)

    a_dis = tagged_distractors(
        a_text,
        0,
        [
            (root(a * a + b * b - given * given), None),  # dropped the factor of 2 in the law
            (Fraction(2 * (a + b) - given), None),  # used the linear sum d1 + d2 = 2(a + b)
            (Fraction(given), None),  # assumed the diagonals are equal, as in a rectangle
            (root(2 * (a * a + b * b)) - given, None),  # took the square root before subtracting
        ],
    )
    b_dis = tagged_distractors(
        b_text,
        1,
        [
            (180 - Fraction(angle), None),  # found the supplementary angle ABC instead of DAB
            (Fraction(math.radians(angle)), None),  # calculator left in radians
            *(
                [(Fraction(math.degrees(math.acos(Fraction(a * a + b * b - bd * bd, a * b)))), None)]
                if abs(a * a + b * b - bd * bd) < a * b
                else []
            ),  # forgot the 2 in the denominator of the cosine rule
        ],
    )
    c_dis = tagged_distractors(
        c_text,
        0,
        [
            (Fraction(a * b), None),  # base times side, no sine
            (Fraction(area_exact) / 2, None),  # the area of one triangle
            (Fraction(a * b * abs(math.cos(math.radians(angle)))), None),  # cosine in place of sine
        ],
    )
    return dict(
        text=first,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=other,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"{given_name}² + {other_name}² = 2({a}² + {b}²) = {2 * (a * a + b * b)}.",
                    f"{other_name}² = {2 * (a * a + b * b)} − {given * given} = {other * other}, so {other_name} = {other}.",
                ],
                distractors=a_dis,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=numeric_answer(angle_shown),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, 180),
                follow_through=f"acos(({a}**2 + {b}**2 - part_a**2)/(2*{a}*{b}))*180/pi",
                working=[
                    f"cos A = ({a}² + {b}² − {bd}²) ÷ (2 × {a} × {b}).",
                    f"A = {b_text}° (1 d.p.).",
                ],
                distractors=b_dis,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=numeric_answer(rounded(Fraction(area_exact), 0)),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{a}*{b}*sin(part_b*pi/180)",
                working=[f"Area = ab sin A = {a} × {b} × sin {b_text}° = {c_text} square {unit}."],
                distractors=c_dis,
            ),
        ],
    )
