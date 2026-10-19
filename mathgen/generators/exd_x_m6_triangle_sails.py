"""Exam question: a triangular sail, a second sail of the same area, and the cost of fabric for both.

A three-part chain. Part (a) is the area of a triangle whose base, perpendicular height and slanting side are
all stated, so the learner must choose the right two lengths. Part (b) works backwards from that area to the
height of a second triangle on a different base. Part (c) prices the fabric for both sails. Parts (b) and (c)
carry `follow_through`.

Built backwards from whole numbers: the area is an integer, the second base divides twice the area, and the
height that results is a whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase

from .exd_x_m6_support import TooFewDistractors, number_part, shows_no_key, tagged

_STEMS = (
    "A sailmaker in Port Lincoln cuts a triangular sail with a base of {b} m, a perpendicular height of {h} m "
    "and a slanting side of {s} m. A second triangular sail is to have the same area on a base of {b2} m.",
    "A triangular pennant for a surf club at Noosa has a slanting side of {s} m, a base of {b} m and a "
    "perpendicular height of {h} m. A second pennant with the same area is to have a base of {b2} m.",
    "A shade sail over a playground in Mackay is a triangle with a base of {b} m and a slanting side of {s} m. "
    "Its perpendicular height is {h} m. A second sail of the same area is to be made with a base of {b2} m.",
)


@generator("exd_x_m6_triangle_sails", level=12, presets={"exam": dict(tier="exam")})
def exd_x_m6_triangle_sails(rng: random.Random, tier: str = "exam") -> dict:
    """Triangle area, the height of a second triangle of equal area, and the cost of fabric for both."""
    check(tier == "exam", "exd_x_m6_triangle_sails: only the exam tier exists")
    for _ in attempts(1000, "a sail whose figures work out exactly"):
        try:
            question = _attempt(rng)
        except TooFewDistractors:
            continue
        if question is not None:
            return question


def _attempt(rng: random.Random) -> dict | None:
    b, h = rng.randint(6, 40), rng.randint(4, 30)
    if (b * h) % 2:
        return None
    s = h + rng.randint(1, 6)
    area = b * h // 2
    bases = [d for d in range(4, 61) if (2 * area) % d == 0 and d != b and (2 * area) // d != h]
    if not bases:
        return None
    b2 = rng.choice(bases)
    h2 = 2 * area // b2
    price = rng.randint(4, 25)
    cost = 2 * area * price
    stem = phrase(rng, _STEMS, b=b, h=h, s=s, b2=b2)
    prompts = (
        "Find the area of the first piece of cloth, in square metres.",
        "Find the perpendicular height of the second sail, in metres.",
        phrase(
            rng,
            (
                "Fabric costs ${price} per square metre. Find the cost of the fabric for both, in dollars.",
                "The cloth is sold at ${price} for each square metre. What is the total cost of the cloth for the "
                "two, in dollars?",
            ),
            price=price,
        ),
    )
    if not shows_no_key([stem, *prompts], [area, h2, cost]):
        return None
    check(Fraction(b2 * h2, 2) == area, "triangle_sails: second sail has a different area")

    a_cands = [
        tagged(Fraction(b * h), "triangle-without-half"),
        tagged(Fraction(b * s, 2), "slant-side-as-height", 0),
        tagged(Fraction(b + h + s)),  # adds the lengths
    ]
    b_cands = [
        tagged(Fraction(area, b2), "height-from-area-without-doubling", 0),
        tagged(Fraction(h)),  # repeats the first sail's height
        tagged(Fraction(2 * area - b2)),  # took the second base off the doubled area
    ]
    c_cands = [
        tagged(Fraction(area * price)),  # prices one sail only
        tagged(Fraction(2 * b * h * price), "triangle-without-half"),
        tagged(Fraction(2 * area + price)),  # adds the price instead of multiplying
    ]
    return dict(
        text=stem,
        parts=[
            number_part(
                prompts[0],
                Fraction(area),
                a_cands,
                marks=1,
                working=[f"Use the base and the perpendicular height: {b} × {h} ÷ 2 = {area} m²."],
            ),
            number_part(
                prompts[1],
                Fraction(h2),
                b_cands,
                marks=2,
                follow_through=f"2*part_a/{b2}",
                working=[f"{b2} × h ÷ 2 = {area}, so h = 2 × {area} ÷ {b2} = {h2} m."],
            ),
            number_part(
                prompts[2],
                Fraction(cost),
                c_cands,
                marks=1,
                follow_through=f"2*part_a*{price}",
                working=[
                    f"The two pieces have equal area, so the total is 2 × {area} = {2 * area} m².",
                    f"{2 * area} × ${price} = ${cost}.",
                ],
            ),
        ],
    )
