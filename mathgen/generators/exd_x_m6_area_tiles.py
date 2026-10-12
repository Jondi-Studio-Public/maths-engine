"""Exam question: paving a rectangular area measured in metres with pavers measured in centimetres.

A three-part chain. Part (a) turns the area of the courtyard into square centimetres (the learner must square
the length factor, not apply it once), part (b) uses that area to count the pavers, and part (c) prices them.
Parts (b) and (c) carry `follow_through`, so a wrong area still earns the later marks when it is used correctly.

Built backwards: the number of pavers along each side is drawn first, so the courtyard is an exact whole
number of pavers in both directions and every key is an integer.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase

from .exd_x_m6_support import TooFewDistractors, number_part, shows_no_key, shown, tagged

_SIZES = (20, 25, 30, 40, 50)

_STEMS = (
    "A courtyard in Fremantle is {L} m long and {W} m wide. It is to be paved with rectangular pavers, each "
    "{p} cm by {q} cm, laid edge to edge with none cut.",
    "A school verandah in Toowoomba measures {L} m by {W} m. It is covered with rectangular pavers of "
    "{p} cm by {q} cm, with no gaps and no cutting.",
    "A rectangular patio in Launceston is {L} m by {W} m. The builder lays pavers measuring {p} cm by {q} cm "
    "edge to edge, using whole pavers only.",
)


def _metres(cm: int) -> str:
    return shown(Fraction(cm, 100))


@generator("exd_x_m6_area_tiles", level=12, presets={"exam": dict(tier="exam")})
def exd_x_m6_area_tiles(rng: random.Random, tier: str = "exam") -> dict:
    """Courtyard area in cm², the number of pavers that cover it, and their cost."""
    check(tier == "exam", "exd_x_m6_area_tiles: only the exam tier exists")
    for _ in attempts(1000, "a courtyard whose figures do not give the answers away"):
        try:
            question = _attempt(rng)
        except TooFewDistractors:
            continue
        if question is not None:
            return question


def _attempt(rng: random.Random) -> dict | None:
    p, q = rng.choice(_SIZES), rng.choice(_SIZES)
    nx, ny = rng.randint(5, 18), rng.randint(4, 14)
    price = rng.randint(3, 15)
    length_cm, width_cm = nx * p, ny * q
    area_cm2 = length_cm * width_cm
    count = area_cm2 // (p * q)
    cost = count * price
    check(count == nx * ny and area_cm2 % (p * q) == 0, "area_tiles: the pavers do not tile the courtyard")
    stem = phrase(rng, _STEMS, L=_metres(length_cm), W=_metres(width_cm), p=p, q=q)
    prompt_a = "Find the area to be paved, in square centimetres."
    prompt_b = "How many pavers are needed to cover this area?"
    prompt_c = f"Each paver costs ${price}. Find the total cost of the pavers, in dollars."
    if not shows_no_key([stem, prompt_a, prompt_b, prompt_c], [area_cm2, count, cost]):
        return None

    area_m2 = Fraction(area_cm2, 10_000)
    a_cands = [
        tagged(area_m2, "forgot-unit-conversion", 0),  # multiplied the metre figures and stopped
        tagged(area_m2 * 100, "unit-factor-wrong-power", 0),  # used 100 once instead of squaring it
        tagged(area_m2 * 10**6, "unit-factor-wrong-power", 0),  # used 1000 for each side
    ]
    b_cands = [
        tagged(Fraction(area_cm2, p), "area-divided-by-one-side", 0),
        tagged(Fraction(area_cm2, q), "area-divided-by-one-side", 0),
        tagged(Fraction(count, 100), "unit-factor-wrong-power", 0),
        tagged(Fraction(count * 100), "unit-factor-wrong-power", 0),
    ]
    c_cands = [
        tagged(area_m2 * price, "cost-from-area-not-count", 0),
        tagged(Fraction(area_cm2 * price), "cost-from-area-not-count", 0),
        tagged(Fraction(cost, 100), "unit-factor-wrong-power", 0),
        tagged(Fraction(cost * 100), "unit-factor-wrong-power", 0),
    ]
    return dict(
        text=stem,
        parts=[
            number_part(
                prompt_a,
                Fraction(area_cm2),
                a_cands,
                marks=2,
                working=[
                    f"Length = {length_cm} cm and width = {width_cm} cm.",
                    f"Area = {length_cm} × {width_cm} = {area_cm2} cm².",
                ],
            ),
            number_part(
                prompt_b,
                Fraction(count),
                b_cands,
                marks=1,
                follow_through=f"part_a/{p * q}",
                working=[f"One paver covers {p} × {q} = {p * q} cm².", f"{area_cm2} ÷ {p * q} = {count} pavers."],
            ),
            number_part(
                prompt_c,
                Fraction(cost),
                c_cands,
                marks=1,
                follow_through=f"part_b*{price}",
                working=[f"{count} × ${price} = ${cost}."],
            ),
        ],
    )
