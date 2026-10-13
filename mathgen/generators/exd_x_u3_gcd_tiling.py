"""Exam-tier source for `euclidean-algorithm-gcd`: tiling a rectangle with the largest whole square tiles.

The learner is not told to find a greatest common divisor. A rectangular floor is to be covered by identical
square tiles with no cutting, so the side of the largest tile is the gcd of the two dimensions (part a). The
number of tiles then needs the area ratio (part b), and the cost needs the count (part c). Parts (b) and (c) carry
`follow_through`: a learner who chose a smaller tile in (a) still earns (b) and (c) for using it correctly.

Built backwards from the gcd and two coprime multipliers, and checked with `math.gcd`.
"""

from __future__ import annotations

import math
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

_FLOORS = [
    ("a courtyard at a house in Launceston", "pavers"),
    ("the floor of a laundry in Mackay", "tiles"),
    ("a bathroom floor in Bendigo", "tiles"),
    ("a patio at a holiday unit in Noosa", "pavers"),
    ("a school hall entrance in Darwin", "tiles"),
]
_SIDES = [12, 15, 18, 20, 24, 25, 30, 32, 35, 36, 40, 45, 48, 50, 60]


def _money(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 2), 2)


def _euclid_remainders(a: int, b: int) -> list[int]:
    """Non-zero remainders of the Euclidean algorithm on (a, b), in order; the last is the gcd."""
    out = []
    while b:
        a, b = b, a % b
        if b:
            out.append(b)
    return out or [a]


@generator("exd_x_u3_gcd_tiling", level=9, presets={"exam": dict(tier="exam")})
def exd_x_u3_gcd_tiling(rng, tier: str = "exam") -> dict:
    """Exam tier: largest square tile, the number needed, and their cost."""
    check(tier == "exam", f"unknown tier {tier!r}")
    place, noun = rng.choice(_FLOORS)
    for _ in attempts(1000, "coprime multipliers giving a clean tiling"):
        g = rng.choice(_SIDES)
        m, n = rng.randint(4, 24), rng.randint(4, 24)
        if m == n or math.gcd(m, n) != 1 or abs(m - n) < 3:
            continue
        length, width = g * max(m, n), g * min(m, n)
        remainders = _euclid_remainders(length, width)
        if len(remainders) < 2 or remainders[-1] != g:
            continue
        count = (length // g) * (width // g)
        price = Fraction(rng.randint(30, 199), 20)  # $1.50 .. $9.95 in 5c steps
        if {str(g), str(count), _money(count * price)} & {str(length), str(width), _money(price)}:
            continue
        break
    check(math.gcd(length, width) == g, "gcd disagrees with math.gcd")
    cost = count * price

    g_text, count_text, cost_text = str(g), str(count), _money(cost)
    stem = (
        f"{place[0].upper() + place[1:]} measures {length} cm by {width} cm. It is to be covered completely with identical "
        f"square {noun}, with none cut. Each one costs ${_money(price)}."
    )
    smallest_prime = next(p for p in range(2, g + 1) if g % p == 0)

    def whole(value: int, tag: str | None = None):
        return Distractor(str(value), tag, value)

    def money(value: Fraction):
        n = round_half_up(value, 2)
        return Distractor(_money(n), None, n)

    a_cand = [
        whole(remainders[-2], "gcd-stopped-early"),
        whole(g // smallest_prime, "common-divisor-not-greatest"),
        whole(length * width // g, "lcm-for-gcd"),
        whole(width),
    ]
    b_cand = [
        whole(length * width // g, "side-ratio-not-squared"),
        whole(length // g + width // g, "sides-counted-not-area"),
        whole(2 * (length // g + width // g)),
    ]
    c_cand = [
        money(price * (length * width // g)),
        money(price * (length // g + width // g)),
        money(price * (length // g)),  # only one side counted
    ]
    a_d = pick_distractors(g_text, a_cand, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(count_text, b_cand, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(cost_text, c_cand, 3, tolerance=GRADER_TOLERANCE)
    check(len(a_d) >= 2 and len(b_d) >= 2 and len(c_d) >= 2, "too few distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the side length, in centimetres, of the largest square {noun} that can be used.",
                answer_kind="number",
                answer=g,
                answer_text=g_text,
                answer_latex=latex_answer(g_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    "The side must divide both dimensions, so the largest side is their greatest common divisor.",
                    f"Euclidean algorithm on {length} and {width}: remainders {', '.join(map(str, remainders))}.",
                    f"The last non-zero remainder is {g_text}, so the side is {g_text} cm.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=f"Using your side length from (a), find how many {noun} are needed.",
                answer_kind="number",
                answer=count,
                answer_text=count_text,
                answer_latex=latex_answer(count_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{length}*{width}/part_a**2",
                working=[f"({length} ÷ {g_text}) × ({width} ÷ {g_text}) = {count_text}."],
                distractors=b_d,
            ),
            dict(
                prompt=f"Using your answer to (b), find the total cost of the {noun}, in dollars.",
                answer_kind="number",
                answer=numeric_answer(round_half_up(cost, 2)),
                answer_text=cost_text,
                answer_latex=latex_answer(cost_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_b*{_money(price)}",
                working=[f"{count_text} × ${_money(price)} = ${cost_text}."],
                distractors=c_d,
            ),
        ],
    )
