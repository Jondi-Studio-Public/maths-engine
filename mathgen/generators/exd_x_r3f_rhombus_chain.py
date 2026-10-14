"""Generates `exd_x_r3f_rhombus_chain` questions: an exam chain on a parallelogram whose diagonals meet at right angles.

A parallelogram ABCD has sides AB = (a1 x + b1) cm and BC = (c1 x + d1) cm and its diagonals meet at right angles
at M, with the diagonal AC given. The chain:

(a) the value of x                                                  number, 2 marks   AB = BC (a rhombus)
(b) the length of AB                                                number, 1 mark    a1 x + b1
(c) the length of BD, to 1 decimal place                            number, 2 marks   2 sqrt(AB^2 - (AC/2)^2)
(d) the area of ABCD, to 1 decimal place                            number, 2 marks   AC . BD / 2

Each part follows through from the one before it, so a learner whose x is wrong still earns (b), (c) and (d)
for using it correctly. The numbers are built backwards from a Pythagorean triple (AM, BM, side), so on the
canonical route BD and the area are whole numbers, shown to 1 decimal place because a learner's own route need
not be. Answers to (c) and (d) that a wrong x would leave without a real value earn nothing.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator, join_terms, phrase
from ._exd_r3f_parts import number_part
from ._exd_distinct_parts import distinct_part_answers


def _whole(num: int, den: int):
    """num/den when it is a whole number, else None."""
    return num // den if den != 0 and num % den == 0 else None


# (half of AC, half of BD, side): integer right triangles AM, BM, AB with the side up to 60.
_TRIPLES = [
    (a * k, b * k, c * k)
    for a, b, c in ((3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (12, 35, 37), (9, 40, 41))
    for k in range(1, 13)
    if c * k <= 60
]

_STEMS = (
    "ABCD is a parallelogram with AB = ({ab}) cm and BC = ({bc}) cm. Its diagonals meet at right angles "
    "at M, and the diagonal AC has length {ac} cm.",
    "In the parallelogram ABCD, AB = ({ab}) cm and BC = ({bc}) cm. The diagonals AC and BD intersect at M at "
    "right angles, and AC = {ac} cm.",
    "The sides of a parallelogram ABCD are AB = ({ab}) cm and BC = ({bc}) cm, and its diagonals meet at right "
    "angles at M. The diagonal AC is {ac} cm long.",
)


@generator("exd_x_r3f_rhombus_chain", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3f_rhombus_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Solve for x from the rhombus condition, then the side, the other diagonal and the area."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(5000, "side expressions with moderate constants"):
        half_ac, half_bd, side = rng.choice(_TRIPLES)
        if rng.random() < 0.5:
            half_ac, half_bd = half_bd, half_ac
        x = rng.randint(2, 12)
        a1, c1 = rng.sample(range(2, 10), 2)
        b1, d1 = side - a1 * x, side - c1 * x
        if abs(b1) <= 40 and abs(d1) <= 40 and b1 != 0 and d1 != 0:
            break
    ac = 2 * half_ac
    bd = 2 * half_bd
    area = ac * bd // 2

    # Independent checks: the sides really are equal at x, and the diagonals and sides satisfy Pythagoras.
    check(a1 * x + b1 == c1 * x + d1 == side, "the side expressions disagree")
    check(half_ac**2 + half_bd**2 == side**2, "the triple is not a right triangle")
    check(math.isclose(2 * math.sqrt(side**2 - half_ac**2), bd), "BD disagrees with Pythagoras")

    ab = join_terms([(a1, "x"), (b1, "")])
    bc = join_terms([(c1, "x"), (d1, "")])
    stem = phrase(rng, _STEMS, ab=ab, bc=bc, ac=ac)

    x_candidates = [
        (_whole(-(b1 + d1), a1 + c1), "wrong-sign-equation"),
        (-x, "sign-error-moving-term"),
        (d1 - b1, "forgot-to-divide-by-coefficient"),
        (_whole(d1 + b1, a1 - c1), "sign-error-moving-term"),
        (_whole(d1 - b1, a1 + c1), "added-coefficients-instead-of-subtracting"),
        (_whole(d1 - b1, a1), "x-term-on-one-side-ignored"),
        (_whole(d1 - b1, c1), "x-term-on-one-side-ignored"),
    ]
    side_candidates = [
        (x, "reports-x-not-side"),
        (a1 * x, "dropped-constant-term"),
        (x + b1, "coefficient-not-applied-to-x"),
        (c1 * x - d1, "sign-error-on-constant"),
    ]
    bd_candidates = [
        (half_bd, "omits-doubling-the-half-diagonal"),
        (2 * math.sqrt(side**2 + half_ac**2), "pythagoras-added-instead-of-subtracted"),
        (side, "reports-side-for-diagonal"),
        (ac, "reports-given-diagonal-as-answer"),
        (math.sqrt(max(side**2 - ac**2, 0)) * 2 if side > ac else None, "full-diagonal-used-as-pythagoras-leg"),
    ]
    area_candidates = [
        (ac * bd, "rhombus-area-diagonals-not-halved"),
        (ac * bd / 4, "rhombus-area-halved-twice"),
        (side * side, "rhombus-area-as-side-squared"),
        (side * half_bd, "rhombus-area-as-side-times-half-diagonal"),
    ]

    def exact(candidates):
        """Candidates as exact fractions: a float from a square root is read back as the decimal it prints as."""
        return [(None if v is None else Fraction(round(v, 6)).limit_denominator(10**6), tag) for v, tag in candidates]

    return dict(
        text=stem,
        parts=[
            number_part(
                "Find the value of x.",
                x,
                exact(x_candidates),
                marks=2,
                working=[
                    f"The diagonals meet at right angles, so ABCD is a rhombus and AB = BC: {ab} = {bc}, x = {x}."
                ],
            ),
            number_part(
                "Hence find the length of AB, in cm.",
                side,
                exact(side_candidates),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"({a1})*part_a + ({b1})",
                working=[f"AB = {a1}({x}) + ({b1}) = {side} cm."],
            ),
            number_part(
                "Find the length of the diagonal BD, correct to 1 decimal place.",
                bd,
                exact(bd_candidates),
                places=1,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"2*sqrt(part_b**2 - {half_ac}**2)",
                working=[
                    f"AM = {half_ac} cm and AB = {side} cm, so BM = \u221a({side}\u00b2 - {half_ac}\u00b2) = {half_bd} cm.",
                    f"BD = 2 \u00d7 {half_bd} = {bd}.0 cm.",
                ],
            ),
            number_part(
                "Hence find the area of ABCD, in cm\u00b2, correct to 1 decimal place.",
                area,
                exact(area_candidates),
                places=1,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{half_ac}*part_c",
                working=[f"Area = \u00bd \u00d7 {ac} \u00d7 {bd} = {area}.0 cm\u00b2."],
            ),
        ],
    )
