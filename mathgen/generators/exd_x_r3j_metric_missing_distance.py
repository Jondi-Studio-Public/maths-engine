"""Generates `exd_x_r3j_metric_missing_distance` questions: which values of one unknown distance give a metric.

Skill: `ucv1-metric-axiom-check` (exam tier). The practice tiers show a complete distance table and ask which
axiom fails, a four-way recognition. An exam instead leaves one distance as a parameter t and asks what the
axioms force on it:

(a) the smallest t for which every triangle inequality holds                 number, 2 marks
(b) the largest t for which every triangle inequality holds                  number, 2 marks
(c) how many positive integer values of t make d a metric                    number, 1 mark, part_b - part_a + 1

## Construction

On {A, B, C, D} five distances are fixed, those not involving the pair A, D, and d(A, D) = t. The two triples
without t, (A, B, C) and (B, C, D), are drawn with a strict triangle, so only the triples (A, B, D) and (A, C, D)
constrain t: |d(A, B) - d(B, D)| <= t <= d(A, B) + d(B, D), and likewise through C. So the least t is the larger of
the two differences and the greatest t is the smaller of the two sums. The draw is accepted when the least t is at
least 1 (so t > 0 holds without a separate argument), the interval holds between 2 and 9 integers, and the binding
triangle on each side is not the same, so both bounds need both triples.
"""

from __future__ import annotations

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3j_kit import number_part

_STEMS = [
    "A function d is defined on the four points A, B, C and D with d(x, x) = 0 and d(x, y) = d(y, x) for all x and y. "
    "Its other values are d(A, B) = {ab}, d(A, C) = {ac}, d(B, C) = {bc}, d(B, D) = {bd}, d(C, D) = {cd} "
    "and d(A, D) = t, where t > 0.",
    "On four points A, B, C and D let d be symmetric with d(x, x) = 0. The distances are d(A, B) = {ab}, "
    "d(A, C) = {ac}, d(B, C) = {bc}, d(B, D) = {bd} and d(C, D) = {cd}, and d(A, D) = t is a positive number.",
    "Let d be a symmetric function on the four points A, B, C and D with d(x, x) = 0, where d(A, B) = {ab}, d(A, C) = {ac}, "
    "d(B, C) = {bc}, d(B, D) = {bd}, d(C, D) = {cd}, and d(A, D) = t > 0.",
]


def _triangle(x: int, y: int, z: int) -> bool:
    return x < y + z and y < x + z and z < x + y


@generator("exd_x_r3j_metric_missing_distance", level=17, presets={"exam": {}})
@distinct_part_answers
def exd_x_r3j_metric_missing_distance(rng):
    for _ in attempts(3000, "five distances that leave an interval of two to nine integers for t"):
        ab, ac, bc, bd, cd = (rng.randint(2, 40) for _ in range(5))
        if not (_triangle(ab, ac, bc) and _triangle(bc, bd, cd)):
            continue
        low_b, low_c = abs(ab - bd), abs(ac - cd)
        high_b, high_c = ab + bd, ac + cd
        low, high = max(low_b, low_c), min(high_b, high_c)
        if low >= 1 and 2 <= high - low + 1 <= 9 and low_b != low_c and high_b != high_c:
            break
    count = high - low + 1

    # independent route: test every integer t against all twelve triangle inequalities
    points = "ABCD"
    table = {("A", "B"): ab, ("A", "C"): ac, ("B", "C"): bc, ("B", "D"): bd, ("C", "D"): cd}

    def dist(x, y, t):
        if x == y:
            return 0
        if {x, y} == {"A", "D"}:
            return t
        return table.get((x, y), table.get((y, x)))

    def is_metric(t):
        return all(dist(x, z, t) <= dist(x, y, t) + dist(y, z, t) for x in points for y in points for z in points)

    valid = [t for t in range(1, 200) if is_metric(t)]
    check(valid == list(range(low, high + 1)), "the interval for t disagrees with a direct check")
    check(is_metric(low) and is_metric(high) and not is_metric(low - 1) and not is_metric(high + 1), "bounds not tight")

    stem = rng.choice(_STEMS).format(ab=ab, ac=ac, bc=bc, bd=bd, cd=cd)
    weaker_low = min(low_b, low_c)
    weaker_high = max(high_b, high_c)
    parts = [
        number_part(
            "Find the smallest value of t for which d satisfies the triangle inequality.",
            low,
            0,
            2,
            [
                f"Triple A, B, D: t ≥ |{ab} − {bd}| = {low_b}. Triple A, C, D: t ≥ |{ac} − {cd}| = {low_c}.",
                f"The larger bound is {low}.",
            ],
            [
                (weaker_low, "took-looser-bound"),
                (ab - bd if ab - bd != low else ac - cd, "forgot-absolute-value"),
                (high, "swapped-lower-and-upper-bound"),
                (abs(ab - ac), "bound-from-wrong-pair"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            "Find the largest value of t for which d satisfies the triangle inequality.",
            high,
            0,
            2,
            [
                f"Triple A, B, D: t ≤ {ab} + {bd} = {high_b}. Triple A, C, D: t ≤ {ac} + {cd} = {high_c}.",
                f"The smaller bound is {high}.",
            ],
            [
                (weaker_high, "took-looser-bound"),
                (low, "swapped-lower-and-upper-bound"),
                (ab + ac, "bound-from-wrong-pair"),
                (high + low, "added-the-two-bounds"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            "For how many positive integer values of t is d a metric?",
            count,
            0,
            1,
            [f"The integers from {low} to {high} inclusive: {high} − {low} + 1 = {count}."],
            [
                (high - low, "off-by-one-count"),
                (max(high - low - 1, 0), "excluded-attainable-endpoints"),
                (high, "counted-from-one-to-upper-bound"),
                (high - low + 2, "off-by-one-count"),
            ],
            follow_through="part_b - part_a + 1",
        ),
    ]
    return dict(text=stem, parts=parts)
