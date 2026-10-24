"""``exd_x_w7f_tessellation_chain`` (``f10-tessellation-vertex-angles``, AC9M6SP03): a floor pattern in which two named
regular polygons and one more meet at every corner point with no gaps.

(a) one interior angle of the first polygon, (b) one of the second, (c) the angle left for the third polygon (follows
(a) and (b)), (d) the number of sides of the third polygon (follows (c)). The vertex configurations come from the
same search the single-question generator uses; each key is checked from the exterior angle too.
"""

from __future__ import annotations

from mathgen.core import check, generator

from ._w7f_kit import number_part, retry
from .fr2a_tessellation_vertex_build import _a, _vertex_configs, interior, interior_by_exterior, sides_for_angle

TOPIC = "exd_x_w7f_tessellation_chain"


def _chain(rng):
    configs = [c for c in _vertex_configs(3) if len(set(c)) >= 2]
    cfg = list(rng.choice(configs))
    rng.shuffle(cfg)
    a, b, c = cfg
    if a == b:
        for i in range(3):
            for j in range(3):
                if i != j and cfg[i] != cfg[j]:
                    a, b = cfg[i], cfg[j]
                    c = cfg[3 - i - j]
    ta, tb = interior(a), interior(b)
    for n, t in ((a, ta), (b, tb)):
        check(t == interior_by_exterior(n) and t.denominator == 1, "interior angle routes disagree")
    left = 360 - ta - tb
    check(sides_for_angle(left) == c and interior_by_exterior(c) == left, "third polygon differs between the routes")
    ta, tb, left = int(ta), int(tb), int(left)
    stem = (
        f"A floor pattern is made from regular polygon tiles. At every corner point {_a(a)}, {_a(b)} and one more "
        "regular polygon meet with no gaps and no overlaps."
    )

    def angle_wrong(n):
        t = int(interior(n))
        return [
            (360 // n, "fr2a-exterior-for-interior"),
            (180 * (n - 2), "fr2a-angle-sum-for-one-angle"),
            (180 // n, "fr2a-straight-angle-over-sides"),
            (360 - t, "fr2a-reflex-for-interior"),
        ]

    part_a = number_part(
        f"What is the size of one interior angle of {_a(a)}, in degrees?",
        ta,
        1,
        [f"Each exterior angle is 360 / {a} = {360 // a} degrees.", f"The interior angle is 180 - {360 // a} = {ta}."],
        angle_wrong(a),
    )
    part_b = number_part(
        f"What is the size of one interior angle of {_a(b)}, in degrees?",
        tb,
        1,
        [f"Each exterior angle is 360 / {b} = {360 // b} degrees.", f"The interior angle is 180 - {360 // b} = {tb}."],
        angle_wrong(b),
    )
    part_c = number_part(
        "What is the size of the corner of the third polygon, in degrees?",
        left,
        2,
        ["The angles round a point add to 360 degrees.", f"360 - {ta} - {tb} = {left}."],
        [
            (180 - ta - tb, "fr2a-straight-line-for-point"),
            (ta + tb, "fr2a-angles-total-for-gap"),
            (360 - ta, "fr2a-one-angle-taken-for-all"),
            (360 - tb, "fr2a-one-angle-taken-for-all"),
        ],
        follow="360-part_a-part_b",
    )
    part_d = number_part(
        "How many sides does the third polygon have?",
        c,
        2,
        [
            f"Its exterior angle is 180 - {left} = {180 - left} degrees.",
            f"The number of sides is 360 / {180 - left} = {c}.",
        ],
        [
            (left, "fr2a-angle-for-sides"),
            (360 // left if 360 % left == 0 else None, "fr2a-360-over-interior"),
            (c + 1, "fr2a-count-off-by-one"),
            (c - 1, "fr2a-count-off-by-one"),
            (a + b, "fr2a-sides-added"),
            (180 - left, "fr2a-exterior-for-interior"),
        ],
        follow="360/(180-part_c)",
    )
    side = rng.randint(3, 12)
    part_e = number_part(
        f"A tile of the third polygon has sides {side} cm long. What is the distance round the edge of one tile, in cm?",
        c * side,
        1,
        [f"The tile has {c} sides of {side} cm.", f"{c} x {side} = {c * side}."],
        [
            (c + side, "tess-adds-the-sides-and-the-length"),
            (left, "tess-gives-the-angle-for-the-distance"),
            (side * side, "tess-squares-the-side"),
            ((c - 1) * side, "tess-misses-a-side"),
            ((c + 1) * side, "tess-misses-a-side"),
            (side, "tess-gives-the-length-of-one-side"),
        ],
        follow=f"{side}*part_d",
    )
    parts = [part_a, part_b, part_c, part_d, part_e]
    if any(p is None for p in parts):
        return None
    return dict(text=stem, parts=parts)


@generator(TOPIC, level=6, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_tessellation_chain(rng, tier: str = "exam"):
    """Two interior angles, the angle left at the point, and the sides of the third polygon."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _chain, "a corner point with three distractors per part")
