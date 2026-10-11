"""Exam tier for "calculate the distance between two Cartesian points".

The practice tiers give the two points directly (or three vertices and ask for a perimeter). Here the
learner must first find a point and only then measure: either the length of a median of a triangle (the
midpoint of the opposite side is not given), or the distance between the midpoints of two segments. That is
two ideas composed -- the midpoint formula and the distance formula -- with the learner choosing the order of
steps, and the result is usually a surd that needs simplifying.

Built backwards from the midpoint(s): the midpoint ``M`` is drawn as an integer point, the two segment
endpoints are placed symmetrically about it, and the target point is offset from ``M`` by ``(dx, dy)``, so
the exact distance is ``sqrt(dx**2 + dy**2)``. The key is re-derived from the rendered coordinates with
exact arithmetic.

Distractors, each a named misconception:

- measured to an endpoint of a segment instead of to its midpoint;
- summed (rather than averaged) the endpoint coordinates when forming a midpoint;
- forgot the square root (reported ``dx**2 + dy**2``).
"""

from __future__ import annotations

import random
import re

from sympy import Rational, Symbol, sqrt

from ..core import check, check_equivalent, generator, phrase, pick_distractors

_MEDIAN = (
    "Triangle ABC has vertices A({ax}, {ay}), B({bx}, {by}) and C({cx}, {cy}). "
    "Find the exact length of the median from C to side AB.",
    "In triangle ABC, A = ({ax}, {ay}), B = ({bx}, {by}) and C = ({cx}, {cy}). "
    "Calculate the exact length of the median drawn from vertex C.",
    "The median from vertex C of triangle ABC joins C to the midpoint of AB. "
    "Given A({ax}, {ay}), B({bx}, {by}) and C({cx}, {cy}), find the exact length of this median.",
)
_MIDPOINTS = (
    "M is the midpoint of A({ax}, {ay}) and B({bx}, {by}), and N is the midpoint of C({cx}, {cy}) and "
    "D({dx}, {dy}). Find the exact distance MN.",
    "Let M be the midpoint of the segment from A({ax}, {ay}) to B({bx}, {by}) and N the midpoint of the "
    "segment from C({cx}, {cy}) to D({dx}, {dy}). Calculate the exact length of MN.",
)


def _surd(d2: int):
    return sqrt(Rational(d2))


def _surd_text(d2: int) -> str:
    return str(_surd(d2))


def _pair_around(rng: random.Random, mx: int, my: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Two distinct integer points with the integer midpoint ``(mx, my)``."""
    while True:
        px, py = rng.randint(1, 6), rng.randint(-6, 6)
        if (px, py) != (0, 0):
            return (mx - px, my - py), (mx + px, my + py)


@generator(
    "exd_x_a4_median_length",
    level=10,
    presets={"exam": dict(tier="exam")},
)
def exd_x_a4_median_length(rng: random.Random, tier: str = "exam") -> dict:
    """Median length, or distance between two midpoints. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    mx, my = rng.randint(-8, 8), rng.randint(-8, 8)
    a, b = _pair_around(rng, mx, my)
    while True:
        dx, dy = rng.randint(-9, 9), rng.randint(-9, 9)
        d2 = dx * dx + dy * dy
        # A genuine surd (never a whole number), and not a bare square root of a small prime
        # every time: a square factor to pull out makes simplifying part of the task.
        if d2 and int(d2**0.5) ** 2 != d2:
            break
    use_median = rng.random() < 0.5
    if use_median:
        c = (mx + dx, my + dy)
        text = phrase(rng, _MEDIAN, ax=a[0], ay=a[1], bx=b[0], by=b[1], cx=c[0], cy=c[1])
        target = c
    else:
        # N is the midpoint of C, D; place N at offset (dx, dy) from M.
        nx, ny = mx + dx, my + dy
        c, d = _pair_around(rng, nx, ny)
        text = phrase(rng, _MIDPOINTS, ax=a[0], ay=a[1], bx=b[0], by=b[1], cx=c[0], cy=c[1], dx=d[0], dy=d[1])
        target = (nx, ny)

    # Independent route: parse the rendered coordinates and recompute.
    nums = [int(v) for v in re.findall(r"-?\d+", text.replace("ABC", "").replace("MN", ""))]
    if use_median:
        pa, pb, pc = (nums[0], nums[1]), (nums[2], nums[3]), (nums[4], nums[5])
        mid = Rational(pa[0] + pb[0], 2), Rational(pa[1] + pb[1], 2)
        got = sqrt((pc[0] - mid[0]) ** 2 + (pc[1] - mid[1]) ** 2)
    else:
        pa, pb, pc, pd = (nums[0], nums[1]), (nums[2], nums[3]), (nums[4], nums[5]), (nums[6], nums[7])
        m1 = Rational(pa[0] + pb[0], 2), Rational(pa[1] + pb[1], 2)
        m2 = Rational(pc[0] + pd[0], 2), Rational(pc[1] + pd[1], 2)
        got = sqrt((m2[0] - m1[0]) ** 2 + (m2[1] - m1[1]) ** 2)
    key = _surd(d2)
    check_equivalent(got, key, "distance re-derived from the printed points disagrees with the construction")
    check(target == (mx + dx, my + dy), "target point drifted from the construction")

    answer_text = str(key)
    # Misconceptions.
    sx, sy = a[0] + b[0], a[1] + b[1]  # summed, never halved
    if use_median:
        to_end1 = (target[0] - a[0]) ** 2 + (target[1] - a[1]) ** 2
        to_end2 = (target[0] - b[0]) ** 2 + (target[1] - b[1]) ** 2
        summed = (target[0] - sx) ** 2 + (target[1] - sy) ** 2
        cands = [_surd_text(to_end1), _surd_text(to_end2), _surd_text(summed), str(d2)]
    else:
        n_sum = (c[0] + d[0], c[1] + d[1])
        summed = (n_sum[0] - sx) ** 2 + (n_sum[1] - sy) ** 2
        to_a = (target[0] - a[0]) ** 2 + (target[1] - a[1]) ** 2
        to_c = (mx - c[0]) ** 2 + (my - c[1]) ** 2
        cands = [_surd_text(to_a), _surd_text(summed), _surd_text(to_c), str(d2)]
    return dict(
        text=text,
        latex=text,
        answer=answer_text,
        answer_text=answer_text,
        working=[
            f"The first midpoint is ({mx}, {my}); the point it is measured to is {target}.",
            f"d = sqrt(({dx})^2 + ({dy})^2) = sqrt({d2}) = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, cands),
    )
