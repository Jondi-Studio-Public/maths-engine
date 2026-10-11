"""Exam tier for vector equations of lines: where two lines meet.

The practice tiers ask for a direction vector from two points. Here two lines are specified (one by
two points, the other by a point and direction, or both in the form ``r = c + t d``) and the learner
forms each vector equation, equates them and solves for the two parameters to find the intersection.

Built backwards from the intersection point ``P``: draw integer parameters ``s`` and ``t``, directions
``e`` and ``d`` that are not parallel, then set the base points to ``P - s e`` and ``P - t d``.
Answer kind is ``pair`` (``x = .., y = ..``).
"""

from __future__ import annotations

import random

from ..core import GRADER_TOLERANCE, check, generator, pick_distractors

_POINTS_AND_VECTOR = (
    "The first line passes through A({ax}, {ay}) and B({bx}, {by}). The second line has vector equation "
    "r = ({cx}, {cy}) + t({dx}, {dy}). Find the point where the two lines intersect.",
    "One line passes through the points A({ax}, {ay}) and B({bx}, {by}); another is r = ({cx}, {cy}) + t({dx}, {dy}). "
    "Find the coordinates of their point of intersection.",
)
_TWO_VECTOR = (
    "Two lines have vector equations r = ({ax}, {ay}) + s({ex}, {ey}) and r = ({cx}, {cy}) + t({dx}, {dy}). "
    "Find the point where they intersect.",
    "The lines r = ({ax}, {ay}) + s({ex}, {ey}) and r = ({cx}, {cy}) + t({dx}, {dy}) meet at one point. Find it.",
)


def _pair_text(x, y) -> str:
    return f"x = {x}, y = {y}"


def _build(rng: random.Random, kind: str) -> dict | None:
    while True:
        px, py = rng.randint(-8, 8), rng.randint(-8, 8)
        ex, ey = rng.randint(-4, 4), rng.randint(-4, 4)
        dx, dy = rng.randint(-4, 4), rng.randint(-4, 4)
        s = rng.choice([-3, -2, -1, 1, 2, 3])
        t = rng.choice([-3, -2, -1, 1, 2, 3])
        if (ex, ey) == (0, 0) or (dx, dy) == (0, 0) or ex * dy - ey * dx == 0:
            continue
        break
    ax, ay = px - s * ex, py - s * ey
    cx, cy = px - t * dx, py - t * dy
    # Independent check: both lines pass through P at their own parameter, and P is the only common point.
    check((ax + s * ex, ay + s * ey) == (px, py) and (cx + t * dx, cy + t * dy) == (px, py))
    check(ex * dy - ey * dx != 0, "lines are parallel")
    swapped = (py, px)  # (x, y) reported as (y, x)
    params = (s, t)  # reported the parameter values instead of the point
    wrong_line = (cx + s * dx, cy + s * dy)  # put the first line's parameter into the second line
    wrong_line_2 = (ax + t * ex, ay + t * ey)  # put the second line's parameter into the first line
    cands = [params, wrong_line, wrong_line_2, swapped]
    answer_text = _pair_text(px, py)
    distractors = pick_distractors(answer_text, [_pair_text(*c) for c in cands], tolerance=GRADER_TOLERANCE)
    if len(distractors) < 2:
        return None  # caller redraws
    if kind == "points":
        bx, by = ax + ex, ay + ey
        text = rng.choice(_POINTS_AND_VECTOR).format(ax=ax, ay=ay, bx=bx, by=by, cx=cx, cy=cy, dx=dx, dy=dy)
    else:
        text = rng.choice(_TWO_VECTOR).format(ax=ax, ay=ay, ex=ex, ey=ey, cx=cx, cy=cy, dx=dx, dy=dy)
    return dict(
        text=text,
        latex=text,
        answer={"x": float(px), "y": float(py)},
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
    )


@generator(
    "exd_x_g1_vector_line",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_g1_vector_line(rng: random.Random, tier: str = "exam") -> dict:
    """Intersection point of two lines given by two points / a point and direction."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(200):
        built = _build(rng, "points" if rng.random() < 0.5 else "vectors")
        if built is not None:
            return built
    raise ValueError("no usable draw in 200 attempts")
