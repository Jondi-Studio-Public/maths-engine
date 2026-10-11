"""Exam tier for "calculate the midpoint of two Cartesian points".

The practice tiers ask for a midpoint, or for the other end of a segment given one end and its midpoint.
Here the learner is given three vertices of a parallelogram and must find the fourth. Nothing in the prompt
mentions a midpoint: the learner has to recognise that the diagonals of a parallelogram bisect each other,
choose which pair of vertices forms a diagonal, and use that shared midpoint to reach the missing vertex.

Built backwards: draw a non-degenerate parallelogram ``V0 V1 V2 V3`` (named in order around the shape) and
hide one vertex. The key is re-derived from the three printed vertices alone with exact fractions via the
shared-midpoint identity, and as a second route by requiring both pairs of opposite sides to be equal vectors.

Distractors, each a named misconception:

- stopped at the midpoint of the known diagonal (never reflected it to reach the far vertex);
- treated a side as the diagonal (one of the two other ways to combine the three known vertices).
"""

from __future__ import annotations

import random
import re

from sympy import Rational

from ..core import check, generator, phrase, pick_distractors, rational_text

_LABELS = "ABCD"

_TEMPLATES = (
    "ABCD is a parallelogram with {known}. Find the coordinates of {target}.",
    "Three vertices of the parallelogram ABCD (named in order around the shape) are {known}. Find {target}.",
    "{known_cap} are three of the four vertices of parallelogram ABCD, labelled in order. "
    "Find the coordinates of the fourth vertex, {target}.",
)


def _point_text(label: str, p: tuple[int, int]) -> str:
    return f"{label}({p[0]}, {p[1]})"


def _join(parts: list[str]) -> str:
    return f"{parts[0]}, {parts[1]} and {parts[2]}"


@generator(
    "exd_x_a4_parallelogram_vertex",
    level=9,
    presets={"exam": dict(tier="exam")},
)
def exd_x_a4_parallelogram_vertex(rng: random.Random, tier: str = "exam") -> dict:
    """Fourth vertex of a parallelogram from three. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    while True:
        a = (rng.randint(-8, 8), rng.randint(-8, 8))
        u = (rng.randint(-7, 7), rng.randint(-7, 7))  # side AB
        w = (rng.randint(-7, 7), rng.randint(-7, 7))  # side AD
        if u[0] * w[1] - u[1] * w[0] != 0:  # not degenerate
            break
    pts = [a, (a[0] + u[0], a[1] + u[1]), (a[0] + u[0] + w[0], a[1] + u[1] + w[1]), (a[0] + w[0], a[1] + w[1])]
    hidden = rng.randrange(4)
    known_idx = [i for i in range(4) if i != hidden]
    known = [_point_text(_LABELS[i], pts[i]) for i in known_idx]
    text = phrase(
        rng,
        _TEMPLATES,
        known=_join(known),
        known_cap=_join(known),
        target=_LABELS[hidden],
    )
    key = pts[hidden]

    # Independent route 1: from the PRINTED text only, shared midpoint of the diagonal that does not
    # contain the hidden vertex, solved for the hidden vertex.
    printed = {m[0]: (int(m[1]), int(m[2])) for m in re.findall(r"([A-D])\((-?\d+), (-?\d+)\)", text)}
    check(len(printed) == 3 and _LABELS[hidden] not in printed, "exactly the three known vertices are printed")
    opposite = _LABELS[(hidden + 2) % 4]
    n1, n2 = _LABELS[(hidden + 1) % 4], _LABELS[(hidden + 3) % 4]
    mid = (Rational(printed[n1][0] + printed[n2][0], 2), Rational(printed[n1][1] + printed[n2][1], 2))
    got = (2 * mid[0] - printed[opposite][0], 2 * mid[1] - printed[opposite][1])
    check(got == key, "shared-midpoint route disagrees with the construction")
    # Route 2: opposite sides are equal vectors, with the fourth vertex in place.
    full = dict(printed)
    full[_LABELS[hidden]] = key
    ab = (full["B"][0] - full["A"][0], full["B"][1] - full["A"][1])
    dc = (full["C"][0] - full["D"][0], full["C"][1] - full["D"][1])
    check(ab == dc, "opposite sides are not equal vectors")

    answer_text = f"x = {key[0]}, y = {key[1]}"
    o, p, q = printed[opposite], printed[n1], printed[n2]
    wrong = [
        ((Rational(p[0] + q[0], 2)), Rational(p[1] + q[1], 2)),  # midpoint of the diagonal, not reflected
        (p[0] + o[0] - q[0], p[1] + o[1] - q[1]),  # treated the wrong pair as a diagonal
        (q[0] + o[0] - p[0], q[1] + o[1] - p[1]),  # and the other wrong pair
    ]
    cands = [f"x = {rational_text(x)}, y = {rational_text(y)}" for x, y in wrong]
    return dict(
        text=text,
        latex=text,
        answer={"x": key[0], "y": key[1]},
        answer_text=answer_text,
        working=[
            f"The diagonals bisect each other. Midpoint of {n1}{n2} = ({rational_text(mid[0])}, {rational_text(mid[1])}).",
            f"{_LABELS[hidden]} = 2 x midpoint - {opposite} = ({key[0]}, {key[1]}).",
        ],
        distractors=pick_distractors(answer_text, cands),
    )
