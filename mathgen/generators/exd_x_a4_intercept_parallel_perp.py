"""Exam tier for "interpret the vertical intercept".

The practice tiers read the intercept off a table or a flat-fee story. Here no table and no stated gradient
is given: a line ``L`` is described as passing through a point and being PARALLEL or PERPENDICULAR to the line
through two other points. The learner has to choose the right relationship between gradients (equal, or
negative reciprocal), find the gradient of the reference line, and then back-solve the output at ``x = 0``
from the point on ``L``. That is three linked steps with a method choice, not a larger table.

Built backwards: draw the gradient ``m = p/q`` of the reference line in lowest terms, build the reference
points from a displacement ``(q*j, p*j)``, pick the gradient of ``L`` (``m`` or ``-q/p``) and an x-coordinate
for the point on ``L`` that makes the intercept a whole number. The key is re-derived from the printed points
with exact fractions.

Distractors, each a named misconception:

- gave the y-coordinate of the point on ``L`` as the intercept;
- used the other relationship (negative reciprocal instead of equal, or equal instead of negative
  reciprocal) when it still gives a whole number;
- subtracted ``m*x`` with the wrong sign when back-solving;
- reported the x-intercept (the skill's "confusing x- and y-intercepts" misconception);
- reported the intercept with its sign reversed.
"""

from __future__ import annotations

import math
import random
import re
from fractions import Fraction

from ..core import check, generator, join_terms, phrase, pick_distractors

_TEMPLATES = (
    "The line through A({ax}, {ay}) and B({bx}, {by}) is {rel} line L. "
    "Line L passes through C({cx}, {cy}). What is the y-intercept of L?",
    "Line L passes through C({cx}, {cy}) and is {rel} the line through A({ax}, {ay}) and B({bx}, {by}). "
    "Find the value of y where L crosses the y-axis.",
    "Points A({ax}, {ay}) and B({bx}, {by}) lie on a line that is {rel} L. "
    "If L passes through C({cx}, {cy}), find the y-intercept of L.",
)


@generator(
    "exd_x_a4_intercept_parallel_perp",
    level=11,
    presets={"exam": dict(tier="exam")},
)
def exd_x_a4_intercept_parallel_perp(rng: random.Random, tier: str = "exam") -> dict:
    """y-intercept of a line parallel or perpendicular to another. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    perpendicular = rng.random() < 0.5
    while True:
        p = rng.choice([-1, 1]) * rng.randint(1, 4)
        q = rng.randint(1, 4)
        if math.gcd(abs(p), q) != 1 or (abs(p) == 1 and q == 1):
            continue  # lowest terms; a gradient of +/-1 makes the two relationships too alike
        break
    # A pair of points with gradient p/q.
    j = rng.choice([-3, -2, -1, 1, 2, 3])
    ax, ay = rng.randint(-6, 6), rng.randint(-6, 6)
    bx, by = ax + q * j, ay + p * j
    slope_ref = Fraction(p, q)
    slope_l = Fraction(-q, p) if perpendicular else slope_ref
    # x-coordinate of C so that the intercept is a whole number: a nonzero multiple of the gradient's denominator.
    k = rng.choice([-3, -2, -1, 1, 2, 3])
    cx = slope_l.denominator * k
    b = rng.randint(-12, 12)
    cy = int(b + slope_l * cx)
    rel = "perpendicular to" if perpendicular else "parallel to"
    text = phrase(rng, _TEMPLATES, ax=ax, ay=ay, bx=bx, by=by, cx=cx, cy=cy, rel=rel)

    # Independent route: parse the printed points, work out the relationship from the words.
    printed = {m[0]: (int(m[1]), int(m[2])) for m in re.findall(r"([ABC])\((-?\d+), (-?\d+)\)", text)}
    check(sorted(printed) == ["A", "B", "C"], "exactly the three printed points")
    (pax, pay), (pbx, pby), (pcx, pcy) = printed["A"], printed["B"], printed["C"]
    m_ref = Fraction(pby - pay, pbx - pax)
    m_l = (-1 / m_ref) if "perpendicular" in text else m_ref
    got = pcy - m_l * pcx
    check(got == b and got.denominator == 1, "re-derived intercept disagrees with the construction")

    other_slope = slope_ref if perpendicular else Fraction(-q, p)
    x_int = -Fraction(b) / slope_l
    raw = [
        cy,
        cy - other_slope * cx,  # other relationship between the gradients
        cy + slope_l * cx,  # wrong sign when back-solving
        x_int,  # x-intercept
        -b,  # sign reversed
    ]
    cands = [str(int(v)) for v in raw if Fraction(v).denominator == 1]
    return dict(
        text=text,
        latex=text,
        answer=b,
        answer_text=str(b),
        working=[
            f"Gradient of the reference line = ({join_terms([(pby, ''), (-pay, '')])}) / ({join_terms([(pbx, ''), (-pax, '')])}) = {slope_ref}.",
            f"Gradient of L = {slope_l}.",
            f"Intercept = {cy} - ({slope_l})({cx}) = {b}.",
        ],
        distractors=pick_distractors(str(b), cands),
    )
