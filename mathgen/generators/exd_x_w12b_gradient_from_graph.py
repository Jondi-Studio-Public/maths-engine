"""Exam chain for "determine gradient from a graph": a segment between two marked points A and B.

A straight line is drawn through two marked points A and B on a grid whose axes may use different scales. The chain
finds the gradient, the equation, the midpoint and the length of the segment AB:

(a) the gradient of the line                                       fraction,   2 marks
(b) the equation of the line                                       expression, 2 marks, follow-through (a)
(c) the midpoint of AB                                             pair,       1 mark
(d) the length of AB, to 1 decimal place                           number,     2 marks, follow-through (a)

A and B are chosen so the rise and the run (in axis units) are both even, so the midpoint is a lattice point. The
length is checked against ``math.hypot`` and is kept away from a rounding tie.

## Distractors

* (a): run over rise, the sign reversed, the rise on its own, the gradient counted in grid squares.
* (b): either sign reversed, the reciprocal gradient, A's y-value as the intercept.
* (c): the coordinates added and not halved, the differences halved, the coordinates swapped.
* (d): the rise and run added, the squares added and not rooted, the run alone, the rise alone.
"""

from __future__ import annotations

import math

from ..core import generator, paren_negative, phrase, round_half_up
from ._exd_x_sc4_kit import num_part, txt
from ._exd_x_w12b_kit import (
    F,
    check,
    draw_line,
    eq_text,
    expr_part,
    frac_part,
    frac_text,
    line_figure,
    marked,
    no_leak,
    pair_part,
    point_text,
    real_mc,
    real_pt,
    retry,
)

_SLOPES = [F(s) for s in (1, 2, 3, -1, -2, -3)] + [F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(2, 3), F(-2, 3)]
_SCALES = ((1, 1), (1, 1), (1, 2), (2, 1), (1, 5))

_STEMS = (
    "The points A and B are marked on a straight line drawn on this coordinate grid.",
    "A straight line passes through the marked points A and B on the grid below.",
    "On the coordinate grid, the line through the marked points A and B is drawn.",
)


def _build(rng):
    win, M, C, pts = draw_line(
        rng, slopes=_SLOPES, c_range=(-4, 4), y_axis=None, scales=_SCALES, min_points=3, min_du=2
    )
    pairs = [
        (a, b)
        for i, a in enumerate(pts)
        for b in pts[i + 1 :]
        if (b[0] - a[0]) * win.xs % 2 == 0 and (b[1] - a[1]) * win.ys % 2 == 0 and b[0] - a[0] >= 2
    ]
    check(bool(pairs), "no pair of points with a whole midpoint")
    pa, pb = rng.choice(pairs)
    m, c = real_mc(win, M, C)
    (xa, ya), (xb, yb) = real_pt(win, pa), real_pt(win, pb)
    dx, dy = xb - xa, yb - ya
    check(m != 0 and F(dy, dx) == m and ya == m * xa + c, "points are not on the line")
    mid = ((xa + xb) // 2, (ya + yb) // 2)
    length = F(math.hypot(dx, dy)).limit_denominator(10**9)
    scaled = length * 10
    check(abs(scaled - math.floor(scaled) - F(1, 2)) > F(1, 20), "length too close to a rounding tie")
    check(round_half_up(length, 1) not in (abs(dx), abs(dy)), "length reads as one component")
    fig = line_figure(
        win,
        [(pa, pb)],
        "a straight line through two marked points on a coordinate grid",
        points=marked(win, [pa, pb], ["A", "B"]),
    )
    stem = phrase(rng, _STEMS)

    part_a = frac_part(
        "Find the gradient of the line AB.",
        m,
        2,
        [f"A is ({xa}, {ya}) and B is ({xb}, {yb}).", f"gradient = {dy} / {dx} = {frac_text(m)}."],
        [
            (1 / m, "inverted-gradient"),
            (-m, "line-gradient-sign-reversed"),
            (F(dy), "gradient-rise-only"),
            (F(M), "gradient-read-in-grid-squares"),
        ],
    )
    part_b = expr_part(
        "Hence find the equation of the line AB in the form y = mx + c.",
        m,
        c,
        2,
        [f"Using A: {ya} = {frac_text(m)} × ({xa}) + c, so c = {frac_text(c)}.", f"The equation is {eq_text(m, c)}."],
        [
            ((-m, ya + m * xa), "line-gradient-sign-reversed"),
            ((m, -c), "line-intercept-sign-reversed"),
            ((1 / m, ya - xa / m), "inverted-gradient"),
            ((m, F(ya)), "intercept-read-as-point-height"),
        ],
        follow=f"part_a*x+({ya})-part_a*({xa})",
    )
    part_c = pair_part(
        "Find the coordinates of the midpoint of AB.",
        mid,
        1,
        [f"x = ({xa} + ({xb})) ÷ 2 = {mid[0]}, y = ({ya} + ({yb})) ÷ 2 = {mid[1]}."],
        [
            ((xa + xb, ya + yb), "midpoint-sum-not-halved"),
            ((dx // 2, dy // 2), "midpoint-half-the-difference"),
            ((mid[1], mid[0]), "components-swapped"),
        ],
    )
    part_d = num_part(
        "Hence find the length of AB, correct to 1 decimal place.",
        length,
        1,
        2,
        [
            f"Length = √({paren_negative(dx)}² + {paren_negative(dy)}²) = √{dx * dx + dy * dy}.",
            f"= {txt(length, 1)} (1 d.p.).",
        ],
        [
            (abs(dx) + abs(dy), "adds-instead-of-using-pythagoras"),
            (dx * dx + dy * dy, "distance-squared-not-rooted"),
            (abs(dx), "ignores-the-vertical-component"),
            (abs(dy), "ignores-the-horizontal-component"),
        ],
        follow_through=f"{abs(dx)}*sqrt(1+part_a**2)",
        key_places=True,
    )
    parts = [part_a, part_b, part_c, part_d]
    no_leak(fig, parts)
    return dict(text=stem, figure=fig, parts=parts)


@generator("exd_x_w12b_gradient_from_graph", level=12, presets={"exam": dict(tier="exam")})
def exd_x_w12b_gradient_from_graph(rng, tier="exam"):
    """Gradient, equation, midpoint and length of a segment between two marked points; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _build, "a segment with a whole midpoint")
