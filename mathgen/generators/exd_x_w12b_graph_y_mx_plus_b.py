"""Exam chain for "match linear relations in slope-intercept form to their graphs": read, write, then use the rule.

A straight line is drawn with two points marked and the y-axis in view. The chain reads the gradient and the
y-intercept off the figure, writes the equation, and then uses it in one of two ways, chosen by the seed:

(a) the gradient                                                   fraction,   1 mark
(b) the y-intercept                                                fraction,   1 mark
(c) the equation y = mx + c                                        expression, 2 marks, follow-through (a), (b)
(d) either the x-intercept, or the value of y at an x outside the
    grid                                                           fraction,   2 marks, follow-through (a), (b)

The x-intercept is minus the intercept over the gradient and is not always a whole number. The x outside the grid is
a multiple of the gradient's denominator, so y is whole.

## Distractors

* (a): run over rise, the sign reversed, the intercept given as the gradient.
* (b): the gradient given as the intercept, the sign reversed, the x-intercept, the y-value of a marked point.
* (c): the gradient and intercept swapped, either sign reversed, the intercept left out.
* (d) x-intercept: the sign lost, the ratio inverted, the y-intercept given. (d) value: the intercept left out,
  the intercept subtracted, the gradient and intercept swapped.
"""

from __future__ import annotations

from fractions import Fraction as F

from ..core import generator, phrase
from ._exd_x_w12b_kit import (
    F,
    beyond_window,
    check,
    draw_line,
    eq_text,
    expr_part,
    frac_part,
    frac_text,
    line_figure,
    marked,
    no_leak,
    pick_pair,
    real_mc,
    real_pt,
    retry,
)

_SLOPES = [F(s) for s in (1, 2, 3, -1, -2, -3)] + [
    F(1, 2),
    F(-1, 2),
    F(3, 2),
    F(-3, 2),
    F(1, 3),
    F(-1, 3),
    F(2, 3),
    F(-2, 3),
]

_STEMS = (
    "A straight line is drawn on the coordinate grid, with two points marked on it.",
    "The grid shows a straight line. Two points on the line are marked.",
    "Study the straight line on this coordinate grid. Two points on it are marked.",
)


def _build(rng):
    win, M, C, pts = draw_line(
        rng, slopes=_SLOPES, c_range=(-5, 5), y_axis=True, c_visible=True, c_nonzero=True, min_points=2, min_du=2
    )
    pa, pb = pick_pair(rng, pts)
    m, c = real_mc(win, M, C)
    (xa, ya), (xb, yb) = real_pt(win, pa), real_pt(win, pb)
    check(m != 0 and c != 0 and F(yb - ya, xb - xa) == m and ya == m * xa + c, "points are not on the line")
    xint = -c / m
    fig = line_figure(win, [(pa, pb)], "a straight line on a coordinate grid", points=marked(win, [pa, pb]))
    stem = phrase(rng, _STEMS)

    part_a = frac_part(
        "Find the gradient of the line.",
        m,
        1,
        [
            f"The marked points are ({xa}, {ya}) and ({xb}, {yb}).",
            f"gradient = {yb - ya} / {xb - xa} = {frac_text(m)}.",
        ],
        [(1 / m, "inverted-gradient"), (-m, "line-gradient-sign-reversed"), (c, "swaps-slope-intercept")],
    )
    part_b = frac_part(
        "Write down the y-intercept of the line.",
        c,
        1,
        [f"The line crosses the y-axis at y = {frac_text(c)}."],
        [
            (m, "swaps-slope-intercept"),
            (-c, "line-intercept-sign-reversed"),
            (xint, "intercepts-confused"),
            (F(ya), "intercept-read-as-point-height"),
        ],
    )
    part_c = expr_part(
        "Hence write the equation of the line in the form y = mx + c.",
        m,
        c,
        2,
        [f"m = {frac_text(m)} and c = {frac_text(c)}.", f"The equation is {eq_text(m, c)}."],
        [
            ((c, m), "swaps-slope-intercept"),
            ((-m, c), "line-gradient-sign-reversed"),
            ((m, -c), "line-intercept-sign-reversed"),
            ((m, 0), "intercept-left-out"),
        ],
        follow="part_a*x+part_b",
    )
    if rng.random() < 0.5:
        part_d = frac_part(
            "Find the x-coordinate of the point where the line crosses the x-axis.",
            xint,
            2,
            [
                f"On the x-axis y = 0, so 0 = {eq_text(m, c)[4:]}.",
                f"x = {frac_text(-c)} ÷ {frac_text(m)} = {frac_text(xint)}.",
            ],
            [(c / m, "x-intercept-sign-reversed"), (-m / c, "x-intercept-ratio-inverted"), (c, "intercepts-confused")],
            follow="-part_b/part_a",
            sensible=(-500, 500),
        )
    else:
        x_far = beyond_window(rng, win, m.denominator, lo=6, hi=30)
        y_far = m * x_far + c
        check(y_far.denominator == 1 and abs(y_far) <= 200, "y at the far x must be whole")
        part_d = frac_part(
            f"Use your equation to find the value of y when x = {x_far}.",
            y_far,
            2,
            [f"y = {frac_text(m)} × {x_far} + ({frac_text(c)}) = {frac_text(y_far)}."],
            [
                (m * x_far, "dropped-intercept-in-prediction"),
                (m * x_far - c, "line-intercept-sign-reversed"),
                (c * x_far + m, "swaps-slope-intercept"),
            ],
            follow=f"part_a*({x_far})+part_b",
            sensible=(-500, 500),
        )
    parts = [part_a, part_b, part_c, part_d]
    no_leak(fig, parts)
    return dict(text=stem, figure=fig, parts=parts)


@generator("exd_x_w12b_graph_y_mx_plus_b", level=11, presets={"exam": dict(tier="exam")})
def exd_x_w12b_graph_y_mx_plus_b(rng, tier="exam"):
    """Read gradient and intercept, write the equation, then use it; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _build, "a line with a visible y-intercept")
