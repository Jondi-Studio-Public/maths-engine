"""Exam chain for proportional graphs ``y = mx``: a line through the origin and a horizontal line that crosses it.

A straight line through the origin is drawn with one point P marked, and a horizontal line is drawn across it. The
axes sometimes use different scales, so the gradient has to be read in axis units rather than in grid squares.

(a) the gradient of the line                                       fraction,   2 marks
(b) the equation of the line                                       expression, 2 marks, follow-through (a)
(c) the value of y at a given x outside the grid                    fraction,   1 mark,  follow-through (a)
(d) the x-coordinate where the line meets the horizontal line      fraction,   2 marks, follow-through (a)

The given x in (c) is a multiple of the gradient's denominator, so y is a whole number. The horizontal line is
labelled with its equation, which is data the question gives, and is drawn through a grid crossing of the line so the
meeting point can be checked against the figure.

## Distractors

* (a): run over rise, the sign reversed, P's y-value as the gradient, the gradient counted in grid squares.
* (b): the reciprocal gradient, the gradient with the wrong sign, an unstated intercept added.
* (c): x divided by the gradient, the gradient added to x, the sign reversed.
* (d): the height multiplied by the gradient, the height given as x, the sign reversed.
"""

from __future__ import annotations

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
    real_mc,
    real_pt,
    retry,
)

_SLOPES = [F(s) for s in (1, 2, 3, -1, -2, -3)] + [F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(1, 3), F(-2, 3), F(2, 3)]
_SCALES = ((1, 1), (1, 1), (1, 2), (1, 5))

_STEMS = (
    "A straight line through the origin is drawn on the grid, with the point P marked on it. A horizontal line "
    "is drawn across it.",
    "The grid shows a straight line through the origin and a horizontal line. The point P is on the sloping line.",
    "A sloping line through the origin and a horizontal line are drawn on this grid. P is a marked point on the "
    "sloping line.",
)


def _build(rng):
    win, M, C, pts = draw_line(rng, slopes=_SLOPES, y_axis=True, origin=True, scales=_SCALES, min_points=3, min_du=1)
    others = [p for p in pts if p != (0, 0)]
    p_pt = rng.choice(others)
    cross = [p for p in others if p != p_pt and win.v0 < p[1] < win.v1]
    check(bool(cross), "no grid crossing for the horizontal line")
    h_pt = rng.choice(cross)
    k, _c0 = real_mc(win, M, C)
    xp, yp = real_pt(win, p_pt)
    xh, yh = real_pt(win, h_pt)
    check(k * xp == yp and k * xh == yh and k != 0, "the points are not on y = kx")
    x_far = beyond_window(rng, win, k.denominator, lo=6, hi=40)
    y_far = k * x_far
    check(y_far.denominator == 1 and abs(y_far) <= 200, "y at the far x must be whole")
    check(abs(yh) >= 1 and F(xh) != yh, "the horizontal line should not sit on the identity")
    fig = line_figure(
        win,
        [((0, 0), p_pt), ((win.u0, h_pt[1]), (win.u1, h_pt[1]))],
        "a straight line through the origin and a horizontal line on a coordinate grid",
        labels=[None, f"y = {yh}"],
        points=marked(win, [p_pt], ["P"]),
    )
    stem = phrase(rng, _STEMS)

    squares = F(M)
    part_a = frac_part(
        "Find the gradient of the sloping line.",
        k,
        2,
        [f"The line passes through the origin and P({xp}, {yp}).", f"gradient = {yp} / {xp} = {frac_text(k)}."],
        [
            (1 / k, "inverted-gradient"),
            (-k, "line-gradient-sign-reversed"),
            (F(yp), "gave-y-value-for-gradient"),
            (squares, "gradient-read-in-grid-squares"),
        ],
    )
    part_b = expr_part(
        "Hence write the equation of the sloping line in the form y = mx.",
        k,
        0,
        2,
        [
            f"A line through the origin has the form y = mx, with m = {frac_text(k)}.",
            f"The equation is {eq_text(k, 0)}.",
        ],
        [
            ((1 / k, 0), "inverted-gradient"),
            ((-k, 0), "line-gradient-sign-reversed"),
            ((k, 1), "adds-an-unstated-intercept"),
        ],
        follow="part_a*x",
    )
    part_c = frac_part(
        f"Use your equation to find the value of y on the line when x = {x_far}.",
        y_far,
        1,
        [f"y = ({frac_text(k)}) × {x_far} = {frac_text(y_far)}."],
        [(x_far / k, "inverted-gradient"), (x_far + k, "added-gradient-to-x"), (-y_far, "answer-sign-reversed")],
        follow=f"part_a*({x_far})",
        sensible=(-500, 500),
    )
    part_d = frac_part(
        f"Find the x-coordinate of the point where the sloping line meets the horizontal line y = {yh}.",
        F(xh),
        2,
        [f"Put y = {yh} into y = {frac_text(k)}x: x = {yh} ÷ ({frac_text(k)}) = {xh}."],
        [(yh * k, "solved-for-x-by-multiplying"), (F(yh), "y-value-given-as-x"), (F(-xh), "answer-sign-reversed")],
        follow=f"({yh})/part_a",
        sensible=(-500, 500),
    )
    parts = [part_a, part_b, part_c, part_d]
    no_leak(fig, parts)
    return dict(text=stem, figure=fig, parts=parts)


@generator("exd_x_w12b_graph_y_mx", level=11, presets={"exam": dict(tier="exam")})
def exd_x_w12b_graph_y_mx(rng, tier="exam"):
    """Read a proportional line, write its equation, use it and meet a horizontal line; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _build, "a proportional line with a horizontal crossing")
