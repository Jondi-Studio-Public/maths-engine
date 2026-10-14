"""Exam chain for "determine a linear equation from a graph": the line, then a parallel line through a point.

A straight line is drawn on a grid with two points marked; the y-axis is in view about half the time and out of the
window otherwise, so the intercept is found by running the line back. A second line is parallel to the first and
passes through a point (p, q) given in words.

(a) the equation of the drawn line                                  expression, 2 marks
(b) the equation of the parallel line through (p, q)                expression, 2 marks, follow-through (a)
(c) the x-intercept of the parallel line                            fraction,   2 marks, follow-through (b)
(d) the vertical distance between the two lines                     fraction,   1 mark,  follow-through (a), (b)

The point (p, q) has p a multiple of the gradient's denominator, so the parallel line's intercept is a whole number,
different from the first line's and not zero.

## Distractors

* (a): either sign reversed, the reciprocal gradient, the gradient and intercept swapped.
* (b): the first line's intercept kept, the perpendicular gradient, q taken as the intercept, the gradient sign
  reversed.
* (c): the sign lost, the ratio inverted, the y-intercept given.
* (d): the intercepts added, the second intercept alone, the signed difference, the horizontal distance.
"""

from __future__ import annotations

from fractions import Fraction as F

from ..core import generator, join_terms, phrase
from ._exd_x_w12b_kit import (
    F,
    check,
    draw_line,
    eq_text,
    expr_part,
    frac_part,
    frac_text,
    icpt_of,
    line_figure,
    marked,
    no_leak,
    pick_pair,
    real_mc,
    real_pt,
    retry,
    slope_of,
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
    "A straight line is drawn on the coordinate grid, with two points marked on it. A second line is parallel to "
    "this line and passes through the point ({p}, {q}).",
    "The grid shows a straight line with two marked points. A second line, parallel to it, goes through the "
    "point ({p}, {q}).",
    "Two points are marked on the straight line drawn on this grid. Another line is parallel to the drawn line "
    "and passes through ({p}, {q}).",
)


def _build(rng):
    hidden = rng.random() < 0.5
    win, M, C, pts = draw_line(
        rng,
        slopes=_SLOPES,
        c_range=(-9, 9) if hidden else (-4, 4),
        y_axis=False if hidden else True,
        c_visible=not hidden,
        c_nonzero=True,
        min_points=2,
        min_du=2,
    )
    pa, pb = pick_pair(rng, pts)
    m, c1 = real_mc(win, M, C)
    (xa, ya), (xb, yb) = real_pt(win, pa), real_pt(win, pb)
    check(m != 0 and F(yb - ya, xb - xa) == m and ya == m * xa + c1, "points are not on the line")
    p = m.denominator * rng.choice([t for t in range(-4, 5) if t])
    c2 = F(rng.choice([v for v in range(-9, 10) if v not in (0, int(c1))]))
    q = m * p + c2
    check(q.denominator == 1 and p != 0 and c2 != c1, "the second line must pass through a lattice point")
    xint = -c2 / m
    gap = abs(c2 - c1)
    fig = line_figure(win, [(pa, pb)], "a straight line on a coordinate grid", points=marked(win, [pa, pb]))
    stem = phrase(rng, _STEMS, p=p, q=int(q))

    part_a = expr_part(
        "Find the equation of the drawn line in the form y = mx + c.",
        m,
        c1,
        2,
        [
            f"The marked points are ({xa}, {ya}) and ({xb}, {yb}), so m = {yb - ya} / {xb - xa} = {frac_text(m)}.",
            f"Running the line to x = 0 gives c = {frac_text(c1)}, so {eq_text(m, c1)}.",
        ],
        [
            ((-m, c1), "line-gradient-sign-reversed"),
            ((m, -c1), "line-intercept-sign-reversed"),
            ((1 / m, c1), "inverted-gradient"),
            ((c1, m), "swaps-slope-intercept"),
        ],
    )
    part_b = expr_part(
        f"Find the equation of the parallel line through ({p}, {int(q)}) in the form y = mx + c.",
        m,
        c2,
        2,
        [
            f"Parallel lines have the same gradient, m = {frac_text(m)}.",
            f"Substituting ({p}, {int(q)}): {int(q)} = {frac_text(m)} × ({p}) + c, so c = {frac_text(c2)}.",
            f"The equation is {eq_text(m, c2)}.",
        ],
        [
            ((m, c1), "parallel-line-keeps-old-intercept"),
            ((-1 / m, q + p / m), "used-perpendicular-gradient"),
            ((m, q), "y-value-as-intercept"),
            ((-m, q + m * p), "line-gradient-sign-reversed"),
        ],
        follow=f"{slope_of('a')}*x+({int(q)})-{slope_of('a')}*({p})",
    )
    part_c = frac_part(
        "Find the x-coordinate of the point where the parallel line crosses the x-axis.",
        xint,
        2,
        [
            f"On the x-axis y = 0, so 0 = {join_terms([(F(m), 'x'), (F(c2), '')])}.",
            f"x = {frac_text(-c2)} ÷ {frac_text(m)} = {frac_text(xint)}.",
        ],
        [(c2 / m, "x-intercept-sign-reversed"), (-m / c2, "x-intercept-ratio-inverted"), (c2, "intercepts-confused")],
        follow=f"-{icpt_of('b')}/{slope_of('b')}",
        sensible=(-500, 500),
    )
    part_d = frac_part(
        "Find the vertical distance between the two lines.",
        gap,
        1,
        [
            f"The lines cross the y-axis at {frac_text(c1)} and {frac_text(c2)}.",
            f"The vertical distance is |{frac_text(c2)} - ({frac_text(c1)})| = {frac_text(gap)}.",
        ],
        [
            (c1 + c2, "parallel-gap-intercepts-added"),
            (c2, "intercepts-confused"),
            (c1 - c2, "answer-sign-reversed"),
            (abs(xint - (-c1 / m)), "parallel-gap-horizontal"),
        ],
        follow=f"abs({icpt_of('b')}-{icpt_of('a')})",
        sensible=(0, 500),
    )
    parts = [part_a, part_b, part_c, part_d]
    no_leak(fig, parts)
    return dict(text=stem, figure=fig, parts=parts)


@generator("exd_x_w12b_equation_from_graph", level=12, presets={"exam": dict(tier="exam")})
def exd_x_w12b_equation_from_graph(rng, tier="exam"):
    """The drawn line's equation, a parallel line through a point, its x-intercept and the gap; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _build, "a drawn line and a parallel line through a lattice point")
