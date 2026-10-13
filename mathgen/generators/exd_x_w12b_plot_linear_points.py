"""Exam chain for "read points representing a relationship": three plotted points on a hidden straight line.

Three points A, B and C are plotted on a labelled grid and lie on one straight line (no line is drawn). The chain
reads them, finds the rule they follow, and then uses the rule beyond the edge of the grid, forwards and backwards:

(a) the coordinates of B                                           pair,     1 mark
(b) the gradient of the line through A, B and C                    fraction, 2 marks
(c) the y-coordinate of the point of the line at a given x          fraction, 2 marks, follow-through (b)
(d) the x-coordinate of the point of the line at a given y          fraction, 2 marks, follow-through (b)

The given x in (c) and the given y in (d) both lie outside the grid, and each is chosen so the answer is a whole
number. The key for (c) and (d) is recomputed from the three plotted points alone, and (a) and (b) are read back from
them, before the question is returned.

## Distractors

* (a): the pair written (y, x), and either coordinate with its sign reversed.
* (b): run over rise, the sign reversed, the rise on its own.
* (c): the intercept left out (gradient times x), the rise added once however far along, the sign reversed.
* (d): multiplying by the gradient instead of dividing, dividing y by the gradient without allowing for the
  starting point, the sign reversed.
"""

from __future__ import annotations

from ..core import generator, join_terms, phrase
from ..figures import cartesian
from ._exd_x_w12b_kit import (
    F,
    beyond_window,
    check,
    draw_line,
    frac_part,
    frac_text,
    marked,
    no_leak,
    pair_part,
    point_text,
    retry,
)

_SLOPES = [F(s) for s in (-3, -2, -1, 1, 2, 3)] + [F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(1, 3), F(-1, 3)]

_STEMS = (
    "Three points, A, B and C, are plotted on the coordinate grid. They lie on the same straight line.",
    "The points A, B and C are plotted on this coordinate grid and all lie on one straight line.",
    "On the grid, the points A, B and C lie on a straight line. The line itself is not drawn.",
)


def _build(rng):
    win, _M, _C, pts = draw_line(rng, slopes=_SLOPES, c_range=(-4, 4), y_axis=None, min_points=3, min_du=2)
    i, j, k = sorted(rng.sample(range(len(pts)), 3))
    a, b, c = pts[i], pts[j], pts[k]
    check(c[0] - a[0] >= 2, "A and C too close")
    (xa, ya), (xb, yb), (xc, yc) = a, b, c
    m = F(yc - ya, xc - xa)
    check(m != 0 and F(yb - ya) == m * (xb - xa), "A, B and C are not on one line")
    den = m.denominator
    x_far = beyond_window(rng, win, den, lo=6, hi=30)
    y_far = ya + m * (x_far - xa)
    x_at = beyond_window(rng, win, den, lo=6, hi=30)
    y_target = ya + m * (x_at - xa)
    check(y_far.denominator == 1 and y_target.denominator == 1, "predicted values must be whole")
    check(abs(y_far) <= 80 and abs(y_target) <= 80, "predicted values too large")
    check(x_far != x_at and y_far != y_target, "the two predictions must differ")
    check(y_target != x_at and y_far != x_far, "a value equal to its own argument gives the question away")
    fig = cartesian(
        x_range=win.x_range,
        y_range=win.y_range,
        alt="three points plotted on a coordinate grid",
        grid=1,
        equal_scale=True,
        points=marked(win, [a, b, c], ["A", "B", "C"]),
    )
    stem = phrase(rng, _STEMS)

    part_a = pair_part(
        "Write down the coordinates of B.",
        (xb, yb),
        1,
        [f"Reading across then up, B is at {point_text(xb, yb)}."],
        [((yb, xb), "components-swapped"), ((-xb, yb), "answer-sign-reversed"), ((xb, -yb), "answer-sign-reversed")],
    )
    part_b = frac_part(
        "Find the gradient of the line through A, B and C.",
        m,
        2,
        [
            f"A is {point_text(xa, ya)} and C is {point_text(xc, yc)}.",
            f"gradient = rise / run = {yc - ya} / {xc - xa} = {frac_text(m)}.",
        ],
        [(1 / m, "inverted-gradient"), (-m, "line-gradient-sign-reversed"), (F(yc - ya), "gradient-rise-only")],
    )
    part_c = frac_part(
        f"The same line is extended beyond the grid. Use your gradient to find the y-coordinate of the point on the "
        f"line where x = {x_far}.",
        y_far,
        2,
        [
            f"From A, x changes by {x_far - xa}, so y changes by ({frac_text(m)}) × {x_far - xa} = {frac_text(m * (x_far - xa))}.",
            f"y = {join_terms([(ya, ''), (int(m * (x_far - xa)), '')])} = {frac_text(y_far)}.",
        ],
        [
            (m * x_far, "dropped-intercept-in-prediction"),
            (F(yc + (yc - ya)), "prediction-rise-added-once"),
            (-y_far, "answer-sign-reversed"),
        ],
        follow=f"{ya}+part_b*({x_far}-({xa}))",
        sensible=(-500, 500),
    )
    part_d = frac_part(
        f"Use your gradient to find the x-coordinate of the point on the line where y = {frac_text(y_target)}.",
        F(x_at),
        2,
        [
            f"From A, y changes by {frac_text(y_target - ya)}, so x changes by {frac_text(y_target - ya)} ÷ ({frac_text(m)}) = {x_at - xa}.",
            f"x = {join_terms([(xa, ''), (x_at - xa, '')])} = {x_at}.",
        ],
        [
            (xa + (y_target - ya) * m, "solved-for-x-by-multiplying"),
            (y_target / m, "inverse-prediction-ignores-intercept"),
            (F(-x_at), "answer-sign-reversed"),
        ],
        follow=f"{xa}+({frac_text(y_target)}-({ya}))/part_b",
        sensible=(-500, 500),
    )
    parts = [part_a, part_b, part_c, part_d]
    no_leak(fig, parts)
    return dict(text=stem, figure=fig, parts=parts)


@generator("exd_x_w12b_plot_linear_points", level=10, presets={"exam": dict(tier="exam")})
def exd_x_w12b_plot_linear_points(rng, tier="exam"):
    """Read three plotted points, find the gradient, then predict forwards and backwards; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _build, "three plotted points with whole-number predictions")
