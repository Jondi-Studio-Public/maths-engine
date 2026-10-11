"""Exam-tier source for `quadratic-vertex`: find the turning point of a parabola that is described, not given.

The equation is never shown. The learner is told where the parabola meets the x-axis and one more fact
about it (where it meets the y-axis, or another point it passes through), and must

1. use the second fact to find the stretch factor ``a`` in ``y = a(x - p)(x - q)``, then
2. place the turning point on the axis of symmetry halfway between the intercepts, and evaluate there.

Built backwards from ``a``, ``p`` and ``q``. The intercepts always have the same parity, so the
midpoint and the turning-point height are integers.
"""

from __future__ import annotations

import random

from sympy import Rational, expand, symbols

from ..core import GRADER_TOLERANCE, check, check_equivalent, generator, pick_distractors

x = symbols("x")


@generator("exd_x_a1_quadratic_vertex", level=9, presets={"exam": dict(tier="exam")})
def exd_x_a1_quadratic_vertex(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: turning point of a parabola given by its x-intercepts and one more fact."""
    check(tier == "exam", f"unknown tier {tier!r}")
    while True:
        a = rng.choice([-3, -2, -1, 2, 3])
        p = rng.randint(-7, 7)
        q = p + 2 * rng.randint(1, 5)  # same parity, p < q
        if p == 0 or q == 0 or p + q == 0:
            continue
        h = (p + q) // 2
        break
    k = -a * ((q - p) // 2) ** 2
    check_equivalent(
        expand(a * (x - p) * (x - q)).subs(x, h), k, "turning point height disagrees with the expanded parabola"
    )
    check_equivalent(
        Rational(-(expand(a * (x - p) * (x - q))).coeff(x, 1), 2 * a), h, "axis of symmetry disagrees with -b/2a"
    )

    mode = rng.choice(["y_intercept", "point", "graph"])
    if mode == "y_intercept":
        shown_y = a * p * q
        text = (
            f"A parabola crosses the x-axis at x = {p} and x = {q}, and crosses the y-axis at y = {shown_y}. "
            "Find its turning point (vertex)."
        )
    else:
        while True:
            m = rng.randint(-8, 8)
            if m not in (p, q) and m != h:
                break
        shown_y = a * (m - p) * (m - q)
        if mode == "point":
            text = (
                f"A parabola crosses the x-axis at x = {p} and x = {q}, and passes through the point "
                f"({m}, {shown_y}). Find its turning point (vertex)."
            )
        else:
            text = (
                f"The graph of a quadratic function has x-intercepts ({p}, 0) and ({q}, 0) and passes through "
                f"({m}, {shown_y}). Find the coordinates of its turning point."
            )

    answer_text = f"({h}, {k})"
    cands = [
        f"({-h}, {k})",  # sign error on the x-coordinate
        f"({h}, {-k})",  # sign error on the y-coordinate
        f"({h}, {-(((q - p) // 2) ** 2)})",  # took a = 1, skipping the stretch factor
        f"({h}, {shown_y})",  # read the given y-value as the turning-point height
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"x": h, "y": k},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"y = a(x - {p})(x - {q}); the axis of symmetry is x = {h}.", f"Turning point {answer_text}."],
        distractors=pick_distractors(answer_text, cands, tolerance=GRADER_TOLERANCE),
    )
