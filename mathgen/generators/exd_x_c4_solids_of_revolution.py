"""Exam source for volumes of solids of revolution: a region between a parabola and a line.

The region is enclosed by ``y = c*x^2`` and ``y = c*m*x`` and is rotated about the x-axis or the y-axis. The
limits are not given: the learner finds the intersections, decides which curve is outer, and sets up a washer
(difference of squares, not square of a difference).

Closed forms (checked against sympy integration on every question):

    about the x-axis:  V = pi * integral_0^m [(c m x)^2 - (c x^2)^2] dx = 2 pi c^2 m^5 / 15
    about the y-axis:  V = pi * integral_0^{c m^2} [y/c - (y/(c m))^2] dy = pi c m^4 / 6
"""

from __future__ import annotations

import random

from sympy import Symbol, integrate, pi, sqrt

from ..core import check_distinct, check_equivalent, generator, latex_prompt, pick_distractors

_X = Symbol("x", positive=True)
_Y = Symbol("y", positive=True)


def _fmt(expr) -> str:
    return str(expr).replace("**", "^")


def _about_x(c: int, m: int) -> tuple:
    outer, inner = c * m * _X, c * _X**2
    answer = pi * integrate(outer**2 - inner**2, (_X, 0, m))
    wrong = [
        pi * integrate((outer - inner) ** 2, (_X, 0, m)),  # squared the difference instead of differencing squares
        integrate(outer**2 - inner**2, (_X, 0, m)),  # forgot pi
        pi * integrate(outer**2, (_X, 0, m)),  # ignored the hole: the line alone
    ]
    return answer, wrong


def _about_y(c: int, m: int) -> tuple:
    top = c * m**2
    # At height y the parabola is at radius sqrt(y/c) (outer) and the line at y/(c m) (inner).
    outer, inner = sqrt(_Y / c), _Y / (c * m)
    answer = pi * integrate(outer**2 - inner**2, (_Y, 0, top))
    wrong = [
        pi * integrate((outer - inner) ** 2, (_Y, 0, top)),  # squared the difference instead of differencing squares
        integrate(outer**2 - inner**2, (_Y, 0, top)),  # forgot pi
        _about_x(c, m)[0],  # used the x-axis washer formula
    ]
    return answer, wrong


def _build(rng: random.Random) -> dict:
    c = rng.randint(1, 3)
    m = rng.randint(2, 5)
    axis = rng.choice(["x", "y"])
    answer, wrong = _about_x(c, m) if axis == "x" else _about_y(c, m)
    closed = 2 * pi * c**2 * m**5 / 15 if axis == "x" else pi * c * m**4 / 6
    check_equivalent(answer, closed)
    for w in wrong:
        check_distinct(answer, w)
    answer_text = _fmt(answer)
    distractors = pick_distractors(answer_text, [_fmt(w) for w in wrong])

    para = f"y = {c}x^2" if c > 1 else "y = x^2"
    line = f"y = {c * m}x"
    if rng.random() < 0.5:
        text = (
            f"The finite region enclosed between the parabola {para} and the line {line} is rotated about the "
            f"{axis}-axis. Find the exact volume of the solid formed. Give your answer in terms of pi."
        )
    else:
        text = (
            f"The curves {para} and {line} bound a region. Rotate it through one full turn about the "
            f"{axis}-axis and find the exact volume of the solid. Give your answer in terms of pi."
        )
    return {
        "text": text,
        "latex": latex_prompt(text),
        "answer": answer_text,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": [
            f"The curves meet where {c}x^2 = {c * m}x, so x = 0 and x = {m}",
            "Use the washer method: outer radius squared minus inner radius squared",
            f"V = {answer_text}",
        ],
    }


@generator("exd_x_c4_solids_of_revolution", level=17, presets={"exam": dict(tier="exam")})
def exd_x_c4_solids_of_revolution(rng: random.Random, tier: str = "exam") -> dict:
    """Washer volume of the region between a parabola and a line, limits found by the learner."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
