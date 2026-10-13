"""Exam-tier source for `least-squares-line`: fit a line to raw paired data, then use it.

Five equally spaced x-values and their y-values are given as a table in a short context. The learner

1. finds the slope of the least-squares line (no sums are supplied),
2. finds the intercept, using the slope and the means,
3. predicts y at a new x beyond the data, and
4. finds the residual (observed minus predicted) for one of the observations.

Built backwards: choose the line, then add a residual pattern that is orthogonal to both the constant and x, so the
least-squares line of the data is exactly the chosen one and every answer is on the 0.5 grid.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import (
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)
from .exd_x_u6_support import keys_clear, tagged

# Residual patterns over five equally spaced x-values; each sums to zero and is orthogonal to x.
_PATTERNS = [(1, -2, 0, 2, -1), (1, -1, -1, 1, 0), (0, 1, -2, 1, 0)]

# intro, x label, y label, x description, unit text, y description, x0 choices, spacing, slope halves, intercept halves
_CONTEXTS = [
    (
        "A cafe in Fremantle records the daily maximum temperature and the number of cold drinks it sells on five days.",
        "Temperature (°C)",
        "Cold drinks sold",
        "the temperature",
        " °C",
        "the number of cold drinks sold",
        [16, 18, 20, 22],
        2,
        range(5, 17),
        range(-60, -10),
    ),
    (
        "A student in Adelaide records the hours spent revising each week and the mark scored in five maths tests.",
        "Revision (hours)",
        "Test mark",
        "the revision time",
        " hours",
        "the test mark",
        [2, 3, 4, 5],
        1,
        range(3, 11),
        range(40, 80),
    ),
    (
        "A fencing crew in Toowoomba records the number of workers on site and the length of fence built on five days.",
        "Workers",
        "Fence built (m)",
        "the number of workers",
        " workers",
        "the length of fence built (m)",
        [2, 3, 4, 5],
        1,
        range(5, 15),
        range(0, 20),
    ),
    (
        "A market gardener near Launceston records the weekly rainfall and the mass of lettuce harvested over five weeks.",
        "Rainfall (mm)",
        "Harvest (kg)",
        "the rainfall",
        " mm",
        "the mass of lettuce harvested (kg)",
        [10, 15, 20, 25],
        5,
        range(1, 7),
        range(10, 60),
    ),
]

# Contexts whose y-values are counts of things, so every observed y must be a whole number. The line is still exact:
# only draws whose chosen line gives whole-number y-values at all five x-values are kept (the slope and intercept of
# the line itself may still be halves).
_WHOLE_NUMBER_Y = {"Cold drinks sold"}


def _text(value: Fraction) -> str:
    return stated_precision_string(value, 1)


def _shown(value: Fraction) -> str:
    return str(int(value)) if value.denominator == 1 else stated_precision_string(value, 1)


def _build(rng: random.Random):
    (intro, x_label, y_label, x_desc, unit, y_desc, x0s, step, slopes, intercepts) = rng.choice(_CONTEXTS)
    x0 = rng.choice(x0s)
    xs = [x0 + step * i for i in range(5)]
    m = Fraction(rng.choice(list(slopes)), 2)
    c = Fraction(rng.choice(list(intercepts)), 2)
    coeffs = [rng.randint(-2, 2) for _ in _PATTERNS]
    if not any(coeffs):
        return None
    e = [sum(k * pat[i] for k, pat in zip(coeffs, _PATTERNS)) for i in range(5)]
    ys = [c + m * x + ei for x, ei in zip(xs, e)]
    if min(ys) <= 0 or len(set(ys)) < 5 or len(set(e)) < 3:
        return None
    if y_label in _WHOLE_NUMBER_Y and any(y.denominator != 1 for y in ys):
        return None

    # Independent least-squares fit on the data as the learner sees it.
    xbar, ybar = Fraction(sum(xs), 5), sum(ys) / 5
    sxx = sum((x - xbar) ** 2 for x in xs)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    check(sxy / sxx == m and ybar - m * xbar == c, "chosen line is not the least-squares line of the data")

    x_pred = xs[-1] + step * rng.choice([1, 2])
    pred = c + m * x_pred
    k = rng.choice([i for i in range(5) if e[i] != 0])
    resid = ys[k] - (c + m * xs[k])
    check(resid == e[k])

    stem = f"{intro} {x_label}: {', '.join(str(x) for x in xs)}. {y_label}: {', '.join(_shown(y) for y in ys)}."
    p1 = f"Find the slope of the least-squares line for predicting {y_desc} from {x_desc}."
    p2 = "Using your slope, find the y-intercept of the line."
    p3 = f"Use your line to predict {y_desc} when {x_desc} is {x_pred}{unit}."
    p4 = (
        f"For the observation where {x_desc} was {xs[k]}{unit}, the observed value was {_shown(ys[k])}. "
        "Using your line, find the residual (observed value minus predicted value)."
    )
    if not keys_clear(" ".join([stem, p1, p2, p3, p4]), [m, c, pred, resid]):
        return None
    if 0 in (m, c, pred):
        return None

    def tag(value, mid):
        value = round_half_up(value, 1)
        return tagged(_text(value), mid, value)

    end_slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    a_d = pick_distractors(
        _text(m),
        [tag(c, "swaps-slope-intercept"), tag(end_slope, "slope-from-end-points")],
        3,
    )
    b_d = pick_distractors(
        _text(c),
        [
            tag(m, "swaps-slope-intercept"),
            tag(ys[0], "uses-first-point-intercept"),
            tag(ybar, "intercept-is-mean-of-y"),
        ],
        3,
    )
    c_d = pick_distractors(
        _text(pred),
        [
            tag(m * x_pred, "dropped-intercept-in-prediction"),
            tag(m + c * x_pred, "swaps-slope-intercept"),
            tag(ys[-1], None),
        ],
        3,
    )
    d_d = pick_distractors(
        _text(resid),
        [tag(-resid, "residual-sign-reversed"), tag(c + m * xs[k], None), tag(ys[k], None)],
        3,
    )
    if min(len(a_d), len(b_d), len(c_d), len(d_d)) < 2:
        return None

    def part(prompt, value, marks, distractors, working, follow_through=None):
        entry = dict(
            prompt=prompt,
            answer_kind="number",
            answer=numeric_answer(value),
            answer_text=_text(value),
            answer_latex=latex_answer(_text(value)),
            marks=marks,
            working=working,
            distractors=distractors,
        )
        if follow_through:
            entry["follow_through"] = follow_through
        return entry

    return dict(
        text=stem,
        parts=[
            part(
                p1,
                m,
                3,
                a_d,
                [
                    f"Mean x = {_shown(xbar)}, mean y = {_shown(ybar)}.",
                    f"Sxy / Sxx = {_shown(sxy)} / {_shown(sxx)} = {_text(m)}.",
                ],
            ),
            part(
                p2,
                c,
                2,
                b_d,
                [f"Intercept = mean y - slope × mean x = {_shown(ybar)} - {_text(m)} × {_shown(xbar)} = {_text(c)}."],
                follow_through=f"{ybar} - part_a*{xbar}",
            ),
            part(
                p3,
                pred,
                1,
                c_d,
                [f"y = {_text(c)} + {_text(m)} × {x_pred} = {_text(pred)}."],
                follow_through=f"part_b + part_a*{x_pred}",
            ),
            part(
                p4,
                resid,
                2,
                d_d,
                [
                    f"Predicted = {_text(c)} + {_text(m)} × {xs[k]} = {_text(c + m * xs[k])}.",
                    f"Residual = {_shown(ys[k])} - {_text(c + m * xs[k])} = {_text(resid)}.",
                ],
                follow_through=f"{ys[k]} - (part_b + part_a*{xs[k]})",
            ),
        ],
    )


@generator("exd_x_u6_regression_prediction", level=14, presets={"exam": dict(tier="exam")})
def exd_x_u6_regression_prediction(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: least-squares line from raw data, a prediction and a residual, as a four-part chain."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a data set with enough distinct distractors"):
        question = _build(rng)
        if question is not None:
            return question
