"""Exam-tier source for `senior-find-the-line-of-best-fit-by-eye`: from a drawn line to a prediction and a residual.

A scatter of six data points is stated, with a line of best fit drawn by eye through two marked points A and B on
the line (A and B are read off the graph, so they are not necessarily data points). The learner works the way a
bivariate-data exam question does:

(a) the gradient of the line, from A and B;
(b) the intercept c in y = mx + c, using their gradient and point A;
(c) a prediction from the line at a stated x, using their gradient and intercept;
(d) the residual (observed minus predicted) at one of the data points, using their line.

Parts (b) to (d) carry `follow_through`, so a learner who slips on the gradient still earns the marks for using it
correctly. Built backwards from the line and the points, so every figure is exact (halves at most).
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    paren_negative,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

# (setting, x name, y name, x low, x high, gradients, y at the lowest x, y floor, y ceiling)
_CONTEXTS = (
    (
        "A kiosk at Cottesloe Beach recorded the daily maximum temperature and the number of bottles of water "
        "it sold on six summer days.",
        "daily maximum temperature (degrees Celsius)",
        "bottles of water sold",
        24,
        38,
        (4, 5, 6, 8, 10),
        (25, 50),
        10,
        250,
    ),
    (
        "A running club in Hobart recorded how many hours each of six members trained per week and the time each "
        "took to run 5 km.",
        "weekly training (hours)",
        "time for 5 km (minutes)",
        2,
        12,
        (Fraction(-1, 2), Fraction(-3, 2), -2),
        (34, 42),
        20,
        50,
    ),
    (
        "A dealer in Adelaide listed the age and asking price of six used hatchbacks of the same model.",
        "age of the car (years)",
        "asking price ($ thousands)",
        1,
        9,
        (Fraction(-3, 2), -2, Fraction(-5, 2), -3),
        (25, 36),
        4,
        40,
    ),
    (
        "A school in Townsville logged the hours of sunshine each day and the energy its solar panels produced on "
        "six days.",
        "sunshine (hours)",
        "energy produced (kWh)",
        4,
        12,
        (Fraction(3, 2), 2, Fraction(5, 2), 3),
        (8, 20),
        2,
        90,
    ),
)


def _txt(value: Fraction) -> str:
    """An answer at its stated precision: 1 decimal place, rounded half up."""
    return stated_precision_string(round_half_up(Fraction(value), 1), 1)


def _plain(value: Fraction) -> str:
    """A given figure in its shortest exact decimal form (every figure shown is a multiple of 0.5)."""
    return exact_decimal_string(Fraction(value), 2)


def _paren(value: Fraction) -> str:
    """`_plain`, bracketed when negative, for a figure that follows an operator."""
    return f"({_plain(value)})" if Fraction(value) < 0 else _plain(value)


def _plus(value: Fraction) -> str:
    """``+ v`` or ``- v`` for a figure added to something, so no ``+ -``."""
    return f"{'-' if Fraction(value) < 0 else '+'} {_plain(abs(Fraction(value)))}"


def _dis(value: Fraction, tag: str | None) -> Distractor:
    value = Fraction(value)
    return Distractor(_txt(value), tag, exact_value(round_half_up(value, 1)))


def _pt(x, y) -> str:
    return f"({x}, {_plain(y)})"


def _clear_of(candidates: list[Distractor], carried: list[Fraction]) -> list[Distractor]:
    """The candidates that are not within grader tolerance of any value reached by carrying a wrong answer on."""
    return [x for x in candidates if all(abs(Fraction(x.text) - v) > GRADER_TOLERANCE for v in carried)]


@generator("exd_x_sc8_best_fit_line_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_sc8_best_fit_line_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: gradient, intercept, prediction and residual for a line of best fit drawn through two points."""
    check(tier == "exam", f"unknown tier {tier!r}")
    setting, xname, yname, lo, hi, grads, base_range, floor, ceil = rng.choice(_CONTEXTS)
    for _ in attempts(3000, "a consistent line, points and data"):
        m = Fraction(rng.choice(grads))
        base = rng.randint(*base_range)

        def line(x, m=m, base=base):
            return base + m * (x - lo)

        # Marked points A and B on the line, with whole y values and a clear run between them.
        x1 = rng.randint(lo, hi - 4)
        x2 = rng.randint(x1 + 4, hi)
        y1, y2 = line(x1), line(x2)
        if y1.denominator != 1 or y2.denominator != 1:
            continue
        xs = sorted(rng.sample(range(lo, hi + 1), 6))
        amp = max(2, round(abs(m) * Fraction(3, 2)))
        noise = [rng.randint(-amp, amp) for _ in xs]
        if abs(sum(noise)) > amp or min(noise) >= 0 or max(noise) <= 0:
            continue
        ys = [round(line(x)) + e for x, e in zip(xs, noise)]
        if any(not floor <= v <= ceil for v in ys) or any(not floor <= line(x) <= ceil for x in range(lo, hi + 1)):
            continue
        residuals = {x: Fraction(y) - line(x) for x, y in zip(xs, ys)}
        px = rng.choice(xs)
        if abs(residuals[px]) < 1:
            continue
        # Predict inside the range of the data (interpolation), never beyond it.
        x0_choices = [x for x in range(min(xs) + 1, max(xs)) if x not in xs and x not in (x1, x2)]
        if not x0_choices:
            continue
        x0 = rng.choice(x0_choices)
        break
    py = ys[xs.index(px)]
    c = y1 - m * x1
    check(c == line(0), "intercept disagrees with the line")
    check(line(x2) - line(x1) == m * (x2 - x1), "gradient disagrees with the two points")
    pred0 = m * x0 + c
    check(pred0 == line(x0), "prediction disagrees with the line")
    res = Fraction(py) - (m * px + c)
    check(res == residuals[px], "residual disagrees")

    table = ", ".join(_pt(x, y) for x, y in zip(xs, ys))
    stem = (
        f"{setting} Let x be the {xname} and y the {yname}. The points (x, y) are {table}. "
        f"A line of best fit y = mx + c is drawn by eye through the marked points A{_pt(x1, y1)} and "
        f"B{_pt(x2, y2)}."
    )

    dy, dx = y2 - y1, x2 - x1
    a_cands = [
        _dis(Fraction(dx) / dy, "inverted-ratio"),
        _dis(-m, "gradient-sign-from-point-order"),
        _dis(Fraction(dy), "gradient-rise-only"),
        _dis(m * 2 if m != 0 else 1, None),
    ]
    b_cands = [
        _dis(Fraction(y1), "intercept-read-as-point-height"),
        _dis(y1 + m * x1, "intercept-sign-error"),
        _dis(x1 - m * y1, "swapped-x-and-y"),
        _dis(c + 1, None),
        _dis(c - 1, None),
        _dis(y2 - m * x1, None),
    ]
    c_cands = [
        _dis(m * x0, "dropped-constant-term"),
        _dis(c * x0 + m, "gradient-and-intercept-swapped"),
        _dis((x0 - c) / m, "swapped-x-and-y"),
        _dis(m + c, "substituted-without-multiplying"),
        _dis(m * (x0 + 1) + c, None),
        _dis(m * (x0 - 1) + c, None),
    ]
    d_cands = [
        _dis(-res, "residual-sign-reversed"),
        _dis(m * px + c, "gave-prediction-not-residual"),
        _dis(Fraction(py), "gave-observed-not-residual"),
        _dis(res + 1, None),
        _dis(res - 1, None),
        _dis(Fraction(py) - (m * (px + 1) + c), None),
    ]
    key_a, key_b, key_c, key_d = _txt(m), _txt(c), _txt(pred0), _txt(res)
    a_dis = pick_distractors(key_a, a_cands, 3, tolerance=GRADER_TOLERANCE)
    # A later distractor must not be the value a learner reaches by carrying an earlier wrong answer forward
    # correctly: that reply earns follow-through marks, so it cannot also be offered as a wrong answer.
    grads = [m] + [Fraction(x.text) for x in a_dis]
    b_cands = _clear_of(b_cands, [y1 - g * x1 for g in grads])
    b_dis = pick_distractors(key_b, b_cands, 3, tolerance=GRADER_TOLERANCE)
    inters = [c] + [Fraction(x.text) for x in b_dis]
    pairs = [(m, k) for k in inters] + [(g, y1 - g * x1) for g in grads]
    c_cands = _clear_of(c_cands, [g * x0 + k for g, k in pairs])
    c_dis = pick_distractors(key_c, c_cands, 3, tolerance=GRADER_TOLERANCE)
    d_cands = _clear_of(d_cands, [py - (g * px + k) for g, k in pairs])
    d_dis = pick_distractors(key_d, d_cands, 3, tolerance=GRADER_TOLERANCE)
    check(min(len(a_dis), len(b_dis), len(c_dis), len(d_dis)) >= 2, "too few distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the gradient m of the line, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(m),
                answer_text=key_a,
                answer_latex=latex_answer(key_a),
                marks=2,
                working=[
                    f"m = ({_plain(y2)} - {_paren(y1)}) / ({x2} - {paren_negative(x1)}) = {_plain(dy)} / {dx} = {key_a}."
                ],
                distractors=a_dis,
            ),
            dict(
                prompt="Hence find the value of c in the equation y = mx + c, using point A, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(c),
                answer_text=key_b,
                answer_latex=latex_answer(key_b),
                marks=1,
                follow_through=f"{y1} - part_a*{x1}",
                working=[f"{_plain(y1)} = {_plain(m)} x {paren_negative(x1)} + c, so c = {key_b}."],
                distractors=b_dis,
            ),
            dict(
                prompt=f"Use your line to predict y when x = {x0}, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(pred0),
                answer_text=key_c,
                answer_latex=latex_answer(key_c),
                marks=1,
                follow_through=f"part_a*{x0} + part_b",
                working=[f"y = {_plain(m)} x {paren_negative(x0)} {_plus(c)} = {key_c}."],
                distractors=c_dis,
            ),
            dict(
                prompt=f"Find the residual (observed y minus predicted y) for the data point with x = {px}, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(res),
                answer_text=key_d,
                answer_latex=latex_answer(key_d),
                marks=2,
                follow_through=f"{py} - (part_a*{px} + part_b)",
                working=[
                    f"Predicted y = {_plain(m)} x {paren_negative(px)} {_plus(c)} = {_plain(m * px + c)}.",
                    f"Residual = {py} - {_paren(m * px + c)} = {key_d}.",
                ],
                distractors=d_dis,
            ),
        ],
    )
