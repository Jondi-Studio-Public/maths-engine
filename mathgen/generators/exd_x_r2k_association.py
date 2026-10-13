"""Exam question: describe an association from raw summary statistics.

Instead of being handed r, the learner is handed the five summary sums for n pairs of values. Part (a) builds r from
them (the sums of squares and cross-products about the means), part (b) turns it into the percentage of variation
explained, and part (c) states direction, form and strength, using r and the stated bands when the residual plot is
random, and the stated closeness to the curve when the residual plot is curved (r measures linear strength only).

Built from a generated data set of integer pairs, so the five sums are consistent by construction. Draws are kept only
when r rounds cleanly to 2 places, sits well away from every strength band edge, and gives the same percentage in (b)
whether it is taken from the rounded or the unrounded r.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator
from ..answers import Distractor
from .exd_x_r2k_support import retry_build, clear_of_boundary, number_part, rd

# (x name, y name, subject, x low, x high, y at x low, y at x high)
_CONTEXTS = (
    ("hours of study per week", "exam mark (%)", "students", 2, 20, 38, 88),
    ("age of the car (years)", "resale value ($ thousands)", "cars", 1, 14, 31, 6),
    ("daily screen time (hours)", "sleep per night (hours)", "teenagers", 1, 12, 9, 6),
    ("weekly training (hours)", "5 km run time (minutes)", "runners", 2, 14, 34, 22),
    ("daily maximum temperature (degrees Celsius)", "ice creams sold", "summer days", 18, 38, 45, 190),
    ("distance from the city centre (km)", "weekly rent ($)", "rental units", 2, 30, 640, 360),
    ("daily hours of sunshine", "solar energy generated (kWh)", "days", 2, 12, 5, 32),
)
_BANDS = "For a linear relationship, judge strength from the size of r: |r| ≥ 0.75 is strong, 0.5 ≤ |r| < 0.75 is moderate, 0.25 ≤ |r| < 0.5 is weak."
_CURVE_BANDS = (
    "For a non-linear relationship, judge strength from how closely the points follow the curve: lying almost "
    "exactly on the curve is strong, scattered fairly close to the curve is moderate, and noticeably scattered around "
    "the curve is weak."
)
_FORMAT = (
    "State your answer as three fields, in this order: direction (positive or negative), form (linear or non-linear), "
    "strength (strong, moderate or weak), separated by commas -- for example 'positive, linear, strong'. Use exactly "
    "these spellings, including the hyphen in non-linear."
)
_CLOSENESS = {
    "strong": "the points lie almost exactly on the curve",
    "moderate": "the points are scattered fairly close to the curve",
    "weak": "the points are noticeably scattered around the curve",
}
_NEXT_BAND = {"strong": ("moderate",), "moderate": ("strong", "weak"), "weak": ("moderate",)}


def _band(r: float) -> str:
    a = abs(r)
    return "strong" if a >= 0.75 else "moderate" if a >= 0.5 else "weak"


def _edge_clear(r: float) -> bool:
    return abs(r) > 0.3 and all(abs(abs(r) - e) > 0.02 for e in (0.25, 0.5, 0.75))


def _sums(xs, ys):
    n = len(xs)
    return n, sum(xs), sum(ys), sum(x * x for x in xs), sum(y * y for y in ys), sum(x * y for x, y in zip(xs, ys))


def _r(n, sx, sy, sxx, syy, sxy) -> float:
    """r from the sums, by the centred cross-products."""
    s_xy = Fraction(sxy) - Fraction(sx * sy, n)
    s_xx = Fraction(sxx) - Fraction(sx * sx, n)
    s_yy = Fraction(syy) - Fraction(sy * sy, n)
    return float(s_xy) / math.sqrt(float(s_xx * s_yy))


def _build_k_association(rng: random.Random) -> dict | None:
    """r from raw sums, then r squared as a percentage, then direction, form and strength."""
    for _ in attempts(2000, "a data set with a clean r"):
        xname, yname, subject, xlo, xhi, ylo, yhi = rng.choice(_CONTEXTS)
        n = rng.randint(8, min(12, xhi - xlo + 1))
        xs = rng.sample(range(xlo, xhi + 1), n)
        span = abs(yhi - ylo)
        noise = rng.uniform(0.04, 0.95) * span
        ys = [round(ylo + (yhi - ylo) * (x - xlo) / (xhi - xlo) + rng.gauss(0, noise)) for x in xs]
        if len(set(ys)) < 4 or min(ys) < 0:
            continue
        sums = _sums(xs, ys)
        r = _r(*sums)
        r_shown = rd(Fraction(repr(r)), 2)
        if (r > 0) != (yhi > ylo):
            continue  # keep the direction the real-world context would give
        if not (_edge_clear(r) and clear_of_boundary(Fraction(repr(r)), 2, Fraction(1, 5))):
            continue
        pct_exact = rd(100 * Fraction(repr(r)) ** 2, 1)
        pct = rd(100 * r_shown**2, 1)
        if pct_exact != pct or not clear_of_boundary(100 * r_shown**2, 1):
            continue
        break
    a_val = r_shown
    direction = "positive" if a_val > 0 else "negative"
    curved = rng.random() < 0.3
    if curved:
        strength = rng.choice(("strong", "moderate", "weak"))
        form = "non-linear"
    else:
        strength = _band(float(a_val))
        form = "linear"
    n, sx, sy, sxx, syy, sxy = sums

    if curved:
        resid = (
            f"The residual plot for the least-squares line shows a clear curved pattern, not random scatter. "
            f"As x increases, y tends to {'increase' if a_val > 0 else 'decrease'}. "
            f"Relative to the best-fitting curve, {_CLOSENESS[strength]}."
        )
        rules = f"{_BANDS} {_CURVE_BANDS}"
    else:
        resid = "The residual plot for the least-squares line shows random scatter about zero, with no pattern."
        rules = f"{_BANDS} {_CURVE_BANDS}"
    stem = (
        f"A study recorded x, the {xname}, and y, the {yname}, for {n} {subject}. The summary statistics are "
        f"Σx = {sx}, Σy = {sy}, Σx² = {sxx}, Σy² = {syy} and Σxy = {sxy}. {rules} {resid}"
    )

    s_xy = Fraction(sxy) - Fraction(sx * sy, n)
    s_xx = Fraction(sxx) - Fraction(sx * sx, n)
    s_yy = Fraction(syy) - Fraction(sy * sy, n)
    r_true = Fraction(repr(r))
    uncentred = Fraction(repr(sxy / math.sqrt(sxx * syy)))
    a_wrongs = [
        (-r_true, "sign-error-in-r"),
        (s_xy / s_xx, "slope-reported-as-r"),
        (uncentred, "forgot-to-centre-sums"),
        (s_xy / (s_xx * s_yy), "no-square-root-in-r-denominator"),
    ]
    pa = number_part(
        "Calculate the correlation coefficient r, correct to 2 decimal places.",
        r_true,
        2,
        2,
        [
            f"Sxy = {sxy} - {sx} x {sy} / {n} = {float(s_xy):.3f}; Sxx = {float(s_xx):.3f}; Syy = {float(s_yy):.3f}.",
            f"r = Sxy / sqrt(Sxx x Syy) = {a_val}.",
        ],
        a_wrongs,
    )
    b_wrongs = [
        (100 * abs(a_val), "r-given-as-r-squared-percentage"),
        (100 - 100 * a_val**2, "unexplained-variation-for-explained"),
    ]
    pb = number_part(
        "Use your value of r from (a) to find the percentage of the variation in y that is explained by x, "
        "correct to 1 decimal place.",
        100 * a_val**2,
        1,
        1,
        [f"r squared = {a_val}^2 = {float(a_val**2):.4f}, so {pct}% of the variation is explained."],
        b_wrongs,
        follow_through="100*part_a**2",
        sensible_range=(0, 100),
    )
    if pa is None or pb is None:
        return None

    answer = [direction, form, strength]
    flip = "negative" if direction == "positive" else "positive"
    c_wrongs: list[tuple[list, str]] = [([flip, form, strength], "direction-sign-flip")]
    if curved:
        c_wrongs.append(([direction, "linear", strength], "trusts-large-r-for-linearity"))
        by_r = _band(float(a_val))
        if by_r != strength:
            c_wrongs.append(([direction, form, by_r], "r-used-for-nonlinear-strength"))
        for off in _NEXT_BAND[strength]:
            c_wrongs.append(([direction, form, off], "strength-band-off-by-one"))
    else:
        c_wrongs.append(([direction, "non-linear", strength], "form-confusion"))
        for off in _NEXT_BAND[strength]:
            c_wrongs.append(([direction, form, off], "strength-band-off-by-one"))
    key_text = ", ".join(answer)
    options = []
    seen = {key_text}
    for value, tag in c_wrongs:
        text = ", ".join(value)
        if text in seen:
            continue
        seen.add(text)
        options.append(Distractor(text, tag, value))
    pc = dict(
        prompt=f"Using your answers, state the direction, form and strength of the association. {_FORMAT}",
        answer_kind="ordered_list",
        answer=answer,
        answer_text=key_text,
        answer_latex="plain:" + key_text,
        marks=2,
        working=[
            f"r = {a_val} is {direction}.",
            "The residual plot is curved, so the form is non-linear and strength comes from the closeness to the curve."
            if curved
            else f"The residuals are random, so the form is linear and |r| = {abs(a_val)} is {strength}.",
        ],
        distractors=options[:3],
    )
    return dict(text=stem, parts=[pa, pb, pc])


@generator("exd_x_r2k_association", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2k_association(rng: random.Random, tier: str = "exam") -> dict:
    """r from raw sums, then r squared as a percentage, then direction, form and strength."""
    check(tier == "exam", "exd_x_r2k_association: only the exam tier exists")
    return retry_build(rng, _build_k_association, "exd_x_r2k_association")
