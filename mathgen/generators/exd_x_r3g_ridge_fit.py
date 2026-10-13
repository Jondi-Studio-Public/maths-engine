"""Exam tier for "compute a one-feature ridge coefficient".

The practice tiers give the sums Sxx and Sxy outright. Here the learner starts from the centred observations
themselves and carries one fit all the way to a prediction, as a first statistical-learning exam would set it:

(a) Sxx, the sum of the squares of the x values;
(b) Sxy, the sum of the products x y;
(c) the ridge coefficient beta = Sxy / (Sxx + lambda), an exact fraction, from (a) and (b);
(d) the prediction beta * x for a new x, from (c).

The data are five integer pairs whose x values and y values each sum to zero, so the no-intercept model is the
right one. Every answer is exact: (a) and (b) are integers, (c) and (d) are rationals.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, generator
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3g_kit import clashes, fraction_part, number_part

_STEMS = (
    "Ridge regression fits the model y = βx, with no intercept, to the five centred observations (x, y): {data}. "
    "The penalty is λ = {lam}.",
    "A one-feature model y = βx is fitted with a ridge penalty λ = {lam}. The data have been centred (the x values "
    "and the y values each have mean zero), and the observations (x, y) are {data}.",
    "Five centred observations (x, y) are recorded: {data}. A ridge fit of y = βx, with no intercept and penalty "
    "λ = {lam}, is to be made.",
)


def _centred(rng: random.Random, spread: int) -> list[int]:
    head = [rng.randint(-spread, spread) for _ in range(4)]
    return head + [-sum(head)]


@generator("exd_x_r3g_ridge_fit", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3g_ridge_fit(rng: random.Random, tier: str = "exam") -> dict:
    """From five centred observations: Sxx, Sxy, the ridge coefficient, then a prediction."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a centred data set with distinct, non-trivial sums"):
        xs, ys = _centred(rng, 4), _centred(rng, 4)
        if max(map(abs, xs)) > 6 or max(map(abs, ys)) > 6 or len(set(xs)) < 4:
            continue
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        lam = rng.randint(2, 9)
        new_x = rng.choice((-3, -2, -1, 1, 2, 3, 4))
        beta = Fraction(sxy, sxx + lam)
        if not 10 <= sxx <= 45 or abs(sxy) < 4 or beta.denominator == 1 or Fraction(sxy, sxx).denominator == 1:
            continue
        pred = beta * new_x
        if pred.denominator == 1 or abs(pred) < Fraction(1, 2):
            continue
        data = ", ".join(f"({x}, {y})" for x, y in zip(xs, ys))
        stem = rng.choice(_STEMS).format(data=data, lam=lam)
        pd = f"Use your ridge coefficient to predict y when x = {new_x}."
        if clashes((sxx, sxy), stem, pd):
            continue
        break

    ols = Fraction(sxy, sxx)
    parts = [
        number_part(
            "Find the sum of the squares of the x values.",
            sxx,
            0,
            1,
            ["Sxx = " + " + ".join(f"({x})²" for x in xs) + f" = {sxx}."],
            [
                (sum(abs(x) for x in xs), "summed-sizes-not-squares"),
                (sxx - xs[-1] ** 2, "omitted-entry-in-total"),
                (sum(y * y for y in ys), "ridge-wrong-variable-squared"),
                (2 * sum(abs(x) for x in xs), "squared-as-doubled"),
                (sxx - max(x * x for x in xs), "omitted-entry-in-total"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            "Find the sum of the products x y over the five observations.",
            sxy,
            0,
            2,
            ["Sxy = " + " + ".join(f"({x})({y})" for x, y in zip(xs, ys)) + f" = {sxy}."],
            [
                (sum(y * y for y in ys), "ridge-wrong-variable-squared"),
                (sum(abs(x * y) for x, y in zip(xs, ys)), "ridge-product-signs-ignored"),
                (sxy - xs[-1] * ys[-1], "omitted-entry-in-total"),
                (-sxy, "sign-of-result-reversed"),
                (sum(x * y for x, y in zip(xs, ys[1:] + ys[:1])), "ridge-observations-mispaired"),
            ],
        ),
        fraction_part(
            "Find the ridge coefficient β = Sxy / (Sxx + λ), as a fraction in its simplest form.",
            beta,
            3,
            [f"β = {sxy} / ({sxx} + {lam}) = {sxy} / {sxx + lam}.", f"β = {beta}."],
            [
                (ols, "ridge-penalty-omitted"),
                (Fraction(sxy, sxx - lam), "subtracted-instead-of-added") if sxx != lam else None,
                (Fraction(sxy + lam, sxx), "ridge-penalty-added-to-numerator"),
                (Fraction(sxy, sxx + lam * lam), "ridge-penalty-squared"),
            ],
            follow_through=f"part_b/(part_a + {lam})",
        ),
        fraction_part(
            pd,
            pred,
            2,
            [f"ŷ = βx = ({beta})({new_x}) = {pred}."],
            [
                (beta, "ridge-prediction-gives-the-coefficient"),
                (ols * new_x, "ridge-penalty-omitted"),
                (new_x / beta, "divided-instead-of-multiplied"),
                (beta * (new_x + 1), "ridge-prediction-wrong-input"),
            ],
            follow_through=f"part_c*({new_x})",
        ),
    ]
    return dict(text=stem, parts=parts)
