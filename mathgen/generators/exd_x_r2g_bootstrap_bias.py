"""Exam question: bias and a basic confidence interval from a list of bootstrap means (`ucv1-bootstrap-summary`).

One scenario: a sample statistic (a sample mean, a whole number) and twenty bootstrap means of it, listed in the
order they were drawn. The parts chain:

* (a) the mean of the twenty bootstrap means (2 marks, exact to 2 decimal places);
* (b) the bootstrap estimate of the bias, mean of the bootstrap means minus the sample statistic (follow-through
  from (a));
* (c) the bias-corrected estimate, the sample statistic minus the bias (follow-through from (b));
* (d) the lower bound of the 90% basic (reverse-percentile) interval, 2 * statistic - q95, where the percentile
  convention is stated in the prompt: with twenty values the k-th smallest is the 5k-th percentile, so q05 is the
  smallest and q95 the 19th smallest value. This needs the sorted list and the statistic, not the earlier parts.

The bootstrap means are whole numbers, so every answer is exact at 2 decimal places (a whole number for (d)).
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator
from .exd_x_r2g_support import build_part, redraws

_B = 20
_CONTEXTS = [
    (
        "A council surveys households about weekly grocery spending. The sample mean is {th} dollars.",
        "dollars",
        (150, 300),
    ),
    (
        "A water authority samples daily household water use. The sample mean is {th} litres.",
        "litres",
        (120, 260),
    ),
    (
        "A consumer group records the price of unleaded petrol at a sample of Queensland service stations. "
        "The sample mean is {th} cents per litre.",
        "cents per litre",
        (170, 230),
    ),
    (
        "A school records how long a sample of students take to travel to school. The sample mean is {th} minutes.",
        "minutes",
        (20, 70),
    ),
]
_LEAD = [
    "To judge the sampling variability, an analyst draws {b} bootstrap resamples and records the mean of each, "
    "in the order drawn: {values}.",
    "An analyst draws {b} bootstrap resamples from the sample and records each resample's mean, in the order "
    "drawn: {values}.",
    "The sample is resampled {b} times with replacement; the {b} bootstrap means, in the order drawn, are: {values}.",
]
_CONVENTION = (
    "Use the convention that, of the {b} bootstrap means sorted into order, the k-th smallest is the (5k)th "
    "percentile, with no interpolation."
)


def _draw(rng: random.Random) -> dict:
    for _ in range(1000):
        stem, unit, (lo, hi) = rng.choice(_CONTEXTS)
        th = rng.randint(lo, hi)
        spread = max(2, round(th * rng.uniform(0.015, 0.04)))
        shift = rng.choice([-1, 1]) * rng.randint(1, max(2, spread // 2))
        values = [round(rng.gauss(th + shift, spread)) for _ in range(_B)]
        total = sum(values)
        bias = Fraction(total, _B) - th
        ordered = sorted(values)
        if abs(bias) < Fraction(3, 10) or abs(bias) > spread:
            continue
        if ordered[0] == ordered[1] or ordered[-2] == ordered[-3] or ordered[-1] == ordered[-2]:
            continue  # keep the lowest and 19th-smallest values distinct from their neighbours
        if len({th, *values}) < 14:
            continue
        return dict(stem=stem, unit=unit, th=th, values=values, total=total, ordered=ordered)
    check(False, "no acceptable bootstrap draw")


@generator("exd_x_r2g_bootstrap_bias", level=14, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_r2g_bootstrap_bias(rng: random.Random, tier: str = "exam") -> dict:
    """Bias, bias-corrected estimate and a basic interval from twenty bootstrap means."""
    check(tier == "exam", "exd_x_r2g_bootstrap_bias: only the exam tier exists")
    d = _draw(rng)
    th, values, total, ordered = d["th"], d["values"], d["total"], d["ordered"]
    mean = Fraction(total, _B)
    bias = mean - th
    corrected = th - bias
    q05, q95, q90 = ordered[0], ordered[-2], ordered[-3]
    lower = 2 * th - q95
    median = Fraction(ordered[9] + ordered[10], 2)
    unit = d["unit"]

    stem = f"{d['stem'].format(th=th)} {rng.choice(_LEAD).format(b=_B, values=', '.join(str(v) for v in values))}"
    parts = [
        build_part(
            "Find the mean of the 20 bootstrap means, correct to 2 decimal places.",
            mean,
            kind="number",
            dp=2,
            marks=2,
            sensible_range=(0, None),
            working=[
                f"The 20 values sum to {total}.",
                f"Mean = {total} / 20 = {mean.numerator / mean.denominator:.2f}.",
            ],
            wrong=[
                (Fraction(total, _B - 1), "divides-by-n-minus-one-for-mean"),
                (median, "median-for-mean"),
                (Fraction(total), "total-instead-of-mean"),
                (Fraction(ordered[0] + ordered[-1], 2), "midrange-for-mean"),
            ],
        ),
        build_part(
            "Hence find the bootstrap estimate of the bias, correct to 2 decimal places.",
            bias,
            kind="number",
            dp=2,
            marks=1,
            follow_through=f"part_a - {th}",
            working=[
                "Bias = (mean of the bootstrap means) - (sample mean).",
                f"= {float(mean):.2f} - {th} = {float(bias):.2f}.",
            ],
            wrong=[
                (-bias, "answer-sign-reversed"),
                (mean, "bias-taken-as-bootstrap-mean"),
                (median - th, "median-for-mean"),
                (bias * 100 / th, "bias-as-percentage"),
            ],
        ),
        build_part(
            f"Hence find the bias-corrected estimate (the sample mean minus the estimated bias), in {unit}, "
            "correct to 2 decimal places.",
            corrected,
            kind="number",
            dp=2,
            marks=1,
            sensible_range=(0, None),
            follow_through=f"{th} - part_b",
            working=[f"Corrected estimate = {th} - ({float(bias):.2f}) = {float(corrected):.2f}."],
            wrong=[
                (th + bias, "bias-correction-wrong-direction"),
                (th, "bias-correction-omitted"),
                (th - 2 * bias, "bias-correction-doubled"),
            ],
        ),
        build_part(
            "The 90% basic bootstrap confidence interval for the population mean is (2m - q95, 2m - q05), where m "
            f"is the sample mean and q05 and q95 are the 5th and 95th percentiles of the bootstrap means. "
            f"{_CONVENTION.format(b=_B)} Find the lower bound of this interval, as a whole number of {unit}.",
            lower,
            kind="number",
            marks=2,
            working=[
                f"Sorted, the smallest value is {q05} and the 19th smallest is {q95}.",
                f"Lower bound = 2 x {th} - {q95} = {lower}.",
            ],
            wrong=[
                (q05, "percentile-interval-for-basic"),
                (2 * th - q05, "basic-interval-ends-swapped"),
                (2 * th - q90, "percentile-rank-off-by-one"),
                (q95, "percentile-interval-for-basic"),
                (2 * th - ordered[-1], "percentile-rank-off-by-one"),
                (2 * th - ordered[1], "percentile-rank-off-by-one"),
            ],
        ),
    ]
    return dict(text=stem, parts=parts)
