"""Exam-tier chain for `ucv1-validation-calibration-check`: how a binned calibration error can hide cancelling errors.

A model gives ten predicted probabilities (multiples of 0.1) with the observed 0/1 outcomes. The chain asks for

(a) the mean predicted probability,
(b) the overall confidence gap: mean prediction minus mean outcome (positive means over-confident),
(c) the binned calibration error with two equal-width bins, and
(d) how much larger (c) is than the size of (b): the miscalibration that cancels in the overall gap.

## Why every answer is exact

With N points and predictions in tenths, a bin's weighted gap is ``(n_k / N) * |mean_pred - mean_outcome|``
= ``|S_k - O_k| / N`` where ``S_k`` is the sum of the bin's predictions and ``O_k`` its count of positive outcomes.
Working in tenths, ``G_k = S_k - 10 * O_k`` is an integer, so the calibration error is ``(|G1| + |G2|) / 100``
and the confidence gap is ``(G1 + G2) / 100``: both exact at two decimal places, with no rounding to carry forward.

The two bins are drawn so their signed gaps point opposite ways (one bin over-confident, the other
under-confident), which is what makes (d) positive and the question worth asking.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator

from .exd_x_r2i_support import number_part, shown

_N = 10
_PREDICTIONS = [1, 2, 3, 4, 6, 7, 8, 9]  # tenths; 0.5 is left out so no point sits on the bin edge

_INTROS = [
    "A model predicts the probability that each of {n} patients will return to hospital within 30 days. The "
    "predictions p and the observed outcomes o (1 if the patient returned, 0 if not) are:",
    "A classifier gives the probability that each of {n} emails is spam. The predicted probabilities p and the "
    "observed outcomes o (1 if spam, 0 if not) are:",
    "A forecasting model gives the probability of rain on each of {n} days. The predictions p and the observed "
    "outcomes o (1 if it rained, 0 if not) are:",
    "A validation set has {n} cases, each with a predicted probability p of a fault and an observed outcome o "
    "(1 if a fault occurred, 0 if not):",
]
_BINS = [
    "Sort the points into two equal-width bins by predicted probability: bin 1 holds p < 0.5 and bin 2 holds p ≥ 0.5.",
    "Use two equal-width bins on p: bin 1 is p < 0.5 and bin 2 is p ≥ 0.5.",
]


def _decimal(tenths: int) -> str:
    return f"{tenths // 10}.{tenths % 10}"


@generator("exd_x_r2i_calibration_chain", level=20, presets={"exam": {"tier": "exam"}})
def exd_x_r2i_calibration_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: confidence gap and binned calibration error on ten points, with the cancellation they hide."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "ten points whose two bins are miscalibrated in opposite directions"):
        preds = [rng.choice(_PREDICTIONS) for _ in range(_N)]
        bias = rng.choice([1, 2, 2, 3])  # chance (in tenths) of a positive outcome, shifted per bin below
        outs = []
        for p in preds:
            # outcome more likely than predicted in bin 1, less likely in bin 2, so the bins pull apart
            chance = min(9, max(1, p + (bias if p < 5 else -bias)))
            outs.append(1 if rng.randrange(10) < chance else 0)
        low = [i for i in range(_N) if preds[i] < 5]
        high = [i for i in range(_N) if preds[i] >= 5]
        if not 3 <= len(low) <= 7:
            continue
        g1 = sum(preds[i] for i in low) - 10 * sum(outs[i] for i in low)
        g2 = sum(preds[i] for i in high) - 10 * sum(outs[i] for i in high)
        if g1 * g2 < 0 and g1 + g2 != 0 and min(abs(g1), abs(g2)) >= 4 and abs(g1) + abs(g2) <= 40:
            break

    pairs = ", ".join(f"({_decimal(p)}, {o})" for p, o in zip(preds, outs))
    n_low, n_high = len(low), len(high)
    mean_pred = Fraction(sum(preds), 10 * _N)
    mean_out = Fraction(sum(outs), _N)
    gap = mean_pred - mean_out
    ece = Fraction(abs(g1) + abs(g2), 100)
    cancel = ece - abs(gap)
    check(gap == Fraction(g1 + g2, 100), "gap disagrees with the bin totals")
    check(cancel == Fraction(2 * min(abs(g1), abs(g2)), 100) and cancel > 0, "cancellation disagrees with the bins")

    # Independent route to the calibration error, from the bin means and shares of the points.
    def bin_gap(idx):
        m = Fraction(sum(preds[i] for i in idx), 10 * len(idx))
        o = Fraction(sum(outs[i] for i in idx), len(idx))
        return Fraction(len(idx), _N), m - o

    (w1, d1), (w2, d2) = bin_gap(low), bin_gap(high)
    check(w1 * abs(d1) + w2 * abs(d2) == ece, "calibration error disagrees with the weighted bin gaps")

    stem = f"{rng.choice(_INTROS).format(n=_N)} {pairs}. {rng.choice(_BINS)}"
    mean_out_text = f"{mean_out.numerator}/{mean_out.denominator}" if mean_out.denominator != 1 else str(mean_out)
    sums = f"{sum(preds) / 10:.1f}"

    unweighted = (abs(d1) + abs(d2)) / 2
    swapped = w2 * abs(d1) + w1 * abs(d2)
    mean_of_bin_means = (
        Fraction(sum(preds[i] for i in low), 10 * n_low) + Fraction(sum(preds[i] for i in high), 10 * n_high)
    ) / 2

    parts = [
        number_part(
            "Find the mean predicted probability of all ten points, as a decimal.",
            mean_pred,
            2,
            1,
            [f"Sum of the ten predictions = {sums}.", f"Mean prediction = {sums} ÷ 10 = {shown(mean_pred, 2)}."],
            [
                (mean_out, "gave-observed-not-predicted"),
                (mean_of_bin_means, "unweighted-bin-average"),
                (Fraction(sum(preds), 10 * (_N - 1)), "divided-by-wrong-amount"),
                (Fraction(sum(preds), 10), "divided-by-wrong-amount"),
                (Fraction(sum(preds[i] for i in low), 10 * n_low), "used-one-bin-not-all-points"),
                (Fraction(sum(preds[i] for i in high), 10 * n_high), "used-one-bin-not-all-points"),
            ],
            sensible_range=(0, 1),
        ),
        number_part(
            "Find the confidence gap: the mean predicted probability minus the mean observed outcome. "
            "A positive gap means the model is over-confident overall.",
            gap,
            2,
            2,
            [
                f"Mean outcome = {sum(outs)} ÷ 10 = {shown(mean_out, 1)}.",
                f"Gap = {shown(mean_pred, 2)} − {shown(mean_out, 1)} = {shown(gap, 2)}.",
            ],
            [
                (-gap, "reversed-difference"),
                (abs(gap), "dropped-sign-of-difference"),
                (mean_out, "gave-one-measure-not-difference"),
                (mean_pred, "gave-one-measure-not-difference"),
            ],
            follow_through=f"part_a - {mean_out_text}",
        ),
        number_part(
            "Find the calibration error with these two bins: for each bin take the absolute difference between "
            "its mean predicted probability and its mean observed outcome, then weight each bin by its share of "
            "the ten points. Give your answer to 2 decimal places.",
            ece,
            2,
            2,
            [
                f"Bin 1 has {n_low} points: mean prediction − mean outcome = {d1}.",
                f"Bin 2 has {n_high} points: mean prediction − mean outcome = {d2}.",
                f"Error = {w1} × |{d1}| + {w2} × |{d2}| = {shown(ece, 2)}.",
            ],
            [
                (abs(gap), "skipped-binning"),
                (unweighted, "unweighted-bin-average"),
                (swapped, "weighted-by-wrong-share"),
                (abs(d1) + abs(d2), "unweighted-bin-average"),
            ],
            sensible_range=(0, 1),
        ),
        number_part(
            "The calibration error in (c) is larger than the size of the confidence gap in (b), because errors of "
            "opposite sign cancel in the overall gap. Find how much larger. Give your answer to 2 decimal places.",
            cancel,
            2,
            1,
            [f"Difference = {shown(ece, 2)} − |{shown(gap, 2)}| = {shown(cancel, 2)}."],
            [
                (ece + abs(gap), "added-instead-of-subtracted"),
                (ece, "gave-one-measure-not-difference"),
                (abs(gap), "gave-one-measure-not-difference"),
            ],
            follow_through="part_c - Abs(part_b)",
            sensible_range=(0, 1),
            min_distractors=1,
        ),
    ]
    return dict(text=stem, parts=parts)
