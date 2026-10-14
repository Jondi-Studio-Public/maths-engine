"""Exam chain for `f10-boxplot-distribution`: quartiles, an outlier fence and a comparison from the raw data of two groups.

The practice tiers compute Q1, Q3 and the IQR for one small data set. The exam question gives the raw values (unsorted)
for two groups, A and B, and works through the box plot rules:

(a) the first quartile of group A                                   1 d.p., 1 mark
(b) the third quartile of group A                                   1 d.p., 1 mark
(c) the upper fence Q3 + 1.5 x IQR for group A                      1 d.p., 2 marks, follow-through (a), (b)
(d) the mean of group A once its one outlier is removed             1 d.p., 2 marks
(e) how much the IQRs of the two groups differ (larger minus smaller) 1 d.p., 2 marks, follow-through (a), (b)

The quartile method is stated in the question: split the sorted data at the median, leave the median out of both halves
when the number of values is odd, and take Q1 and Q3 as the medians of the lower and upper halves.

## Construction

Group A has 9 or 11 values, one of which is a planted high outlier (it lies above the upper fence by at least 2 and no
other value is within 2 of the fence), and the rest lie well inside both fences. The mean of the other values is exact
to one decimal place. Group B has 8 to 12 values and a different IQR from A by at least 2. Quartiles are computed by
the halves rule in the generator and cross-checked against a position formula; (d) is cross-checked with exact
fractions.

## Distractors

(a) and (b): the inclusive method (median kept in both halves), the value at the quarter position with no averaging,
the other quartile, the median.
(c) Q3 + IQR, 1.5 x IQR alone, the lower fence.
(d) the mean of all the values, dividing the reduced total by the original count, the median.
(e) the sum of the IQRs, the difference of the ranges, the IQR of A alone.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase
from mathgen.generators._exd_x_r2x_kit import num_part

_SCENES = (
    dict(
        text="Two classes, A and B, timed how long each student took to finish the same puzzle. The times, in "
        "minutes, were recorded.",
        unit="minutes",
        lo=12,
        hi=40,
    ),
    dict(
        text="Two cycling clubs, A and B, recorded how far each rider travelled last weekend. The distances, in "
        "kilometres, were recorded.",
        unit="kilometres",
        lo=20,
        hi=70,
    ),
    dict(
        text="Two Year 10 classes, A and B, were asked how many minutes they spent on homework last night. The times "
        "were recorded.",
        unit="minutes",
        lo=10,
        hi=50,
    ),
)
_METHOD = (
    "Use the method that splits the ordered data at the median and leaves the median out of both halves when the "
    "number of values is odd."
)


def _median(v):
    v = sorted(v)
    m = len(v) // 2
    return Fraction(v[m]) if len(v) % 2 else Fraction(v[m - 1] + v[m], 2)


def _quartiles(data):
    v = sorted(data)
    half = len(v) // 2
    return _median(v[:half]), _median(v), _median(v[half + (len(v) % 2) :])


def _inclusive(data):
    v = sorted(data)
    half = (len(v) + 1) // 2
    return _median(v[:half]), _median(v[len(v) - half :])


def _list(v):
    return ", ".join(str(x) for x in v)


@generator("exd_x_r2x_boxplot_two_groups", level=10, presets={"exam": {"tier": "exam"}})
def exd_x_r2x_boxplot_two_groups(rng, tier: str = "exam"):
    """Quartiles, the upper fence, the mean without an outlier, then the difference in IQR."""
    check(tier == "exam", f"unknown tier {tier!r}")
    sc = rng.choice(_SCENES)
    for _ in attempts(5000, "group A with exactly one clear high outlier"):
        n = rng.choice([9, 11])
        core = sorted(rng.randint(sc["lo"], sc["hi"]) for _ in range(n - 1))
        if len(set(core)) < n - 4 or Fraction(sum(core), n - 1) * 10 % 1:
            continue
        o = max(core) + rng.randint(14, 45)
        data = sorted(core + [o])
        q1, med, q3 = _quartiles(data)
        iqr = q3 - q1
        fence = q3 + Fraction(3, 2) * iqr
        low_fence = q1 - Fraction(3, 2) * iqr
        if iqr < 4 or not (o > fence + 2 and core[-1] < fence - 2 and core[0] > low_fence + 2):
            continue
        if (q1 * 2).denominator > 2 or (q3 * 2).denominator > 2 or (fence * 10).denominator != 1:
            continue
        break
    for _ in attempts(2000, "group B with a different spread"):
        nb = rng.choice([8, 9, 10, 11, 12])
        b = sorted(rng.randint(sc["lo"] - 4, sc["hi"] + 12) for _ in range(nb))
        b1, bmed, b3 = _quartiles(b)
        if abs((b3 - b1) - iqr) < 2 or (b3 - b1) <= 0:
            continue
        break
    iqr_b = b3 - b1
    mean_all = Fraction(sum(data), n)
    mean_wo = Fraction(sum(core), n - 1)
    diff = abs(iqr_b - iqr)

    # Second routes: quartiles by position, the mean by exact fractions.
    k = n // 2
    low_half, up_half = data[:k], data[n - k :]
    check(q1 == _median(low_half) and q3 == _median(up_half), "quartiles disagree with the halves")
    check(mean_wo * (n - 1) == sum(data) - o, "mean without the outlier disagrees")
    check([v for v in data if v > fence] == [o], "exactly one value must be an outlier")

    order_a = list(data)
    rng.shuffle(order_a)
    order_b = list(b)
    rng.shuffle(order_b)
    stem = f"{sc['text']} Group A: {_list(order_a)}. Group B: {_list(order_b)}. {_METHOD}"
    inc1, inc3 = _inclusive(data)
    f1 = lambda v: v  # noqa: E731
    parts = [
        num_part(
            "Find the first quartile (Q1) of group A. Give the answer to 1 decimal place.",
            q1,
            1,
            1,
            [
                f"Ordered, group A is {_list(data)}.",
                f"The lower half is {_list(low_half)}, whose median is {float(q1):g}.",
            ],
            [
                (inc1, "r2x-box-median-kept-in-half"),
                (Fraction(data[(n + 1) // 4]), "r2x-box-position-no-averaging"),
                (q3, "r2x-box-quartiles-swapped"),
                (med, "r2x-box-median-for-quartile"),
            ],
        ),
        num_part(
            "Find the third quartile (Q3) of group A. Give the answer to 1 decimal place.",
            q3,
            1,
            1,
            [f"The upper half is {_list(up_half)}, whose median is {float(q3):g}."],
            [
                (inc3, "r2x-box-median-kept-in-half"),
                (Fraction(data[(3 * (n + 1)) // 4]), "r2x-box-position-no-averaging"),
                (q1, "r2x-box-quartiles-swapped"),
                (med, "r2x-box-median-for-quartile"),
            ],
        ),
        num_part(
            "Using your answers to (a) and (b), find the upper fence, Q3 + 1.5 × IQR, for group A. Give the answer to "
            "1 decimal place.",
            fence,
            1,
            2,
            [
                f"IQR = {float(q3):g} - {float(q1):g} = {float(iqr):g}.",
                f"Upper fence = {float(q3):g} + 1.5 × {float(iqr):g} = {float(fence):g}.",
            ],
            [
                (q3 + iqr, "r2x-box-fence-multiplier-one"),
                (Fraction(3, 2) * iqr, "r2x-box-fence-without-q3"),
                (low_fence, "r2x-box-lower-fence-for-upper"),
                (q3 + Fraction(3, 2), "r2x-box-fence-without-iqr"),
            ],
            follow="part_b + 1.5*(part_b - part_a)",
        ),
        num_part(
            "Group A has one outlier, a value above the upper fence. Find the mean of group A once the outlier is "
            "removed. Give the answer to 1 decimal place.",
            mean_wo,
            1,
            2,
            [
                f"{o} is the only value above {float(fence):g}.",
                f"The other {n - 1} values total {sum(core)}, so the mean is {float(mean_wo):g}.",
            ],
            [
                (mean_all, "r2x-box-outlier-kept"),
                (Fraction(sum(core), n), "r2x-box-divided-by-original-count"),
                (_median(core), "r2x-box-median-for-mean"),
                (Fraction(sum(data) - data[0], n - 1), "r2x-box-wrong-value-removed"),
            ],
        ),
        num_part(
            "Using your answers to (a) and (b), find how much the interquartile ranges of the two groups differ "
            "(larger minus smaller). Give the answer to 1 decimal place.",
            diff,
            1,
            2,
            [
                f"Group A: IQR = {float(iqr):g}. Group B: Q1 = {float(b1):g}, Q3 = {float(b3):g}, so IQR = {float(iqr_b):g}.",
                f"The IQRs differ by {float(diff):g}.",
            ],
            [
                (iqr + iqr_b, "r2x-box-iqrs-added"),
                (abs((data[-1] - data[0]) - (b[-1] - b[0])), "r2x-box-range-for-iqr"),
                (iqr, "r2x-box-one-group-only"),
                (iqr_b, "r2x-box-one-group-only"),
            ],
            follow=f"Abs({float(iqr_b)} - (part_b - part_a))",
        ),
    ]
    return dict(text=stem, parts=parts)
