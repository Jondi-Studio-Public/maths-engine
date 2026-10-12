"""Exam question: the shape of a grouped distribution, argued from its mean and median.

A frequency table in equal-width classes is given. The learner estimates the mean from the class midpoints (a),
estimates the median by linear interpolation in the median class (b), takes their difference (c) and uses its sign
and size to decide the shape (d). A right-skewed table has a mean above its median, a left-skewed one below, and a
symmetric one has them equal, so the decision is a conclusion drawn from the numbers rather than a look at the table.

Built from a chosen profile: a single-peaked run of frequencies (rising to the peak, then falling), mirrored for a
left skew, or a palindrome for a symmetric table. Draws where the skew is too slight to call from the mean and
median (a gap under a tenth of the class width) are rejected, and so is any draw whose rounded values sit close to a
rounding boundary. The difference in (c) is taken between the two rounded answers, as the prompt says.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator
from .exd_x_r2k_support import retry_build, clear_of_boundary, word_part, number_part, rd, tx

_CONTEXTS = (
    ("A service desk recorded how long, in minutes, each of {n} customers waited. ", 4, (2, 5, 10)),
    ("{n} students sat a test marked out of 100. ", 0, (10,)),
    ("The masses, in kilograms, of {n} parcels handled by a courier depot were recorded. ", 6, (2, 5)),
    ("A cafe recorded the amount, in dollars, spent by each of {n} customers. ", 3, (5, 10)),
    ("{n} commuters were asked how long, in minutes, their trip to work takes. ", 3, (5, 10)),
    ("The daily steps, in hundreds, taken by {n} members of a walking club were recorded. ", 5, (10, 20)),
)
_CLOSERS = (
    "The results were grouped into classes of equal width: {table}.",
    "The data were grouped in equal classes as follows: {table}.",
    "Grouped into equal-width classes, the frequencies were: {table}.",
)
_CONVENTION = (
    "A distribution is skewed right (positively skewed) when its longer tail points to larger values, and "
    "skewed left (negatively skewed) when its longer tail points to smaller values. Treat a distribution as "
    "symmetric if its mean and median are within one tenth of a class width of each other."
)
_LABELS = {
    "right": "right",
    "left": "left",
    "sym": "symmetric",
    "bimodal": "bimodal",
}


def _profile(rng: random.Random, kind: str) -> list[int] | None:
    """Frequencies rising to one peak then falling (right skew); mirrored for left; a palindrome for symmetric."""
    if kind == "sym":
        half = rng.randint(3, 4)
        rise = sorted(rng.sample(range(6, 40), half))
        freqs = rise + rise[-2::-1]
        for i in rng.sample(range(len(freqs)), rng.randint(1, 3)):
            freqs[i] += rng.choice((-1, 1))
        return freqs
    peak = rng.randint(1, 2)
    up = sorted(rng.sample(range(8, 26), peak + 1))
    cur = up[-1] + rng.randint(2, 12)
    up[-1] = cur
    tail = []
    ratio = rng.uniform(0.6, 0.8)
    for _ in range(9 - len(up)):
        if cur <= 3:
            break
        cur = max(2, min(cur - 1, round(cur * ratio)))
        tail.append(cur)
    freqs = up + tail
    if len(freqs) < 7 or not all(freqs[i] > freqs[i + 1] for i in range(peak, len(freqs) - 1)):
        return None
    return freqs if kind == "right" else freqs[::-1]


def _stats(freqs: list[int], lo: int, width: int) -> tuple[Fraction, Fraction, int, int]:
    """(mean, median, median class index, n) by class midpoints and interpolation."""
    n = sum(freqs)
    mids = [Fraction(2 * (lo + i * width) + width, 2) for i in range(len(freqs))]
    mean = sum(f * m for f, m in zip(freqs, mids)) / n
    cum = 0
    for i, f in enumerate(freqs):
        if cum + f >= Fraction(n, 2):
            lower = lo + i * width
            median = lower + (Fraction(n, 2) - cum) / f * width
            return mean, median, i, n
        cum += f
    raise AssertionError("median class not found")


def _class_text(lo: int, width: int, i: int) -> str:
    return f"{lo + i * width} to under {lo + (i + 1) * width}"


def _build_k_histogram(rng: random.Random) -> dict | None:
    """Mean, median, their difference and the shape they point to, for a grouped frequency table."""
    for _ in attempts(1000, "a table with a clear mean-median gap"):
        kind = rng.choice(("right",) * 4 + ("left",) * 4 + ("sym",))
        sentence, lo_mult, widths = rng.choice(_CONTEXTS)
        width = rng.choice(widths)
        lo = width * rng.randint(0, lo_mult)
        freqs = _profile(rng, kind)
        if freqs is None:
            continue
        mean, median, med_i, n = _stats(freqs, lo, width)
        gap = mean - median
        if kind == "sym":
            if abs(gap) >= Fraction(width, 20):
                continue
        else:
            if (kind == "right") != (gap > 0) or abs(gap) < Fraction(width, 8):
                continue
        if not (clear_of_boundary(mean, 1) and clear_of_boundary(median, 1)):
            continue
        break
    a_val, b_val = rd(mean, 1), rd(median, 1)
    c_val = a_val - b_val

    table = ", ".join(f"{_class_text(lo, width, i)}: {f}" for i, f in enumerate(freqs))
    stem = sentence.format(n=n) + rng.choice(_CLOSERS).format(table=table) + " " + _CONVENTION

    mids = [Fraction(2 * (lo + i * width) + width, 2) for i in range(len(freqs))]
    k = len(freqs)
    cum_before = sum(freqs[:med_i])
    m_lower = lo + med_i * width
    mean_wrongs = [
        (sum(mids) / k, "ignored-frequencies-in-mean"),
        (sum(f * (lo + i * width) for i, f in enumerate(freqs)) / n, "used-class-limit-not-midpoint"),
        (sum(f * (lo + (i + 1) * width) for i, f in enumerate(freqs)) / n, "used-class-limit-not-midpoint"),
    ]
    median_wrongs = [
        (mids[med_i], "median-class-not-interpolated"),
        (Fraction(m_lower), "median-class-not-interpolated"),
        (m_lower + (Fraction(n) - cum_before) / freqs[med_i] * width, "forgot-to-halve-n"),
        (Fraction(n, 2), "position-reported-as-median"),
    ]
    diff_wrongs = [
        (-c_val if c_val != 0 else None, "reversed-difference"),
        (abs(c_val) if c_val < 0 else None, "dropped-sign-of-difference"),
        (a_val / b_val if b_val else None, "ratio-not-difference"),
    ]
    if c_val == 0:
        diff_wrongs = [
            (a_val / b_val, "ratio-not-difference"),
            (a_val, "gave-one-measure-not-difference"),
            (b_val, "gave-one-measure-not-difference"),
        ]

    pa = number_part(
        "Estimate the mean of the data, using the class midpoints, correct to 1 decimal place.",
        mean,
        1,
        2,
        [
            "Midpoints: " + ", ".join(tx(m, 1) for m in mids) + ".",
            f"Mean = (sum of frequency x midpoint) / {n} = {tx(mean, 1)}.",
        ],
        mean_wrongs,
    )
    pb = number_part(
        "Estimate the median by linear interpolation within the median class, correct to 1 decimal place.",
        median,
        1,
        2,
        [
            f"Half of {n} is {tx(Fraction(n, 2), 1)}; the median class is {_class_text(lo, width, med_i)}.",
            f"Median = {m_lower} + ({tx(Fraction(n, 2), 1)} - {cum_before}) / {freqs[med_i]} x {width} = {tx(median, 1)}.",
        ],
        median_wrongs,
    )
    pc = number_part(
        "Using your answers to (a) and (b), find the mean minus the median, correct to 1 decimal place.",
        c_val,
        1,
        1,
        [f"{tx(a_val, 1)} - {tx(b_val, 1)} = {tx(c_val, 1)}."],
        diff_wrongs,
        follow_through="part_a - part_b",
    )
    if pa is None or pb is None or pc is None:
        return None

    other = {"right": "left", "left": "right"}
    if kind == "sym":
        shape_wrongs = [
            (_LABELS["right"], "symmetric-as-skewed"),
            (_LABELS["left"], "symmetric-as-skewed"),
            (_LABELS["bimodal"], "unimodal-as-bimodal"),
        ]
    else:
        shape_wrongs = [
            (_LABELS[other[kind]], "skew-direction-confusion"),
            (_LABELS["sym"], "skewed-as-symmetric"),
            (_LABELS["bimodal"], "unimodal-as-bimodal"),
        ]
    if kind == "sym":
        why = f"The mean and median differ by {tx(abs(c_val), 1)}, under a tenth of the class width ({width / 10}), so the distribution is symmetric."
    elif kind == "right":
        why = "The mean is above the median: the long tail is to the right, so the distribution is skewed right."
    else:
        why = "The mean is below the median: the long tail is to the left, so the distribution is skewed left."
    pd = word_part(
        "Use the sign and size of this difference, and the table, to state the shape of the distribution. "
        "Answer with one word: right (skewed right), left (skewed left), symmetric or bimodal.",
        _LABELS[kind],
        shape_wrongs,
        1,
        [why],
    )

    # Independent check of the shape from the table itself: the longer tail is on the skewed side.
    peak = freqs.index(max(freqs))
    before, after = peak, len(freqs) - peak - 1
    if kind == "right":
        check(after > before, "histogram: right skew has no longer right tail")
    elif kind == "left":
        check(before > after, "histogram: left skew has no longer left tail")

    return dict(text=stem, parts=[pa, pb, pc, pd])


@generator("exd_x_r2k_histogram", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2k_histogram(rng: random.Random, tier: str = "exam") -> dict:
    """Mean, median, their difference and the shape they point to, for a grouped frequency table."""
    check(tier == "exam", "exd_x_r2k_histogram: only the exam tier exists")
    return retry_build(rng, _build_k_histogram, "exd_x_r2k_histogram")
