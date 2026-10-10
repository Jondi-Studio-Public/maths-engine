"""Exam-tier questions for the interval estimate of a proportion.

The easier tiers hand the learner the sample proportion and ask for a margin of error or a sample size. Here the
learner is shown only the two ENDS of a published confidence interval, and must see that the sample proportion is
the midpoint and the margin of error is half the width before the sample-size formula applies:

    p_hat = (lower + upper) / 2        E = (upper - lower) / 2        n >= z^2 p_hat (1 - p_hat) / E^2

The key is the smallest whole sample size whose margin of error does not exceed E. The stated end points are
exact decimals (p_hat has two decimal places and E four), the key is computed in exact ``Fraction`` arithmetic,
and a draw is kept only when ``z^2 p_hat (1 - p_hat) / E^2`` is clear of a whole number, so the ceiling is never
decided by rounding noise.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, check_distinct, generator, phrase, pick_distractors

_Z = {"90%": Fraction(1645, 1000), "95%": Fraction(196, 100), "99%": Fraction(2576, 1000)}


def _ceil_div(num: Fraction) -> int:
    return math.ceil(num)


@generator("exd_x_s4_interval", level=12, presets={"exam": dict(tier="exam")})
def exd_x_s4_interval(rng, tier="exam"):
    for _ in range(2000):
        conf = rng.choice(list(_Z))
        z = _Z[conf]
        p = Fraction(rng.randint(20, 80), 100)
        e = Fraction(rng.randint(150, 450), 10000)
        n_exact = z * z * p * (1 - p) / (e * e)
        frac = n_exact - math.floor(n_exact)
        if Fraction(1, 20) < frac < Fraction(19, 20):
            break
    else:
        raise RuntimeError("exd_x_s4_interval: no draw clear of a whole number")
    lo, hi = p - e, p + e
    key = math.ceil(n_exact)

    # Independent route: the smallest n whose margin of error z*sqrt(p(1-p)/n) is at most e, by comparing
    # squares (z^2 p (1-p) / n <= e^2) and walking n up from below the formula value.
    n = max(1, key - 3)
    while z * z * p * (1 - p) / n > e * e:
        n += 1
    check(n == key, "key disagrees with the direct search for the smallest sufficient n")
    check(z * z * p * (1 - p) / (key - 1) > e * e, "key - 1 must be too small")

    pct = rng.random() < 0.4
    if pct:
        lo_s, hi_s = f"{float(lo) * 100:.2f}%", f"{float(hi) * 100:.2f}%"
    else:
        lo_s, hi_s = f"{float(lo):.4f}", f"{float(hi):.4f}"
    text = phrase(
        rng,
        (
            "A survey reports a {conf} confidence interval for a population proportion of ({lo} to {hi}). "
            "Using z = {z}, what is the smallest sample size for which the margin of error is no more than "
            "the half-width of this interval?",
            "A {conf} confidence interval for a proportion is quoted as ({lo}, {hi}), built as p-hat plus or "
            "minus z times the square root of p-hat(1 - p-hat)/n with z = {z}. Find the minimum sample size "
            "n that gives a margin of error no larger than the half-width of the interval.",
            "The end points of a {conf} confidence interval for a proportion are {lo} and {hi}. "
            "Take z = {z}. What is the least number of observations needed for the margin of error to be at "
            "most the half-width of this interval?",
        ),
        conf=conf,
        lo=lo_s,
        hi=hi_s,
        z=f"{float(z):g}",
    )

    # Misconceptions: the whole width is used as the margin of error; the lower end point is used as the
    # sample proportion; z is not squared; the ceiling is replaced by a floor.
    width = hi - lo
    cands = [
        _ceil_div(z * z * p * (1 - p) / (width * width)),
        _ceil_div(z * z * lo * (1 - lo) / (e * e)),
        _ceil_div(p * (1 - p) / (e * e)),
        math.floor(n_exact),
    ]
    answer_text = str(key)
    distractors = pick_distractors(answer_text, [str(c) for c in cands], n=3)
    check_distinct(key, *[int(d) for d in distractors])
    check(len(distractors) >= 3, "need three principled distractors")

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"p-hat = ({lo_s} + {hi_s})/2 = {float(p):g} and E = ({hi_s} - {lo_s})/2 = {float(e):g}"
            if not pct
            else f"p-hat = {float(p):g} and E = {float(e):g} (the end points are given as percentages)",
            f"n >= z^2 p-hat(1 - p-hat)/E^2 = {float(n_exact):.2f}, so the smallest whole n is {key}.",
        ],
        distractors=distractors,
    )
