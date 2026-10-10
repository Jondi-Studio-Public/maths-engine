"""Exam questions on the sample standard deviation where the data are not simply listed.

Two forms, chosen by the seed:

* `sums` -- only the number of values, their sum and the sum of their squares are given. The learner
  must choose the computational form `s^2 = (sum_sq - sum^2 / n) / (n - 1)`; there is no list of
  deviations to work through.
* `missing` -- all but one of the values are listed, together with the sample mean of the whole set.
  The learner must first recover the missing value from the mean, then compute the sample standard
  deviation of the complete set.

Both are built backwards from an integer data set, so every stated total is exact. The key is rounded
to 2 decimal places only at the end, and draws that sit near a rounding tie are rejected. Distractors
are graded against the typed-number tolerance of half a hundredth.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import check, generator, phrase, pick_distractors, rounds_unambiguously, stated_precision_tolerance

_TOL = stated_precision_tolerance(2)

_SUMS = (
    "A sample of {n} measurements has a total of {s} and the sum of the squares of the measurements is {q}. "
    "Calculate the sample standard deviation, to 2 decimal places.",
    "For a sample of {n} values, the sum of the values is {s} and the sum of their squares is {q}. Find the "
    "sample standard deviation of the values, correct to 2 decimal places.",
    "Summary of a sample: n = {n}, sum of values = {s}, sum of squared values = {q}. What is the sample "
    "standard deviation, to 2 decimal places?",
)
_MISSING = (
    "A sample of {n} values has a sample mean of {mean}. {k} of the values are {listed}. Find the sample "
    "standard deviation of all {n} values, to 2 decimal places.",
    "The sample mean of {n} readings is {mean}. The readings that are known are {listed}; one reading is "
    "missing. Calculate the sample standard deviation of the full set of {n} readings, to 2 decimal places.",
)
_COUNT_WORDS = {4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}


def _s2(values) -> Fraction:
    n = len(values)
    mean = Fraction(sum(values), n)
    return sum((Fraction(v) - mean) ** 2 for v in values) / (n - 1)


def _fmt2(x: float) -> str:
    return f"{x:.2f}"


def _finish(text, working, values, cand_vals):
    n = len(values)
    var = _s2(values)
    sd = math.sqrt(float(var))
    if not rounds_unambiguously(sd, 2, margin=0.1):
        return None
    key = _fmt2(round(sd, 2))
    cands = [_fmt2(round(c, 2)) for c in cand_vals]
    distractors = pick_distractors(key, cands, tolerance=_TOL)
    if len(distractors) < 3:
        return None
    # Independent check: the pairwise-difference identity s^2 = sum_{i<j} (xi - xj)^2 / (n (n - 1)).
    pair = sum((Fraction(values[i] - values[j])) ** 2 for i in range(n) for j in range(i + 1, n)) / (n * (n - 1))
    check(pair == var, "sd_summary: pairwise identity disagrees with the sample variance")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=float(key),
        answer_text=key,
        answer_latex=key,
        distractors=distractors[:3],
        working=working,
    )


def _sums(rng):
    n = rng.randint(8, 12)
    values = [rng.randint(8, 60) for _ in range(n)]
    s, q = sum(values), sum(v * v for v in values)
    var = _s2(values)
    ssd = Fraction(q) - Fraction(s * s, n)  # sum of squared deviations
    check(ssd / (n - 1) == var, "sd_summary: computational form disagrees with the definition")
    if ssd == 0:
        return None
    text = phrase(rng, _SUMS, n=n, s=s, q=q)
    working = [
        f"Sum of squared deviations = {q} - {s}^2 / {n} = {float(ssd):.4f}.",
        f"Sample variance = {float(ssd):.4f} / {n - 1} = {float(var):.4f}.",
        f"Sample standard deviation = square root of {float(var):.4f}.",
    ]
    cands = [
        math.sqrt(float(ssd) / n),  # divides by n
        float(var),  # forgets the square root
        math.sqrt(q / (n - 1)),  # forgets to subtract the correction for the mean
        math.sqrt(q / (n - 1) - s / n),  # subtracts the mean instead of the squared mean
    ]
    return _finish(text, working, values, cands)


def _missing(rng):
    n = rng.randint(5, 8)
    values = [rng.randint(8, 50) for _ in range(n)]
    total = sum(values)
    if (total * 10) % n != 0:
        return None
    mean = Fraction(total, n)
    miss_pos = n - 1
    hidden = values[miss_pos]
    listed_vals = values[:miss_pos]
    rng.shuffle(listed_vals)
    listed = ", ".join(map(str, listed_vals))
    mean_s = f"{float(mean):.1f}".rstrip("0").rstrip(".")
    # the hidden value is recovered from the mean
    recovered = n * mean - sum(listed_vals)
    check(recovered == hidden and recovered > 0, "sd_summary: recovered missing value is wrong")
    text = phrase(rng, _MISSING, n=n, mean=mean_s, k=_COUNT_WORDS[n - 1], listed=listed)
    var = _s2(values)
    ssd = var * (n - 1)
    only_listed = math.sqrt(float(_s2(listed_vals)))  # ignores the missing value altogether
    with_mean = values[:miss_pos] + [mean]  # fills the gap with the mean instead of the recovered value
    cands = [
        math.sqrt(float(ssd) / n),  # divides by n
        float(var),  # forgets the square root
        only_listed,
        math.sqrt(float(_s2(with_mean))),
    ]
    working = [
        f"Missing value = {n} x {mean_s} - ({' + '.join(map(str, listed_vals))}) = {hidden}.",
        f"Sum of squared deviations from the mean {mean_s} = {float(ssd):.4f}.",
        f"Sample variance = {float(ssd):.4f} / {n - 1} = {float(var):.4f}; take the square root.",
    ]
    return _finish(text, working, values, cands)


def _build(rng: random.Random) -> dict:
    for _ in range(2000):
        out = _sums(rng) if rng.random() < 0.5 else _missing(rng)
        if out is not None:
            return out
    raise RuntimeError("exd_x_s5_sd_summary: no draw satisfied the constraints in 2000 attempts")


@generator("exd_x_s5_sd_summary", level=10, presets={"exam": dict(tier="exam")})
def exd_x_s5_sd_summary(rng: random.Random, tier: str = "exam") -> dict:
    """Sample standard deviation from summary sums, or with one value missing."""
    check(tier == "exam", "exd_x_s5_sd_summary: only the exam tier exists")
    return _build(rng)
