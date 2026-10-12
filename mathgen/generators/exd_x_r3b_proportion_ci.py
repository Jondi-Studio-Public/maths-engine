"""Generates `exd_x_r3b_proportion_ci`: a multi-part exam question that works back from a proportion interval.

The stem reports a normal-approximation confidence interval ``(L, U)`` for a proportion and the critical value z it
was built with. The chain is the one a statistics paper sets, working back from the printed interval to the survey:

(a) the sample proportion p_hat = (L + U) / 2, 4 decimal places            number, 1 mark
(b) the margin of error (U - L) / 2, 4 decimal places                      number, 1 mark
(c) the sample size n = z^2 p_hat (1 - p_hat) / margin^2, whole number     number, 2 marks, from part_a and part_b
(d) the smallest sample size for a stated smaller margin, same p_hat       number, 2 marks, ceiling(z^2 p_hat (1 - p_hat) / t^2)

The draw is the shared one (`_draw_base`: a hidden n and k, bounds rounded to 4 decimal places) kept only when L + U
is even in the fourth decimal place, so p_hat and the margin are exact at 4 places. A draw is kept only when n
recovered from the printed numbers is the hidden n, and when every sensible route to (d) (the printed midpoint, the
sample's own k / n, and scaling n by (margin / target)^2) reaches the same smallest whole number.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator, phrase, round_half_up, stated_precision_string
from .exd_x_r3b_common import number_part
from .p17b_proportion_ci_inverse_build import _draw_base
from ._exd_r3_distinct import distinct_part_answers

_CONTEXTS = (
    "A survey of households in a regional town reports {art} {level}% confidence interval for the proportion who recycle "
    "all their glass: ({L}, {U}). The interval was found using z = {z}.",
    "A quality-control check on a batch of batteries gives {art} {level}% confidence interval for the proportion that "
    "are faulty: ({L}, {U}). The critical value used was z = {z}.",
    "A poll of voters in an electorate gives {art} {level}% confidence interval of ({L}, {U}) for the proportion "
    "supporting a proposed bike path, using z = {z}.",
    "Students at a large school were surveyed, and {art} {level}% confidence interval for the proportion who catch "
    "public transport to school is ({L}, {U}). It was built with z = {z}.",
    "A health clinic reports {art} {level}% confidence interval of ({L}, {U}) for the proportion of patients who "
    "complete a course of treatment, using z = {z}.",
)
_TARGETS = (30, 35, 40, 45, 55, 60, 65, 70)  # target margin as a percentage of the margin; never 50


def _fmt4(value: Fraction) -> str:
    return stated_precision_string(value, 4)


def _ceil(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _build(rng: random.Random) -> dict:
    for _ in attempts(4000, "a recoverable interval with a stable target sample size"):
        base = _draw_base(rng)
        z, lo, hi, level = base["z"], base["L4"], base["U4"], base["level"]
        if int((lo + hi) * 10000) % 2:
            continue
        p_hat, margin = (lo + hi) / 2, (hi - lo) / 2
        n_exact = z * z * p_hat * (1 - p_hat) / (margin * margin)
        n_key = int(round_half_up(n_exact, 0))
        if (
            abs(n_exact - n_key) > Fraction(2, 5)
            or n_key != base["n"]
            or base["k"] != int(round_half_up(p_hat * n_key, 0))
        ):
            continue
        target = None
        for pct in rng.sample(_TARGETS, len(_TARGETS)):
            t = round_half_up(margin * Fraction(pct, 100), 4)
            if t <= 0:
                continue
            by_midpoint = z * z * p_hat * (1 - p_hat) / (t * t)
            p_kn = Fraction(base["k"], n_key)
            by_kn = z * z * p_kn * (1 - p_kn) / (t * t)
            by_scale = n_key * (margin / t) ** 2
            frac = by_midpoint - int(by_midpoint)
            if not (Fraction(1, 20) <= frac <= Fraction(19, 20)):
                continue
            if len({_ceil(by_midpoint), _ceil(by_kn), _ceil(by_scale)}) != 1 or _ceil(by_midpoint) >= 10000:
                continue
            target = t
            break
        if target is not None:
            break
    n_big = _ceil(z * z * p_hat * (1 - p_hat) / (target * target))
    check(n_big > n_key, "the tighter target does not need a larger sample")
    check(math.isclose(float(n_exact), float(z) ** 2 * float(p_hat) * (1 - float(p_hat)) / float(margin) ** 2), "n")

    z_text = stated_precision_string(z, 3)
    stem = phrase(rng, _CONTEXTS, art="an" if level == 80 else "a", level=level, L=_fmt4(lo), U=_fmt4(hi), z=z_text)
    zq = f"({z.numerator}/{z.denominator})"
    t_text = _fmt4(target)
    tq = f"({target.numerator}/{target.denominator})"
    p_text, m_text = _fmt4(p_hat), _fmt4(margin)

    def n_with(zz, p, m, *, complement=True, root=True):
        value = zz * zz * p * ((1 - p) if complement else 1) / (m * m if root else m)
        return value

    wrong_n = {
        "margin-as-full-width": n_with(z, p_hat, margin * 2),
        "missing-z-squared": z * p_hat * (1 - p_hat) / (margin * margin),
        "missing-complement": n_with(z, p_hat, margin, complement=False),
        "no-square-root-inversion": z * z * p_hat * (1 - p_hat) / margin,
    }
    part_a = number_part(
        "Find the sample proportion p̂, the midpoint of the interval, correct to 4 decimal places.",
        p_hat,
        4,
        1,
        [f"p̂ = ({_fmt4(lo)} + {_fmt4(hi)}) / 2 = {p_text}."],
        [
            ("bound-as-p-hat", lo),
            ("bound-as-p-hat", hi),
            ("swapped-centre-and-margin", margin),
            ("complement-reported", 1 - p_hat),
        ],
        sensible_range=(0, 1),
    )
    part_b = number_part(
        "Find the margin of error of the interval, correct to 4 decimal places.",
        margin,
        4,
        1,
        [f"Margin = ({_fmt4(hi)} - {_fmt4(lo)}) / 2 = {m_text}."],
        [
            ("margin-as-full-width", margin * 2),
            ("swapped-centre-and-margin", p_hat),
            ("halved-twice", margin / 2),
            ("bound-as-p-hat", hi),
        ],
        sensible_range=(0, 1),
    )
    part_c = number_part(
        "Find the whole-number sample size n that produced this interval.",
        Fraction(n_key),
        0,
        2,
        [f"n = z² p̂ (1 - p̂) / margin² = {z_text}² × {p_text} × (1 - {p_text}) / {m_text}² = {n_key}."],
        [(mid, value) for mid, value in wrong_n.items()],
        follow_through=f"{zq}**2*part_a*(1 - part_a)/part_b**2",
        sensible_range=(0, None),
    )
    part_d = number_part(
        f"A follow-up study wants the margin of error to be at most {t_text}, with the same sample proportion and "
        "confidence level. Find the smallest whole-number sample size that achieves this.",
        Fraction(n_big),
        0,
        2,
        [
            f"N ≥ z² p̂ (1 - p̂) / {t_text}² = {float(z * z * p_hat * (1 - p_hat) / (target * target)):.2f}.",
            f"The smallest whole number is {n_big}.",
        ],
        [
            ("rounds-margin-target-down", Fraction(n_big - 1)),
            ("linear-margin-rescale", n_key * margin / target),
            ("missing-z-squared", z * p_hat * (1 - p_hat) / (target * target)),
            ("missing-complement", z * z * p_hat / (target * target)),
        ],
        follow_through=f"ceiling({zq}**2*part_a*(1 - part_a)/{tq}**2)",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r3b_proportion_ci", level=16, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3b_proportion_ci(rng: random.Random, tier: str = "exam") -> dict:
    """A proportion interval worked back to p̂, the margin, the survey size and a larger follow-up size."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
