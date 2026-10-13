"""Exam source for `z-score`: a four-part chain that recovers a normal distribution from two z-scores.

The stem gives two observations of an approximately normal quantity together with their z-scores.

- (a) the standard deviation, from the difference of the two observations over the difference of their z-scores
  (2 marks);
- (b) the mean (1 mark, ``follow_through`` from (a));
- (c) the z-score of a third observation (2 marks, ``follow_through`` from (a) and (b));
- (d) the probability of exceeding (or falling below) that third observation, to 4 decimal places (2 marks,
  ``follow_through`` from (c), through ``erf``).

## Construction

The mean and standard deviation are drawn first as whole numbers. A z-score is a whole number of hundredths
chosen so that ``mean + z * sd`` is a whole number, which depends on the standard deviation, so every
observation shown is an integer and every z-score is exact to 2 decimal places. The probability in (d) is
computed from the exact z-score with `math.erf`, then re-derived with `statistics.NormalDist`, and kept clear of
a rounding boundary.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from statistics import NormalDist

from ..answers import Distractor
from ..core import (
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_SQRT2 = math.sqrt(2.0)
_MARGIN = 1e-5

_CONTEXTS = [
    dict(
        intro="The marks in a trial examination are approximately normally distributed.",
        thing="mark",
        entity="student",
        unit="",
        mus=range(55, 76, 1),
        sigmas=(5, 8, 10, 12, 15),
    ),
    dict(
        intro="The masses of loaves from a bakery are approximately normally distributed.",
        thing="mass",
        entity="loaf",
        unit=" g",
        mus=range(600, 801, 10),
        sigmas=(4, 5, 8, 10, 12),
    ),
    dict(
        intro="The heights of seedlings at a nursery after eight weeks are approximately normally distributed.",
        thing="height",
        entity="seedling",
        unit=" mm",
        mus=range(100, 401, 10),
        sigmas=(5, 8, 10, 12, 15, 20, 25),
    ),
    dict(
        intro="The times taken by commuters to travel from a suburb to the city are approximately normally "
        "distributed.",
        thing="journey time",
        entity="commuter",
        unit=" minutes",
        mus=range(25, 61, 1),
        sigmas=(4, 5, 8, 10, 12, 15),
    ),
]


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


def _z_text(k: int) -> str:
    return stated_precision_string(Fraction(k, 100), 2)


def _step(sigma: int) -> int:
    """Smallest step in hundredths of a z-score that keeps ``mu + z * sigma`` a whole number."""
    return 100 // math.gcd(100, sigma)


def _num(x: Fraction, places: int, tag: str) -> Distractor:
    r = round_half_up(x, places)
    return Distractor(stated_precision_string(r, places) if places else str(int(r)), tag, r)


@generator("exd_x_r2e_zscore_chain", level=12, presets={"exam": dict(tier="exam")})
def zscore_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Recover sd and mean from two z-scores, standardise a third value, and find a normal tail probability."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    upper = rng.random() < 0.5
    for _ in attempts(4000, "a z-score chain with clear probabilities"):
        ctx = rng.choice(_CONTEXTS)
        mu = rng.choice(ctx["mus"])
        sigma = rng.choice(ctx["sigmas"])
        step = _step(sigma)
        ks = [step * j for j in range(-(260 // step), 260 // step + 1)]
        k1, k2, k3 = rng.sample(ks, 3)
        if k1 > k2:
            k1, k2 = k2, k1
        if k2 - k1 < 80 or abs(k3) < 20 or k1 == 0 or k2 == 0 or k1 + k2 == 0 or k2 - k1 == 100 or abs(k2) == 100:
            continue
        x1, x2, x3 = (mu + Fraction(k * sigma, 100) for k in (k1, k2, k3))
        if min(x1, x2, x3) <= 0 or any(x.denominator != 1 for x in (x1, x2, x3)):
            continue
        z3 = k3 / 100
        tail = _phi(-z3) if upper else _phi(z3)
        if abs(tail * 1e4 - math.floor(tail * 1e4) - 0.5) < 0.02:
            continue
        # Second route: the distribution object.
        nd = NormalDist(mu, sigma)
        tail2 = 1 - nd.cdf(float(x3)) if upper else nd.cdf(float(x3))
        check(abs(tail - tail2) < 1e-9, "probability routes disagree")
        break

    x1, x2, x3 = int(x1), int(x2), int(x3)
    z1, z2 = Fraction(k1, 100), Fraction(k2, 100)
    check(Fraction(x2 - x1) / (z2 - z1) == sigma, "standard deviation from the two observations disagrees")
    check(x1 - z1 * sigma == mu, "mean from the first observation disagrees")
    p4 = round_half_up(Fraction(tail), 4)
    p_text = stated_precision_string(p4, 4)
    z3_text = _z_text(k3)

    thing, unit = ctx["thing"], ctx["unit"]
    stem = (
        f"{ctx['intro']} A {thing} of {x1}{unit} has a z-score of {_z_text(k1)}, and a {thing} of {x2}{unit} "
        f"has a z-score of {_z_text(k2)}."
    )

    a_ratio = round_half_up(Fraction(x2 - x1) / (z1 + z2), 0)
    a_cands = [
        Distractor(str(int(a_ratio)), "adds-z-scores", a_ratio),
        Distractor(str(x2 - x1), "forgets-to-divide-by-z-difference", x2 - x1),
        Distractor(
            str(int(round_half_up(Fraction(x2 - x1) / abs(z2), 0))),
            "uses-only-one-z-score",
            round_half_up(Fraction(x2 - x1) / abs(z2), 0),
        ),
        Distractor(
            str(int(round_half_up(Fraction(x2 - x1) * (z2 - z1), 0))),
            "multiplied-instead-of-divided",
            round_half_up(Fraction(x2 - x1) * (z2 - z1), 0),
        ),
    ]
    b_cands = [
        Distractor(str(int(x1 + z1 * sigma)), "sign-error-moving-term", x1 + z1 * sigma),
        Distractor(str(x1), "forgets-the-z-score-term", x1),
        Distractor(str(int(round_half_up(x1 - z1))), "forgets-the-standard-deviation", round_half_up(x1 - z1)),
    ]
    c_cands = [
        _num(Fraction(x3 - mu), 2, "forgets-the-standard-deviation"),
        _num(Fraction(x3, sigma), 2, "omits-centering"),
        _num(Fraction(mu - x3, sigma), 2, "z-sign-reversed"),
        _num(Fraction(x3 - mu, sigma * sigma), 2, "variance-as-standard-deviation"),
    ]
    wrong_tail = 1 - float(p4)
    d_cands = [
        _num(Fraction(wrong_tail), 4, "gives-the-complement"),
        _num(Fraction(2 * float(p4)), 4, "doubles-for-two-tails"),
        _num(Fraction(abs(0.5 - float(p4))), 4, "uses-area-from-the-mean"),
        _num(Fraction(_phi(-abs(z3)) if not upper else _phi(abs(z3))), 4, "gives-the-complement"),
    ]
    picks = [
        pick_distractors(str(sigma), a_cands, 3),
        pick_distractors(str(mu), b_cands, 3),
        pick_distractors(z3_text, c_cands, 3, tolerance=stated_precision_tolerance(2)),
        pick_distractors(p_text, d_cands, 3, tolerance=stated_precision_tolerance(4)),
    ]
    check(all(len(p) >= 2 for p in picks), f"a part has too few distractors: {[len(p) for p in picks]}")

    side = "greater" if upper else "less"
    erf_term = "(1 - erf(part_c/sqrt(2)))/2" if upper else "(1 + erf(part_c/sqrt(2)))/2"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the standard deviation of the distribution.",
                answer_kind="number",
                answer=sigma,
                answer_text=str(sigma),
                answer_latex=latex_answer(str(sigma)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"x = mean + z x sd, so {x2} - {x1} = ({_z_text(k2)} - ({_z_text(k1)})) x sd.",
                    f"sd = {sigma}.",
                ],
                distractors=picks[0],
            ),
            dict(
                prompt="Hence find the mean of the distribution.",
                answer_kind="number",
                answer=mu,
                answer_text=str(mu),
                answer_latex=latex_answer(str(mu)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{x1} - ({z1})*part_a",
                working=[f"mean = {x1} - ({_z_text(k1)}) x {sigma} = {mu}."],
                distractors=picks[1],
            ),
            dict(
                prompt=f"Find the z-score of a {thing} of {x3}{unit}.",
                answer_kind="number",
                answer=numeric_answer(Fraction(k3, 100)),
                answer_text=z3_text,
                answer_latex=latex_answer(z3_text),
                marks=2,
                follow_through=f"({x3} - part_b)/part_a",
                working=[f"z = ({x3} - {mu}) / {sigma} = {z3_text}."],
                distractors=picks[2],
            ),
            dict(
                prompt=f"Using your z-score from part (c), find the probability that a randomly chosen {ctx['entity']} "
                f"has a {thing} {side} than {x3}{unit}. Give your answer correct to 4 decimal places.",
                answer_kind="number",
                answer=numeric_answer(p4),
                answer_text=p_text,
                answer_latex=latex_answer(p_text),
                marks=2,
                sensible_range=(0, 1),
                follow_through=erf_term,
                working=[f"P(Z {'>' if upper else '<'} {z3_text}) = {p_text}."],
                distractors=picks[3],
            ),
        ],
    )
