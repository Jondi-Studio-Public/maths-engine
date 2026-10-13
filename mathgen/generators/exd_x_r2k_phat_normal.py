"""Exam question: the sampling distribution of a sample proportion, used to judge an observed sample.

A population proportion p and a sample size n are given, then a sample in which x items had the property. Part (a)
finds the standard deviation of the sample proportion, sqrt(p(1-p)/n); part (b) standardises the observed sample
proportion x/n with it; part (c) finds the probability, under the normal model, of a sample proportion at least that
far from p in the direction observed. Parts (b) and (c) carry `follow_through`.

Built from a target z: x is the whole number nearest to n p + z sd n, and a draw is kept only when z is the same to 2
places whether it is worked from the rounded standard deviation or the exact one, and neither z nor the probability
sits close to a rounding boundary. The probability is worked from z rounded to 2 places, as the prompt says.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator
from .exd_x_r2k_support import (
    retry_build,
    POSITIVE,
    PROBABILITY,
    clear_of_boundary,
    exact_from_float,
    normal_sf,
    number_part,
    rd,
    tx,
)

# sentence about the population, noun for a member, property phrase for the sample sentence
_CONTEXTS = (
    ("A large fish farm finds that {pc}% of its fish are over the legal size.", "fish", "were over the legal size"),
    (
        "In a large school, {pc}% of students travel to school by public transport.",
        "students",
        "travel by public transport",
    ),
    (
        "A large electronics supplier knows that {pc}% of its phone chargers pass the first test.",
        "chargers",
        "passed the first test",
    ),
    ("In a city, {pc}% of households recycle all their glass.", "households", "recycle all their glass"),
    (
        "A seed company reports that {pc}% of its tomato seeds germinate within a week.",
        "seeds",
        "germinated within a week",
    ),
)


def _build_k_phat_normal(rng: random.Random) -> dict | None:
    """SD of the sample proportion, the z of an observed sample, and the tail probability."""
    sentence, noun, phrase_ = rng.choice(_CONTEXTS)
    for _ in attempts(3000, "a sample with a stable z and tail probability"):
        pc = rng.randrange(20, 81, 5)
        p = Fraction(pc, 100)
        n = rng.randint(40, 250)
        sd = Fraction(repr(math.sqrt(float(p * (1 - p)) / n)))
        z0 = rng.uniform(0.5, 3.0) * rng.choice((-1, 1))
        x = round(n * float(p) + z0 * float(sd) * n)
        if not 0 < x < n:
            continue
        obs = Fraction(x, n)
        diff = obs - p
        sd_shown = rd(sd, 4)
        z_shown = rd(diff / sd_shown, 2)
        z_exact = Fraction(repr(float(diff) / float(sd)))
        if rd(z_exact, 2) != z_shown or not clear_of_boundary(diff / sd_shown, 2, Fraction(1, 5)):
            continue
        if abs(z_shown) < Fraction(1, 2) or not clear_of_boundary(sd, 4, Fraction(1, 5)):
            continue
        tail_raw = Fraction(repr(normal_sf(abs(float(z_shown)))))
        tail = exact_from_float(normal_sf(abs(float(z_shown))), 4)
        if tail == 0 or not clear_of_boundary(tail_raw, 4, Fraction(1, 5)):
            continue
        break
    above = diff > 0
    stem = (
        sentence.format(pc=pc)
        + f" A random sample of {n} {noun} is taken, and {x} of them {phrase_}. The sample is large enough for the "
        f"sample proportion p̂ to be taken as approximately normal."
    )
    direction = "at least" if above else "at most"
    pa = number_part(
        f"Find the standard deviation of the sample proportion p̂ for samples of size {n}, correct to 4 decimal places.",
        sd,
        4,
        1,
        [f"sd = √(p(1 - p)/n) = √({pc / 100:.2f} x {1 - pc / 100:.2f} / {n}) = {tx(sd, 4)}."],
        [
            (Fraction(repr(math.sqrt(float(p * (1 - p))))), "omits-n"),
            (p * (1 - p) / n, "omits-sqrt"),
            (Fraction(repr(math.sqrt(float(p) / n))), "forgot-one-minus-p"),
            (Fraction(repr(math.sqrt(float(p * (1 - p)) / (n - 1)))), "wrong-n-in-root"),
        ],
        sensible_range=POSITIVE,
    )
    pb = number_part(
        f"Hence find the z-score of the observed sample proportion {x}/{n}, using your standard deviation from (a), "
        f"correct to 2 decimal places.",
        diff / sd_shown,
        2,
        2,
        [
            f"p̂ = {x}/{n} = {float(obs):.4f}; z = ({float(obs):.4f} - {pc / 100:.2f}) / {tx(sd_shown, 4)} = {tx(z_shown, 2)}."
        ],
        [
            (-diff / sd_shown, "reversed-difference"),
            (diff / Fraction(repr(math.sqrt(float(p * (1 - p))))), "used-population-sd-not-standard-error"),
            (diff / (sd_shown**2), "divides-by-variance"),
        ],
        follow_through=f"({x}/{n} - {pc}/100)/part_a",
        sensible_range=(-12, 12),  # keeps a wild z from being fed to the tail-probability expression
    )
    pc_ = number_part(
        f"Using your z-score from (b), find the probability, correct to 4 decimal places, that a "
        f"sample of {n} has a sample proportion {direction} {x}/{n}, if the population proportion really is "
        f"{pc / 100:.2f}.",
        tail,
        4,
        2,
        [f"P(Z {'>' if above else '<'} {float(z_shown):.2f}) = {tx(tail, 4)}."],
        [
            (2 * tail_raw, "doubled-for-one-tail"),
            (1 - tail_raw, "gave-cumulative-for-tail"),
            (Fraction(1, 2) - tail_raw, "gave-area-from-centre"),
        ],
        follow_through="erfc(Abs(part_b)/sqrt(2))/2",
        sensible_range=PROBABILITY,
    )
    if pa is None or pb is None or pc_ is None:
        return None
    return dict(text=stem, parts=[pa, pb, pc_])


@generator("exd_x_r2k_phat_normal", level=13, presets={"exam": dict(tier="exam")})
def exd_x_r2k_phat_normal(rng: random.Random, tier: str = "exam") -> dict:
    """SD of the sample proportion, the z of an observed sample, and the tail probability."""
    check(tier == "exam", "exd_x_r2k_phat_normal: only the exam tier exists")
    return retry_build(rng, _build_k_phat_normal, "exd_x_r2k_phat_normal")
