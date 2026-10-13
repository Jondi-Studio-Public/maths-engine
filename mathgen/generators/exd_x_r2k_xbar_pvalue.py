"""Exam question: how unusual is a sample mean? Standard error, z, and the two-sided tail probability.

A large sample reports its mean and its own standard deviation s; the population mean is claimed. Part (a) is the
estimated standard error s / sqrt(n), part (b) the standardised value z of the sample mean using it, and part (c) the
probability, from the normal model, of a sample mean at least this far from the claimed mean in either direction
(2 x P(Z > |z|), with z rounded to 2 places as the prompt says). Parts (b) and (c) carry `follow_through`.

Built from the standardised value: a target z is drawn, the sample mean is set to put z there (to 2 places), and a draw
is kept only when z is the same to 2 places whether it is worked from the rounded standard error or the exact one, and
neither it nor the probability sits close to a rounding boundary.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator
from .exd_x_r2k_support import (
    retry_build,
    PROBABILITY,
    POSITIVE,
    clear_of_boundary,
    exact_from_float,
    normal_sf,
    normal_two_tail,
    number_part,
    rd,
    tx,
)

# description, subject of the sentence, unit, mu choices, s range
_CONTEXTS = (
    ("delivery times", "deliveries", "minutes", (22, 25, 28, 30, 35, 40), (3.0, 9.0)),
    ("battery lives", "batteries", "hours", (10, 12, 14, 15, 18, 20), (1.0, 3.5)),
    ("fill volumes", "bottles", "millilitres", (250, 300, 330, 375, 500, 600), (2.0, 9.0)),
    ("journey times to work", "commuters", "minutes", (30, 35, 40, 45, 50), (5.0, 14.0)),
    ("masses", "parcels", "kilograms", (4, 5, 6, 8, 10, 12), (0.4, 1.8)),
)


def _build_k_xbar_pvalue(rng: random.Random) -> dict | None:
    """Standard error, standardised sample mean, and the chance of a sample mean this far from the claimed mean."""
    thing, subject, unit, mus, s_range = rng.choice(_CONTEXTS)
    for _ in attempts(3000, "a sample with a stable z and tail probability"):
        n = rng.randint(30, 120)
        mu = rng.choice(mus)
        s = Fraction(round(rng.uniform(*s_range) * 100), 100)
        se = Fraction(repr(float(s) / math.sqrt(n)))
        z0 = rng.uniform(0.4, 3.3) * rng.choice((-1, 1))
        xbar = Fraction(round((mu + z0 * float(se)) * 100), 100)
        if xbar == mu:
            continue
        diff = xbar - mu
        se_shown = rd(se, 3)
        z_shown = rd(diff / se_shown, 2)
        z_exact = float(diff) / float(se)
        if rd(Fraction(repr(z_exact)), 2) != z_shown or not clear_of_boundary(diff / se_shown, 2, Fraction(1, 5)):
            continue
        if abs(z_shown) < Fraction(3, 10):
            continue
        tail = exact_from_float(normal_two_tail(float(z_shown)), 4)
        raw = Fraction(repr(normal_two_tail(float(z_shown))))
        if tail == 0 or not clear_of_boundary(raw, 4, Fraction(1, 5)):
            continue
        break
    check(abs(float(z_shown) - z_exact) < 0.011, "xbar: z disagrees with the exact route")
    one_tail = Fraction(repr(normal_sf(abs(float(z_shown)))))
    stem = (
        f"A random sample of {n} {subject} gave a mean x̄ = {tx(xbar, 2)} {unit} and a standard deviation "
        f"s = {tx(s, 2)} {unit}. The population mean is claimed to be μ = {mu} {unit}. The sample is large, so the "
        f"sampling distribution of the sample mean may be taken to be approximately normal."
    )
    pa = number_part(
        "Find the estimated standard error of the sample mean, s / √n, correct to 3 decimal places.",
        se,
        3,
        1,
        [f"s / √n = {tx(s, 2)} / √{n} = {tx(se, 3)}."],
        [
            (s / n, "divides-by-n"),
            (
                Fraction(repr(float(s) / math.sqrt(n - 1)))
                if abs(float(s) / math.sqrt(n - 1) - float(se)) > 0.01
                else None,
                "wrong-n-in-root",
            ),
            (s, "omits-sqrt-n"),
            (s * Fraction(repr(math.sqrt(n))), "multiplies-by-sqrt-n"),
        ],
        sensible_range=POSITIVE,
    )
    pb = number_part(
        "Hence find z, the standardised value of the sample mean, equal to (x̄ − μ) divided by your standard error, "
        "correct to 2 decimal places.",
        diff / se_shown,
        2,
        2,
        [f"z = ({tx(xbar, 2)} - {mu}) / {tx(se_shown, 3)} = {tx(z_shown, 2)}."],
        [
            (-diff / se_shown, "reversed-difference"),
            (diff / s, "omits-sqrt-n"),
            (diff / (s / n), "divides-by-n"),
            (diff * se_shown, "multiplies-instead-of-dividing"),
        ],
        follow_through=f"({xbar} - {mu})/part_a",
        sensible_range=(-12, 12),  # keeps a wild z from being fed to the tail-probability expression
    )
    pc = number_part(
        "Using your value of z from (b), find the probability, correct to 4 decimal places, of "
        "getting a sample mean at least this far from μ in either direction, if the population mean really is μ.",
        tail,
        4,
        2,
        [f"P(|Z| > {abs(float(z_shown)):.2f}) = 2 x {float(one_tail):.5f} = {tx(tail, 4)}."],
        [
            (one_tail, "one-tail-for-two-tails"),
            (1 - Fraction(repr(normal_two_tail(float(z_shown)))), "gave-complement-of-tail"),
            (1 - one_tail, "gave-cumulative-for-tail"),
        ],
        follow_through="erfc(Abs(part_b)/sqrt(2))",
        sensible_range=PROBABILITY,
    )
    if pa is None or pb is None or pc is None:
        return None
    return dict(text=stem, parts=[pa, pb, pc])


@generator("exd_x_r2k_xbar_pvalue", level=13, presets={"exam": dict(tier="exam")})
def exd_x_r2k_xbar_pvalue(rng: random.Random, tier: str = "exam") -> dict:
    """Standard error, standardised sample mean, and the chance of a sample mean this far from the claimed mean."""
    check(tier == "exam", "exd_x_r2k_xbar_pvalue: only the exam tier exists")
    return retry_build(rng, _build_k_xbar_pvalue, "exd_x_r2k_xbar_pvalue")
