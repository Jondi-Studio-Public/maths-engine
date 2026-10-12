"""Exam tier for "check mechanical energy across two supplied states" of a one-dimensional oscillator.

The practice tiers compute the energy difference between two stated states. Here a single state of a block on a
spring is given, and the learner uses conservation of energy to work out the rest of the motion, as a first-year
mechanics exam would set it:

(a) the total mechanical energy, from the displacement and speed in the stem;
(b) the amplitude (the displacement at which the speed is zero), from (a);
(c) the maximum speed (the speed at equilibrium), from (a) or (b);
(d) the speed at a different displacement, from (a).

## Construction

Take the angular frequency w to be a whole number, so the stiffness is k = m w^2. A Pythagorean triple
(x, s, a) with a^2 = x^2 + s^2 gives displacements and speeds on a grid of 0.1: with amplitude A = a / 10 metres,
the speed at displacement x = p / 10 is w s / 10 metres per second. Every answer is then exact, with (a) on a grid of
0.01 J and (b), (c) and (d) on a grid of 0.1.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_x_r3_distinct import distinct_answers
from ._exd_x_r3g_kit import clashes, number_part

# amplitude (in tenths of a metre) -> displacements (in tenths) at which the speed is a whole number of tenths times w
_TRIPLES = {
    5: ((3, 4), (4, 3)),
    10: ((6, 8), (8, 6)),
    13: ((5, 12), (12, 5)),
    15: ((9, 12), (12, 9)),
    17: ((8, 15), (15, 8)),
    20: ((12, 16), (16, 12)),
}
_STEMS = (
    "A block of mass {m} kg is attached to a horizontal spring of stiffness {k} N/m and slides on a smooth surface. "
    "At one instant the block is {x1} m from its equilibrium position and is moving at {v1} m/s.",
    "A trolley of mass {m} kg is fixed to a light spring of stiffness {k} N/m and oscillates along a frictionless "
    "track. When it is {x1} m from equilibrium its speed is {v1} m/s.",
    "A {m} kg mass on a spring of stiffness {k} N/m oscillates without friction. The mass is {x1} m from the "
    "equilibrium position when its speed is {v1} m/s.",
)


def _g(tenths: int | Fraction) -> str:
    return f"{float(Fraction(tenths) / 10):g}"


@generator("exd_x_r3g_oscillator_energy", level=12, presets={"exam": dict(tier="exam")})
@distinct_answers
def exd_x_r3g_oscillator_energy(rng: random.Random, tier: str = "exam") -> dict:
    """One state of a spring-mass oscillator: its energy, amplitude, maximum speed and speed at another position."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "an oscillator whose energy sits on the 0.01 J grid"):
        a = rng.choice(tuple(_TRIPLES))
        (p1, s1), (p2, s2) = rng.sample(_TRIPLES[a], 2)
        m, w = rng.randint(1, 8), rng.randint(2, 5)
        k = m * w * w
        if (m * w * w * a * a) % 2:
            continue
        x1, v1, x2 = Fraction(p1, 10), Fraction(w * s1, 10), Fraction(p2, 10)
        energy = Fraction(1, 2) * m * v1**2 + Fraction(1, 2) * k * x1**2
        amp, vmax, v2 = Fraction(a, 10), Fraction(w * a, 10), Fraction(w * s2, 10)
        stem = rng.choice(_STEMS).format(m=m, k=k, x1=_g(p1), v1=_g(w * s1))
        pd = f"Find the speed of the mass when it is {_g(p2)} m from equilibrium, in m/s, correct to 1 decimal place."
        if clashes((f"{float(energy):.2f}", _g(a), _g(w * a), _g(w * s2)), stem, pd):
            continue
        break

    check(energy == Fraction(1, 2) * k * amp**2 == Fraction(1, 2) * m * vmax**2, "energy is not conserved")
    check(Fraction(1, 2) * m * v2**2 + Fraction(1, 2) * k * x2**2 == energy, "energy at the second position differs")

    def f2(v):
        return Fraction(v)

    parts = [
        number_part(
            "Find the total mechanical energy of the system, in J, correct to 2 decimal places.",
            energy,
            2,
            2,
            [
                "E = ½mv² + ½kx².",
                f"E = ½({m})({_g(w * s1)})² + ½({k})({_g(p1)})² = {float(energy):.2f} J.",
            ],
            [
                (m * v1**2 + k * x1**2, "dropped-factor-of-two"),
                (Fraction(1, 2) * m * v1**2, "energy-one-term-only"),
                (Fraction(1, 2) * k * x1**2, "energy-one-term-only"),
                (Fraction(1, 2) * k * v1**2 + Fraction(1, 2) * m * x1**2, "energy-mass-and-stiffness-swapped"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            "Find the amplitude of the oscillation, in m, correct to 1 decimal place.",
            amp,
            1,
            2,
            ["At the amplitude the speed is zero, so E = ½kA².", f"A = √(2 × {float(energy):.2f} / {k}) = {_g(a)} m."],
            [
                (_sqrt_frac(energy, k), "dropped-factor-of-two"),
                (vmax, "amplitude-and-max-speed-swapped"),
                (2 * energy / k, "forgot-square-root"),
                (x1 + x2, "amplitude-adds-the-displacements"),
            ],
            follow_through=f"sqrt(2*part_a/{k})",
            sensible_range=(0, None),
        ),
        number_part(
            "Find the maximum speed of the mass, in m/s, correct to 1 decimal place.",
            vmax,
            1,
            2,
            [
                "At equilibrium all of the energy is kinetic, so E = ½mv².",
                f"v = √(2 × {float(energy):.2f} / {m}) = {_g(w * a)} m/s.",
            ],
            [
                (_sqrt_frac(energy, m), "dropped-factor-of-two"),
                (amp * Fraction(k, m), "max-speed-frequency-as-k-over-m"),
                (amp, "amplitude-and-max-speed-swapped"),
                (v1, "max-speed-reported-as-given-speed"),
            ],
            follow_through=f"part_b*sqrt({Fraction(k, m)})",
            sensible_range=(0, None),
        ),
        number_part(
            pd,
            v2,
            1,
            2,
            [f"½({m})v² + ½({k})({_g(p2)})² = {float(energy):.2f}.", f"v = {_g(w * s2)} m/s."],
            [
                (vmax, "speed-at-position-reports-max-speed"),
                (w * x2, "speed-uses-displacement-in-place-of-remainder"),
                (_sqrt_frac(energy - Fraction(1, 2) * k * x2**2, m), "dropped-factor-of-two"),
                (v1, "max-speed-reported-as-given-speed"),
            ],
            follow_through=f"sqrt((2*part_a - {k * x2 * x2})/{m})",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)


def _sqrt_frac(numerator: Fraction, denominator: int) -> Fraction:
    """sqrt(numerator / denominator), to 6 decimal places: only ever a distractor value, never a key."""
    value = float(Fraction(numerator) / denominator) ** 0.5
    return Fraction(round(value * 10**6), 10**6)
