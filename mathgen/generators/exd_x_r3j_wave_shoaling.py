"""Generates `exd_x_r3j_wave_shoaling` questions: ocean waves slow down as they enter shallower water.

Skill: `ucv1-wave-parameter-relation` (exam tier). The practice tiers apply v = f lambda once, or once with a
change of medium; an exam chains the relation through a change of medium:

(a) the frequency f from the number of cycles counted in a time                  number, 1 mark, 2 d.p.
(b) the speed v1 in deep water from f and the crest spacing L                    number, 1 mark, 2 d.p., f L
(c) the wavelength in shallow water, the frequency unchanged, speed v2          number, 2 marks, 2 d.p., L v2 / v1
(d) how many complete wavelengths fit along a shallow section of length D       number, 1 mark,  floor(D / part_c)

## Construction

f is drawn from 0.1, 0.2 and 0.25 Hz (swell periods of 10, 5 and 4 s, so 1 / f has at most one decimal place) with a
crest spacing L near the deep-water value 1.56 T^2, the number of cycles N and the time T are chosen to give f, and v2 is
a one-decimal speed below v1. Then v1 = f L has
at most two decimal places and lambda2 = v2 / f too, so every canonical answer is exact at its stated 2 decimal
places and the follow-through route (rounded earlier answers) agrees with the exact route. D is a whole number of
metres chosen so that D / lambda2 is not close to a whole number, which keeps part (d) from depending on rounding.
"""

from __future__ import annotations

from fractions import Fraction
from math import floor

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3j_kit import number_part

# (frequency in Hz, plausible range of crest spacing in metres for ocean swell of that period: about 1.56 T^2)
_SWELL = [
    (Fraction(1, 10), 140, 170),
    (Fraction(1, 5), 35, 43),
    (Fraction(1, 4), 22, 27),
]
_TIMES = [40, 60, 80, 90, 100, 120, 150, 180]
_SETTINGS = [
    "Ocean swell approaches a jetty at Manly.",
    "Waves roll in towards the beach at Cottesloe.",
    "Swell approaches the headland at Byron Bay.",
    "Waves approach a reef at Port Douglas.",
]


def _plain(value: Fraction) -> str:
    """`value` with no trailing zeros, as a stem would quote it."""
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text


@generator("exd_x_r3j_wave_shoaling", level=17, presets={"exam": {}})
@distinct_part_answers
def exd_x_r3j_wave_shoaling(rng):
    for _ in attempts(2000, "a frequency, spacing, shallow speed and section length that avoid rounding edges"):
        f, low, high = rng.choice(_SWELL)
        t_cycles = rng.choice(_TIMES)
        if (f * t_cycles).denominator != 1:
            continue
        n = int(f * t_cycles)
        spacing = rng.randint(low, high)
        v1 = f * spacing
        v2 = Fraction(rng.randint(int(v1 * 10 * Fraction(5, 10)), int(v1 * 10 * Fraction(9, 10))), 10)
        if v2 <= 0 or v2 >= v1:
            continue
        lam2 = v2 / f
        section = rng.randint(int(3.2 * lam2) + 1, int(9 * lam2))
        ratio = section / lam2
        if abs(ratio - round(ratio)) < Fraction(1, 20) or floor(ratio) < 3:
            continue
        break
    check(lam2 == spacing * v2 / v1, "the two routes to the shallow wavelength disagree")
    count = floor(ratio)

    stem = (
        f"{rng.choice(_SETTINGS)} A buoy in deep water rises and falls {n} times in {t_cycles} seconds, "
        f"and successive crests are {spacing} m apart. The waves then enter shallower water, where their "
        f"frequency is unchanged but their speed falls to {_plain(v2)} m/s. "
        f"The shallow section is {section} m long."
    )
    places = "correct to 2 decimal places"
    parts = [
        number_part(
            f"Find the frequency of the waves, in hertz, {places}.",
            f,
            2,
            1,
            [f"f = {n} ÷ {t_cycles} = {float(f):.2f} Hz."],
            [
                (Fraction(t_cycles, n), "inverted-frequency-and-period"),
                (Fraction(n * t_cycles), "multiplied-instead-of-divided"),
                (Fraction(n, t_cycles * 2), "halved-the-frequency"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the speed of the waves in deep water, in m/s, {places}.",
            v1,
            2,
            1,
            [f"v = fλ = {float(f):.2f} × {spacing} = {float(v1):.2f} m/s."],
            [
                (Fraction(spacing) / f, "divided-instead-of-multiplied"),
                (Fraction(spacing * t_cycles), "multiplied-instead-of-divided"),
                (f + spacing, "added-instead-of-multiplied"),
            ],
            follow_through=f"part_a*{spacing}",
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the wavelength of the waves in the shallower water, in metres, {places}.",
            lam2,
            2,
            2,
            [f"The frequency is unchanged, so λ = v/f = {_plain(v2)} ÷ {float(f):.2f} = {float(lam2):.2f} m."],
            [
                (Fraction(spacing), "wavelength-unchanged-in-new-medium"),
                (spacing * v1 / v2, "speed-ratio-inverted"),
                (v2 * f, "multiplied-instead-of-divided"),
                (v2, "speed-given-as-wavelength"),
            ],
            follow_through=f"{spacing}*{v2.numerator}/{v2.denominator}/part_b",
            sensible_range=(0, None),
        ),
        number_part(
            "How many complete wavelengths fit along the shallow section?",
            count,
            0,
            1,
            [f"{section} ÷ {float(lam2):.2f} = {float(ratio):.2f}, so {count} complete wavelengths."],
            [
                (count + 1, "rounded-up-complete-wavelengths"),
                (floor(Fraction(section, spacing)), "used-deep-water-wavelength"),
                (count - 1, "off-by-one-count"),
            ],
            follow_through=f"floor({section}/part_c)",
        ),
    ]
    return dict(text=stem, parts=parts)
