"""Exam chain that builds a sinusoidal model of rotating motion from a description, then uses it.

A wheel's highest and lowest points above the ground and the time it takes to complete a stated number of
revolutions are given; a rider starts at the lowest point at t = 0, so h(t) = L + A - A cos(2 pi t / T). Four parts:

(a) the amplitude A = (H - L) / 2;
(b) the period T = (time) / (revolutions), to 2 decimal places;
(c) the height after a stated time, to 1 decimal place;
(d) the first time the height reaches a stated value, to 2 decimal places.

The values of A and T are exact (T has at most two decimal places by construction), so (c) and (d) follow from the
exact model and from the displayed (a) and (b) alike.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase

from .exd_x_r2j_support import clear_of_boundary, number_part

#: (what, ground phrase, lowest range, amplitude range, time unit, time unit singular)
_SCENARIOS = [
    ("a Ferris wheel", "above the ground", (2, 6), (8, 30), "minute"),
    ("a wind turbine blade", "above the ground", (20, 50), (10, 40), "second"),
    ("a water wheel", "above the water", (1, 3), (2, 5), "second"),
]

_STEM = [
    "A point on {what} moves steadily in a circle. Its highest point is {high} m and its lowest point is "
    "{low} m, both measured {ground}. It completes {n} revolutions in {m} {tunit}s. The point is at its lowest at time t = 0, and "
    "its height h metres at time t {tunit}s is modelled by h(t) = D - A cos(2πt/T).",
    "A point on {what} moves in a circle, rising to {high} m and falling to {low} m {ground}. It makes {n} "
    "revolutions in {m} {tunit}s. It is at its lowest when t = 0, and its height h metres at time t {tunit}s is "
    "modelled by h(t) = D - A cos(2πt/T).",
]


def _draw(rng: random.Random):
    what, ground, low_range, amp_range, tunit = rng.choice(_SCENARIOS)
    low = rng.randint(*low_range)
    amp = Fraction(rng.randint(amp_range[0] * 2, amp_range[1] * 2), 2)
    n = rng.choice([2, 4, 5, 10])
    m = rng.randint(3 * n, 12 * n)
    return what, ground, tunit, low, amp, n, m


def _text(x: Fraction) -> str:
    return f"{float(x):g}"


@generator("exd_x_r2j_periodic_motion_chain", level=14, presets={"exam": dict(tier="exam")})
def exd_x_r2j_periodic_motion_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Amplitude and period from a description of rotating motion, then the height at a time and the first time at a height."""
    check(tier == "exam", "exd_x_r2j_periodic_motion_chain: only the exam tier exists")
    for _ in attempts(1000, "an acceptable periodic-motion draw"):
        what, ground, tunit, low, amp, n, m = _draw(rng)
        period = Fraction(m, n)
        t0 = Fraction(rng.randint(2, int(period * 30)), 10)
        high = low + 2 * amp
        h1 = Fraction(round(float(low + amp * rng.uniform(0.15, 1.85)) * 10), 10)
        r = float((low + amp - h1) / amp)
        if abs(r) >= 0.95 or abs(float(h1) - float(low + amp)) < 0.2 * float(amp):
            continue
        height = float(low + amp) - float(amp) * math.cos(2 * math.pi * float(t0 / period))
        t_first = float(period) * math.acos(r) / (2 * math.pi)
        if (
            clear_of_boundary(Fraction(height), 1)
            and clear_of_boundary(Fraction(t_first), 2)
            and abs(height - low) > 0.1 * float(amp)
        ):
            break
    stem = phrase(
        rng,
        _STEM,
        what=what,
        high=_text(high),
        low=low,
        ground=ground,
        n=n,
        m=m,
        tunit=tunit,
    )
    check(
        abs(low + float(amp) - float(amp) * math.cos(2 * math.pi * t_first / float(period)) - float(h1)) < 1e-9,
        "first time does not give the stated height",
    )
    mean = low + amp
    t0_text, h1_text = _text(t0), _text(h1)
    ft_c = f"{low} + part_a - part_a*cos(2*pi*{t0_text}/part_b)"
    ft_d = f"part_b*acos(({low} + part_a - {h1_text})/part_a)/(2*pi)"
    parts = [
        number_part(
            prompt=f"Find the amplitude A, in metres.",
            exact=amp,
            places=1 if amp.denominator != 1 else 0,
            marks=1,
            working=[f"A = ({_text(high)} - {low}) / 2 = {_text(amp)} m."],
            wrong=[
                (high - low, "range-for-amplitude"),
                (mean, "mean-for-amplitude"),
                (high / 2, "half-highest-for-amplitude"),
                (Fraction(high + low, 2) - low / 2, "mean-for-amplitude"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            prompt=f"Find the period T, in {tunit}s, correct to 2 decimal places.",
            exact=period,
            places=2,
            marks=1,
            working=[f"One revolution takes {m} / {n} = {float(period):.2f} {tunit}s, so T = {float(period):.2f}."],
            wrong=[
                (Fraction(n, m), "inverted-ratio"),
                (Fraction(m), "total-time-for-period"),
                (Fraction(m * n), "multiplies-instead-of-dividing"),
                (Fraction(m, n + 1), "off-by-one-count"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            prompt=f"Using your values of A and T, find the height of the point after {t0_text} {tunit}{'' if t0_text == '1' else 's'}, in metres, "
            "correct to 1 decimal place.",
            exact=Fraction(height),
            places=1,
            marks=2,
            working=[
                f"D = {low} + {_text(amp)} = {_text(mean)}, so h(t) = {_text(mean)} - {_text(amp)}cos(2πt/{float(period):g}).",
                f"h({t0_text}) = {height:.1f} m.",
            ],
            wrong=[
                (
                    Fraction(float(mean) + float(amp) * math.cos(2 * math.pi * float(t0 / period))),
                    "reflected-about-mean",
                ),
                (
                    Fraction(float(amp) - float(amp) * math.cos(2 * math.pi * float(t0 / period))),
                    "dropped-constant-term",
                ),
                (Fraction(float(mean) - float(amp) * math.cos(float(t0))), "forgot-period-scaling"),
                (
                    Fraction(float(mean) - float(amp) * math.sin(2 * math.pi * float(t0 / period))),
                    "sine-cosine-swapped",
                ),
            ],
            follow_through=ft_c,
            sensible_range=(0, None),
        ),
        number_part(
            prompt=f"Find the first time the height of the point reaches {h1_text} m, in {tunit}s, correct to 2 "
            "decimal places.",
            exact=Fraction(t_first),
            places=2,
            marks=3,
            working=[
                f"{_text(mean)} - {_text(amp)}cos(2πt/{float(period):g}) = {h1_text}, so cos(2πt/{float(period):g}) = {r:.4f}.",
                f"t = {float(period):g} × arccos({r:.4f}) / (2π) = {t_first:.2f} {tunit}s.",
            ],
            wrong=[
                (Fraction(float(period) * math.acos(r) / math.pi), "period-factor-slip"),
                (Fraction(float(period)) - Fraction(t_first), "later-root-instead-of-first"),
                (Fraction(float(period) * math.acos(-r) / (2 * math.pi)), "arccos-sign-slip"),
                (Fraction(float(period) / 2), "half-period-guess"),
            ],
            follow_through=ft_d,
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
