"""Exam tier for classifying a linear oscillator: a three-part chain that needs the regime, never names it.

A mass-spring-damper m x'' + b x' + k x = 0 is given with m and b in grams and grams per second:

(a) the critical damping coefficient 2 sqrt(mk) in kg/s (the unit conversion is part of the task);
(b) the damping ratio zeta = b / (2 sqrt(mk)), to 2 decimal places;
(c) the smaller of |Re s| over the roots of m s^2 + b s + k = 0. Which formula applies depends on whether zeta is below,
    at or above 1, so the learner has to classify the regime to answer. With w0 = sqrt(k/m), the answer is
    w0 * zeta for zeta <= 1 and w0 * (zeta - sqrt(zeta^2 - 1)) above it; one expression covers all three,
    w0 * (zeta - sqrt(max(zeta^2 - 1, 0))), and that is the follow-through from the learner's (b).

Built backwards: m = s p^2 kg and k = s q^2 N/m make 2 sqrt(mk) = 2spq and w0 = q/p exact, and zeta is drawn first
(below, at or above 1) with two decimal places so b is a whole number of grams per second.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator, round_half_up
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3h_kit import number_part

_MARGIN = Fraction(1, 50)
_STEMS = (
    "A mass-spring-damper system has mass m = {m} g, damping constant b = {b} g/s and spring stiffness k = {k} N/m. "
    "Its displacement x metres satisfies m x'' + b x' + k x = 0, with m and b in SI units.",
    "A {m} g block hangs from a spring of stiffness {k} N/m and moves through a fluid that gives a damping "
    "constant of {b} g/s. Its displacement x metres from equilibrium obeys m x'' + b x' + k x = 0, with m and b in "
    "SI units.",
    "A car-door closer is modelled as m x'' + b x' + k x = 0 with m = {m} g, b = {b} g/s and k = {k} N/m, where "
    "x metres is the displacement. Work in SI units.",
)


def _clear(value: Fraction, places: int) -> bool:
    return abs((value * 10**places) % 1 - Fraction(1, 2)) > _MARGIN


def _root(value: Fraction) -> Fraction:
    return Fraction(math.sqrt(value)) if value >= 0 else Fraction(0)


@generator("exd_x_r3h_oscillator_damping", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3h_oscillator_damping(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "an oscillator with exact critical damping and a clear decay rate"):
        s, p, q = rng.choice((1, 2, 3)), rng.choice((1, 2, 3)), rng.randint(1, 6)
        regime = rng.choices(("under", "critical", "over"), weights=(4, 2, 4))[0]
        if regime == "under":
            z100 = rng.randrange(20, 100, 5)
        elif regime == "critical":
            z100 = 100
        else:
            z100 = rng.randrange(105, 261, 5)
        mass, stiffness = s * p * p, s * q * q
        if mass > 20:
            continue
        zeta = Fraction(z100, 100)
        crit = Fraction(2 * s * p * q)
        omega = Fraction(q, p)
        decay = omega * (zeta - _root(max(zeta * zeta - 1, Fraction(0))))
        if _clear(decay, 2):
            break
    damping = zeta * crit  # kg/s
    check(damping * 1000 == int(damping * 1000), "damping is a whole number of g/s")
    check(crit * crit == 4 * mass * stiffness, "critical damping from m and k")
    check(abs(float(omega) ** 2 - stiffness / mass) < 1e-12, "natural frequency")
    check(
        abs(float(decay) - min(abs(r.real) for r in _roots(mass, float(damping), stiffness))) < 1e-9,
        "decay rate from the characteristic roots",
    )
    m_g, b_g = mass * 1000, int(damping * 1000)
    # A block hanging on the spring stretches it by m g / k at rest: keep that under about two metres.
    stems = _STEMS if stiffness >= 5 * mass else (_STEMS[0], _STEMS[2])
    stem = rng.choice(stems).format(m=m_g, b=b_g, k=stiffness)
    bk = Fraction(b_g, 1000)
    omega_text = f"({omega.numerator}/{omega.denominator})"
    decay_ft = f"{omega_text}*(part_b - sqrt(Max(part_b**2 - 1, 0)))"
    fast = omega * (zeta + _root(max(zeta * zeta - 1, Fraction(0))))
    wd = omega * _root(max(1 - zeta * zeta, Fraction(0)))
    return dict(
        text=stem,
        parts=[
            dict(
                **number_part(
                    "Find the damping constant, in kg/s, at which this system would be critically damped.",
                    crit,
                    0,
                    2,
                    [f"m = {mass} kg. Critical damping: b = 2 sqrt(mk) = 2 sqrt({mass} x {stiffness}) = {crit} kg/s."],
                    [
                        (2 * _root(Fraction(m_g * stiffness)), "skipped-unit-conversion"),
                        (crit / 2, "dropped-factor-of-two"),
                        (4 * mass * stiffness, "forgot-square-root"),
                        (mass * stiffness, "used-product-mk-for-critical-damping"),
                    ],
                    sensible_range=(0, None),
                )
            ),
            dict(
                **number_part(
                    "Find the damping ratio of the system, the ratio of its damping constant (in kg/s) to the "
                    "value in (a), correct to 2 decimal places.",
                    zeta,
                    2,
                    1,
                    [f"b = {bk} kg/s, so the ratio is {bk} / {crit} = {zeta}."],
                    [
                        (1 / zeta, "inverted-ratio"),
                        (Fraction(b_g) / crit, "skipped-unit-conversion"),
                        (2 * zeta, "dropped-factor-of-two"),
                        (zeta * zeta, "squared-damping-ratio"),
                    ],
                    follow_through=f"({bk.numerator}/{bk.denominator})/part_a",
                    sensible_range=(0, None),
                )
            ),
            dict(
                **number_part(
                    "The characteristic equation m s^2 + b s + k = 0 has two roots, each with a negative real part. "
                    "Find the smaller of the two values of |Re s|, in s^-1, correct to 2 decimal places.",
                    decay,
                    2,
                    4,
                    [
                        f"Natural frequency {omega} s^-1 and damping ratio {zeta}: this system is {regime}damped"
                        if regime != "critical"
                        else f"Natural frequency {omega} s^-1 and damping ratio 1: critically damped",
                        f"The smaller |Re s| is {decay} s^-1.",
                    ],
                    [
                        (Fraction(b_g, 1000 * mass), "dropped-factor-of-two"),
                        (omega, "gave-natural-frequency-for-decay-rate"),
                        (fast, "gave-other-root"),
                        (omega * zeta, "used-underdamped-decay-rate-for-all-regimes"),
                        (wd, "gave-damped-frequency-for-decay-rate"),
                        (-decay, "sign-of-result-reversed"),
                    ],
                    follow_through=decay_ft,
                    sensible_range=(0, None),
                )
            ),
        ],
    )


def _roots(mass: int, damping: float, stiffness: int):
    disc = complex(damping * damping - 4 * mass * stiffness) ** 0.5
    return [(-damping + disc) / (2 * mass), (-damping - disc) / (2 * mass)]
