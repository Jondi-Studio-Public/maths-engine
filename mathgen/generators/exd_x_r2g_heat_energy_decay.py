"""Exam question: decay of one heat-equation mode and the time for its energy to fall (`ucv1-energy-estimate`).

A rod on [0, L] obeys u_t = k u_xx. One end is held at 0; the other is either held at 0 or insulated. The initial
profile is a single eigenfunction, u(x, 0) = A sin(kappa x), with kappa = m pi / L (both ends held at 0) or
kappa = (2j - 1) pi / (2 L) (insulated far end). Separation of variables gives u = A sin(kappa x) exp(-lambda t)
with lambda = k kappa^2. The parts chain:

* (a) lambda written as c pi^2, so c = k (kappa/pi)^2 (fraction);
* (b) the initial energy E(0), the integral of u(x, 0)^2 over [0, L], which is A^2 L / 2 (fraction);
* (c) the rate dE/dt at t = 0, which is -2 lambda E(0) = -2 c E(0) pi^2, written K pi^2 (follow-through from (a) and (b));
* (d) the time at which E falls to E(0)/r, which is ln(r) / (2 lambda) (follow-through from (a); 3 decimal places).

The decimal answer of (d) is kept at least 0.01 of a thousandth from a rounding boundary, so the follow-through route
and the exact route always agree.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import check, generator, round_half_up
from .exd_x_r2g_support import build_part, redraws

_LENGTHS = [4, 5, 6, 8, 10, 12]
_RATIOS = [2, 3, 5, 10]
_ROD = [
    "A thin metal rod lies along the interval [0, {L}] (lengths in cm). Its temperature u(x, t) in degrees Celsius, "
    "with t in minutes, satisfies u_t = {k} u_xx. {ends}",
    "The temperature u(x, t) of a metal bar of length {L} cm obeys the heat equation u_t = {k} u_xx for 0 < x < {L}, "
    "with t in minutes. {ends}",
]
_ENDS = {
    "dd": "Both ends are held at 0 degrees.",
    "dn": "The end x = 0 is held at 0 degrees and the end x = {L} is insulated, so u_x({L}, t) = 0.",
}


def _kappa_over_pi(kind: str, length: int, mode: int) -> Fraction:
    return Fraction(mode, length) if kind == "dd" else Fraction(2 * mode - 1, 2 * length)


def _draw(rng: random.Random) -> dict:
    for _ in range(1000):
        kind = rng.choice(["dd", "dn"])
        length = rng.choice(_LENGTHS)
        mode = rng.randint(1, 3)
        k = rng.randint(2, 9)
        amp = rng.randint(2, 9)
        ratio = rng.choice(_RATIOS)
        kp = _kappa_over_pi(kind, length, mode)
        c = k * kp * kp
        t = math.log(ratio) / (2 * float(c) * math.pi**2)
        if not 0.1 <= t <= 25:
            continue
        scaled = t * 1000
        if abs(scaled - math.floor(scaled) - 0.5) < 0.01:
            continue
        return dict(kind=kind, length=length, mode=mode, k=k, amp=amp, ratio=ratio, kp=kp, c=c, t=t)
    check(False, "no acceptable heat-mode draw")


def _sin_text(kp: Fraction) -> str:
    num = "" if kp.numerator == 1 else str(kp.numerator)
    return f"sin({num}pi x/{kp.denominator})" if kp.denominator != 1 else f"sin({num}pi x)"


@generator("exd_x_r2g_heat_energy_decay", level=18, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_r2g_heat_energy_decay(rng: random.Random, tier: str = "exam") -> dict:
    """Decay constant, initial energy, initial energy-decay rate and the time for the energy to fall by a factor."""
    check(tier == "exam", "exd_x_r2g_heat_energy_decay: only the exam tier exists")
    d = _draw(rng)
    kind, length, k, amp, ratio, kp, c = (d[x] for x in ("kind", "length", "k", "amp", "ratio", "kp", "c"))
    energy0 = Fraction(amp * amp * length, 2)
    rate = -2 * c * energy0
    t_exact = d["t"]
    t_value = round_half_up(Fraction(t_exact), 3)
    ends = _ENDS[kind].format(L=length)
    stem = (
        rng.choice(_ROD).format(L=length, k=k, ends=ends)
        + f" At t = 0 the temperature is u(x, 0) = {amp} {_sin_text(kp)}."
    )
    ln_r = f"ln {ratio}"
    parts = [
        build_part(
            "For t > 0 the solution is u(x, t) = A sin(kappa x) exp(-lambda t). Write lambda = c pi^2 and find c as a "
            "fraction in lowest terms if it is not a whole number.",
            c,
            kind="fraction",
            marks=2,
            sensible_range=(0, None),
            working=[
                f"kappa = {kp} pi, and sin(kappa x) must satisfy the end conditions.",
                f"lambda = k kappa^2 = {k} x ({kp})^2 pi^2, so c = {c}.",
            ],
            wrong=[
                (k * kp, "forgot-to-square-wavenumber"),
                (kp * kp, "forgot-diffusivity"),
                (2 * c, "energy-rate-for-decay-rate"),
                (c / k, "forgot-diffusivity"),
                (Fraction(k * kp * kp, 2), "halves-the-decay-rate"),
            ],
        ),
        build_part(
            "Find the initial energy E(0), the integral of u(x, 0)^2 over [0, "
            f"{length}], as a fraction in lowest terms if it is not a whole number.",
            energy0,
            kind="fraction",
            marks=1,
            sensible_range=(0, None),
            working=[
                f"The integral of sin^2 over [0, {length}] is {length}/2 for this mode.",
                f"E(0) = {amp}^2 x {length}/2 = {energy0}.",
            ],
            wrong=[
                (Fraction(amp * length, 2), "amplitude-not-squared"),
                (Fraction(amp * amp * length), "forgot-half-from-integral-of-sin-squared"),
                (Fraction(amp * amp, 2), "forgot-interval-length"),
                (Fraction(amp * amp * length, 4), "halves-the-integral-twice"),
            ],
        ),
        build_part(
            "Hence write dE/dt at t = 0 in the form K pi^2 and find K, as a fraction in lowest terms if it is not "
            "a whole number.",
            rate,
            kind="fraction",
            marks=1,
            follow_through="-2*part_a*part_b",
            working=[
                f"E(t) = E(0) exp(-2 lambda t), so dE/dt at 0 is -2 lambda E(0) = -2 x {c} x {energy0} pi^2.",
                f"K = {rate}.",
            ],
            wrong=[
                (-c * energy0, "forgot-factor-two-in-energy-rate"),
                (2 * c * energy0, "answer-sign-reversed"),
                (-2 * c, "omits-initial-energy"),
                (-2 * energy0, "omits-decay-constant"),
            ],
        ),
        build_part(
            f"Hence find the time t, in minutes, at which the energy has fallen to E(0)/{ratio}. Give your answer "
            "correct to 3 decimal places.",
            t_value,
            kind="number",
            dp=3,
            marks=2,
            sensible_range=(0, None),
            follow_through=f"log({ratio})/(2*pi**2*part_a)",
            working=[
                f"E(t) = E(0) exp(-2 lambda t) = E(0)/{ratio} when 2 lambda t = {ln_r}.",
                f"t = {ln_r} / (2 x {c} pi^2) = {float(t_value):.3f} minutes.",
            ],
            wrong=[
                (Fraction(math.log(ratio) / (float(c) * math.pi**2)), "energy-decays-like-u-not-u-squared"),
                (Fraction(math.log(ratio) / (4 * float(c) * math.pi**2)), "doubles-the-energy-decay-rate"),
                (Fraction(math.log10(ratio) / (2 * float(c) * math.pi**2)), "log-base-confusion"),
                (Fraction(ratio / (2 * float(c) * math.pi**2)), "solves-linear-not-exponential"),
            ],
        ),
    ]
    return dict(text=stem, parts=parts)
