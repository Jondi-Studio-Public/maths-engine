"""Exam tier for converting a sum of sinusoids to a single ``R cos(x - alpha)``.

The practice tiers hand over ``a cos(x) + b sin(x)`` with integer ``a`` and ``b``. Here the function is a sum of
two phase-shifted waves, ``A cos(x -/+ p) + B sin(x -/+ q)``, so the learner must first expand each term with
the compound-angle identities to read off ``a`` and ``b``, and only then apply the auxiliary-angle conversion.

    A cos(x + u) = (A cos u) cos x - (A sin u) sin x
    B sin(x + u) = (B sin u) cos x + (B cos u) sin x

``R`` and ``alpha`` are irrational in general, so both are asked to 1 decimal place, and a draw is rejected
unless both values sit clear of a rounding boundary (so float noise cannot change the printed digits).
"""

from __future__ import annotations

import math
import random

from ..core import GRADER_TOLERANCE, check, generator, join_terms, pick_distractors

_ANGLES = (15, 20, 25, 30, 35, 40, 50, 55, 60, 65, 70)


def _coefficients(a1: int, fn1: str, s1: int, p: int, a2: int, fn2: str, s2: int, q: int) -> tuple[float, float]:
    """(a, b) with  a1*fn1(x + s1*p) + a2*fn2(x + s2*q)  ==  a cos x + b sin x."""
    a = b = 0.0
    for amp, fn, shift in ((a1, fn1, s1 * p), (a2, fn2, s2 * q)):
        r = math.radians(shift)
        if fn == "cos":  # cos(x + u) = cos x cos u - sin x sin u
            a += amp * math.cos(r)
            b -= amp * math.sin(r)
        else:  # sin(x + u) = sin x cos u + cos x sin u
            a += amp * math.sin(r)
            b += amp * math.cos(r)
    return a, b


def _clear_of_boundary(value: float, eps: float = 0.02) -> bool:
    return abs((value * 10) % 1 - 0.5) > eps


def _term(amp: int, fn: str, shift_sign: int, mag: int, first: bool) -> str:
    inner = f"{fn}(x {'+' if shift_sign > 0 else '-'} {mag}°)"
    if first:
        return f"{amp} {inner}"
    return f"{'+' if amp > 0 else '-'} {abs(amp)} {inner}"


def _build(rng: random.Random) -> dict:
    while True:
        fns = rng.choice([("cos", "sin"), ("sin", "cos"), ("cos", "cos"), ("sin", "sin")])
        amps = [rng.randint(2, 9), rng.choice([-1, 1]) * rng.randint(2, 9)]
        signs = [rng.choice([-1, 1]), rng.choice([-1, 1])]
        mags = [rng.choice(_ANGLES), rng.choice(_ANGLES)]
        a, b = _coefficients(amps[0], fns[0], signs[0], mags[0], amps[1], fns[1], signs[1], mags[1])
        r_exact = math.hypot(a, b)
        alpha_exact = math.degrees(math.atan2(b, a))
        if r_exact < 1.5 or not (_clear_of_boundary(r_exact) and _clear_of_boundary(alpha_exact)):
            continue
        alpha_exact = alpha_exact if alpha_exact > -180 else 180.0
        r_val = round(r_exact, 1)
        alpha_val = round(alpha_exact, 1) + 0.0
        if abs(alpha_val) in (0.0, 180.0):
            continue
        # Principled wrong answers, each a named slip.
        na, nb = _coefficients(amps[0], fns[0], signs[0], mags[0], amps[1], "cos", signs[1], mags[1])
        wrong_fn = (math.hypot(na, nb), math.degrees(math.atan2(nb, na)))  # reads sin(x+u) as cos(x+u)
        naive = math.degrees(math.atan(b / a) if a else math.copysign(math.pi / 2, b))  # no quadrant correction
        cands = [
            (abs(amps[0]) + abs(amps[1]), alpha_val),  # adds the amplitudes
            (r_val, -alpha_val),  # sign of alpha reversed
            (r_val, naive),
            wrong_fn,
            (r_val, math.degrees(math.atan2(a, b))),  # swaps a and b
        ]
        strings = [f"R = {r:.1f}, alpha = {round(al, 1) + 0.0:.1f}" for r, al in cands]
        answer_text = f"R = {r_val:.1f}, alpha = {alpha_val:.1f}"
        distractors = pick_distractors(answer_text, strings, n=3, tolerance=GRADER_TOLERANCE)
        if len(distractors) < 3:
            continue
        break

    expr = " ".join(_term(amps[i], fns[i], signs[i], mags[i], first=(i == 0)) for i in range(2))
    text = (
        f"Write f(x) = {expr} in the form R cos(x - alpha). "
        f"Give R to 1 d.p. and alpha in degrees to 1 d.p., using -180 < alpha <= 180."
    )
    # Independent re-derivation by numeric evaluation: f(x) must equal R cos(x - alpha) at spread-out points.
    for x_deg in (0, 37, 111, 250):
        x = math.radians(x_deg)
        lhs = sum(
            amp * (math.cos if fn == "cos" else math.sin)(x + math.radians(sg * mg))
            for amp, fn, sg, mg in zip(amps, fns, signs, mags)
        )
        rhs = r_exact * math.cos(x - math.radians(alpha_exact))
        check(abs(lhs - rhs) < 1e-9, "R cos(x - alpha) does not reproduce f(x)")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"R": r_val, "alpha": alpha_val},
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        working=[
            "Expand each term with the compound-angle identities and collect: f(x) = a cos x + b sin x.",
            f"a = {a:.3f}, b = {b:.3f}, so R = sqrt(a^2 + b^2) = {r_val:.1f} and alpha = atan2(b, a) = {alpha_val:.1f}.",
        ],
        distractors=distractors,
    )


@generator(
    "exd_x_f5_auxsum",
    level=14,
    presets={"exam": dict(tier="exam")},
)
def exd_x_f5_auxsum(rng: random.Random, tier: str = "exam") -> dict:
    """Sum of two phase-shifted sinusoids written as a single R cos(x - alpha)."""
    if tier != "exam":
        raise ValueError(f"unknown tier: {tier}")
    return _build(rng)
