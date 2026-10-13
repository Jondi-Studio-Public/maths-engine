"""Exam tier for the moment of a force: a beam on a pivot, then the balancing force.

The practice tiers multiply one force by one distance. Here two forces act on opposite sides of a pivot, one
of them sometimes along a line at 30 degrees to the beam, so the learner must choose which part of the force
turns the beam and which way each moment turns, report a signed net moment, and then use it to place a third
force that balances the beam.

## Construction

Forces are multiples of 5 N and distances multiples of 0.5 m. The 30 degree case uses sin 30 = 1/2, so the net
moment is a multiple of 1.25 N m and shows exactly at 2 decimal places (stated to 1 here, with every drawn
value landing on one decimal). The balancing force is drawn so that the distance net / F3 terminates within 2
decimal places; the key for each part is exact arithmetic on what the stem prints.
"""

from __future__ import annotations

import random
import re
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_FORCES = list(range(10, 81, 5))
_HALVES = [Fraction(n, 2) for n in range(1, 9)]  # 0.5 .. 4.0 m
_BALANCERS = (10, 20, 25, 40, 50, 80, 100)

_SETTINGS = (
    "A light rigid plank rests on a pivot P at a playground.",
    "A light rigid beam rests on a pivot P in a workshop.",
    "A light rigid walkway plank rests on a trestle pivot P on a building site.",
    "A light rigid beam is supported on a pivot P in a science laboratory.",
)
_ANGLE_CLAUSES = {
    90: "at right angles to the beam",
    30: "along a line at 30° to the beam",
}


def _dp(value: Fraction, places: int) -> str:
    return stated_precision_string(round_half_up(value, places), places)


def _tag(dp: int, key_text: str, candidates: list[tuple[str, Fraction | None]]) -> list:
    out = []
    for ident, value in candidates:
        if value is None:
            continue
        shown = round_half_up(value, dp)
        if shown == 0:
            continue
        out.append(Distractor(stated_precision_string(shown, dp), ident, shown))
    return pick_distractors(key_text, out, 3, tolerance=stated_precision_tolerance(dp))


def _clashes(prompt: str, keys: list[Fraction]) -> bool:
    printed = {Fraction(m) for m in re.findall(r"\d+(?:\.\d+)?", prompt)}
    return any(abs(key) in printed for key in keys)


def _build(rng: random.Random) -> dict | None:
    f1, f2 = rng.choice(_FORCES), rng.choice(_FORCES)
    d1, d2 = rng.choice(_HALVES), rng.choice(_HALVES)
    angle = rng.choice([30, 90])
    sin = Fraction(1, 2) if angle == 30 else Fraction(1)
    left = f1 * d1
    right = f2 * d2 * sin
    net = right - left  # clockwise positive
    if net == 0:
        return None
    f3 = rng.choice(_BALANCERS)
    x = abs(net) / f3
    if (x * 100).denominator != 1 or not Fraction(1, 5) <= x <= 4:
        return None

    stem = (
        f"{rng.choice(_SETTINGS)} A downward force of {f1} N acts at {exact_decimal_string(d1, 1)} m to the left of "
        f"P. A downward force of {f2} N acts at {exact_decimal_string(d2, 1)} m to the right of P, {_ANGLE_CLAUSES[angle]}."
    )
    a_prompt = "Taking clockwise as positive, find the net moment about P, in N·m, correct to 1 decimal place."
    b_prompt = (
        f"A third downward force of {f3} N, at right angles to the beam, is placed so that the beam balances. "
        "Using your answer to (a), find its distance from P, in m, correct to 2 decimal places."
    )
    net_text, x_text = _dp(net, 1), _dp(x, 2)
    if _clashes(f"{stem} {a_prompt} {b_prompt}", [net, x]):
        return None

    a_distractors = _tag(
        1,
        net_text,
        [
            ("ignored-force-angle", f2 * d2 - left if angle == 30 else None),
            ("added-opposing-moments", right + left),
            ("sign-convention-reversed", -net),
        ],
    )
    b_distractors = _tag(
        2,
        x_text,
        [
            ("balanced-one-force-only", Fraction(left, 1) / f3),
            ("ignored-force-angle", abs(f2 * d2 - left) / f3 if angle == 30 else None),
            ("added-opposing-moments", (right + left) / f3),
            ("balanced-one-force-only", right / f3),
        ],
    )
    if len(a_distractors) < 2 or len(b_distractors) < 2:
        return None
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=a_prompt,
                answer_kind="number",
                answer=numeric_answer(round_half_up(net, 1)),
                answer_text=net_text,
                answer_latex=latex_answer(net_text),
                marks=2,
                working=[
                    f"Moment of the {f1} N force: {f1} × {exact_decimal_string(d1, 1)} = {exact_decimal_string(left, 2)} N·m anticlockwise.",
                    f"Moment of the {f2} N force: {exact_decimal_string(right, 3)} N·m clockwise"
                    + (
                        " (only the part of the force at right angles to the beam turns it: F sin 30°)."
                        if angle == 30
                        else "."
                    ),
                    f"Net moment = {exact_decimal_string(right, 3)} − {exact_decimal_string(left, 2)} = {net_text} N·m.",
                ],
                distractors=a_distractors,
            ),
            dict(
                prompt=b_prompt,
                answer_kind="number",
                answer=numeric_answer(round_half_up(x, 2)),
                answer_text=x_text,
                answer_latex=latex_answer(x_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"Abs(part_a)/{f3}",
                working=[
                    "The third force must supply a moment equal in size and opposite in direction to the net moment.",
                    f"{f3} × x = {exact_decimal_string(abs(net), 3)}, so x = {x_text} m.",
                ],
                distractors=b_distractors,
            ),
        ],
    )


@generator("exd_x_u2_lever_balance", level=19, presets={"exam": dict(tier="exam")})
def exd_x_u2_lever_balance(rng: random.Random, tier: str = "exam") -> dict:
    """Net moment of two forces on a pivoted beam, then the distance of a balancing force. See the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a beam with a clean net moment and balancing distance"):
        built = _build(rng)
        if built is not None:
            return built
