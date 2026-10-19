"""Exam-tier source for `f10-logarithmic-scales`: a decibel scale read in both directions.

The practice tiers place a value part of the way between two labelled marks on a logarithmic scale. An exam
question sets the scale in a real context, the decibel scale, where each 10 dB step multiplies the sound
intensity by 10. Part (a) turns a difference in levels into an intensity ratio; part (b) uses that ratio to find
the louder sound's intensity (following through from (a)); part (c) reads the scale the other way, from an
intensity ratio back to a level.

Built backwards from the number of 10 dB steps, so every ratio is an exact power of ten. Intensities are given in
microwatts per square metre so that every answer is a whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, draw_until, exact_decimal_string, generator, latex_answer, numeric_answer, pick_distractors

_QUIET = ["a quiet library", "a bedroom at night", "a still garden", "a country road at dawn"]
_LOUD = ["a vacuum cleaner", "a lawn mower", "a busy cafe", "a food blender", "a motorbike passing"]
_VERY = ["a chainsaw", "a jet taking off", "a stadium crowd", "a pneumatic drill"]
_COEFFS = [Fraction(n, 10) for n in range(11, 100) if n % 10]


@generator("exd_x_n3_decibel_scale", level=10, presets={"exam": dict(tier="exam")})
def exd_x_n3_decibel_scale(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: difference in decibels to intensity ratio, to intensity, and back from a ratio to a level."""
    check(tier == "exam", f"unknown tier {tier!r}")

    def draw(r):
        return r.choice([20, 25, 30, 35, 40, 45, 50]), r.randint(2, 5), r.choice(_COEFFS), r.randint(1, 3)

    def accept(t):
        l1, steps, c, j = t
        shown = {l1, l1 + 10 * steps, 10**j, 10}
        keys = {10**steps, int(c * 10**steps), l1 + 10 * steps + 10 * j}
        return l1 + 10 * steps <= 100 and l1 + 10 * steps + 10 * j <= 130 and not (shown & keys)

    l1, steps, c, j = draw_until(rng, draw, accept, what="levels and a coefficient")
    l2 = l1 + 10 * steps
    ratio = 10**steps
    louder = c * ratio  # microwatts per square metre, a whole number because steps >= 2
    check(louder.denominator == 1, "louder intensity is not a whole number")
    l3 = l2 + 10 * j
    times = 10**j
    quiet_place, loud_thing, very = rng.choice(_QUIET), rng.choice(_LOUD), rng.choice(_VERY)
    c_text = exact_decimal_string(c, 1)

    stem = (
        "Sound levels are measured in decibels (dB). On this scale an increase of 10 dB means the sound "
        f"intensity is multiplied by 10. The sound in {quiet_place} is {l1} dB and has an intensity of "
        f"{c_text} microwatts per square metre. The sound from {loud_thing} is {l2} dB."
    )
    a_text, b_text, c3_text = str(ratio), str(int(louder)), str(l3)
    a_dist = pick_distractors(
        a_text,
        [str(10 * steps), str(steps), str(2**steps), str(10 ** (steps - 1)), str(10 ** (steps + 1))],
        3,
    )
    # The key is a whole number, so a distractor is only worth offering if it is one too: a value that needs a
    # decimal point would give the answer away by its format. Each wrong multiplier is kept when it lands on a whole
    # number of microwatts per square metre.
    b_wrong = [
        c * 10 * steps,  # difference in decibels as the multiplier
        c * steps,  # number of steps as the multiplier
        c * 10 ** (steps - 1),
        Fraction(ratio),  # the ratio itself, forgetting to multiply by the quiet sound's intensity
        c * 2**steps,
        c * 10 ** (steps + 1),
    ]
    b_dist = pick_distractors(b_text, [str(int(v)) for v in b_wrong if v.denominator == 1], 3)
    c_dist = pick_distractors(
        c3_text,
        [
            str(l2 + j),
            str(l2 + times),
            str(l2 * j),
            str(l2 + 10 * (j + 1)),
            str(l2 + 10 * (j - 1)) if j > 1 else str(l2 + 5),
        ],
        3,
    )
    check(min(len(a_dist), len(b_dist), len(c_dist)) >= 2, "too few distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"How many times as intense as the sound in {quiet_place} is the sound from {loud_thing}?",
                answer_kind="number",
                answer=ratio,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"{l2} − {l1} = {10 * steps} dB is {steps} steps of 10 dB, so the intensity is multiplied by 10^{steps} = {ratio}."
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=f"Find the intensity of the sound from {loud_thing}, in microwatts per square metre.",
                answer_kind="number",
                answer=numeric_answer(louder),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*{c_text}",
                working=[f"{c_text} × {ratio} = {b_text} microwatts per square metre."],
                distractors=b_dist,
            ),
            dict(
                prompt=f"The sound from {very} is {times} times as intense as the sound from {loud_thing}. "
                "Find its level in decibels.",
                answer_kind="number",
                answer=l3,
                answer_text=c3_text,
                answer_latex=latex_answer(c3_text),
                marks=2,
                working=[
                    f"{times} = 10^{j}, which is {j} step{'' if j == 1 else 's'} of 10 dB: {l2} + {10 * j} = {l3} dB."
                ],
                distractors=c_dist,
            ),
        ],
    )
