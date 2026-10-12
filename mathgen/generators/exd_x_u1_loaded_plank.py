"""Exam-tier source for `beam-equilibrium-seesaw`: a uniform plank on an off-centre pivot, with its own weight.

A uniform plank of known weight and length rests on a pivot that is nearer the left end than the middle. A
load hangs at the left end and a downward force at the right end holds the plank horizontal. Unlike the plain
see-saw, the plank's own weight acts at its centre, which is on the far side of the pivot from the load, so it
adds a moment to the right.

* (a) the moment of the plank's weight about the pivot, w (L/2 - p);
* (b) the force at the right end that holds the plank horizontal, from taking moments about the pivot:
  (W p - part_a) / (L - p), followed through from (a);
* (c) the reaction at the pivot, from the vertical balance of forces: W + w + F, followed through from (b).

Built backwards from whole-number weights and a whole-number balancing force, so every key is an exact integer.
The draw keeps the plank's weight moment, the load and the force all different and non-zero, and keeps each wrong
route (arm measured from the end, plank weight ignored or placed at the end, a reaction missing a load) distinct
from its key.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import check, generator, pick_distractors, round_half_up, stated_precision_string

_SETUPS = [
    "A uniform plank of weight {w} N and length {L} m rests horizontally on a pivot {p} m from its left end. "
    "A child of weight {W} N sits on the left end of the plank. A downward force F is applied at the right end "
    "so that the plank stays horizontal and in equilibrium.",
    "A uniform wooden beam, {L} m long and weighing {w} N, is balanced horizontally on a pivot placed {p} m from "
    "its left end. A {W} N weight hangs from the left end. A downward force F at the right end keeps the "
    "beam horizontal.",
    "On a playground, a uniform plank of length {L} m and weight {w} N is supported on a pivot {p} m from its left "
    "end. A {W} N load rests on the left end, and a downward force F at the right end holds the plank in "
    "equilibrium.",
]
_A_PROMPTS = [
    "Find the moment of the weight of the plank about the pivot, in N·m.",
    "Calculate the moment about the pivot of the plank's own weight, in N·m.",
]
_B_PROMPTS = [
    "Hence find F, in N.",
    "Using your answer to (a) and taking moments about the pivot, find the force F, in N.",
]
_C_PROMPTS = [
    "Find the reaction of the pivot on the plank, in N.",
    "Find the upward force the pivot exerts on the plank, in N.",
]


def _rs(value: Fraction) -> str:
    return f"({value.numerator}/{value.denominator})"


def _length(value: Fraction) -> str:
    return f"{float(value):g}"


def _draw(rng: random.Random):
    for _ in range(5000):
        length = Fraction(rng.choice([3, 4, 5, 6, 8]))
        p = Fraction(rng.randint(1, int(length) * 2 - 1), 2)
        if not p < length / 2 - Fraction(1, 2):
            continue
        w = rng.choice(range(40, 301, 10))
        force = rng.choice(range(10, 121, 5))
        load = Fraction(force * (length - p) + w * (length / 2 - p)) / p
        moment = w * (length / 2 - p)
        if load.denominator != 1 or load % 5 or not 50 <= load <= 600 or moment.denominator != 1:
            continue
        if len({moment, load, force, w}) < 4:
            continue
        return length, p, int(w), int(load), force, int(moment)
    raise RuntimeError("no suitable plank in 5000 draws")


@generator("exd_x_u1_loaded_plank", level=19, presets={"exam": dict(tier="exam")})
def exd_x_u1_loaded_plank(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: moment of a plank's weight, the balancing force, and the pivot reaction."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(200):
        question = _build(rng)
        if question is not None:
            return question
    raise RuntimeError("no plank question with enough distractors in 200 builds")


def _build(rng: random.Random) -> dict | None:
    length, p, w, load, force, moment = _draw(rng)
    reaction = load + w + force

    # Independent route: resolve moments about the right end instead of the pivot, and check the force balance.
    check(reaction * (length - p) == load * length + w * length / 2, "moments about the right end fail")
    check(load * p == moment + force * (length - p), "moments about the pivot fail")
    check(reaction - load - w - force == 0, "vertical forces do not balance")

    def shown(value: Fraction, tag):
        rounded = round_half_up(Fraction(value), 2)
        if rounded.denominator == 1:
            text = str(int(rounded))
        else:
            text = stated_precision_string(rounded, 2)
        return Distractor(text, tag, exact_value(rounded))

    arm = length - p
    a_cands = [
        shown(w * length / 2, "moment-arm-from-end"),
        shown(w * p, "moment-arm-pivot-to-end"),
        shown(w * length, "moment-arm-full-length"),
        shown(w * (length / 2 + p), None),
    ]
    b_cands = [
        shown(Fraction(load * p, arm), "ignored-beam-weight"),
        shown((load * p - moment) / (length / 2), "force-arm-taken-as-half-length"),
        shown(Fraction(load * arm, p), "flipped-distance-ratio"),
        shown(Fraction(load), "assumed-equal-force"),
        shown((load * p - w * arm) / arm, "beam-weight-at-end"),
    ]
    c_cands = [
        shown(Fraction(load + w), "reaction-omits-balancing-force"),
        shown(Fraction(load + force), "reaction-omits-beam-weight"),
        shown(Fraction(load + w + force) - load, None),
        shown(Fraction(w + force), "reaction-omits-load"),
    ]
    a_dist = pick_distractors(str(moment), a_cands, 3)
    b_dist = [d for d in pick_distractors(str(force), b_cands, 5) if Fraction(d.text) > 0][:3]
    c_dist = pick_distractors(str(reaction), c_cands, 3)
    if min(len(a_dist), len(b_dist), len(c_dist)) < 2:
        return None

    fmt = dict(w=w, L=_length(length), p=_length(p), W=load)
    return dict(
        text=rng.choice(_SETUPS).format(**fmt),
        parts=[
            dict(
                prompt=rng.choice(_A_PROMPTS),
                answer_kind="number",
                answer=moment,
                answer_text=str(moment),
                marks=1,
                sensible_range=(0, None),
                working=[
                    f"The plank's weight acts at its centre, {_length(length / 2)} m from the left end, which is "
                    f"{_length(length / 2 - p)} m beyond the pivot.",
                    f"Moment = {w} × {_length(length / 2 - p)} = {moment} N·m.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=rng.choice(_B_PROMPTS),
                answer_kind="number",
                answer=force,
                answer_text=str(force),
                marks=3,
                sensible_range=(0, None),
                follow_through=f"({load}*{_rs(p)} - part_a)/{_rs(arm)}",
                working=[
                    f"Anticlockwise moment of the load: {load} × {_length(p)} = {int(load * p)} N·m.",
                    f"Clockwise: {moment} + F × {_length(arm)}.",
                    f"F = ({int(load * p)} - {moment}) / {_length(arm)} = {force} N.",
                ],
                distractors=b_dist,
            ),
            dict(
                prompt=rng.choice(_C_PROMPTS),
                answer_kind="number",
                answer=int(reaction),
                answer_text=str(int(reaction)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{load} + {w} + part_b",
                working=[f"Vertical balance: R = {load} + {w} + {force} = {int(reaction)} N."],
                distractors=c_dist,
            ),
        ],
    )
