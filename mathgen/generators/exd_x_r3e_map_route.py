"""Exam-tier source for `senior-use-scales-to-find-distances-such-as-on`: a route measured on a map.

A track is drawn as two straight sections of d1 and d2 centimetres on a map of scale 1 : S. The chain:

(a) the real length of the first section, in metres          number, 1 mark   whole number of metres
(b) the real length of the whole track, in km, 2 d.p.        number, 2 marks   (part_a + d2 S / 100) / 1000
(c) the time at a steady speed, to the nearest minute        number, 2 marks   part_b / v * 60
(d) the track's length on a map of scale 1 : S2, in cm       number, 2 marks   part_b * 100000 / S2   (1 d.p.)

Scales are multiples of 10 000 and map lengths have one decimal place, so (a) is a whole number of metres and (b)
is exact at 2 decimal places: the learner's rounded (b) is the true (b), and every follow-through is exact. The
draw keeps (c) and (d) clear of a rounding boundary and (d) off an exact tie.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor
from ..core import (
    check,
    generator,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)
from ._exd_r3_distinct import distinct_part_answers

_WALK = dict(scales=[10000, 20000, 50000], speeds=[3, 4, 5], max_km=18)
_RIDE = dict(scales=[50000, 100000], speeds=[12, 15, 18, 20], max_km=60)
# Not 1 : 100000: on that map the length in cm equals the length in km, so (d) would repeat (b).
_SECOND_SCALES = [10000, 20000, 25000, 50000, 200000]


def _num_options(key_text: str, key: Fraction, candidates, places: int, n: int = 3):
    """Distractors at `places` decimal places, at least half a unit of the last place from the key and each other."""
    out, seen = [], [key]
    step = Fraction(1, 10**places) if places else Fraction(1)
    for tag, value in candidates:
        value = round_half_up(Fraction(value), places)
        if value <= 0 or value > 10**9 or any(abs(value - s) < step for s in seen):
            continue
        seen.append(value)
        text = stated_precision_string(value, places) if places else str(int(value))
        out.append(Distractor(text, tag, value=int(value) if value.denominator == 1 else _frac(value)))
    return pick_distractors(key_text, out, n, tolerance=step / 2)


def _frac(value: Fraction):
    import sympy

    return sympy.Rational(value.numerator, value.denominator)


@generator("exd_x_r3e_map_route", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_map_route(rng, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    ride = rng.random() < 0.35
    cfg = _RIDE if ride else _WALK
    for _ in range(3000):
        S = rng.choice(cfg["scales"])
        d1, d2 = Fraction(rng.randint(25, 140), 10), Fraction(rng.randint(25, 140), 10)
        v = rng.choice(cfg["speeds"])
        S2 = rng.choice([s for s in _SECOND_SCALES if s != S])
        leg1_m = d1 * S / 100
        total_km = (d1 + d2) * S / 100000
        if leg1_m.denominator != 1 or total_km * 100 != int(total_km * 100) or not 1 <= total_km <= cfg["max_km"]:
            continue
        minutes = total_km / v * 60
        second = total_km * 100000 / S2
        frac_c = (minutes - int(minutes)) - Fraction(1, 2)
        frac_d = (second * 10 - int(second * 10)) - Fraction(1, 2)
        if abs(frac_c) < Fraction(1, 20) or frac_d == 0 or not 2 <= second <= 60:
            continue
        key_a, key_b = int(leg1_m), total_km
        key_c, key_d = round_half_up(minutes, 0), round_half_up(second, 1)
        a_opts = _num_options(
            str(key_a),
            Fraction(key_a),
            [
                ("scale-power-of-ten-slip", key_a * 100),
                ("scale-power-of-ten-slip", key_a * 10),
                ("scale-power-of-ten-slip", Fraction(key_a, 10) if key_a % 10 == 0 else 0),
                ("scale-power-of-ten-slip", Fraction(key_a, 1000) if key_a % 1000 == 0 else 0),
            ],
            0,
        )
        b_text = stated_precision_string(key_b, 2)
        b_opts = _num_options(
            b_text,
            key_b,
            [
                ("used-first-leg-only", Fraction(key_a, 1000)),
                ("scale-power-of-ten-slip", key_b * 10),
                ("scale-power-of-ten-slip", key_b / 10),
                ("scale-power-of-ten-slip", key_b * 1000),
            ],
            2,
        )
        c_opts = _num_options(
            str(int(key_c)),
            key_c,
            [
                ("hours-not-converted-to-minutes", total_km / v),
                ("decimal-hours-read-as-minutes", total_km / v * 100),
                ("speed-time-inverted", total_km * v * 60),
                ("used-first-leg-only", Fraction(key_a, 1000) / v * 60),
            ],
            0,
        )
        d_text = stated_precision_string(key_d, 1)
        d_opts = _num_options(
            d_text,
            key_d,
            [
                ("kept-the-original-scale", total_km * 100000 / S),
                ("scale-power-of-ten-slip", second * 10),
                ("scale-power-of-ten-slip", second / 10),
                ("used-first-leg-only", Fraction(key_a) * 100 / S2),
            ],
            1,
        )
        if min(len(a_opts), len(b_opts), len(c_opts), len(d_opts)) >= 3:
            break
    else:
        check(False, "no route with enough distractors")

    # Independent route: the whole track in centimetres on the ground, then converted.
    ground_cm = (d1 + d2) * S
    check(Fraction(ground_cm, 100000) == total_km and Fraction(ground_cm, S2) == second, "scale conversion disagrees")
    walker = "walker" if not ride else "cyclist"
    if ride:
        stem = phrase(
            rng,
            (
                "A cycling map has a scale of 1 : {S}. A rail trail between two towns is drawn as two straight "
                "sections measuring {d1} cm and {d2} cm on the map.",
                "On a cycling map with a scale of 1 : {S}, a trail is drawn as two straight sections, one "
                "{d1} cm long and the other {d2} cm long.",
            ),
            S=S,
            d1=float(d1),
            d2=float(d2),
        )
        speed = (
            f"A cyclist averages {v} km/h along the trail. How long does the whole trail take, to the nearest minute?"
        )
    else:
        stem = phrase(
            rng,
            (
                "A bushwalking map has a scale of 1 : {S}. A track from the car park to a lookout is drawn as two "
                "straight sections measuring {d1} cm and {d2} cm on the map.",
                "On a bushwalking map with a scale of 1 : {S}, a track is drawn as two straight sections, one "
                "{d1} cm long and the other {d2} cm long.",
            ),
            S=S,
            d1=float(d1),
            d2=float(d2),
        )
        speed = (
            f"A {walker} averages {v} km/h along the track. How long does the whole track take, to the nearest minute?"
        )
    what = "trail" if ride else "track"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the real length of the first section, in metres.",
                answer_kind="number",
                answer=key_a,
                answer_text=str(key_a),
                marks=1,
                working=[f"{float(d1)} cm x {S} = {int(d1 * S)} cm = {key_a} m."],
                distractors=a_opts,
            ),
            dict(
                prompt=f"Find the real length of the whole {what}, in kilometres, correct to 2 decimal places.",
                answer_kind="number",
                answer=numeric_answer(key_b),
                answer_text=b_text,
                marks=2,
                working=[f"({float(d1)} + {float(d2)}) cm x {S} = {int(ground_cm)} cm = {b_text} km."],
                distractors=b_opts,
                follow_through=f"(part_a + {int(d2 * S / 100)})/1000",
            ),
            dict(
                prompt=speed,
                answer_kind="number",
                answer=int(key_c),
                answer_text=str(int(key_c)),
                marks=2,
                working=[f"{b_text} / {v} = {float(total_km / v):.4f} hours, which is about {int(key_c)} minutes."],
                distractors=c_opts,
                follow_through=f"part_b/{v}*60",
            ),
            dict(
                prompt=(
                    f"The same {what} is drawn on a map with a scale of 1 : {S2}. Find its length on that map, "
                    "in centimetres, correct to 1 decimal place."
                ),
                answer_kind="number",
                answer=numeric_answer(key_d),
                answer_text=d_text,
                marks=2,
                working=[
                    f"{b_text} km = {int(total_km * 100000)} cm, and {int(total_km * 100000)} / {S2} = {d_text} cm."
                ],
                distractors=d_opts,
                follow_through=f"part_b*100000/{S2}",
            ),
        ],
    )
