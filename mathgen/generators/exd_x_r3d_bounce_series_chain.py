"""Exam source of `senior-use-geometric-sequences-to-model-and-analyse-numerically`: a bouncing ball, in four linked steps.

The practice tiers give the drop height and rebound fraction and ask for a distance. An exam question gives only the
first two rebound heights, so the model has to be recovered before it is used:

(a) the rebound ratio r, as a decimal to 2 places (2nd rebound over 1st);
(b) the height the ball was dropped from, h1 / r (follows through from (a));
(c) the total vertical distance when the ball hits the ground for the k-th time: the drop, plus the rise and the fall of
    each of the k - 1 rebounds, a finite geometric series (follows through from (a) and (b));
(d) the total distance the ball travels before it comes to rest, the sum to infinity H (1 + r) / (1 - r) (follows
    through from (a) and (b)).

Built backwards from r and H, both exact to 2 decimal places, with the first two rebound heights also exact to 2
decimal places. Every displayed answer is kept clear of a rounding boundary, so the exact route and the route that
uses the rounded values from earlier parts agree. An independent check sums the heights term by term.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    round_half_up,
    stated_precision_string,
)

from .exd_x_r3d_common import tagged
from ._exd_r3_distinct import distinct_part_answers

_RATIOS = tuple(Fraction(v, 100) for v in (50, 60, 70, 75, 80))
_ORDINALS = {4: "4th", 5: "5th", 6: "6th"}
_STEMS = (
    "A ball is dropped and bounces repeatedly, each rebound reaching a fixed fraction of the height of the previous "
    "fall. Its 1st rebound reaches {h1} m and its 2nd rebound reaches {h2} m.",
    "A rubber ball is dropped onto a hard floor. Each time it rebounds to the same fraction of the height it has just "
    "fallen. The 1st rebound reaches {h1} m and the 2nd reaches {h2} m.",
    "A ball is released from rest and bounces on a hard floor. Every rebound reaches a constant fraction of the "
    "previous fall. The first rebound is {h1} m high and the second is {h2} m high.",
)
_PROMPT_A = (
    "Find the ratio of each rebound height to the height of the previous fall, correct to 2 decimal places.",
    "Find the common ratio r of the rebound heights, correct to 2 decimal places.",
)
_PROMPT_B = (
    "Hence find the height from which the ball was dropped, in metres, correct to 2 decimal places.",
    "Using your ratio, find the drop height in metres, correct to 2 decimal places.",
)
_PROMPT_C = (
    "Counting the initial drop once and both the upward and downward travel of every rebound, find the total vertical "
    "distance travelled when the ball hits the ground for the {k} time. Give your answer in metres, correct to 2 "
    "decimal places.",
    "Find the total vertical distance, in metres, travelled by the ball by the time it lands for the {k} time "
    "(the initial drop counts once; each rebound counts up and down). Give your answer correct to 2 decimal places.",
)
_PROMPT_D = (
    "Find the total vertical distance, in metres, the ball travels before it comes to rest, correct to 2 decimal places.",
    "If the ball bounces for ever, find the total vertical distance it travels, in metres, correct to 2 decimal places.",
)


def _two(value: Fraction) -> str:
    return stated_precision_string(round_half_up(value, 2), 2)


def _clear(value: Fraction) -> bool:
    """True when `value` is at least 0.02 of a hundredth away from a rounding boundary."""
    offset = (value * 100) % 1
    return abs(offset - Fraction(1, 2)) > Fraction(1, 50)


@generator("exd_x_r3d_bounce_series_chain", level=16, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3d_bounce_series_chain(rng: random.Random, tier: str = "exam") -> dict:
    """r, the drop height, the distance to the k-th landing and the distance to rest, from two rebound heights."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(3000, "a drop height and ratio with exact 2-decimal rebounds"):
        r = rng.choice(_RATIOS)
        # Both rebound heights are exact to 2 decimal places when 100 H is a multiple of the square of r's denominator.
        step = r.denominator**2
        h = Fraction(step * rng.randint(-(-100 // step), 1200 // step), 100)
        h1, h2 = h * r, h * r * r
        if (h1 * 100).denominator != 1 or (h2 * 100).denominator != 1:
            continue
        k = rng.choice([4, 5, 6])
        total_k = h + 2 * sum(h * r**i for i in range(1, k))
        total_inf = h * (1 + r) / (1 - r)
        if not (_clear(total_k) and _clear(total_inf)):
            continue
        fmt = _two

        def rebounds(count, once=False):
            return sum(h * r**i for i in range(1, count + 1)) * (1 if once else 2)

        a_cands = [
            (round_half_up(h1 / h2, 2), "inverted-ratio"),
            (round_half_up(h1 - h2, 2), "difference-instead-of-ratio"),
            (round_half_up(1 - r, 2), "complement-of-ratio"),
        ]
        b_cands = [
            (round_half_up(h1 * r, 2), "rebound-height-multiplied-by-ratio"),
            (round_half_up(h1 + (h1 - h2), 2), "arithmetic-instead-of-geometric"),
            (round_half_up(h1 / (r * r), 2), "undone-for-too-many-periods"),
            (h1, "first-rebound-as-drop-height"),
        ]
        c_cands = [
            (round_half_up(h + rebounds(k - 1, once=True), 2), "rebound-travel-counted-once"),
            (round_half_up(h + rebounds(k), 2), "one-term-too-many-in-series"),
            (round_half_up(2 * h + rebounds(k - 1), 2), "initial-drop-counted-twice"),
            (round_half_up(h + rebounds(k - 2), 2), "one-term-too-few-in-series"),
        ]
        d_cands = [
            (round_half_up(h / (1 - r), 2), "rebound-travel-counted-once"),
            (round_half_up(2 * h / (1 - r), 2), "initial-drop-counted-twice"),
            (round_half_up(h * (1 - r) / (1 + r), 2), "series-sum-formula-inverted"),
        ]
        a_d = tagged(_two(r), a_cands, fmt, tolerance=GRADER_TOLERANCE)
        b_d = tagged(_two(h), b_cands, fmt, tolerance=GRADER_TOLERANCE)
        c_d = tagged(_two(total_k), c_cands, fmt, tolerance=GRADER_TOLERANCE)
        d_d = tagged(_two(total_inf), d_cands, fmt, tolerance=GRADER_TOLERANCE)
        if len(a_d) >= 3 and len(b_d) >= 3 and len(c_d) >= 3 and len(d_d) >= 3:
            break
    # Independent route: add the heights term by term (the ball falls h, then rises and falls h r^i each rebound).
    fall_rise = [h] + [h * r**i for i in range(1, k) for _ in range(2)]
    check(sum(fall_rise) == total_k, "term-by-term distance disagrees with the series formula")
    check(abs(float(total_inf) - float(h + 2 * sum(h * r**i for i in range(1, 400)))) < 1e-6, "infinite sum disagrees")
    check(h2 / h1 == r, "the ratio of the rebounds must be r")

    stem = rng.choice(_STEMS).format(h1=_two(h1), h2=_two(h2))
    ft_c = f"part_b*(1 + 2*part_a*(1 - part_a**{k - 1})/(1 - part_a))"
    ft_d = "part_b*(1 + part_a)/(1 - part_a)"

    def part(prompt, value, marks, distractors, working, follow_through=None, sensible_range=(0, None)):
        out = dict(
            prompt=prompt,
            answer_kind="number",
            answer=float(round_half_up(value, 2)),
            answer_text=_two(value),
            answer_latex=latex_answer(_two(value)),
            marks=marks,
            sensible_range=sensible_range,
            working=working,
            distractors=distractors,
        )
        if follow_through:
            out["follow_through"] = follow_through
        return out

    return dict(
        text=stem,
        parts=[
            part(
                rng.choice(_PROMPT_A), r, 1, a_d, [f"r = {_two(h2)} ÷ {_two(h1)} = {_two(r)}."], sensible_range=(0, 1)
            ),
            part(
                rng.choice(_PROMPT_B),
                h,
                1,
                b_d,
                [
                    f"The 1st rebound is r times the drop height, so the drop height is {_two(h1)} ÷ {_two(r)} = {_two(h)} m."
                ],
                follow_through=f"{_two(h1)}/part_a",
            ),
            part(
                rng.choice(_PROMPT_C).format(k=_ORDINALS[k]),
                total_k,
                3,
                c_d,
                [f"Distance = H + 2·H·r·(1 - r^{k - 1})/(1 - r) = {_two(total_k)} m."],
                follow_through=ft_c,
            ),
            part(
                rng.choice(_PROMPT_D),
                total_inf,
                2,
                d_d,
                [f"Distance = H(1 + r)/(1 - r) = {_two(total_inf)} m."],
                follow_through=ft_d,
            ),
        ],
    )
