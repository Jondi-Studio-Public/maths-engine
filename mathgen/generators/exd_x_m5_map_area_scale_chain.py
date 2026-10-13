"""Exam-tier source for scale factors of similar figures: map scales and areas.

The stem gives a length on a map and the real distance it stands for, in kilometres, an area on the map in
cm², and a real area in hectares. The learner must

(a) work out the map scale 1 : n from the length (centimetres against kilometres),
(b) turn a map area into a real area in hectares (square the factor, then 10^8 cm² in a hectare), and
(c) go the other way, from a real area in hectares to an area on the map.

Built backwards from the scale, so every scale in the stem is exact.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    VerificationError,
    attempts,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
)

from .exd_x_m5_support import clear_of_boundary, display, keys_absent, rounded, tagged_distractors

_SCALES = (5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 40_000, 50_000, 60_000, 75_000, 80_000, 100_000)
_FEATURES = (
    ("a lake", "the lake"),
    ("a wetland", "the wetland"),
    ("a wheat paddock", "the paddock"),
    ("a pine plantation", "the plantation"),
    ("a reservoir", "the reservoir"),
)
_RESERVES = ("a national park reserve", "a nature reserve", "a bushland reserve", "a conservation reserve")
_PAIRS = (
    "two lookouts",
    "two bridges",
    "two railway stations",
    "two lighthouses",
    "two campsites",
)
_FIRST = (
    "On a street map, {pair} are {m} cm apart. Their real distance apart is {dist} km. "
    "On the same map, {feature} covers {area} cm². {Reserve} has a real area of {ha} ha.",
    "A bushwalking map shows {pair} {m} cm apart, and they are really {dist} km apart. "
    "{Feature} is drawn on the map with an area of {area} cm², and {reserve} has a real area of {ha} ha.",
    "The map of a council area is drawn to a single scale. {Pair} are {m} cm apart on it and {dist} km apart "
    "in real life. {Feature} covers {area} cm² of the map. Also, {reserve} has a real area of {ha} ha.",
)
_A_PROMPT = (
    "The scale of the map is 1 : n. Find n.",
    "Write the scale of the map as 1 : n and give the value of n.",
)
_B_PROMPT = (
    "Using your scale from (a), find the real area of {the_feature} in hectares, correct to 1 decimal place. "
    "(1 ha = 10 000 m²)",
    "Hence find the real area of {the_feature}, in hectares, correct to 1 decimal place. (1 ha = 10 000 m²)",
)
_C_PROMPT = (
    "Using your scale from (a), find the area of the reserve on the map, in square centimetres, "
    "correct to 1 decimal place.",
    "On the same map, what area in cm² does the reserve cover? Use your scale from (a) and give "
    "your answer correct to 1 decimal place.",
)


@generator("exd_x_m5_map_area_scale_chain", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m5_map_area_scale_chain(rng, tier: str = "exam") -> dict:
    """Exam tier: the scale of a map from a distance, then areas converted both ways."""
    if tier != "exam":
        raise VerificationError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "a map whose answers are clear of rounding boundaries"):
        n = rng.choice(_SCALES)
        m = rng.randint(2, 9)
        dist = Fraction(m * n, 100_000)
        area = Fraction(rng.randint(15, 95), 10)
        f2 = Fraction(n * n, 10**8)  # hectares per cm² of map
        ha = Fraction(rng.randint(int(2 * f2 * 10), int(20 * f2 * 10)), 10)
        real_ha = area * f2
        map_area = ha / f2
        if not (clear_of_boundary(real_ha, 1) and clear_of_boundary(map_area, 1)):
            continue
        a_text, b_text, c_text = str(n), display(real_ha, 1), display(map_area, 1)
        pair = rng.choice(_PAIRS)
        feature, the_feature = rng.choice(_FEATURES)
        reserve = rng.choice(_RESERVES)
        first = rng.choice(_FIRST).format(
            pair=pair,
            Pair=pair[0].upper() + pair[1:],
            m=m,
            dist=exact_decimal_string(dist, 4),
            feature=feature,
            Feature=feature[0].upper() + feature[1:],
            area=display(area, 1),
            reserve=reserve,
            Reserve=reserve[0].upper() + reserve[1:],
            ha=display(ha, 1),
        )
        prompts = [
            rng.choice(_A_PROMPT),
            rng.choice(_B_PROMPT).format(the_feature=the_feature),
            rng.choice(_C_PROMPT),
        ]
        if keys_absent(first, prompts, [a_text, b_text, c_text]):
            break

    a_dis = pick_distractors(
        a_text,
        [
            Distractor(str(n // 100), None, n // 100),  # kilometres read as metres: 1 km over cm
            Distractor(str(n * 10), None, n * 10),  # one factor of ten too many
            Distractor(str(n // 10), None, n // 10),  # one factor of ten too few
        ],
        3,
        tolerance=GRADER_TOLERANCE,
    )
    b_dis = tagged_distractors(
        b_text,
        1,
        [
            (area * n * n / 10_000, None),  # left in square metres
            (area * n / 10**8, None),  # scaled the area by n, not n squared
            (area * n * n / 10**10, None),  # converted to square kilometres
            (area * n * n / 10**7, None),  # took 1 ha as 10^7 cm²
        ],
    )
    c_dis = tagged_distractors(
        c_text,
        1,
        [
            (ha * 10**8 / n, None),  # divided by n, not n squared
            (ha * 10_000 / (n * n), None),  # worked in square metres
            (ha * f2, "inverted-ratio"),  # multiplied where it should divide
        ],
    )
    return dict(
        text=first,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=n,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"{exact_decimal_string(dist, 4)} km = {dist * 100_000} cm, so n = {dist * 100_000} ÷ {m} = {n}."
                ],
                distractors=a_dis,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=numeric_answer(rounded(real_ha, 1)),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{display(area, 1)}*part_a**2/10**8",
                working=[
                    f"Area scale factor = n² = {n}². Real area = {display(area, 1)} × {n}² cm².",
                    f"1 ha = 10⁸ cm², so real area = {b_text} ha (1 d.p.).",
                ],
                distractors=b_dis,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=numeric_answer(rounded(map_area, 1)),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"{display(ha, 1)}*10**8/part_a**2",
                working=[
                    f"{display(ha, 1)} ha = {display(ha, 1)} × 10⁸ cm²; map area = that ÷ {n}² = {c_text} cm² (1 d.p.)."
                ],
                distractors=c_dis,
            ),
        ],
    )
