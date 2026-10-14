"""Generates `sr2b_scale_costs` questions (easy, medium, hard) and the exam chain `sr2b_exam_scale_costs`.

Skill ``scale-drawing-costs`` (``measurement``, answer ``number``): lengths on a plan drawn to a scale of 1 : n are turned
into real lengths and areas (the area scales by the square of the ratio), and then into quantities of material and a
cost. The floor-plan skill for real lengths and areas stops before quantities and money.

* easy: the cost of covering a room's floor, from its plan dimensions in centimetres, the scale and a price per square
  metre.
* medium: the same with an allowance for waste (a percentage extra is ordered).
* hard: either flooring sold in whole boxes of a stated coverage (the area plus waste, divided by the coverage, rounded
  up to a whole box, then priced), or fencing round a courtyard with a gate gap, where the lengths scale by n but the
  gate is a real length and has its own price.
* exam (``sr2b_exam_scale_costs``): (a) the real length of the room, (b) its real area (follows through from (a)),
  (c) the area to order with the waste allowance (follows through from (b)), (d) the cost of the whole boxes needed
  (follows through from (c)).

Every real length is also found in centimetres first and converted at the end, which is the second route to each
answer. Money is in dollars, shown to the number of decimal places the exact answer needs.
"""

from __future__ import annotations

from fractions import Fraction
from math import ceil

from ..core import attempts, check, exact_decimal_string, generator, phrase
from ._sr2b_kit import decimal_dist, number_fields, pick, places_needed, retry_short

_ROOMS = (
    ("a bedroom", "carpet", "costs"),
    ("a lounge room", "timber flooring", "costs"),
    ("a classroom", "vinyl flooring", "costs"),
    ("a studio", "bamboo flooring", "costs"),
    ("a hall", "carpet tiles", "cost"),
    ("an office", "laminate flooring", "costs"),
)
_SCALES = (20, 25, 50, 100)


def _txt(v: Fraction) -> str:
    return exact_decimal_string(v, 2)


def _draw_room(rng, *, allow=lambda rl, rw: True):
    """Real dimensions in metres (multiples of 0.5) and a scale for which the plan lengths have one decimal place."""
    for _ in attempts(1000, "a room whose plan lengths are tidy"):
        s = rng.choice(_SCALES)
        rl = Fraction(rng.randint(7, 18), 2)
        rw = Fraction(rng.randint(5, 14), 2)
        if rl <= rw:
            continue
        plan_l, plan_w = rl * 100 / s, rw * 100 / s
        if (plan_l * 10).denominator != 1 or (plan_w * 10).denominator != 1:
            continue
        if allow(rl, rw):
            return s, rl, rw, plan_l, plan_w
    raise AssertionError  # pragma: no cover


def _real_area_cm(plan_l: Fraction, plan_w: Fraction, s: int) -> Fraction:
    """Second route: scale each plan length to real centimetres, multiply, then convert square centimetres."""
    return (plan_l * s) * (plan_w * s) / 10000


def _area_distractors(plan_l, plan_w, rl, rw, s):
    """Wrong areas, each from a scaling error: (value, tag)."""
    return [
        (plan_l * plan_w * s / 10000, "sr2b-area-scaled-linearly"),
        (plan_l * s * plan_w * s / 100, "sr2b-square-centimetres-to-metres-with-100"),
        (2 * (rl + rw), "sr2b-perimeter-for-area"),
        (plan_l * plan_w, "sr2b-scale-ignored"),
    ]


def _fmt_plan(v: Fraction) -> str:
    return exact_decimal_string(v, 1)


def _intro(rng, room, s, plan_l, plan_w):
    return phrase(
        rng,
        (
            "A plan of {room} is drawn to a scale of 1 : {s}. On the plan the floor is {l} cm long and {w} cm wide.",
            "The floor of {room} is shown on a plan with a scale of 1 : {s}. It measures {l} cm by {w} cm on the plan.",
        ),
        room=room,
        s=s,
        l=_fmt_plan(plan_l),
        w=_fmt_plan(plan_w),
    )


# -- easy -----------------------------------------------------------------------------------------------------


def _easy(rng):
    room, product, verb = rng.choice(_ROOMS)
    s, rl, rw, plan_l, plan_w = _draw_room(rng)
    price = rng.randint(18, 95)
    area = rl * rw
    check(area == _real_area_cm(plan_l, plan_w, s), "area differs from the centimetre route")
    cost = area * price
    dp = places_needed(cost)
    cands = [decimal_dist(a * price, dp, tag) for a, tag in _area_distractors(plan_l, plan_w, rl, rw, s)]
    cands.append(decimal_dist(area, dp, "sr2b-area-given-for-cost"))
    cands = [c for c in cands if c.text != number_fields(cost, dp)["answer_text"]]
    stem = (
        _intro(rng, room, s, plan_l, plan_w)
        + f" {product.capitalize()} {verb} ${price} per square metre. What is the cost, in dollars, of covering the whole floor?"
    )
    return dict(
        text=stem,
        working=[
            f"Real length = {_fmt_plan(plan_l)} × {s} = {_txt(plan_l * s)} cm = {_txt(rl)} m. Real width = {_fmt_plan(plan_w)} × {s} = {_txt(plan_w * s)} cm = {_txt(rw)} m.",
            f"Area = {_txt(rl)} × {_txt(rw)} = {_txt(area)} m².",
            f"Cost = {_txt(area)} × {price} = ${_txt(cost)}.",
        ],
        distractors=pick(number_fields(cost, dp)["answer_text"], cands),
        **number_fields(cost, dp),
    )


# -- medium ---------------------------------------------------------------------------------------------------


def _medium(rng):
    room, product, verb = rng.choice(_ROOMS)
    for _ in attempts(500, "a waste allowance that keeps the cost to cents"):
        s, rl, rw, plan_l, plan_w = _draw_room(rng)
        price = rng.randint(18, 95)
        waste = rng.choice((10, 15, 20, 25))
        area = rl * rw
        cost = area * Fraction(100 + waste, 100) * price
        if (area * Fraction(100 + waste, 100) * 100).denominator != 1:
            continue
        try:
            dp = places_needed(cost)
        except ValueError:
            continue
        break
    check(area == _real_area_cm(plan_l, plan_w, s), "area differs from the centimetre route")
    f = Fraction(100 + waste, 100)
    cands = [decimal_dist(a * f * price, dp, tag) for a, tag in _area_distractors(plan_l, plan_w, rl, rw, s)[:3]]
    cands += [
        decimal_dist(area * price, dp, "sr2b-waste-allowance-left-out"),
        decimal_dist(area * Fraction(100 - waste, 100) * price, dp, "sr2b-waste-subtracted"),
        decimal_dist((area + waste) * price, dp, "sr2b-waste-percentage-added-as-area"),
        decimal_dist(area * (price + waste), dp, "sr2b-waste-added-to-price"),
    ]
    ans = number_fields(cost, dp)["answer_text"]
    stem = (
        _intro(rng, room, s, plan_l, plan_w)
        + f" {product.capitalize()} {verb} ${price} per square metre, and {waste}% more than the floor area is ordered to allow for waste. What is the cost, in dollars, of the {product} ordered?"
    )
    return dict(
        text=stem,
        working=[
            f"Real floor: {_txt(rl)} m by {_txt(rw)} m, so the area is {_txt(area)} m² (each length is multiplied by {s}, so the area by {s}²).",
            f"Amount ordered = {_txt(area)} × {_txt(Fraction(100 + waste, 100))} = {_txt(area * f)} m².",
            f"Cost = {_txt(area * f)} × {price} = ${_txt(cost)}.",
        ],
        distractors=pick(ans, [c for c in cands if c.text != ans]),
        **number_fields(cost, dp),
    )


# -- hard -----------------------------------------------------------------------------------------------------


def _hard_boxes(rng):
    room, product, verb = rng.choice(_ROOMS)
    for _ in attempts(500, "an order whose box count and cost are tidy"):
        s, rl, rw, plan_l, plan_w = _draw_room(rng)
        waste = rng.choice((10, 15, 20))
        cover = rng.choice((Fraction(3, 2), Fraction(2), Fraction(5, 2), Fraction(3), Fraction(4)))
        box_price = Fraction(rng.choice((34, 42, 48, 55, 63, 72, 85)))
        area = rl * rw
        need = area * Fraction(100 + waste, 100)
        boxes = ceil(need / cover)
        if need / cover == boxes or boxes < 3 or (need * 100).denominator != 1:
            continue
        break
    cost = boxes * box_price
    dp = places_needed(cost)
    check(area == _real_area_cm(plan_l, plan_w, s), "area differs from the centimetre route")
    check(boxes * cover >= need > (boxes - 1) * cover, "box count is not the smallest that covers the order")
    f = Fraction(100 + waste, 100)
    cands = [
        decimal_dist(need / cover * box_price, dp, "sr2b-boxes-not-rounded-up"),
        decimal_dist((boxes - 1) * box_price, dp, "sr2b-boxes-rounded-down"),
        decimal_dist(ceil(area / cover) * box_price, dp, "sr2b-waste-allowance-left-out"),
        decimal_dist(ceil(plan_l * plan_w * s / 10000 * f / cover) * box_price, dp, "sr2b-area-scaled-linearly"),
        decimal_dist(boxes * cover * box_price, dp, "sr2b-box-coverage-priced-as-box"),
    ]
    ans = number_fields(cost, dp)["answer_text"]
    cover_t, price_t = _txt(cover), _txt(box_price)
    stem = (
        _intro(rng, room, s, plan_l, plan_w)
        + f" The {product} is sold in boxes that each cover {cover_t} m² and cost ${price_t} a box. {waste}% more than the floor area is ordered to allow for waste, and only whole boxes can be bought. What is the total cost, in dollars?"
    )
    return dict(
        text=stem,
        working=[
            f"Real floor: {_txt(rl)} m by {_txt(rw)} m = {_txt(area)} m².",
            f"With waste: {_txt(area)} × {_txt(Fraction(100 + waste, 100))} = {_txt(need)} m², and {_txt(need)} ÷ {cover_t} = {float(need / cover):.2f}, so {boxes} whole boxes are needed.",
            f"Cost = {boxes} × ${price_t} = ${_txt(cost)}.",
        ],
        distractors=pick(ans, [c for c in cands if c.text != ans]),
        **number_fields(cost, dp),
    )


def _hard_fence(rng):
    for _ in attempts(500, "a fencing cost to cents"):
        s, rl, rw, plan_l, plan_w = _draw_room(rng)
        gate = rng.choice((Fraction(1), Fraction(3, 2), Fraction(2)))
        fence = rng.randint(22, 60)
        gate_price = rng.choice((120, 150, 185, 240))
        perimeter = 2 * (rl + rw)
        cost = (perimeter - gate) * fence + gate_price
        try:
            dp = places_needed(cost)
        except ValueError:
            continue
        break
    check(perimeter == 2 * (plan_l * s + plan_w * s) / 100, "perimeter differs from the centimetre route")
    cands = [
        decimal_dist(perimeter * fence + gate_price, dp, "sr2b-gate-gap-not-removed"),
        decimal_dist((perimeter - gate) * fence, dp, "sr2b-gate-price-left-out"),
        decimal_dist(perimeter * fence, dp, "sr2b-gate-ignored"),
        decimal_dist(
            (2 * (plan_l + plan_w) * s / 10 - gate) * fence + gate_price, dp, "sr2b-centimetres-to-metres-with-10"
        ),
        decimal_dist((perimeter - gate) * fence + 2 * gate_price, dp, "sr2b-gate-price-counted-twice"),
    ]
    ans = number_fields(cost, dp)["answer_text"]
    stem = (
        f"A plan of a rectangular courtyard is drawn to a scale of 1 : {s}. On the plan the courtyard is {_fmt_plan(plan_l)} cm long and {_fmt_plan(plan_w)} cm wide. "
        f"A fence is to go all the way round the courtyard, except for a gate opening {_txt(gate)} m wide. The fencing costs ${fence} per metre and the gate costs ${gate_price}. "
        "What is the total cost of the fencing and the gate, in dollars?"
    )
    return dict(
        text=stem,
        working=[
            f"Real sides: {_txt(rl)} m and {_txt(rw)} m, so the distance round is 2 × ({_txt(rl)} + {_txt(rw)}) = {_txt(perimeter)} m.",
            f"Fencing needed = {_txt(perimeter)} - {_txt(gate)} = {_txt(perimeter - gate)} m, costing {_txt(perimeter - gate)} × {fence} = ${_txt((perimeter - gate) * fence)}.",
            f"Total = ${_txt((perimeter - gate) * fence)} + ${gate_price} = ${_txt(cost)}.",
        ],
        distractors=pick(ans, [c for c in cands if c.text != ans]),
        **number_fields(cost, dp),
    )


@generator(
    "sr2b_scale_costs",
    level=11,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def sr2b_scale_costs(rng, tier="easy"):
    """Quantities and costs from a scale plan; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_boxes, _hard_boxes, _hard_fence])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("sr2b_exam_scale_costs", level=11, presets={"exam": dict(tier="exam")})
@retry_short
def sr2b_exam_scale_costs(rng, tier="exam"):
    """Exam chain: real length, real area, area to order, cost of whole boxes."""
    check(tier == "exam", f"unknown tier {tier!r}")
    room, product, verb = rng.choice(_ROOMS)
    for _ in attempts(1000, "an exam order with tidy values at every step"):
        s, rl, rw, plan_l, plan_w = _draw_room(rng)
        waste = 20  # one fifth more keeps every carried area to two places
        cover = rng.choice((Fraction(3, 2), Fraction(2), Fraction(5, 2), Fraction(3)))
        box_price = Fraction(rng.choice((34, 42, 48, 55, 63, 72)))
        area = rl * rw
        need = area * Fraction(100 + waste, 100)
        boxes = ceil(need / cover)
        if need / cover == boxes or boxes < 3 or (need * 100).denominator != 1:
            continue
        try:
            places_needed(need)
            places_needed(boxes * box_price)
        except ValueError:
            continue
        break
    cost = boxes * box_price
    check(area == _real_area_cm(plan_l, plan_w, s), "area differs from the centimetre route")
    check(boxes * cover >= need > (boxes - 1) * cover, "box count is not the smallest that covers the order")
    # parts b and c are carried into later parts, so they are stated to the two places any carried reply can need
    dp_l, dp_a, dp_n, dp_c = places_needed(rl), 2, 2, places_needed(cost)
    f = Fraction(100 + waste, 100)
    a_cands = [
        decimal_dist(plan_l * s, dp_l, "sr2b-centimetres-not-converted-to-metres"),
        decimal_dist(plan_l * s / 10, dp_l, "sr2b-centimetres-to-metres-with-10"),
        decimal_dist(plan_l, dp_l, "sr2b-scale-ignored"),
        decimal_dist(plan_l / s, dp_l, "sr2b-scale-applied-the-wrong-way"),
        decimal_dist(rw, dp_l, "sr2b-other-side-used"),
    ]
    b_cands = [
        decimal_dist(plan_l * plan_w * s / 10000, dp_a, "sr2b-area-scaled-linearly"),
        decimal_dist(rl * plan_w, dp_a, "sr2b-one-side-scaled"),
        decimal_dist(2 * (rl + rw), dp_a, "sr2b-perimeter-for-area"),
        decimal_dist(plan_l * plan_w, dp_a, "sr2b-scale-ignored"),
    ]
    c_cands = [
        decimal_dist(area, dp_n, "sr2b-waste-allowance-left-out"),
        decimal_dist(area * Fraction(100 - waste, 100), dp_n, "sr2b-waste-subtracted"),
        decimal_dist(area + waste, dp_n, "sr2b-waste-percentage-added-as-area"),
        decimal_dist(area * Fraction(waste, 100), dp_n, "sr2b-waste-only"),
    ]
    d_cands = [
        decimal_dist(need / cover * box_price, dp_c, "sr2b-boxes-not-rounded-up"),
        decimal_dist((boxes - 1) * box_price, dp_c, "sr2b-boxes-rounded-down"),
        decimal_dist(ceil(area / cover) * box_price, dp_c, "sr2b-waste-allowance-left-out"),
        decimal_dist(boxes * cover * box_price, dp_c, "sr2b-box-coverage-priced-as-box"),
    ]
    cover_t, price_t = _txt(cover), _txt(box_price)
    stem = (
        f"A plan of {room} is drawn to a scale of 1 : {s}. On the plan the floor is {_fmt_plan(plan_l)} cm long and {_fmt_plan(plan_w)} cm wide. "
        f"The {product} is sold in boxes that each cover {cover_t} m² and cost ${price_t} a box."
    )
    wtxt = f"{waste}% more than the floor area is ordered to allow for waste"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the real length of the floor, in metres.",
                marks=1,
                sensible_range=(0, None),
                working=[f"{_fmt_plan(plan_l)} cm × {s} = {_txt(plan_l * s)} cm = {_txt(rl)} m."],
                distractors=pick(
                    number_fields(rl, dp_l)["answer_text"],
                    [c for c in a_cands if c.text != number_fields(rl, dp_l)["answer_text"]],
                ),
                **number_fields(rl, dp_l),
            ),
            dict(
                prompt="Hence find the real area of the floor, in square metres.",
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*{_txt(rw)}",
                working=[
                    f"Real width = {_fmt_plan(plan_w)} × {s} ÷ 100 = {_txt(rw)} m. Area = {_txt(rl)} × {_txt(rw)} = {_txt(area)} m²."
                ],
                distractors=pick(
                    number_fields(area, dp_a)["answer_text"],
                    [c for c in b_cands if c.text != number_fields(area, dp_a)["answer_text"]],
                ),
                **number_fields(area, dp_a),
            ),
            dict(
                prompt=f"{wtxt[0].upper()}{wtxt[1:]}. Find the area to order, in square metres.",
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_b*{_txt(f)}",
                working=[f"{_txt(area)} × {_txt(f)} = {_txt(need)} m²."],
                distractors=pick(
                    number_fields(need, dp_n)["answer_text"],
                    [c for c in c_cands if c.text != number_fields(need, dp_n)["answer_text"]],
                ),
                **number_fields(need, dp_n),
            ),
            dict(
                prompt="Only whole boxes can be bought. Find the total cost of the boxes needed, in dollars.",
                marks=2,
                sensible_range=(0, None),
                follow_through=f"ceiling(part_c/{_txt(cover)})*{_txt(box_price)}",
                working=[
                    f"{_txt(need)} ÷ {cover_t} is more than {boxes - 1}, so {boxes} boxes are needed: {boxes} × ${price_t} = ${_txt(cost)}."
                ],
                distractors=pick(
                    number_fields(cost, dp_c)["answer_text"],
                    [c for c in d_cands if c.text != number_fields(cost, dp_c)["answer_text"]],
                ),
                **number_fields(cost, dp_c),
            ),
        ],
    )
