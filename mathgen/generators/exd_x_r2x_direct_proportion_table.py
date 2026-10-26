"""Exam chain for `f10-direct-proportion-scale-model`: a table of values, the constant, and two uses of it.

The practice tiers (one multiple choice question) decide whether a stated rate or scale gives a directly proportional
relationship and compute one value. The exam question gives a three-row table in a story, with an unknown constant
of proportionality ``k`` in ``y = k x`` (every row has the same ``y / x``), and asks:

(a) the constant of proportionality k, from the table             2 d.p.,  2 marks
(b) the value of y at a new x, from k                              2 d.p.,  1 mark,  follow-through (a)
(c) the x that gives a y that is a stated amount more than (b)    whole,   2 marks, follow-through (a), (b)

## Construction

``k`` is drawn from a list of two-decimal constants for the setting, the three table inputs and the new input are
distinct whole numbers, and the extra amount is ``k`` times a whole number, so the answer to (c) is a whole number.
Every figure is exact. The table rows are checked to share one ratio (a second route for k), and (c) is recomputed
by adding the whole-number step to the new input.

## Distractors

(a) the ratio inverted, the first row's output with no division, the difference of two outputs.
(b) division by k in place of multiplication, the output of the nearest table row shifted by the difference in inputs
(additive reasoning), k plus the input.
(c) the extra input taken away instead of added, the extra input alone, the extra output added to the input.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase
from mathgen.generators._exd_x_r2x_kit import num_part

_SCENES = (
    dict(
        constants=[Fraction(v, 100) for v in range(30, 200, 5)],
        xs=range(4, 61),
        step=range(2, 13),
        stem=(
            "A landscape supplier sells crushed gravel in bags, and the price depends only on the mass. A "
            "{x1} kg bag costs ${y1}, a {x2} kg bag costs ${y2} and a {x3} kg bag costs ${y3}.",
        ),
        a="In the rule price = k × mass, find the constant k, in dollars per kilogram.",
        b="Using your answer to (a), find the price of a {xn} kg bag, in dollars.",
        c="Using your answer to (b), a customer pays ${delta} more than the price of the {xn} kg bag. What mass of "
        "gravel does the customer buy, in kilograms?",
        money=True,
        unit_x="kg",
    ),
    dict(
        constants=[Fraction(v, 100) for v in (50, 100, 125, 200, 250, 400, 500)],
        xs=range(8, 121),
        step=range(2, 17),
        stem=(
            "A model of a ferry is built to scale, so each length on the model is in proportion to the real length. A "
            "real length of {x1} m is {y1} cm on the model, {x2} m is {y2} cm and {x3} m is {y3} cm.",
        ),
        a="In the rule model length = k × real length, find the constant k, in centimetres per metre.",
        b="Using your answer to (a), find the model length, in centimetres, of a real length of {xn} m.",
        c="Using your answer to (b), another part of the model is {delta} cm longer than the part in (b). What is the "
        "real length of that other part, in metres?",
        money=False,
        unit_x="m",
    ),
    dict(
        constants=[Fraction(v, 100) for v in range(6, 15)],
        xs=range(50, 460, 10),
        step=range(10, 90, 5),
        stem=(
            "A courier van's fuel use was recorded on three trips. A {x1} km trip used {y1} L of fuel, a {x2} km trip "
            "used {y2} L and a {x3} km trip used {y3} L.",
        ),
        a="In the rule fuel used = k × distance, find the constant k, in litres per kilometre.",
        b="Using your answer to (a), find the fuel used, in litres, on a {xn} km trip.",
        c="Using your answer to (b), another trip uses {delta} L more fuel than the {xn} km trip. How far is that "
        "trip, in kilometres?",
        money=False,
        unit_x="km",
    ),
    dict(
        constants=[Fraction(v, 100) for v in range(2400, 3900, 50)],
        xs=range(3, 13),
        step=range(1, 6),
        stem=(
            "A cafe pays its casual staff by the hour, with no fixed payment. A {x1}-hour shift pays ${y1}, a {x2}-"
            "hour shift pays ${y2} and a {x3}-hour shift pays ${y3}.",
        ),
        a="In the rule pay = k × hours, find the constant k, in dollars per hour.",
        b="Using your answer to (a), find the pay for a {xn}-hour shift, in dollars.",
        c="Using your answer to (b), a different shift pays ${delta} more than the {xn}-hour shift. How many hours is "
        "that shift?",
        money=True,
        unit_x="h",
    ),
)


def _fmt(value: Fraction, money: bool) -> str:
    return f"{float(value):.2f}" if money else f"{float(value):g}"


@generator("exd_x_r2x_direct_proportion_table", level=9, presets={"exam": {"tier": "exam"}})
def exd_x_r2x_direct_proportion_table(rng, tier: str = "exam"):
    """Find k from a table, use it to find an output, then find the input for a larger output."""
    check(tier == "exam", f"unknown tier {tier!r}")
    sc = rng.choice(_SCENES)
    for _ in attempts(1000, "table rows and a new input that are all different"):
        k = rng.choice(sc["constants"])
        xs = sorted(rng.sample(list(sc["xs"]), 4))
        rng.shuffle(xs)
        x1, x2, x3, xn = sorted(xs[:3]) + [xs[3]]
        dx = rng.choice(list(sc["step"]))
        ys = [k * v for v in (x1, x2, x3)]
        y_new = k * xn
        delta = k * dx
        x_target = xn + dx
        if len({x1, x2, x3, xn, x_target}) < 5 or len({k, *ys, y_new, delta}) < 6:
            continue
        if x_target in (x1, x2, x3):
            continue
        break
    check(len({y / x for y, x in zip(ys, (x1, x2, x3))}) == 1, "table rows must share one ratio")
    check((y_new + delta) / k == x_target, "extra output disagrees with the whole-number step")

    money = sc["money"]
    slots = dict(
        x1=x1, x2=x2, x3=x3, xn=xn, delta=_fmt(delta, money),
        y1=_fmt(ys[0], money), y2=_fmt(ys[1], money), y3=_fmt(ys[2], money),
    )  # fmt: skip
    stem = phrase(rng, sc["stem"], **slots)
    near = min((x1, x2, x3), key=lambda v: abs(v - xn))
    y_near = k * near
    parts = [
        num_part(
            sc["a"],
            k,
            2,
            2,
            [f"Each row gives the same ratio: {_fmt(ys[0], money)} ÷ {x1} = {_fmt(k, True)}.", f"k = {_fmt(k, True)}."],
            [
                (1 / k, "r2x-prop-ratio-inverted"),
                (ys[0], "r2x-prop-output-not-ratio"),
                (ys[1] - ys[0], "r2x-prop-difference-not-ratio"),
                (k * 10, "r2x-prop-decimal-shift"),
            ],
        ),
        num_part(
            sc["b"].format(**slots),
            y_new,
            2,
            1,
            [f"y = k × x = {_fmt(k, True)} × {xn} = {_fmt(y_new, True)}."],
            [
                (Fraction(xn) / k, "r2x-prop-divided-not-multiplied"),
                (y_near + (xn - near), "r2x-prop-additive-reasoning"),
                (k + xn, "r2x-prop-added-not-multiplied"),
                (y_new * 10, "r2x-prop-decimal-shift"),
            ],
            follow=f"part_a*{xn}",
        ),
        num_part(
            sc["c"].format(**slots),
            Fraction(x_target),
            0,
            2,
            [
                f"The new output is {_fmt(y_new, True)} + {_fmt(delta, True)} = {_fmt(y_new + delta, True)}.",
                f"x = y ÷ k = {_fmt(y_new + delta, True)} ÷ {_fmt(k, True)} = {x_target}.",
            ],
            [
                (Fraction(xn - dx), "r2x-prop-increase-subtracted"),
                (Fraction(dx), "r2x-prop-increase-only"),
                (Fraction(xn) + round(delta), "r2x-prop-output-added-to-input"),
                (Fraction(near), "r2x-prop-table-input-reused"),
            ],
            follow=f"(part_b + {float(delta)})/part_a",
        ),
    ]
    return dict(text=stem, parts=parts)
