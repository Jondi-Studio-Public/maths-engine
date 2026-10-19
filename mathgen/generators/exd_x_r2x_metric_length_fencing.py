"""Exam chain for `senior-review-metric-units-of-length-their-abbreviations-conversions`: fencing a plot whose sides
are given in different metric units.

The practice tiers convert one length, or add two lengths given in different units. The exam question is the job that
needs those conversions: a rectangular plot has one side in kilometres or centimetres and the other in metres, a gate
(given in its own unit) needs no fencing, fencing is priced per metre, and the learner pays with a note.

(a) the side given in kilometres or centimetres, in metres            exact decimal,  1 mark
(b) the perimeter of the plot in metres                               exact decimal,  2 marks, follow-through (a)
(c) the cost of the fencing, after leaving out the gate               to the cent,    2 marks, follow-through (b)
(d) the change from the note used to pay                              to the cent,    1 mark,  follow-through (c)

## Construction

The converted side is a whole number of metres (from kilometres) or a whole number of centimetres (so two decimal
places in metres), the rate per metre is a whole number of dollars, and the gate is a whole number of metres or of
centimetres. Every figure is therefore exact to two decimal places and nothing is rounded. The note is the next
multiple of 500 dollars (or 100 dollars for the smaller plots) above the cost, plus one more step in half the draws,
so the change is always positive. Each part is recomputed from the stated figures in the generator by a second route
(everything converted to centimetres and worked in whole numbers).
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase
from mathgen.generators._exd_x_r2x_kit import num_part

_PADDOCK = (
    "A farmer is fencing a rectangular paddock that is {long} long and {wide} wide. A gate {gate} wide is fitted "
    "and needs no fencing. The fencing costs ${rate} per metre, and the farmer pays with a ${note} cheque.",
    "A council is fencing a rectangular reserve {long} long and {wide} wide. A gate {gate} wide is part of the "
    "boundary and needs no fencing. Fencing costs ${rate} per metre and the council pays with a ${note} cheque.",
)
_GARDEN = (
    "A rectangular backyard measures {long} by {wide}. A new fence goes right around it, except for a gate {gate} "
    "wide that needs no fencing. The fencing costs ${rate} per metre and the owner pays with ${note} in cash.",
    "A school is fencing a rectangular playground {long} long and {wide} wide. A gate {gate} wide needs no "
    "fencing. The fencing costs ${rate} per metre and the school pays with a ${note} cheque.",
)


def _length(value: Fraction, unit: str) -> str:
    """A length with the shortest exact decimal and its unit."""
    if value.denominator == 1:
        return f"{int(value)} {unit}"
    return f"{float(value):g} {unit}"


@generator("exd_x_r2x_metric_length_fencing", level=10, presets={"exam": {"tier": "exam"}})
def exd_x_r2x_metric_length_fencing(rng, tier: str = "exam"):
    """Convert a side to metres, find the perimeter, price the fencing less the gate, then find the change."""
    check(tier == "exam", f"unknown tier {tier!r}")
    kilometres = rng.random() < 0.5
    for _ in attempts(1000, "a plot whose figures are all distinct"):
        if kilometres:
            side_conv_m = Fraction(rng.randrange(250, 1500, 5))  # metres
            conv_value, conv_unit = side_conv_m / 1000, "km"
            other_m = Fraction(rng.randrange(180, 760, 10))
            other_value, other_unit = other_m, "m"
            gate_m = Fraction(rng.choice([3, 4, 5, 6]))
            gate_value, gate_unit = gate_m, "m"
            rate = rng.randint(8, 22)
            step = 500
            patterns = _PADDOCK
        else:
            side_conv_cm = rng.randrange(450, 2500, 5)
            side_conv_m = Fraction(side_conv_cm, 100)
            conv_value, conv_unit = Fraction(side_conv_cm), "cm"
            other_m = Fraction(rng.randrange(60, 300, 5), 10)
            other_value, other_unit = other_m, "m"
            gate_cm = rng.choice([300, 320, 350, 360, 400])
            gate_m = Fraction(gate_cm, 100)
            gate_value, gate_unit = Fraction(gate_cm), "cm"
            rate = rng.randint(14, 48)
            step = 100
            patterns = _GARDEN
        perimeter = 2 * (side_conv_m + other_m)
        fenced = perimeter - gate_m
        cost = rate * fenced
        if cost.denominator not in (1, 2, 4, 5, 10, 20, 25, 50, 100):
            continue
        note = (math.floor(cost / step) + 1 + rng.randint(0, 1)) * step
        change = note - cost
        shown = {conv_value, other_value, gate_value, rate, note}
        if {side_conv_m, perimeter, cost, change} & shown:
            continue
        break
    # A second route: everything in whole centimetres.
    cm = lambda v: int(v * 100)  # noqa: E731
    check(cm(side_conv_m) + cm(other_m) == cm(perimeter) // 2, "half-perimeter disagrees in centimetres")
    check(Fraction(rate * (cm(perimeter) - cm(gate_m)), 100) == cost, "cost disagrees in centimetres")
    check(note - cost == change and change > 0, "change must be positive")

    long_side, wide_side = (conv_value, other_value) if rng.random() < 0.5 else (other_value, conv_value)
    long_unit, wide_unit = (conv_unit, other_unit) if long_side == conv_value else (other_unit, conv_unit)
    stem = phrase(
        rng,
        patterns,
        long=_length(long_side, long_unit),
        wide=_length(wide_side, wide_unit),
        gate=_length(gate_value, gate_unit),
        rate=rate,
        note=f"{note:,}",
    )
    big_text = _length(conv_value, conv_unit)
    other_text = _length(other_value, other_unit)
    conv_places = 0 if kilometres else 2

    side_wrong = [
        (side_conv_m * 10, "unit-factor-wrong-power"),
        (side_conv_m / 10, "unit-factor-wrong-power"),
        (side_conv_m * 100 if kilometres else side_conv_m * 10, "unit-factor-wrong-power"),
        (conv_value, "r2x-fence-side-not-converted"),
    ]
    half_perimeter = side_conv_m + other_m
    parts = [
        num_part(
            f"Write the length of the {big_text} side in metres.",
            side_conv_m,
            conv_places,
            1,
            [f"1 km = 1000 m, so {big_text} = {float(side_conv_m):g} m."]
            if kilometres
            else [f"100 cm = 1 m, so {big_text} = {float(side_conv_m):g} m."],
            side_wrong,
        ),
        num_part(
            "Using your answer to (a), find the perimeter of the plot in metres.",
            perimeter,
            2 if not kilometres else 0,
            2,
            [
                f"The sides are {float(side_conv_m):g} m and {float(other_m):g} m.",
                f"Perimeter = 2 × ({float(side_conv_m):g} + {float(other_m):g}) = {float(perimeter):g} m.",
            ],
            [
                (half_perimeter, "r2x-fence-half-perimeter"),
                (side_conv_m + 2 * other_m, "r2x-fence-one-side-counted-once"),
                (2 * side_conv_m + other_m, "r2x-fence-one-side-counted-once"),
                (2 * (conv_value + other_m), "r2x-fence-side-not-converted"),
            ],
            follow=f"2*(part_a + {float(other_m)})",
        ),
        num_part(
            "Using your answer to (b), find the cost of the fencing in dollars, to the nearest cent. "
            "The gate is not fenced.",
            cost,
            2,
            2,
            [
                f"Length to fence = {float(perimeter):g} - {float(gate_m):g} = {float(fenced):g} m.",
                f"Cost = {rate} × {float(fenced):g} = {float(cost):.2f} dollars.",
            ],
            [
                (rate * perimeter, "r2x-fence-gate-not-deducted"),
                (rate * (perimeter + gate_m), "r2x-fence-gate-added"),
                (rate * half_perimeter, "r2x-fence-half-perimeter"),
                (rate * (perimeter - gate_value), "r2x-fence-gate-not-converted"),
            ],
            follow=f"{rate}*(part_b - {float(gate_m)})",
        ),
        num_part(
            f"Using your answer to (c), how much change is there from the ${note:,}, in dollars, to the nearest cent?",
            change,
            2,
            1,
            [f"Change = {note} - {float(cost):.2f} = {float(change):.2f} dollars."],
            [
                (note + cost, "r2x-fence-added-not-subtracted"),
                (note - math.floor(cost), "r2x-fence-cents-dropped"),
                (Fraction(note, 2) - cost, "r2x-fence-pay-halved"),
                (cost, "r2x-fence-cost-for-change"),
            ],
            follow=f"{note} - part_c",
        ),
    ]
    return dict(text=stem, parts=parts)
