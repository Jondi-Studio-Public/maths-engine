"""Hectare-family area conversion and straight-edged shape perimeters, both senior ``acara-senior-v8-4``, domain
``measurement``, ``answer: number``.

Two generators:

- ``convert_between_metric_units_of_area`` (``senior-convert-between-metric-
  units-of-area-and-other``, ACMEM023) -- convert an area measurement within
  the **hectare family**: m²<->ha, ha<->km². **This skill's own territory is
  the hectare family only, never the plain metric-prefix ladder
  mm²/cm²/m²/km².** That plain-ladder ladder belongs exclusively to the
  neighbouring skill ``senior-use-metric-units-of-area-their-
  abbreviations-conversions`` (ACMEM021), whose own scope is
  "convert an area measurement between metric units" over mm²/cm²/m²/km².
  ACMEM023's own official curriculum text says "and **other** area units",
  which is read here as hectares -- the one non-power-of-100 area unit
  outside the plain mm²/cm²/m²/km² prefix ladder (1 ha = 10,000 m²;
  1 km² = 100 ha = 1,000,000 m², the km²<->ha factor of 100 derived from the
  km²<->m² and ha<->m² relationships in ``_FACTORS_M2`` below, never invented
  as a bare constant). A hard-tier item never converts directly between m²
  and km² (that single-hop shape is the plain-ladder territory this skill
  explicitly does not cover) -- the hard tier's mixed-unit pair always keeps
  ``ha`` as one of the two stated units.

- ``calculate_perimeters_of_familiar_shapes`` (``senior-calculate-perimeters-
  of-familiar-shapes-including-triangles-squares``, ACMEM091) -- calculate
  the perimeter of a rectangle, square, triangle, irregular polygon, or a
  straight-edged composite of these. **Design option A (straight-edged only)
  was taken over option B (circles/arc-length included)**: this skill never
  builds a circle-circumference or arc-length item, since every straight
  edge in every tier is either given directly or an exact integer, which
  keeps every canonical answer an exact integer with zero irrational-number
  rounding risk -- option B's own precision-tolerance bookkeeping (a fixed
  decimal-place canonical, every distractor cleared against that stated
  tolerance) is never needed here. Circles and arc lengths, named in
  ACMEM091's own raw curriculum text, are therefore explicitly out of scope;

Both generators format ``answer_text`` from an exact ``Fraction``/``int``
value with explicit, bounded fixed-point decimal places (never
``str(float(...))``, which can silently switch to scientific notation below
``1e-4``) -- see ``exact_decimal_string``/``rounded_decimal_string`` in
``mathgen/core.py``. The perimeter
generator's own values are always plain integers by construction, so it
never needs fixed-point formatting at all.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_combine,
    check_conversion,
    exact_decimal_string,
    fixed_point_string,
    generator,
    numeric_answer,
    pick_distractors,
)

# --- convert_between_metric_units_of_area ----------------------------------

# Base factor table, area expressed in m^2. The km^2<->ha factor (100) is
# always derived from this table (Fraction(_FACTORS_M2["km2"],
# _FACTORS_M2["ha"])), never hardcoded as an unrelated constant.
_FACTORS_M2 = {"m2": Fraction(1), "ha": Fraction(10_000), "km2": Fraction(1_000_000)}
_UNIT_LABELS = {"m2": "m²", "ha": "ha", "km2": "km²"}

_FACTOR_M2_HA = _FACTORS_M2["ha"]  # 10,000
_FACTOR_KM2_M2 = _FACTORS_M2["km2"]  # 1,000,000
_FACTOR_HA_KM2 = Fraction(_FACTORS_M2["km2"], _FACTORS_M2["ha"])  # 100, derived


def _same_shape_candidates(answer: Fraction, answer_text: str, misconceptions: list) -> list[str]:
    """Distractor strings that all show the key's own number of decimal places.

    The two named misconceptions come first; a wrong-factor value that cannot
    be shown at that many places (0.02 beside a whole-number key) is dropped
    rather than rounded, since showing it differently from the key gives the
    key away. The decimal-point-slipped-by-one-or-two-places values (a real
    place-value slip with these powers-of-ten factors) and a plain off-by-one
    follow so every draw still has distractors."""
    places = len(answer_text.partition(".")[2])
    pool = list(misconceptions) + [answer * 10, answer / 10, answer * 100, answer / 100, answer + 1, answer - 1]
    out = []
    for value in pool:
        scaled = Fraction(value) * 10**places
        if scaled.denominator == 1 and scaled >= 0:
            out.append(fixed_point_string(scaled.numerator, places))
    return out


def _build_m2_ha(rng, ha_hi: int) -> dict:
    """Easy tier: single-step m^2<->ha conversion, always an exact result
    (the ha figure is chosen first, as an integer, and the m^2 figure is
    derived as an exact multiple of it)."""
    direction = rng.choice(["m2_to_ha", "ha_to_m2"])
    ha_value = Fraction(rng.randint(1, ha_hi))
    m2_value = ha_value * _FACTOR_M2_HA

    if direction == "m2_to_ha":
        source_value, source_unit, target_unit = m2_value, "m2", "ha"
        answer = ha_value
        direction_wrong = source_value * _FACTOR_M2_HA  # multiplies instead of dividing
        wrong_factor = source_value / _FACTOR_KM2_M2  # 1,000,000 instead of 10,000
    else:
        source_value, source_unit, target_unit = ha_value, "ha", "m2"
        answer = m2_value
        direction_wrong = source_value / _FACTOR_M2_HA  # divides instead of multiplying
        wrong_factor = source_value * _FACTOR_KM2_M2  # 1,000,000 instead of 10,000

    check_conversion(source_value, source_unit, target_unit, answer, _FACTORS_M2, label="conversion")

    answer_text = exact_decimal_string(answer, 3)
    source_text = exact_decimal_string(source_value, 3)
    text = f"Convert {source_text} {_UNIT_LABELS[source_unit]} to {_UNIT_LABELS[target_unit]}."
    candidates = _same_shape_candidates(answer, answer_text, [wrong_factor, direction_wrong])

    return dict(
        text=text,
        latex=rf"{source_text}\,\mathrm{{{_UNIT_LABELS[source_unit]}}}=\square\,\mathrm{{{_UNIT_LABELS[target_unit]}}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"1 ha = 10,000 m², so {source_text} {_UNIT_LABELS[source_unit]} = "
            f"{answer_text} {_UNIT_LABELS[target_unit]}."
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_ha_km2(rng, ha_lo: int, ha_hi: int, km2_numerator_lo: int, km2_numerator_hi: int) -> dict:
    """Medium tier: single-step ha<->km^2 conversion (factor 100, derived
    from the km^2 and ha entries of `_FACTORS_M2`, not the m^2<->ha factor
    used at easy) -- decimal results routinely, always terminating within 3
    decimal places by construction."""
    direction = rng.choice(["ha_to_km2", "km2_to_ha"])

    if direction == "ha_to_km2":
        ha_value = Fraction(rng.randint(ha_lo, ha_hi))
        while ha_value % 100 == 0:  # avoid a coincidentally whole km2 result most of the time
            ha_value = Fraction(rng.randint(ha_lo, ha_hi))
        source_value, source_unit, target_unit = ha_value, "ha", "km2"
        answer = ha_value / _FACTOR_HA_KM2
        direction_wrong = source_value * _FACTOR_HA_KM2
        wrong_factor = source_value / _FACTOR_M2_HA
    else:
        km2_value = Fraction(rng.randint(km2_numerator_lo, km2_numerator_hi), 1000)
        source_value, source_unit, target_unit = km2_value, "km2", "ha"
        answer = km2_value * _FACTOR_HA_KM2
        direction_wrong = source_value / _FACTOR_HA_KM2
        wrong_factor = source_value * _FACTOR_M2_HA

    check_conversion(source_value, source_unit, target_unit, answer, _FACTORS_M2, label="conversion")

    answer_text = exact_decimal_string(answer, 3)
    source_text = exact_decimal_string(source_value, 3)
    text = f"Convert {source_text} {_UNIT_LABELS[source_unit]} to {_UNIT_LABELS[target_unit]}."
    candidates = _same_shape_candidates(answer, answer_text, [wrong_factor, direction_wrong])

    return dict(
        text=text,
        latex=rf"{source_text}\,\mathrm{{{_UNIT_LABELS[source_unit]}}}=\square\,\mathrm{{{_UNIT_LABELS[target_unit]}}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"1 km² = 100 ha, so {source_text} {_UNIT_LABELS[source_unit]} = {answer_text} {_UNIT_LABELS[target_unit]}."
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_combine(
    rng,
    m2_multiple: int,
    m2_count_hi: int,
    ha_hi_small: int,
    ha_lo_large: int,
    ha_hi_large: int,
    km2_numerator_lo: int,
    km2_numerator_hi: int,
) -> dict:
    """Hard tier: a mixed-unit combination, never a chained conversion. The
    two stated units are always ``m2``/``ha`` or ``ha``/``km2`` -- ``ha`` is
    always one of the two -- so this never needs, and never builds, a direct
    m^2<->km^2 conversion (that single-hop shape belongs to the plain-ladder
    sibling skill, see module docstring)."""
    pair = rng.choice([("m2", "ha"), ("ha", "km2")])
    operation = rng.choice(["sum", "difference"])

    if pair == ("m2", "ha"):
        m2_value = Fraction(rng.randint(2, m2_count_hi) * m2_multiple)
        ha_value = Fraction(rng.randint(1, ha_hi_small))
        values = {"m2": m2_value, "ha": ha_value}
    else:
        ha_value = Fraction(rng.randint(ha_lo_large, ha_hi_large))
        km2_value = Fraction(rng.randint(km2_numerator_lo, km2_numerator_hi), 1000)
        values = {"ha": ha_value, "km2": km2_value}

    unit_a, unit_b = pair
    target_unit = rng.choice([unit_a, unit_b])
    other_unit = unit_b if target_unit == unit_a else unit_a

    converted_other = values[other_unit] * Fraction(_FACTORS_M2[other_unit], _FACTORS_M2[target_unit])
    same_value = values[target_unit]
    if operation == "sum":
        answer = same_value + converted_other
    else:
        answer = abs(same_value - converted_other)

    check_combine(values, other_unit, target_unit, operation, answer, _FACTORS_M2, label="combine")

    answer_text = exact_decimal_string(answer, 3)
    label_a, label_b = _UNIT_LABELS[unit_a], _UNIT_LABELS[unit_b]
    text_a, text_b = exact_decimal_string(values[unit_a], 3), exact_decimal_string(values[unit_b], 3)
    target_label = _UNIT_LABELS[target_unit]

    if operation == "sum":
        question_text = (
            f"Paddock A is {text_a} {label_a} and Paddock B is {text_b} {label_b}. "
            f"What is their combined area, in {target_label}?"
        )
    else:
        question_text = (
            f"Paddock A is {text_a} {label_a} and Paddock B is {text_b} {label_b}. "
            f"What is the difference in area between them, in {target_label}?"
        )

    # Misconceptions:
    #  - forgets-to-convert: combines the two raw stated values directly,
    #    ignoring the unit mismatch (always wildly wrong given the
    #    10,000x/100x factors in this unit family).
    #  - converts-wrong-operand: converts the operand that was *already* in
    #    the target unit (using the same-to-other ratio) and leaves the
    #    operand that actually needed converting untouched.
    forgets_to_convert = values[unit_a] + values[unit_b] if operation == "sum" else abs(values[unit_a] - values[unit_b])
    ratio_target_to_other = Fraction(_FACTORS_M2[target_unit], _FACTORS_M2[other_unit])
    wrongly_converted_same = same_value * ratio_target_to_other
    other_raw = values[other_unit]
    converts_wrong_operand = (
        wrongly_converted_same + other_raw if operation == "sum" else abs(wrongly_converted_same - other_raw)
    )

    candidates = _same_shape_candidates(answer, answer_text, [forgets_to_convert, converts_wrong_operand])

    return dict(
        text=question_text,
        latex=rf"{text_a}\,\mathrm{{{label_a}}}\;\square\;{text_b}\,\mathrm{{{label_b}}}\to\mathrm{{{target_label}}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Convert whichever value is not already in {target_label}, then "
            f"{'add' if operation == 'sum' else 'subtract'} the two amounts: "
            f"{answer_text} {target_label}.",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "convert_between_metric_units_of_area",
    level=11,
    presets={
        "easy": dict(mode="m2_ha"),
        "medium": dict(mode="ha_km2"),
        "hard": dict(mode="combine"),
    },
)
def convert_between_metric_units_of_area(
    rng,
    mode="m2_ha",
    ha_hi=50,
    ha_lo=11,
    ha_hi_medium=999,
    km2_numerator_lo=1000,
    km2_numerator_hi=49999,
    combine_m2_multiple=500,
    combine_m2_count_hi=40,
    combine_ha_hi_small=30,
    combine_ha_lo_large=11,
    combine_ha_hi_large=999,
):
    """Convert an area measurement within the hectare family (m^2<->ha,
    ha<->km^2); see module docstring for the boundary against the plain
    metric-prefix ladder sibling skill."""
    if mode == "m2_ha":
        return _build_m2_ha(rng, ha_hi)
    if mode == "ha_km2":
        return _build_ha_km2(rng, ha_lo, ha_hi_medium, km2_numerator_lo, km2_numerator_hi)
    if mode == "combine":
        return _build_combine(
            rng,
            combine_m2_multiple,
            combine_m2_count_hi,
            combine_ha_hi_small,
            combine_ha_lo_large,
            combine_ha_hi_large,
            km2_numerator_lo,
            km2_numerator_hi,
        )
    raise ValueError(f"unknown mode: {mode!r}")


# --- calculate_perimeters_of_familiar_shapes --------------------------------


def _verify_straight_edged(sides: list[int], expected: int) -> None:
    """Independent check: sum the sides in reverse order, accumulating one
    at a time -- a different grouping from `sum(sides)`, never a re-run of
    whatever loop/builtin construction used."""
    total = 0
    for side in reversed(sides):
        total += side
    check(total == expected, "perimeter mismatch (straight-edged sum)")


def _verify_composite_perimeter(width: int, height: int, slant: int, expected: int) -> None:
    """Independent check for the hard-tier composite: derive the perimeter
    from each sub-shape's own *full* perimeter minus twice the shared edge
    -- a distinctly different grouping from construction's own direct sum
    of the five boundary edges."""
    rectangle_perimeter = 2 * (width + height)
    triangle_perimeter = height + 2 * slant
    shared_edge = height
    recomputed = rectangle_perimeter + triangle_perimeter - 2 * shared_edge
    check(recomputed == expected, "composite perimeter mismatch")


def _perimeter_distractors(sides: list[int], answer_text: str) -> list[str]:
    """Shared misconception candidates for the easy/medium tiers:
    - uses-area-formula: multiplies the first two given side lengths
      together instead of summing every side (for a rectangle/square this
      is exactly length*width or side*side, the real area).
    - omits-a-side: drops the first given side from the sum.
    - a doubled-total safety-net candidate (not itself a named contract
      misconception -- included only to keep the >=2-distractor floor in
      the rare case the first two candidates coincide, e.g. a 3 cm square
      where 3*3 == 3*3/043 precedent for a floor-guaranteeing
      extra candidate)."""
    total = sum(sides)
    omits_a_side = total - sides[0]
    uses_area_formula = sides[0] * sides[1] if len(sides) >= 2 else sides[0] * sides[0]
    doubled_total = total * 2
    candidates = [str(uses_area_formula), str(omits_a_side), str(doubled_total)]
    return pick_distractors(answer_text, candidates)


def _build_easy_perimeter(rng, side_hi: int) -> dict:
    """Easy: rectangle or square perimeter."""
    shape = rng.choice(["rectangle", "square"])
    if shape == "square":
        side = rng.randint(2, side_hi)
        sides = [side, side, side, side]
        text = f"A square has sides of length {side} cm. What is its perimeter, in cm?"
        working = [f"Perimeter = 4 x side = 4 x {side} = {sum(sides)} cm."]
    else:
        length = rng.randint(2, side_hi)
        width = rng.randint(2, side_hi)
        while width == length:
            width = rng.randint(2, side_hi)
        sides = [length, width, length, width]
        text = f"A rectangle is {length} cm long and {width} cm wide. What is its perimeter, in cm?"
        working = [f"Perimeter = 2 x (length + width) = 2 x ({length} + {width}) = {sum(sides)} cm."]

    answer = sum(sides)
    _verify_straight_edged(sides, answer)
    answer_text = str(answer)

    return dict(
        text=text,
        latex=rf"\text{{Perimeter of the shape with sides }}{sides}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=_perimeter_distractors(sides, answer_text),
    )


def _sample_triangle(rng, hi: int) -> list[int]:
    while True:
        a, b, c = sorted(rng.randint(3, hi) for _ in range(3))
        if a + b > c:
            return [a, b, c]


def _build_medium_perimeter(rng, poly_side_hi: int) -> dict:
    """Medium: triangle or irregular polygon (3+ independently-stated side
    lengths, summed directly -- a genuinely different reasoning shape from
    easy's "double two given sides", since however many sides are given must
    all be summed, not applied to a fixed formula)."""
    shape = rng.choice(["triangle", "polygon"])
    if shape == "triangle":
        sides = _sample_triangle(rng, poly_side_hi)
        shape_name = "triangle"
    else:
        k = rng.choice([4, 5, 6])
        sides = [rng.randint(2, poly_side_hi) for _ in range(k)]
        shape_name = "polygon"

    answer = sum(sides)
    _verify_straight_edged(sides, answer)
    answer_text = str(answer)
    sides_text = ", ".join(f"{s} cm" for s in sides)
    text = f"A {shape_name} has sides measuring {sides_text}. What is its perimeter, in cm?"

    return dict(
        text=text,
        latex=rf"\text{{Perimeter of the {shape_name} with sides }}{sides}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Sum every given side: {' + '.join(str(s) for s in sides)} = {answer_text} cm."],
        distractors=_perimeter_distractors(sides, answer_text),
    )


_TRIPLES = [(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17), (9, 12, 15), (12, 16, 20)]


def _build_hard_perimeter(
    rng,
    width_lo: int,
    width_hi: int,
    height_lo: int,
    height_hi: int,
    slant_lo: int,
    slant_hi: int,
) -> dict:
    """Hard: a straight-edged composite (a rectangle with a triangular
    point attached to one end, sharing that end's full edge) where the
    shared/internal edge must be identified and excluded from the perimeter
    sum -- a genuine reasoning escalation over medium's plain "sum every
    given side" (which side lengths are given directly is not enough here;
    one given length, the rectangle's end, is explicitly internal and must
    not appear in the total at all)."""
    height = rng.randint(height_lo, height_hi)
    slant = rng.randint(slant_lo, slant_hi)
    while 2 * slant <= height:  # keep the stated triangle geometrically valid
        slant = rng.randint(slant_lo, slant_hi)
    width = rng.randint(width_lo, width_hi)

    answer = 2 * width + height + 2 * slant
    _verify_composite_perimeter(width, height, slant, answer)
    answer_text = str(answer)

    text = (
        f"A sign is shaped like a rectangle {width} cm wide by {height} cm tall, with a "
        f"triangular point attached to its right-hand edge. The whole {height} cm "
        f"right-hand edge of the rectangle is the triangle's base, and each of the "
        f"triangle's two slanted sides is {slant} cm long. What is the total perimeter "
        f"of the sign, in cm?"
    )

    # Misconceptions:
    #  - double-counts-shared-edge: adds the rectangle's own full perimeter
    #    to the triangle's own full perimeter, counting the shared edge
    #    (the rectangle's right-hand side, height) twice.
    #  - omits-a-side: drops the rectangle's width from the sum.
    #  - a width*height area-style filler, guaranteeing the >=2-distractor
    #    floor if the first two coincide (not itself a named misconception).
    double_counts_shared_edge = 2 * (width + height) + (height + 2 * slant)
    omits_a_side = answer - width
    area_style_filler = width * height
    candidates = [str(double_counts_shared_edge), str(omits_a_side), str(area_style_filler)]

    return dict(
        text=text,
        latex=rf"\text{{Perimeter of a }}{width}\times{height}\text{{ rectangle with a triangular point}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            "The rectangle's right-hand edge is shared with the triangle's base, so it is "
            "internal and is not part of the outer perimeter.",
            f"Outer edges: {width} (bottom) + {slant} + {slant} (slanted sides) + {width} (top) + "
            f"{height} (left) = {answer_text} cm.",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "calculate_perimeters_of_familiar_shapes",
    level=11,
    presets={
        "easy": dict(mode="rectangle_or_square"),
        "medium": dict(mode="triangle_or_polygon"),
        "hard": dict(mode="composite"),
    },
)
def calculate_perimeters_of_familiar_shapes(
    rng,
    mode="rectangle_or_square",
    side_hi=30,
    poly_side_hi=25,
    width_lo=4,
    width_hi=20,
    height_lo=4,
    height_hi=16,
    slant_lo=3,
    slant_hi=15,
):
    """Calculate the perimeter of a rectangle, square, triangle, irregular
    polygon, or a straight-edged composite of these; see module docstring
    for the option-A (straight-edged only) design choice."""
    if mode == "rectangle_or_square":
        return _build_easy_perimeter(rng, side_hi)
    if mode == "triangle_or_polygon":
        return _build_medium_perimeter(rng, poly_side_hi)
    if mode == "composite":
        return _build_hard_perimeter(rng, width_lo, width_hi, height_lo, height_hi, slant_lo, slant_hi)
    raise ValueError(f"unknown mode: {mode!r}")
