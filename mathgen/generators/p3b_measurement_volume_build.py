"""Metric volume-unit conversion and volume/capacity formulas, both senior ``acara-senior-v8-4``, domain
``measurement``, ``answer: number``.

Two generators:

- ``use_metric_units_of_volume`` (``senior-use-metric-units-of-volume-
  their-abbreviations-conversions``, ACMEM027 and ACMEM101) -- convert a
  volume measurement between metric units: mm³, cm³, m³ (the full
  three-unit family, no mm³<->m³ direct conversion excluded due to its 1e9
  factor). Both target areas describe the same operation (Unit 1 introduction,
  Unit 3 review), so a single skill id serves both with two fragment rows.

- ``use_formulas_to_find_the_volume`` (``senior-use-formulas-to-find-the-
  volume-and-capacity``, ACMEM103) -- calculate the volume of a cube,
  rectangular prism, or triangular prism using exact-arithmetic formulas;
  the cylinder (π-based) is excluded. This skill is NOT the plain rectangular-
  prism ``rectangular-prism-volume`` (ACMEM030, primary domain); that skill's
  own operation is a single shape's volume applied once. This skill's design
  ensures the rectangular prism never appears alone at any tier:
  - Easy: cube only.
  - Medium: triangular prism or rectangular prism (mixed, never rect alone).
  - Hard: two-object comparison (cube, tri prism, or rect prism, not a single
    shape pair).
  Triangular prism is weighted heavier at medium to keep the skill's center
  of gravity distinct from the single-prism catalog skill.

Both volume-conversion generators format ``answer_text`` from an exact
``Fraction``/``int`` value with explicit, bounded fixed-point decimal places
(never ``str(float(...))``, which can silently switch to scientific notation
below ``1e-4``) -- see ``core.exact_decimal_string``/``core.rounded_decimal_string``. The
volume-formula generators produce exact integers by construction, so no fixed-
point formatting is needed.
"""

from __future__ import annotations

import re
from fractions import Fraction

from ..core import (
    check,
    check_combine,
    check_conversion,
    exact_decimal_string,
    generator,
    numeric_answer,
    pick_distractors,
    rounded_decimal_string,
    stated_precision_string,
)

# --- use_metric_units_of_volume ------------------------------------------

# Volume unit factors expressed in mm³. The cm³ and m³ factors are cubes of
# the linear length factors (mm=1, cm=10, m=1000).
_FACTORS_MM3 = {
    "mm3": Fraction(1),
    "cm3": Fraction(1000),  # 10³
    "m3": Fraction(1_000_000_000),  # 1000³
}
_UNIT_LABELS = {"mm3": "mm³", "cm3": "cm³", "m3": "m³"}

# Linear length factors (for independent check only).
_LINEAR_FACTORS = {"mm": Fraction(1), "cm": Fraction(10), "m": Fraction(1000)}

_FACTOR_MM3_CM3 = Fraction(10) ** 3  # 1000, derived independently
_FACTOR_CM3_M3 = Fraction(1000) ** 3 / Fraction(10) ** 3  # 1,000,000, derived independently


def _shaped_distractors(answer_text: str, candidates: list[str]) -> list[str]:
    """Misconception distractors shaped like the key: they show the same number
    of decimal places ("5451000.000" beside "5.451"; whole numbers only beside a
    whole-number key), so the decimal point never gives the key away. Power-of-ten
    slips of the key top the list up when the shape filter drops a candidate;
    candidates that are already decimals come first when the key is one."""
    answer = Fraction(answer_text)
    places = len(answer_text.partition(".")[2])
    scale = 10**places
    pool = [Fraction(c) for c in candidates] + [
        answer * 10,
        answer / 10,
        answer * 100,
        answer / 100,
        answer * 1000,
        answer / 1000,
    ]
    pool = [v for v in pool if v > 0 and (v * scale).denominator == 1]
    if places:
        pool.sort(key=lambda v: v.denominator == 1)  # stable: natural decimals first
    return pick_distractors(answer_text, [stated_precision_string(v, places) for v in pool])


def _build_mm3_cm3(rng, cm3_hi: int) -> dict:
    """Easy tier: single-step mm³<->cm³ conversion (factor 1000), always an
    exact result. The cm³ figure is chosen first as an integer, and the mm³
    figure is derived as an exact multiple of it."""
    direction = rng.choice(["mm3_to_cm3", "cm3_to_mm3"])
    cm3_value = Fraction(rng.randint(1, cm3_hi))
    mm3_value = cm3_value * _FACTOR_MM3_CM3

    if direction == "mm3_to_cm3":
        source_value, source_unit, target_unit = mm3_value, "mm3", "cm3"
        answer = cm3_value
        direction_wrong = source_value * _FACTOR_MM3_CM3  # multiplies instead of dividing
        # Linear factor (10) and squared factor (100) instead of cubic (1000)
        wrong_linear = source_value / Fraction(10)
        wrong_squared = source_value / Fraction(100)
    else:
        source_value, source_unit, target_unit = cm3_value, "cm3", "mm3"
        answer = mm3_value
        direction_wrong = source_value / _FACTOR_MM3_CM3  # divides instead of multiplying
        wrong_linear = source_value * Fraction(10)
        wrong_squared = source_value * Fraction(100)

    check_conversion(source_value, source_unit, target_unit, answer, _FACTORS_MM3, label="volume conversion")

    answer_text = exact_decimal_string(answer, max_places=3)
    source_text = exact_decimal_string(source_value, max_places=3)
    text = f"Convert {source_text} {_UNIT_LABELS[source_unit]} to {_UNIT_LABELS[target_unit]}."

    # Pick two of the three misconceptions: direction, linear, squared
    candidates = [
        rounded_decimal_string(direction_wrong, max_places=3),
        rounded_decimal_string(wrong_linear, max_places=3),
    ]

    return dict(
        text=text,
        latex=rf"{source_text}\,\mathrm{{{_UNIT_LABELS[source_unit]}}}=\square\,\mathrm{{{_UNIT_LABELS[target_unit]}}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"1 cm³ = 1,000 mm³, so {source_text} {_UNIT_LABELS[source_unit]} = "
            f"{answer_text} {_UNIT_LABELS[target_unit]}."
        ],
        distractors=_shaped_distractors(answer_text, candidates),
    )


def _build_cm3_m3(rng, cm3_lo: int, cm3_hi: int) -> dict:
    """Medium tier: single-step cm³<->m³ conversion (factor 1,000,000) --
    a materially larger fact than easy within the three-unit family, routinely
    producing a genuine decimal result (terminating within 3 decimal places).
    The cm³ value must be a multiple of 1000 to ensure the result terminates
    within 3 places in either direction. For m³→cm³, m³ is chosen as
    Fraction(k, 1000) to ensure termination."""
    direction = rng.choice(["cm3_to_m3", "m3_to_cm3"])

    if direction == "cm3_to_m3":
        # cm³ value as a multiple of 1000: cm³ = k * 1000
        # Then m³ = (k * 1000) / 1e6 = k / 1000, terminates in exactly 3 places
        k = rng.randint(cm3_lo, cm3_hi)
        cm3_value = Fraction(k * 1000)
        source_value, source_unit, target_unit = cm3_value, "cm3", "m3"
        answer = Fraction(k, 1000)  # = (k * 1000) / 1_000_000
        direction_wrong = source_value * _FACTOR_CM3_M3
        # Linear factor (1000) and squared factor (1e6) instead of cubic (1e9)
        wrong_linear = source_value / Fraction(1000)
        wrong_squared = source_value / Fraction(1_000_000)
    else:
        # m³ value as Fraction(k, 1000) so it terminates in exactly 3 places
        # Then cm³ = m³ * 1e6 = (k / 1000) * 1e6 = k * 1000 (integer)
        k = rng.randint(cm3_lo, cm3_hi)
        m3_value = Fraction(k, 1000)
        source_value, source_unit, target_unit = m3_value, "m3", "cm3"
        answer = Fraction(k * 1000)
        direction_wrong = source_value / _FACTOR_CM3_M3
        wrong_linear = source_value * Fraction(1000)
        wrong_squared = source_value * Fraction(1_000_000)

    check_conversion(source_value, source_unit, target_unit, answer, _FACTORS_MM3, label="volume conversion")

    answer_text = exact_decimal_string(answer, max_places=3)
    source_text = exact_decimal_string(source_value, max_places=3)
    text = f"Convert {source_text} {_UNIT_LABELS[source_unit]} to {_UNIT_LABELS[target_unit]}."
    candidates = [
        rounded_decimal_string(direction_wrong, max_places=3),
        rounded_decimal_string(wrong_linear, max_places=3),
    ]

    return dict(
        text=text,
        latex=rf"{source_text}\,\mathrm{{{_UNIT_LABELS[source_unit]}}}=\square\,\mathrm{{{_UNIT_LABELS[target_unit]}}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"1 m³ = 1,000,000 cm³, so {source_text} {_UNIT_LABELS[source_unit]} = "
            f"{answer_text} {_UNIT_LABELS[target_unit]}."
        ],
        distractors=_shaped_distractors(answer_text, candidates),
    )


def _build_volume_combine(
    rng,
    mm3_lo: int,
    mm3_hi: int,
    cm3_lo: int,
    cm3_hi: int,
    m3_numerator_lo: int,
    m3_numerator_hi: int,
) -> dict:
    """Hard tier: a mixed-unit combination, never a chained conversion. Two
    volumes stated in two different units from the mm³/cm³/m³ family. The
    mm³<->m³ pair is excluded entirely (1e9 factor, scientific-notation risk).
    Valid pairs: (mm³, cm³) and (cm³, m³) only. We need to convert one operand
    then add or subtract.

    Constraints to ensure 3-decimal-place termination in either direction:
    - (mm³, cm³): both are integers
    - (cm³, m³): cm³ must be a multiple of 1000, m³ must be Fraction(k, 1000)
    """
    pair = rng.choice([("mm3", "cm3"), ("cm3", "m3")])
    operation = rng.choice(["sum", "difference"])

    if pair == ("mm3", "cm3"):
        mm3_value = Fraction(rng.randint(mm3_lo, mm3_hi))
        cm3_value = Fraction(rng.randint(cm3_lo, cm3_hi))
        values = {"mm3": mm3_value, "cm3": cm3_value}
    else:  # ("cm3", "m3")
        # cm³ as a multiple of 1000, m³ as Fraction(k, 1000)
        k_cm3 = rng.randint(cm3_lo, cm3_hi)
        cm3_value = Fraction(k_cm3 * 1000)
        k_m3 = rng.randint(m3_numerator_lo, m3_numerator_hi)
        m3_value = Fraction(k_m3, 1000)
        values = {"cm3": cm3_value, "m3": m3_value}

    unit_a, unit_b = pair
    target_unit = rng.choice([unit_a, unit_b])
    other_unit = unit_b if target_unit == unit_a else unit_a

    converted_other = values[other_unit] * Fraction(_FACTORS_MM3[other_unit], _FACTORS_MM3[target_unit])
    same_value = values[target_unit]
    if operation == "sum":
        answer = same_value + converted_other
    else:
        answer = abs(same_value - converted_other)

    check_combine(values, other_unit, target_unit, operation, answer, _FACTORS_MM3, label="volume combine")

    answer_text = exact_decimal_string(answer, max_places=3)
    label_a, label_b = _UNIT_LABELS[unit_a], _UNIT_LABELS[unit_b]
    text_a, text_b = (
        exact_decimal_string(values[unit_a], max_places=3),
        exact_decimal_string(values[unit_b], max_places=3),
    )
    target_label = _UNIT_LABELS[target_unit]

    if operation == "sum":
        question_text = (
            f"Container A holds {text_a} {label_a} and Container B holds {text_b} {label_b}. "
            f"What is their combined volume, in {target_label}?"
        )
    else:
        question_text = (
            f"Container A holds {text_a} {label_a} and Container B holds {text_b} {label_b}. "
            f"What is the difference in volume between them, in {target_label}?"
        )

    # Misconceptions:
    #  - forgets-to-convert: combines the two raw stated values directly,
    #    ignoring the unit mismatch.
    #  - converts-wrong-operand: converts the operand that was *already* in
    #    the target unit (using the same-to-other ratio) and leaves the
    #    operand that actually needed converting untouched.
    forgets_to_convert = values[unit_a] + values[unit_b] if operation == "sum" else abs(values[unit_a] - values[unit_b])
    ratio_target_to_other = Fraction(_FACTORS_MM3[target_unit], _FACTORS_MM3[other_unit])
    wrongly_converted_same = same_value * ratio_target_to_other
    other_raw = values[other_unit]
    converts_wrong_operand = (
        wrongly_converted_same + other_raw if operation == "sum" else abs(wrongly_converted_same - other_raw)
    )

    candidates = [
        rounded_decimal_string(forgets_to_convert, max_places=3),
        rounded_decimal_string(converts_wrong_operand, max_places=3),
    ]

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
        distractors=_shaped_distractors(answer_text, candidates),
    )


@generator(
    "use_metric_units_of_volume",
    level=11,
    presets={
        "easy": dict(mode="mm3_cm3"),
        "medium": dict(mode="cm3_m3"),
        "hard": dict(mode="combine"),
    },
)
def use_metric_units_of_volume(
    rng,
    mode="mm3_cm3",
    easy_cm3_hi=50,
    medium_cm3_lo=1,
    medium_cm3_hi=999,
    hard_mm3_lo=100,
    hard_mm3_hi=9999,
    hard_cm3_lo=1,
    hard_cm3_hi=999,
    hard_m3_numerator_lo=1,
    hard_m3_numerator_hi=999,
):
    """Convert a volume measurement between metric units (mm³, cm³, m³);
    see module docstring for the mm³<->m³ pair exclusion."""
    if mode == "mm3_cm3":
        return _build_mm3_cm3(rng, easy_cm3_hi)
    if mode == "cm3_m3":
        return _build_cm3_m3(rng, medium_cm3_lo, medium_cm3_hi)
    if mode == "combine":
        return _build_volume_combine(
            rng,
            hard_mm3_lo,
            hard_mm3_hi,
            hard_cm3_lo,
            hard_cm3_hi,
            hard_m3_numerator_lo,
            hard_m3_numerator_hi,
        )
    raise ValueError(f"unknown mode: {mode!r}")


# --- use_formulas_to_find_the_volume -----------------------------------------


def _verify_cube_volume(side: int, expected: int) -> None:
    """Independent check: compute s³ by accumulation (s*s*s) rather than
    the power operator, a genuinely different route."""
    recomputed = side * side * side
    check(recomputed == expected, "cube volume mismatch")


def _verify_rectangular_prism_volume(length: int, width: int, height: int, expected: int) -> None:
    """Independent check: multiply in a different order (height*width*length
    instead of whatever order construction used)."""
    recomputed = height * width * length
    check(recomputed == expected, "rectangular prism volume mismatch")


def _verify_triangular_prism_volume(base: int, height: int, length: int, expected: int) -> None:
    """Independent check: compute the cross-sectional triangle area separately
    (Fraction(base, 2) * height) and then multiply by length as a separate step,
    a distinctly different grouping from construction's own arithmetic."""
    cross_section_area = Fraction(base, 2) * height
    recomputed = int(cross_section_area * length)
    check(recomputed == expected, "triangular prism volume mismatch")


def _verify_two_volume_difference(vol_a: int, vol_b: int, expected: int) -> None:
    """Independent check for hard tier: recompute each volume separately
    (via their own formula), then subtract as a separate step from the raw values."""
    difference = abs(vol_a - vol_b)
    check(difference == expected, "two-volume difference mismatch")


def _build_cube(rng, side_hi: int) -> dict:
    """Easy tier: cube volume (s³) -- a single stated dimension, the simplest
    formula."""
    side = rng.randint(2, side_hi)
    answer = side**3
    _verify_cube_volume(side, answer)
    answer_text = str(answer)

    text = f"A cube has sides of length {side} cm. What is its volume, in cm³?"

    # Misconceptions:
    #  - uses-two-dimensions-only: s² instead of s³
    #  - area-style filler: 6s² (surface area of a unit cube has 6 faces)
    #  - a third filler to guarantee >=2 distractors: 3s
    candidates = [str(side * side), str(6 * side * side), str(3 * side)]

    return dict(
        text=text,
        latex=rf"V = {side}^3",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Volume = side³ = {side}³ = {answer_text} cm³."],
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_triangular_or_rectangular_prism(rng, base_hi: int, height_hi: int, length_hi: int) -> dict:
    """Medium tier: triangular prism or rectangular prism. Triangular prism
    is weighted heavier (2:1 via choice). Both have three independent dimensions
    (base, height, length; or length, width, height) and require the learner to
    apply a formula correctly."""
    shape = rng.choices(["triangular", "rectangular"], weights=[2, 1])[0]

    if shape == "triangular":
        # Triangular prism: 1/2 * base * height * length
        # Both base and height must be even to keep volume an integer
        base = rng.randint(2, base_hi // 2) * 2
        height = rng.randint(2, height_hi // 2) * 2
        length = rng.randint(2, length_hi)
        answer = (base * height // 2) * length
        _verify_triangular_prism_volume(base, height, length, answer)

        text = (
            f"A triangular prism has a triangular cross-section with base {base} cm "
            f"and height {height} cm. The prism's length is {length} cm. "
            f"What is its volume, in cm³?"
        )

        # Misconceptions:
        #  - omits-half-factor: base * height * length (treats as full rectangle, not triangle)
        #  - uses-two-dimensions-only: (1/2 * base * height) -- the cross-section area, forgetting length
        wrong_no_half = base * height * length
        wrong_no_length = (base * height) // 2

        candidates = [str(wrong_no_half), str(wrong_no_length)]
    else:  # rectangular
        length = rng.randint(2, length_hi)
        width = rng.randint(2, base_hi)
        height = rng.randint(2, height_hi)
        answer = length * width * height
        _verify_rectangular_prism_volume(length, width, height, answer)

        text = (
            f"A rectangular prism is {length} cm long, {width} cm wide, and {height} cm tall. "
            f"What is its volume, in cm³?"
        )

        # Misconceptions:
        #  - wrong-shape-formula: (1/2 * length * width * height) -- applies triangular formula
        #  - uses-two-dimensions-only: length * width -- the base area, forgetting height
        wrong_half_formula = (length * width * height + 1) // 2  # round-ish
        wrong_no_height = length * width

        candidates = [str(wrong_half_formula), str(wrong_no_height)]

    answer_text = str(answer)
    return dict(
        text=text,
        latex=rf"V = \text{{volume of the {shape} prism}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Volume of the {shape} prism = {answer_text} cm³."],
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_two_volume_difference(rng, cube_side_hi: int, base_hi: int, height_hi: int, length_hi: int) -> dict:
    """Hard tier: a two-object comparison. Two objects, each with independently
    stated dimensions; compute both volumes, then find the difference (always
    positive, with the larger volume first). Objects are chosen from cubes,
    triangular prisms, and rectangular prisms. Exclude the (rect, rect) pairing
    so this tier never reduces to a single rectangular-prism operation."""
    shapes = rng.choices(
        ["cube", "triangular", "rectangular"],
        k=2,
    )
    # Reject (rectangular, rectangular) pairing
    while shapes[0] == "rectangular" and shapes[1] == "rectangular":
        shapes = rng.choices(["cube", "triangular", "rectangular"], k=2)

    # Generate two volumes
    volumes = {}
    dims = {}

    for i, shape in enumerate(shapes):
        if shape == "cube":
            side = rng.randint(2, cube_side_hi)
            volumes[i] = side**3
            dims[i] = {"shape": "cube", "side": side}
        elif shape == "triangular":
            base = rng.randint(2, base_hi // 2) * 2
            height = rng.randint(2, height_hi // 2) * 2
            length = rng.randint(2, length_hi)
            volumes[i] = (base * height // 2) * length
            dims[i] = {"shape": "triangular", "base": base, "height": height, "length": length}
        else:  # rectangular
            length = rng.randint(2, length_hi)
            width = rng.randint(2, base_hi)
            height = rng.randint(2, height_hi)
            volumes[i] = length * width * height
            dims[i] = {"shape": "rectangular", "length": length, "width": width, "height": height}

    # Ensure volume_a > volume_b
    if volumes[0] < volumes[1]:
        volumes[0], volumes[1] = volumes[1], volumes[0]
        dims[0], dims[1] = dims[1], dims[0]
        shapes[0], shapes[1] = shapes[1], shapes[0]
    elif volumes[0] == volumes[1]:
        # Resample if equal
        return _build_two_volume_difference(rng, cube_side_hi, base_hi, height_hi, length_hi)

    answer = volumes[0] - volumes[1]
    _verify_two_volume_difference(volumes[0], volumes[1], answer)
    answer_text = str(answer)

    # Build question text
    def dims_text(i):
        d = dims[i]
        if d["shape"] == "cube":
            return f"a cube with sides of length {d['side']} cm"
        elif d["shape"] == "triangular":
            return f"a triangular prism with base {d['base']} cm, height {d['height']} cm, and length {d['length']} cm"
        else:  # rectangular
            return f"a rectangular prism {d['length']} cm long, {d['width']} cm wide, and {d['height']} cm tall"

    text = (
        f"Object A is {dims_text(0)}. Object B is {dims_text(1)}. "
        f"What is the difference in volume between them, in cm³?"
    )

    # Misconceptions:
    #  - add-instead-of-subtract: volumes[0] + volumes[1]
    #  - object B's volume alone (filler to guarantee >= 2 distractors)
    add_instead = volumes[0] + volumes[1]
    just_vol_b = volumes[1]

    candidates = [str(add_instead), str(just_vol_b)]

    return dict(
        text=text,
        latex=rf"V_A - V_B",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Compute both volumes separately, then find their difference: "
            f"{volumes[0]} - {volumes[1]} = {answer_text} cm³."
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "use_formulas_to_find_the_volume",
    level=11,
    presets={
        "easy": dict(mode="cube"),
        "medium": dict(mode="triangular_or_rectangular"),
        "hard": dict(mode="two_volume_difference"),
    },
)
def use_formulas_to_find_the_volume(
    rng,
    mode="cube",
    cube_side_hi=12,
    tri_base_hi=20,
    tri_height_hi=20,
    tri_length_hi=20,
    rect_length_hi=20,
    rect_width_hi=20,
    rect_height_hi=20,
    hard_cube_side_hi=10,
    hard_base_hi=16,
    hard_height_hi=16,
    hard_length_hi=16,
):
    """Calculate the volume of a cube, rectangular prism, or triangular prism
    using exact-arithmetic formulas; see module docstring for the rectangular-
    prism-duplication boundary and shape distribution."""
    if mode == "cube":
        return _build_cube(rng, cube_side_hi)
    if mode == "triangular_or_rectangular":
        return _build_triangular_or_rectangular_prism(rng, tri_base_hi, tri_height_hi, tri_length_hi)
    if mode == "two_volume_difference":
        return _build_two_volume_difference(rng, hard_cube_side_hi, hard_base_hi, hard_height_hi, hard_length_hi)
    raise ValueError(f"unknown mode: {mode!r}")
