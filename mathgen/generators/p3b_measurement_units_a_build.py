"""Metric unit conversion for length review and area use, both senior ``acara-senior-v8-4``, domain
``measurement``, ``answer: number``.

Two generators:

- ``review_metric_units_of_length`` (``senior-review-metric-units-of-length-
  their-abbreviations-conversions``, ACMEM090) -- convert a length measurement
  between metric units (mm/cm/m/km), including a two-operand combine-after-
  converting step. **This skill is a senior-level escalation** over the
  primary ``convert-metric-length`` skill: routinely uses decimal-valued
  inputs/results (not just whole numbers) and non-adjacent unit pairs, and
  escalates to a genuinely harder reasoning shape at hard (a mixed-unit
  combination requiring recognise-then-convert-then-combine reasoning, never
  the primary skill's single-step only conversions).

- ``use_metric_units_of_area`` (``senior-use-metric-units-of-area-their-
  abbreviations-conversions``, ACMEM021) -- convert an area measurement across
  the complete plain metric-prefix ladder (mm²/cm²/m²/km²), including a
  two-operand combine-after-converting step. **This skill's own scope is the
  plain prefix ladder only; the sibling skill senior-convert-between-metric-
  units-of-area-and-other (ACMEM023) exclusively handles the other unit
  family.** Never build a conversion involving the non-prefix area unit in
  this skill; that is ACMEM023's territory.

Both generators format ``answer_text`` from an exact ``Fraction``/``int``
value with explicit, bounded fixed-point decimal places (never
``str(float(...))``, which can silently switch to scientific notation below
``1e-4``) -- see ``exact_decimal_string``/``rounded_decimal_string`` in
``mathgen/core.py``. Every source
value is chosen so the result terminates within 3 decimal places by
construction.
"""

from __future__ import annotations

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

# --- review_metric_units_of_length -------------------------------------------

# Length conversion factors: base unit is mm
_FACTORS_LENGTH_MM = {"mm": Fraction(1), "cm": Fraction(10), "m": Fraction(1000), "km": Fraction(1_000_000)}
_UNIT_LABELS_LENGTH = {"mm": "mm", "cm": "cm", "m": "m", "km": "km"}

_FACTOR_MM_CM = _FACTORS_LENGTH_MM["cm"] / _FACTORS_LENGTH_MM["mm"]  # 10
_FACTOR_CM_M = _FACTORS_LENGTH_MM["m"] / _FACTORS_LENGTH_MM["cm"]  # 100
_FACTOR_M_KM = _FACTORS_LENGTH_MM["km"] / _FACTORS_LENGTH_MM["m"]  # 1000


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


def _build_length_easy(rng) -> dict:
    """Easy tier: single-step, adjacent-unit pair (mm<->cm, cm<->m, or
    m<->km), value chosen so the result is always exact."""
    pair = rng.choice([("mm", "cm"), ("cm", "m"), ("m", "km")])
    direction = rng.choice([0, 1])
    if direction == 0:
        source_unit, target_unit = pair
    else:
        target_unit, source_unit = pair

    factor = _FACTORS_LENGTH_MM[source_unit] / _FACTORS_LENGTH_MM[target_unit]

    # Pick source as a multiple of factor's denominator to ensure exact result
    source_value = Fraction(factor.denominator * rng.randint(1, 20))
    answer = source_value * factor

    check_conversion(source_value, source_unit, target_unit, answer, _FACTORS_LENGTH_MM, label="length conversion")

    answer_text = exact_decimal_string(answer, 3)
    source_text = exact_decimal_string(source_value, 3)
    text = f"Convert {source_text} {_UNIT_LABELS_LENGTH[source_unit]} to {_UNIT_LABELS_LENGTH[target_unit]}."

    # Misconceptions:
    # - conversion-direction: multiplies when should divide or vice versa
    direction_wrong = source_value / factor
    # - assumes-factor-ten: applies 10 regardless (encode as answer * 10 or answer / 10)
    factor_ten = answer * 10

    candidates = [rounded_decimal_string(direction_wrong, 3), rounded_decimal_string(factor_ten, 3)]

    return dict(
        text=text,
        latex=rf"{source_text}\,\mathrm{{{_UNIT_LABELS_LENGTH[source_unit]}}}=\square\,\mathrm{{{_UNIT_LABELS_LENGTH[target_unit]}}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"{source_text} {_UNIT_LABELS_LENGTH[source_unit]} = {answer_text} {_UNIT_LABELS_LENGTH[target_unit]}."
        ],
        distractors=_shaped_distractors(answer_text, candidates),
    )


def _build_length_medium(rng) -> dict:
    """Medium tier: single-step, non-adjacent pair (mm<->m only), a
    materially larger power-of-ten factor producing a genuine decimal result
    routinely."""
    direction = rng.choice([0, 1])
    if direction == 0:
        source_unit, target_unit = "mm", "m"
    else:
        source_unit, target_unit = "m", "mm"

    factor = _FACTORS_LENGTH_MM[source_unit] / _FACTORS_LENGTH_MM[target_unit]

    # For mm->m: factor is 1/1000, pick source to avoid whole result most of time
    if source_unit == "mm":
        source_value = Fraction(rng.randint(1, 9999))
        while source_value % 1000 == 0:
            source_value = Fraction(rng.randint(1, 9999))
    else:
        # For m->mm: factor is 1000, pick small integers
        source_value = Fraction(rng.randint(1, 10))

    answer = source_value * factor

    check_conversion(source_value, source_unit, target_unit, answer, _FACTORS_LENGTH_MM, label="length conversion")

    answer_text = exact_decimal_string(answer, 3)
    source_text = exact_decimal_string(source_value, 3)
    text = f"Convert {source_text} {_UNIT_LABELS_LENGTH[source_unit]} to {_UNIT_LABELS_LENGTH[target_unit]}."

    # Misconceptions
    direction_wrong = source_value / factor
    factor_ten = answer * 10

    candidates = [rounded_decimal_string(direction_wrong, 3), rounded_decimal_string(factor_ten, 3)]

    return dict(
        text=text,
        latex=rf"{source_text}\,\mathrm{{{_UNIT_LABELS_LENGTH[source_unit]}}}=\square\,\mathrm{{{_UNIT_LABELS_LENGTH[target_unit]}}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"{source_text} {_UNIT_LABELS_LENGTH[source_unit]} = {answer_text} {_UNIT_LABELS_LENGTH[target_unit]}."
        ],
        distractors=_shaped_distractors(answer_text, candidates),
    )


def _build_length_hard(rng) -> dict:
    """Hard tier: a mixed-unit combination, not a chained conversion.
    Two lengths stated in two different units from mm/cm/m/km, requiring
    the learner to recognise the unit mismatch, convert one operand, then
    add or subtract. Restrict pairs to: (mm,cm), (cm,m), (m,km), (mm,m)
    -- never cm<->km or mm<->km which would break 3-place termination."""
    pairs = [("mm", "cm"), ("cm", "m"), ("m", "km"), ("mm", "m")]
    unit_a, unit_b = rng.choice(pairs)
    operation = rng.choice(["sum", "difference"])

    # Choose integer values to ensure 3-place termination after conversion
    # For pairs with non-trivial conversion factors, pick carefully:
    # - (mm, cm): factor = 1/10, both can be small integers
    # - (cm, m): factor = 1/100, cm should be multiple of 100 to avoid decimals
    # - (m, km): factor = 1/1000, m should be multiple of 1000
    # - (mm, m): factor = 1/1000, mm should be multiple of 1000
    if (unit_a, unit_b) == ("mm", "cm"):
        value_a = Fraction(rng.randint(10, 900))
        value_b = Fraction(rng.randint(10, 90))
    elif (unit_a, unit_b) == ("cm", "m"):
        value_a = Fraction(rng.randint(100, 9900))  # cm, should be >= 100 for nice conversion
        value_b = Fraction(rng.randint(1, 50))
    elif (unit_a, unit_b) == ("m", "km"):
        value_a = Fraction(rng.randint(1000, 100000))  # m, multiple of 1000
        value_b = Fraction(rng.randint(1, 10))
    elif (unit_a, unit_b) == ("mm", "m"):
        value_a = Fraction(rng.randint(1000, 900000))  # mm, multiple of 1000
        value_b = Fraction(rng.randint(1, 50))
    elif (unit_a, unit_b) == ("cm", "mm"):
        value_a = Fraction(rng.randint(10, 90))
        value_b = Fraction(rng.randint(10, 900))
    elif (unit_a, unit_b) == ("m", "cm"):
        value_a = Fraction(rng.randint(1, 50))
        value_b = Fraction(rng.randint(100, 9900))
    elif (unit_a, unit_b) == ("km", "m"):
        value_a = Fraction(rng.randint(1, 10))
        value_b = Fraction(rng.randint(1000, 100000))
    elif (unit_a, unit_b) == ("m", "mm"):
        value_a = Fraction(rng.randint(1, 50))
        value_b = Fraction(rng.randint(1000, 900000))
    else:
        # Shouldn't reach here given the pairs list, but fallback
        value_a = Fraction(rng.randint(1, 100))
        value_b = Fraction(rng.randint(1, 100))

    values = {unit_a: value_a, unit_b: value_b}

    target_unit = rng.choice([unit_a, unit_b])
    other_unit = unit_b if target_unit == unit_a else unit_a

    converted_other = values[other_unit] * Fraction(_FACTORS_LENGTH_MM[other_unit] / _FACTORS_LENGTH_MM[target_unit])
    same_value = values[target_unit]

    if operation == "sum":
        answer = same_value + converted_other
    else:
        answer = abs(same_value - converted_other)

    check_combine(values, other_unit, target_unit, operation, answer, _FACTORS_LENGTH_MM, label="length combine")

    answer_text = exact_decimal_string(answer, 3)
    label_a, label_b = _UNIT_LABELS_LENGTH[unit_a], _UNIT_LABELS_LENGTH[unit_b]
    text_a, text_b = exact_decimal_string(values[unit_a], 3), exact_decimal_string(values[unit_b], 3)
    target_label = _UNIT_LABELS_LENGTH[target_unit]

    if operation == "sum":
        question_text = (
            f"Path A is {text_a} {label_a} and Path B is {text_b} {label_b}. "
            f"What is their combined length, in {target_label}?"
        )
    else:
        question_text = (
            f"Path A is {text_a} {label_a} and Path B is {text_b} {label_b}. "
            f"What is the difference in length between them, in {target_label}?"
        )

    # Misconceptions for hard tier:
    # - forgets-to-convert: combines the two raw values directly
    forgets_to_convert = value_a + value_b if operation == "sum" else abs(value_a - value_b)

    # - converts-wrong-operand: converts the operand that was already in target unit
    # (using the same-to-other ratio instead of other-to-target ratio)
    ratio_target_to_other = Fraction(_FACTORS_LENGTH_MM[target_unit], _FACTORS_LENGTH_MM[other_unit])
    wrongly_converted_same = same_value * ratio_target_to_other
    other_raw = values[other_unit]
    converts_wrong_operand = (
        wrongly_converted_same + other_raw if operation == "sum" else abs(wrongly_converted_same - other_raw)
    )

    # Guard against answer being 0 (zero length difference)
    if answer == 0:
        forgets_to_convert = Fraction(1)  # dummy non-zero value

    # Also ensure forgets_to_convert and converts_wrong_operand are not zero
    if forgets_to_convert == 0:
        forgets_to_convert = Fraction(1)
    if converts_wrong_operand == 0:
        converts_wrong_operand = Fraction(1)

    candidates = [
        rounded_decimal_string(forgets_to_convert, 3),
        rounded_decimal_string(converts_wrong_operand, 3),
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
    "review_metric_units_of_length",
    level=11,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
    },
)
def review_metric_units_of_length(
    rng,
    mode="easy",
):
    """Convert a length measurement between metric units (mm/cm/m/km),
    including a two-operand combine-after-converting step; see module
    docstring for the senior-level escalation boundary against the
    primary convert-metric-length skill."""
    if mode == "easy":
        return _build_length_easy(rng)
    if mode == "medium":
        return _build_length_medium(rng)
    if mode == "hard":
        return _build_length_hard(rng)
    raise ValueError(f"unknown mode: {mode!r}")


# --- use_metric_units_of_area ------------------------------------------------

# Area conversion factors: base unit is mm²
_FACTORS_AREA_MM2 = {
    "mm2": Fraction(1),
    "cm2": Fraction(100),
    "m2": Fraction(1_000_000),
    "km2": Fraction(1_000_000_000_000),
}
_UNIT_LABELS_AREA = {"mm2": "mm²", "cm2": "cm²", "m2": "m²", "km2": "km²"}

# Length factors for independent check (verify area factor = length_factor²)
_FACTORS_LENGTH_MM_AREA = {"mm": Fraction(1), "cm": Fraction(10), "m": Fraction(1000), "km": Fraction(1_000_000)}


def _build_area_easy(rng) -> dict:
    """Easy tier: single-step, adjacent-unit pair (mm²<->cm² or cm²<->m²),
    exact result."""
    pair = rng.choice([("mm2", "cm2"), ("cm2", "m2")])
    direction = rng.choice([0, 1])
    if direction == 0:
        source_unit, target_unit = pair
    else:
        target_unit, source_unit = pair

    factor = _FACTORS_AREA_MM2[source_unit] / _FACTORS_AREA_MM2[target_unit]

    # Pick source as a multiple of factor's denominator to ensure exact result
    source_value = Fraction(factor.denominator * rng.randint(1, 20))
    answer = source_value * factor

    check_conversion(source_value, source_unit, target_unit, answer, _FACTORS_AREA_MM2, label="area conversion")

    answer_text = exact_decimal_string(answer, 3)
    source_text = exact_decimal_string(source_value, 3)
    text = f"Convert {source_text} {_UNIT_LABELS_AREA[source_unit]} to {_UNIT_LABELS_AREA[target_unit]}."

    # Misconceptions
    direction_wrong = source_value / factor
    uses_linear_factor = answer * 10  # applies linear factor instead of square

    candidates = [rounded_decimal_string(direction_wrong, 3), rounded_decimal_string(uses_linear_factor, 3)]

    return dict(
        text=text,
        latex=rf"{source_text}\,\mathrm{{{_UNIT_LABELS_AREA[source_unit]}}}=\square\,\mathrm{{{_UNIT_LABELS_AREA[target_unit]}}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"{source_text} {_UNIT_LABELS_AREA[source_unit]} = {answer_text} {_UNIT_LABELS_AREA[target_unit]}."],
        distractors=_shaped_distractors(answer_text, candidates),
    )


def _build_area_medium(rng) -> dict:
    """Medium tier: single-step, randomly choose between mm²<->m² and
    m²<->km² (both 1,000,000x factor). Never cm²<->km² (1e10 factor breaks
    3-place termination)."""
    pair = rng.choice([("mm2", "m2"), ("m2", "km2")])
    direction = rng.choice([0, 1])
    if direction == 0:
        source_unit, target_unit = pair
    else:
        source_unit, target_unit = pair

    factor = _FACTORS_AREA_MM2[source_unit] / _FACTORS_AREA_MM2[target_unit]

    # To ensure 3-place termination:
    # - If converting from small to large (mm² to m²): source must be multiple of 1000
    # - If converting from large to small (m² to km²): use integer or Fraction(n, 1000)
    if _FACTORS_AREA_MM2[source_unit] < _FACTORS_AREA_MM2[target_unit]:
        # Converting up, pick multiple of 1000
        source_value = Fraction(rng.randint(1, 999) * 1000)
    else:
        # Converting down, use integer or Fraction(n, 1000)
        if rng.choice([True, False]):
            source_value = Fraction(rng.randint(1, 10))
        else:
            source_value = Fraction(rng.randint(1000, 10000), 1000)

    answer = source_value * factor

    check_conversion(source_value, source_unit, target_unit, answer, _FACTORS_AREA_MM2, label="area conversion")

    answer_text = exact_decimal_string(answer, 3)
    source_text = exact_decimal_string(source_value, 3)
    text = f"Convert {source_text} {_UNIT_LABELS_AREA[source_unit]} to {_UNIT_LABELS_AREA[target_unit]}."

    # Misconceptions
    direction_wrong = source_value / factor
    uses_linear_factor = answer * 10

    candidates = [rounded_decimal_string(direction_wrong, 3), rounded_decimal_string(uses_linear_factor, 3)]

    return dict(
        text=text,
        latex=rf"{source_text}\,\mathrm{{{_UNIT_LABELS_AREA[source_unit]}}}=\square\,\mathrm{{{_UNIT_LABELS_AREA[target_unit]}}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"{source_text} {_UNIT_LABELS_AREA[source_unit]} = {answer_text} {_UNIT_LABELS_AREA[target_unit]}."],
        distractors=_shaped_distractors(answer_text, candidates),
    )


def _build_area_hard(rng) -> dict:
    """Hard tier: mixed-unit combination. Choose pair randomly from exactly:
    (mm², cm²), (cm², m²), (mm², m²), (m², km²). Never (cm², km²) or
    (mm², km²) which have 1e10/1e12 factors risking scientific notation.
    Never the other-unit family. Values chosen to guarantee 3-place termination."""
    pairs = [("mm2", "cm2"), ("cm2", "m2"), ("mm2", "m2"), ("m2", "km2")]
    unit_a, unit_b = rng.choice(pairs)
    operation = rng.choice(["sum", "difference"])

    # Choose values carefully to ensure 3-place termination
    # Use multiples that make the conversion simple:
    # mm2/cm2: factor 100 - mm2 as multiple of 100, cm2 small
    # cm2/m2: factor 10000 - cm2 as multiple of 10000, m2 small
    # mm2/m2: factor 1e6 - mm2 as multiple of 1e6, m2 small
    # m2/km2: factor 1e6 - m2 as multiple of 1e6, km2 small
    if (unit_a, unit_b) == ("mm2", "cm2"):
        value_a = Fraction(rng.randint(1, 99) * 100)  # mm2, multiple of 100
        value_b = Fraction(rng.randint(1, 99))  # cm2, small
    elif (unit_a, unit_b) == ("cm2", "m2"):
        value_a = Fraction(rng.randint(1, 99) * 10000)  # cm2, multiple of 10000
        value_b = Fraction(rng.randint(1, 50))  # m2, small
    elif (unit_a, unit_b) == ("mm2", "m2"):
        value_a = Fraction(rng.randint(1, 50) * 1000000)  # mm2, multiple of 1e6
        value_b = Fraction(rng.randint(1, 50))  # m2, small
    elif (unit_a, unit_b) == ("m2", "km2"):
        value_a = Fraction(rng.randint(1, 50) * 1000000)  # m2, multiple of 1e6
        value_b = Fraction(rng.randint(1, 10))  # km2, small
    elif (unit_a, unit_b) == ("cm2", "mm2"):
        value_a = Fraction(rng.randint(1, 99))  # cm2, small
        value_b = Fraction(rng.randint(1, 99) * 100)  # mm2, multiple of 100
    elif (unit_a, unit_b) == ("m2", "cm2"):
        value_a = Fraction(rng.randint(1, 50))  # m2, small
        value_b = Fraction(rng.randint(1, 99) * 10000)  # cm2, multiple of 10000
    elif (unit_a, unit_b) == ("m2", "mm2"):
        value_a = Fraction(rng.randint(1, 50))  # m2, small
        value_b = Fraction(rng.randint(1, 50) * 1000000)  # mm2, multiple of 1e6
    elif (unit_a, unit_b) == ("km2", "m2"):
        value_a = Fraction(rng.randint(1, 10))  # km2, small
        value_b = Fraction(rng.randint(1, 50) * 1000000)  # m2, multiple of 1e6
    else:
        # Shouldn't reach here
        value_a = Fraction(rng.randint(1, 100))
        value_b = Fraction(rng.randint(1, 100))

    values = {unit_a: value_a, unit_b: value_b}

    target_unit = rng.choice([unit_a, unit_b])
    other_unit = unit_b if target_unit == unit_a else unit_a

    converted_other = values[other_unit] * Fraction(_FACTORS_AREA_MM2[other_unit] / _FACTORS_AREA_MM2[target_unit])
    same_value = values[target_unit]

    if operation == "sum":
        answer = same_value + converted_other
    else:
        answer = abs(same_value - converted_other)

    check_combine(values, other_unit, target_unit, operation, answer, _FACTORS_AREA_MM2, label="area combine")

    answer_text = exact_decimal_string(answer, 3)
    label_a, label_b = _UNIT_LABELS_AREA[unit_a], _UNIT_LABELS_AREA[unit_b]
    text_a, text_b = exact_decimal_string(values[unit_a], 3), exact_decimal_string(values[unit_b], 3)
    target_label = _UNIT_LABELS_AREA[target_unit]

    if operation == "sum":
        question_text = (
            f"Area A is {text_a} {label_a} and Area B is {text_b} {label_b}. "
            f"What is their combined area, in {target_label}?"
        )
    else:
        question_text = (
            f"Area A is {text_a} {label_a} and Area B is {text_b} {label_b}. "
            f"What is the difference in area between them, in {target_label}?"
        )

    # Misconceptions:
    # - forgets-to-convert: combines the two raw values directly
    forgets_to_convert = value_a + value_b if operation == "sum" else abs(value_a - value_b)

    # - converts-wrong-operand: converts the operand already in target unit
    # (using the same-to-other ratio instead of other-to-target ratio)
    ratio_target_to_other = Fraction(_FACTORS_AREA_MM2[target_unit] / _FACTORS_AREA_MM2[other_unit])
    wrongly_converted_same = same_value * ratio_target_to_other
    other_raw = values[other_unit]
    converts_wrong_operand = (
        wrongly_converted_same + other_raw if operation == "sum" else abs(wrongly_converted_same - other_raw)
    )

    # - uses-linear-factor (the headline misconception for area): apply linear
    #   factor instead of its square to the operand that needed converting
    length_factor_other = _FACTORS_LENGTH_MM_AREA[other_unit[:-1]]
    length_factor_target = _FACTORS_LENGTH_MM_AREA[target_unit[:-1]]
    linear_factor = length_factor_target / length_factor_other
    linear_conversion = values[other_unit] * linear_factor
    wrongly_using_linear_combined = (
        same_value + linear_conversion if operation == "sum" else abs(same_value - linear_conversion)
    )

    # Guard against answer being 0
    if answer == 0:
        forgets_to_convert = Fraction(1)
    if forgets_to_convert == 0:
        forgets_to_convert = Fraction(1)
    if converts_wrong_operand == 0:
        converts_wrong_operand = Fraction(1)
    if wrongly_using_linear_combined == 0:
        wrongly_using_linear_combined = Fraction(1)

    candidates = [
        rounded_decimal_string(forgets_to_convert, 3),
        rounded_decimal_string(converts_wrong_operand, 3),
        rounded_decimal_string(wrongly_using_linear_combined, 3),
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
    "use_metric_units_of_area",
    level=11,
    presets={
        "easy": dict(mode="easy"),
        "medium": dict(mode="medium"),
        "hard": dict(mode="hard"),
    },
)
def use_metric_units_of_area(
    rng,
    mode="easy",
):
    """Convert an area measurement across the complete plain metric-prefix
    ladder (mm²/cm²/m²/km²), including a two-operand combine-after-converting
    step; see module docstring for the plain-ladder scope and never the
    other-unit family."""
    if mode == "easy":
        return _build_area_easy(rng)
    if mode == "medium":
        return _build_area_medium(rng)
    if mode == "hard":
        return _build_area_hard(rng)
    raise ValueError(f"unknown mode: {mode!r}")
