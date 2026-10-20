"""Measurement generators (f10-volume-surface-area-prisms-cylinders, f10-decimal-metric-representation).

- f10-volume-surface-area-prisms-cylinders: Calculate volume or surface area of
  cylinders or triangular prisms from supplied dimensions, reporting with units.
  Easy: cylinder volume. Medium: cylinder surface area. Hard: triangular prism
  surface area.

- f10-decimal-metric-representation: Convert a metric measurement to a larger
  unit and express as a decimal to a stated precision. Easy: mm to cm, 1 dp.
  Medium: g/mL to kg/L, 2 dp. Hard: forced rounding scenario, 2 dp.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, fixed_point_string, generator, numeric_answer, pick_distractors, round_half_up


# ============================================================================
# Skill 1: f10-volume-surface-area-prisms-cylinders
# ============================================================================

PI = Fraction(314, 100)

_TRIPLES = [(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17), (9, 12, 15), (12, 16, 20)]


def _verify_cylinder_volume(radius: int, height: int, answer_frac: Fraction) -> None:
    """Independent check: recompute via repeated addition (different route)."""
    raw_total = sum(PI for _ in range(radius * radius * height))
    check(raw_total >= 0, "volume must not be negative")
    rounded_frac = round_half_up(raw_total, 1)
    check(rounded_frac == answer_frac, f"cylinder volume check {float(rounded_frac)} != {float(answer_frac)}")


def _verify_cylinder_surface_area(radius: int, height: int, answer_frac: Fraction) -> None:
    """Independent check: compute two terms separately via repeated addition,
    sum the raw values THEN round once."""
    lateral_raw = sum(PI for _ in range(2 * radius * height))
    ends_raw = sum(PI for _ in range(2 * radius * radius))
    combined_raw = lateral_raw + ends_raw
    check(combined_raw >= 0, "volume must not be negative")
    rounded_frac = round_half_up(combined_raw, 1)
    check(rounded_frac == answer_frac, f"cylinder surface area check {float(rounded_frac)} != {float(answer_frac)}")


def _verify_triangular_prism_surface_area(a: int, b: int, c: int, length: int, answer: int) -> None:
    """Independent check: compute five separate faces and sum them."""
    face1 = Fraction(a * b, 2)
    face2 = Fraction(a * b, 2)
    face3 = a * length
    face4 = b * length
    face5 = c * length
    total = face1 + face2 + face3 + face4 + face5
    check(total == answer, f"triangular prism surface area check {total} != {answer}")


@generator(
    "volume_surface_area_prisms_cylinders",
    level=1,
    presets={
        "easy": dict(mode="cylinder_volume"),
        "medium": dict(mode="cylinder_surface_area"),
        "hard": dict(mode="triangular_prism_surface_area"),
    },
)
def volume_surface_area_prisms_cylinders(rng, mode="cylinder_volume"):
    """Calculate volume or surface area of cylinders or triangular prisms."""
    if mode == "cylinder_volume":
        # Easy: cylinder volume V = pi * r^2 * h
        radius = rng.randint(2, 6)
        height = rng.randint(2, 15)
        # Guard against distractor collisions:
        # - radius_height_swap equals answer when h == r
        # - uses_diameter_as_radius equals radius_height_swap when h == 4r
        while height == radius or height == 4 * radius:
            height = rng.randint(2, 15)

        raw = PI * radius * radius * height
        check(raw >= 0, "volume must not be negative")
        rounded = round_half_up(raw, 1)
        answer = float(rounded)
        answer_text = f"{float(rounded):.1f}"

        text = f"A cylinder has a radius of {radius} cm and a height of {height} cm. Using π = 3.14, what is its volume, in cm³, to 1 decimal place?"
        working = [
            f"Volume = π × radius² × height",
            f"Volume = 3.14 × {radius}² × {height} = 3.14 × {radius * radius} × {height} = {float(raw):.2f}",
            f"To 1 decimal place: {answer_text} cm³",
        ]
        _verify_cylinder_volume(radius, height, rounded)

        # Distractors
        omits_pi = Fraction(radius * radius * height, 1)
        check(omits_pi >= 0, "volume must not be negative")
        omits_pi_rounded = round_half_up(omits_pi, 1)

        uses_diameter = PI * (2 * radius) * (2 * radius) * height
        check(uses_diameter >= 0, "volume must not be negative")
        uses_diameter_rounded = round_half_up(uses_diameter, 1)

        swap = PI * height * height * radius
        check(swap >= 0, "volume must not be negative")
        swap_rounded = round_half_up(swap, 1)

        candidates = [
            str(int(omits_pi_rounded * 10) / 10),
            str(int(uses_diameter_rounded * 10) / 10),
            str(int(swap_rounded * 10) / 10),
        ]
        distractors = pick_distractors(answer_text, candidates)

        # Verify each distractor differs by > 0.05
        for dist_val in candidates:
            dist_float = float(dist_val)
            check(
                abs(float(answer_text) - dist_float) > 0.05, f"distractor {dist_val} too close to answer {answer_text}"
            )

    elif mode == "cylinder_surface_area":
        # Medium: cylinder surface area SA = 2*pi*r^2 + 2*pi*r*h
        radius = rng.randint(2, 6)
        height = rng.randint(3, 15)
        # Guard: height != radius
        while height == radius:
            height = rng.randint(3, 15)

        ends_raw = sum(PI for _ in range(2 * radius * radius))
        lateral_raw = sum(PI for _ in range(2 * radius * height))
        combined_raw = ends_raw + lateral_raw
        check(combined_raw >= 0, "volume must not be negative")
        rounded = round_half_up(combined_raw, 1)
        answer = float(rounded)
        answer_text = f"{float(rounded):.1f}"

        text = f"A cylinder has a radius of {radius} cm and a height of {height} cm. Using π = 3.14, what is its surface area, in cm², to 1 decimal place?"
        working = [
            f"Surface area = 2π × radius² + 2π × radius × height",
            f"= 2 × 3.14 × {radius}² + 2 × 3.14 × {radius} × {height}",
            f"= {float(ends_raw):.2f} + {float(lateral_raw):.2f} = {float(combined_raw):.2f}",
            f"To 1 decimal place: {answer_text} cm²",
        ]
        _verify_cylinder_surface_area(radius, height, rounded)

        # Distractors
        omits_ends = 2 * PI * radius * height
        check(omits_ends >= 0, "volume must not be negative")
        omits_ends_rounded = round_half_up(omits_ends, 1)

        omits_lateral = 2 * PI * radius * radius
        check(omits_lateral >= 0, "volume must not be negative")
        omits_lateral_rounded = round_half_up(omits_lateral, 1)

        omits_pi = 2 * radius * radius + 2 * radius * height
        check(omits_pi >= 0, "volume must not be negative")
        omits_pi_rounded = round_half_up(Fraction(omits_pi, 1), 1)

        candidates = [
            str(int(omits_ends_rounded * 10) / 10),
            str(int(omits_lateral_rounded * 10) / 10),
            str(int(omits_pi_rounded * 10) / 10),
        ]
        distractors = pick_distractors(answer_text, candidates)

        # Verify each distractor differs by > 0.05
        for dist_val in candidates:
            dist_float = float(dist_val)
            check(
                abs(float(answer_text) - dist_float) > 0.05, f"distractor {dist_val} too close to answer {answer_text}"
            )

    else:  # mode == "triangular_prism_surface_area"
        # Hard: triangular prism surface area
        a, b, c = rng.choice(_TRIPLES)
        length = rng.randint(5, 20)

        # SA = a*b + (a + b + c) * length
        answer = a * b + (a + b + c) * length
        answer_text = str(answer)

        text = f"A triangular prism has a right-triangle cross-section with legs {a} cm and {b} cm (hypotenuse {c} cm) and a length of {length} cm. What is its total surface area, in cm²?"
        working = [
            f"Surface area = area of 2 triangles + area of 3 rectangles",
            f"= 2 × ({a} × {b} ÷ 2) + ({a} × {length}) + ({b} × {length}) + ({c} × {length})",
            f"= {a * b} + {a * length} + {b * length} + {c * length}",
            f"= {answer} cm²",
        ]
        _verify_triangular_prism_surface_area(a, b, c, length, answer)

        # Distractors
        omits_triangles = (a + b + c) * length
        forgets_half_factor = 2 * a * b + (a + b + c) * length
        single_end_only = (a * b) // 2 + (a + b + c) * length

        candidates = [str(omits_triangles), str(forgets_half_factor), str(single_end_only)]
        distractors = pick_distractors(answer_text, candidates)

        # Verify each distractor is distinct
        for dist_val in candidates:
            dist_int = int(dist_val)
            check(dist_int != answer, f"distractor {dist_val} equals answer {answer_text}")

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer if mode == "triangular_prism_surface_area" else numeric_answer(rounded),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


# ============================================================================
# Skill 2: f10-decimal-metric-representation
# ============================================================================

_FACTORS = {
    "mm_to_cm": 10,
    "g_to_kg": 1000,
    "ml_to_l": 1000,
}


def _format_fixed(value: Fraction, max_places: int = 1) -> str:
    """Fixed-point formatting from an exact Fraction to exactly max_places decimal places."""
    # Check if value terminates within max_places by scaling
    scaled = value * (10**max_places)
    if scaled.denominator == 1:
        # Value terminates at or before max_places, format to exactly max_places
        return fixed_point_string(scaled.numerator, max_places)
    raise ValueError(f"{value} does not terminate within {max_places} decimal places")


def _format_display(value: Fraction, max_places: int = 1) -> str:
    """Like _format_fixed, but for a *distractor* value: rounds half-up to max_places
    decimal places if the value does not terminate exactly."""
    try:
        return _format_fixed(value, max_places)
    except ValueError:
        # Round half-up using Fraction arithmetic
        scaled = math.floor(value * (10**max_places) + Fraction(1, 2))
        return fixed_point_string(scaled, max_places)


def _verify_decimal_metric(v: int, factor: int, places: int, answer_text: str) -> None:
    """Independent check: compute via Fraction arithmetic, round half-up."""
    frac = Fraction(v, factor)
    scaled = frac * (10**places) + Fraction(1, 2)
    floor_scaled = scaled.numerator // scaled.denominator

    # Reconstruct answer_text from the floor_scaled value to compare (exactly max_places digits)
    expected_text = fixed_point_string(floor_scaled, places)
    check(expected_text == answer_text, f"decimal metric check {expected_text} != {answer_text}")


@generator(
    "decimal_metric_representation",
    level=1,
    presets={
        "easy": dict(mode="single_step"),
        "medium": dict(mode="two_step_family"),
        "hard": dict(mode="forced_rounding"),
    },
)
def decimal_metric_representation(rng, mode="single_step"):
    """Convert metric measurements to larger units with decimal precision."""
    if mode == "single_step":
        # Easy: mm to cm (factor 10), 1 decimal place
        v = rng.randint(90, 200)  # a pencil is 9 to 20 cm long
        factor = 10
        places = 1

        frac = Fraction(v, factor)
        answer_text = _format_fixed(frac, places)
        answer = float(answer_text)

        text = f"A pencil is {v} mm long. Write this length in cm, correct to 1 decimal place."
        working = [
            f"Convert mm to cm: divide by 10",
            f"{v} ÷ 10 = {float(frac):.1f}",
            f"To 1 decimal place: {answer_text} cm",
        ]
        _verify_decimal_metric(v, factor, places, answer_text)

        # Distractors: wrong-conversion-factor (uses factor//10 = 1, no division, 10x too large)
        # and decimal-place-shift (shift decimal point left by 1, divide by 10, 10x too small)
        wrong_factor_10x = Fraction(v, 1)  # No division (divide by 1)
        wrong_factor_10x_text = _format_display(wrong_factor_10x, places)

        # decimal-place-shift: shift the decimal point left by 1 (divide by 10)
        decimal_place_shift = Fraction(v, factor * 10)
        decimal_place_shift_text = _format_display(decimal_place_shift, places)

        # Also include divide-by-100 as an adjacent power-of-ten wrong factor
        wrong_factor_div10 = Fraction(v, 100)
        wrong_factor_div10_text = _format_display(wrong_factor_div10, places)

        candidates = [wrong_factor_10x_text, wrong_factor_div10_text, decimal_place_shift_text]
        distractors = pick_distractors(answer_text, candidates)

    elif mode == "two_step_family":
        # Medium: g to kg or mL to L (factor 1000), 2 decimal places
        family = rng.choice(["g_to_kg", "ml_to_l"])
        factor = 1000
        places = 2

        # Construct v as multiple of 10 so division terminates at 2 places
        # A bag of sugar is 500 g to 5 kg and a bottle of juice 200 mL to 3 L.
        v = 10 * (rng.randint(50, 500) if family == "g_to_kg" else rng.randint(20, 300))

        frac = Fraction(v, factor)
        answer_text = _format_fixed(frac, places)
        answer = float(answer_text)

        if family == "g_to_kg":
            text = f"A bag contains {v} grams of sugar. Write this mass in kilograms, correct to 2 decimal places."
            unit_from, unit_to = "grams", "kilograms"
        else:  # ml_to_l
            text = f"A bottle holds {v} mL of juice. Write this volume in litres, correct to 2 decimal places."
            unit_from, unit_to = "mL", "litres"

        working = [
            f"Convert {unit_from} to {unit_to}: divide by 1000",
            f"{v} ÷ 1000 = {float(frac):.3f}",
            f"To 2 decimal places: {answer_text} {unit_to}",
        ]
        _verify_decimal_metric(v, factor, places, answer_text)

        # Distractors: wrong-conversion-factor (always uses factor//10 = 100, 10x too large)
        # and decimal-place-shift (always reports answer/10, 10x too small)
        wrong_factor_10x = Fraction(v, 100)  # Divide by 100 instead of 1000
        wrong_factor_10x_text = _format_display(wrong_factor_10x, places)

        # decimal-place-shift: shift the decimal point left by 1 (divide by 10)
        decimal_place_shift = Fraction(v, factor * 10)
        decimal_place_shift_text = _format_display(decimal_place_shift, places)

        candidates = [wrong_factor_10x_text, decimal_place_shift_text]
        distractors = pick_distractors(answer_text, candidates)

    else:  # mode == "forced_rounding"
        # Hard: forced rounding, 2 decimal places, g_to_kg or ml_to_l only
        family = rng.choice(["g_to_kg", "ml_to_l"])
        factor = 1000
        places = 2

        # Force either round_up or round_down
        branch = rng.choice(["round_up", "round_down"])

        # Construct v = 1000a + 100b + 10c + d (i.e., in range 1000-9999)
        # When divided by 1000, this gives a.bcd, where the third decimal digit is d
        # To force rounding, control the units digit (d):
        # - If d in [5,9], the third decimal digit rounds up
        # - If d in [0,4], the third decimal digit rounds down
        base = rng.randint(100, 999)  # Gives 1000-9999 range for v
        if branch == "round_up":
            units_digit = rng.randint(5, 9)  # Third decimal digit in [5,9]
        else:  # round_down
            units_digit = rng.randint(0, 4)  # Third decimal digit in [0,4]
        v = base * 10 + units_digit

        frac = Fraction(v, factor)
        # Use _format_display to round, since this tier forces rounding every seed
        answer_text = _format_display(frac, places)
        answer = float(answer_text)

        if family == "g_to_kg":
            text = f"A package weighs {v} grams. Write this mass in kilograms, correct to 2 decimal places."
            unit_from, unit_to = "grams", "kilograms"
        else:  # ml_to_l
            text = f"A container holds {v} mL. Write this volume in litres, correct to 2 decimal places."
            unit_from, unit_to = "mL", "litres"

        working = [
            f"Convert {unit_from} to {unit_to}: divide by 1000",
            f"{v} ÷ 1000 = {float(frac):.3f}",
            f"To 2 decimal places: {answer_text} {unit_to}",
        ]
        _verify_decimal_metric(v, factor, places, answer_text)

        # Distractors
        # truncates_instead_of_rounds: truncate at 2 places (only differs on round_up seeds)
        truncated_scaled = v // (factor // (10**places))  # v // 10 = floor(v/10)
        truncated_frac = Fraction(truncated_scaled, 100)  # Scale back to have 2 decimal places in representation
        truncated_text = _format_display(truncated_frac, places)

        # wrong-conversion-factor: uses factor//10 = 100 (10x too large)
        wrong_factor_10x = Fraction(v, 100)
        wrong_factor_10x_text = _format_display(wrong_factor_10x, places)

        # decimal-place-shift: divide by factor * 10 (10x too small)
        decimal_place_shift = Fraction(v, factor * 10)
        decimal_place_shift_text = _format_display(decimal_place_shift, places)

        candidates = [truncated_text, wrong_factor_10x_text, decimal_place_shift_text]
        distractors = pick_distractors(answer_text, candidates)

        # Verify both branches occur in testing (no assertion here, just construction)
        check(branch in ["round_up", "round_down"], f"invalid branch {branch}")

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
