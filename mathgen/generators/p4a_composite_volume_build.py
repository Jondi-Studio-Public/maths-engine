"""Irregular composite area and triangular-prism volume (f10-irregular-composite-area,
f10-triangular-prism-volume).

- f10-irregular-composite-area: Given an irregular rectilinear shape made of
  rectangular sections, calculate its total area. Easy tier has 2 sections with
  all dimensions stated. Medium tier has 3 sections with all dimensions stated.
  Hard tier has 2 sections where one section's width must be derived from the
  total height and the other section's width.

- f10-triangular-prism-volume: Calculate the volume of a right triangular prism
  from its triangular cross-section and length. Easy tier gives all dimensions
  directly. Medium tier adds a slant-side distractor. Hard tier is inverse:
  given the volume and two of the three dimensions, find the missing one.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, pick_distractors


# ============================================================================
# Skill 1: f10-irregular-composite-area
# ============================================================================


def _verify_composite_area_sum(sections: list[tuple[int, int]], answer: int) -> None:
    """Independent check for easy/medium: recompute via built-in sum()
    instead of manual loop. Distinct computational route from construction."""
    total = sum(L * W for L, W in sections)
    check(total == answer, f"sum {total} != {answer}")


def _verify_composite_area_hard(L1: int, W1: int, L2: int, W2: int, H: int, answer: int) -> None:
    """Independent check for hard: verify the derived W2 via addition
    (W1 + W2 == H) rather than subtraction, then recompute area."""
    check(W1 + W2 == H, f"derived W2: {W1} + {W2} != {H}")
    # Now recompute area independently using the verified W2
    total = L1 * W1 + L2 * W2
    check(total == answer, f"recomputed area {total} != {answer}")


@generator(
    "irregular_composite_area",
    level=1,
    presets={
        "easy": dict(mode="two_sections"),
        "medium": dict(mode="three_sections"),
        "hard": dict(mode="derive_dimension"),
    },
)
def irregular_composite_area(rng, mode="two_sections"):
    """Calculate the total area of an irregular rectilinear shape made of
    rectangular sections. Easy tier: 2 sections, all dimensions stated.
    Medium tier: 3 sections, all dimensions stated. Hard tier: 2 sections,
    one dimension derived from total height."""
    if mode not in ("two_sections", "three_sections", "derive_dimension"):
        raise ValueError(f"unknown mode {mode!r}")

    if mode == "two_sections":
        # Easy: 2 sections, both fully dimensioned
        L1 = rng.randint(2, 12)
        W1 = rng.randint(2, 12)
        L2 = rng.randint(2, 12)
        W2 = rng.randint(2, 12)

        text = f"A shape is made of two rectangular sections. Section 1 is {L1} m by {W1} m. Section 2 is {L2} m by {W2} m. What is the total area of the shape, in square metres?"
        answer = L1 * W1 + L2 * W2
        answer_text = str(answer)
        working = [
            f"Area of section 1 = {L1} × {W1} = {L1 * W1} square metres",
            f"Area of section 2 = {L2} × {W2} = {L2 * W2} square metres",
            f"Total area = {L1 * W1} + {L2 * W2} = {answer} square metres",
        ]
        _verify_composite_area_sum([(L1, W1), (L2, W2)], answer)

        # Misconceptions:
        # - omits_one_section: drop section 2's area
        # - uses_perimeter_instead_of_area: sum perimeters instead
        omits_section2 = L1 * W1
        perim_sum = 2 * (L1 + W1) + 2 * (L2 + W2)

        candidates = [str(omits_section2)]
        if perim_sum != answer:
            candidates.append(str(perim_sum))
        distractors = pick_distractors(answer_text, candidates)

    elif mode == "three_sections":
        # Medium: 3 sections, all fully dimensioned
        L1 = rng.randint(2, 12)
        W1 = rng.randint(2, 12)
        L2 = rng.randint(2, 12)
        W2 = rng.randint(2, 12)
        L3 = rng.randint(2, 12)
        W3 = rng.randint(2, 12)

        text = f"A shape is made of three rectangular sections. Section 1 is {L1} m by {W1} m. Section 2 is {L2} m by {W2} m. Section 3 is {L3} m by {W3} m. What is the total area of the shape, in square metres?"
        answer = L1 * W1 + L2 * W2 + L3 * W3
        answer_text = str(answer)
        working = [
            f"Area of section 1 = {L1} × {W1} = {L1 * W1} square metres",
            f"Area of section 2 = {L2} × {W2} = {L2 * W2} square metres",
            f"Area of section 3 = {L3} × {W3} = {L3 * W3} square metres",
            f"Total area = {L1 * W1} + {L2 * W2} + {L3 * W3} = {answer} square metres",
        ]
        _verify_composite_area_sum([(L1, W1), (L2, W2), (L3, W3)], answer)

        # Misconceptions:
        # - omits_one_section: drop section 3's area
        # - uses_perimeter_instead_of_area: sum perimeters instead
        omits_section3 = L1 * W1 + L2 * W2
        perim_sum = 2 * (L1 + W1) + 2 * (L2 + W2) + 2 * (L3 + W3)

        candidates = [str(omits_section3)]
        if perim_sum != answer:
            candidates.append(str(perim_sum))
        distractors = pick_distractors(answer_text, candidates)

    else:  # mode == "derive_dimension"
        # Hard: 2 sections, must derive W2 from total height
        L1 = rng.randint(2, 12)
        W1 = rng.randint(2, 12)
        L2 = rng.randint(2, 12)
        W2 = rng.randint(2, 12)
        H = W1 + W2

        text = f"A shape is made of two rectangular sections. Section 1 is {L1} m by {W1} m. Section 2 is {L2} m long. Together the two sections reach a total height of {H} m, of which {W1} m belongs to section 1. What is the total area of the shape, in square metres?"
        answer = L1 * W1 + L2 * (H - W1)
        answer_text = str(answer)
        working = [
            f"Area of section 1 = {L1} × {W1} = {L1 * W1} square metres",
            f"Width of section 2 = {H} − {W1} = {H - W1} m",
            f"Area of section 2 = {L2} × {H - W1} = {L2 * (H - W1)} square metres",
            f"Total area = {L1 * W1} + {L2 * (H - W1)} = {answer} square metres",
        ]
        _verify_composite_area_hard(L1, W1, L2, W2, H, answer)

        # Misconceptions:
        # - omits_one_section: drop section 2's area
        # - wrong_missing_dimension: uses H directly instead of (H - W1)
        omits_section2 = L1 * W1
        wrong_width = L2 * H

        candidates = [str(omits_section2), str(wrong_width)]
        distractors = pick_distractors(answer_text, candidates)

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


# ============================================================================
# Skill 2: f10-triangular-prism-volume
# ============================================================================


def _verify_triangular_prism_area_via_fraction(base: int, height_tri: int, answer: int) -> None:
    """Independent check for easy/medium: compute cross-section area via
    Fraction (exact) instead of integer division."""
    area_frac = Fraction(base * height_tri, 2)
    check(area_frac.denominator == 1, f"area {area_frac} has denominator {area_frac.denominator}")
    check(area_frac.numerator == answer, f"area numerator {area_frac.numerator} != {answer}")


def _verify_triangular_prism_via_repeated_addition(cross_section_area: int, length: int, answer: int) -> None:
    """Independent check for easy/medium: recompute volume via repeated addition
    (summing cross_section_area length times) instead of multiplication."""
    total = sum(cross_section_area for _ in range(length))
    check(total == answer, f"repeated addition {total} != {answer}")


def _verify_triangular_prism_inverse(
    base: int, height_tri: int, length: int, volume: int, answer: int, hidden: str
) -> None:
    """Independent check for hard: multiply the recovered dimension back
    into the full formula and assert it equals the stated volume."""
    if hidden == "base":
        recovered_base = answer
        computed_volume = (recovered_base * height_tri // 2) * length
    elif hidden == "height_tri":
        recovered_height = answer
        computed_volume = (base * recovered_height // 2) * length
    else:  # hidden == "length"
        recovered_length = answer
        computed_volume = (base * height_tri // 2) * recovered_length
    check(computed_volume == volume, f"recomputed volume {computed_volume} != stated {volume}")


@generator(
    "triangular_prism_volume",
    level=1,
    presets={
        "easy": dict(mode="direct"),
        "medium": dict(mode="with_slant"),
        "hard": dict(mode="inverse"),
    },
)
def triangular_prism_volume(rng, mode="with_slant"):
    """Calculate the volume of a right triangular prism from its triangular
    cross-section and length. Hard tier is inverse: given volume and two
    dimensions, find the third."""
    if mode not in ("direct", "with_slant", "inverse"):
        raise ValueError(f"unknown mode {mode!r}")

    # Sample base (2..10) and height_tri from {2, 4, 6, 8, 10} to ensure even product
    base = rng.randint(2, 10)
    height_tri = rng.choice([2, 4, 6, 8, 10])
    length = rng.randint(2, 15)

    cross_section_area = base * height_tri // 2
    volume = cross_section_area * length

    if mode == "direct":
        # Easy: all dimensions given directly
        text = f"A triangular prism has a triangular cross-section with a base of {base} cm and a perpendicular height of {height_tri} cm. The prism is {length} cm long. What is its volume, in cubic centimetres?"
        answer = volume
        answer_text = str(answer)
        working = [
            f"Area of triangular cross-section = {base} × {height_tri} ÷ 2 = {cross_section_area} cm²",
            f"Volume = {cross_section_area} × {length} = {answer} cm³",
        ]
        _verify_triangular_prism_area_via_fraction(base, height_tri, cross_section_area)
        _verify_triangular_prism_via_repeated_addition(cross_section_area, length, answer)

        # Misconception: omits_half_factor (treats like rectangular prism)
        omits_half = base * height_tri * length
        check(omits_half != answer, "omits_half_factor should be distinct from answer")

        # - forgets_length: stops at the cross-section area; - adds_length: area + length instead of area x length
        candidates = [str(omits_half), str(cross_section_area), str(cross_section_area + length)]
        distractors = pick_distractors(answer_text, candidates)

    elif mode == "with_slant":
        # Medium: include slant-side distractor
        slant = height_tri + rng.randint(1, 5)
        check(slant != height_tri, "slant side should never equal height")

        text = f"A triangular prism has a triangular cross-section with a base of {base} cm, a perpendicular height of {height_tri} cm, and a sloping side of {slant} cm. The prism is {length} cm long. What is its volume, in cubic centimetres?"
        answer = volume
        answer_text = str(answer)
        working = [
            f"Area of triangular cross-section = {base} × {height_tri} ÷ 2 = {cross_section_area} cm²",
            f"The sloping side is not used in the area formula.",
            f"Volume = {cross_section_area} × {length} = {answer} cm³",
        ]
        _verify_triangular_prism_area_via_fraction(base, height_tri, cross_section_area)
        _verify_triangular_prism_via_repeated_addition(cross_section_area, length, answer)

        # Misconceptions:
        # - omits_half_factor: base * height_tri * length
        # - uses_slant_side: (base * slant // 2) * length
        omits_half = base * height_tri * length
        check(omits_half != answer, "omits_half_factor should be distinct from answer")

        uses_slant = (base * slant // 2) * length
        check(uses_slant != answer, "uses_slant_side should be distinct from answer")

        candidates = [str(omits_half), str(uses_slant), str(cross_section_area)]
        distractors = pick_distractors(answer_text, candidates)

    else:  # mode == "inverse"
        # Hard: given volume and two of {base, height_tri, length}, find the third
        hide_choice = rng.choice(["base", "height_tri", "length"])

        slant = height_tri + rng.randint(1, 5)

        if hide_choice == "base":
            answer = base
            text = f"A triangular prism has a volume of {volume} cm³. Its triangular cross-section has a perpendicular height of {height_tri} cm and a sloping side of {slant} cm. The prism is {length} cm long. What is the base of the triangular cross-section, in centimetres?"
            unknown_name = "base"
            # uses_slant_side: use slant instead of height_tri to recover base
            # Correct formula: base = volume / ((height_tri * length) / 2) = volume * 2 / (height_tri * length)
            # With slant: base = volume * 2 / (slant * length)
            uses_slant_recovered = (volume * 2) // (slant * length)
            # wrong_inverse: forget halving: base = volume / (height_tri * length)
            wrong_inverse = volume // (height_tri * length)
        elif hide_choice == "height_tri":
            answer = height_tri
            text = f"A triangular prism has a volume of {volume} cm³. Its triangular cross-section has a base of {base} cm and a sloping side of {slant} cm. The prism is {length} cm long. What is the perpendicular height of the triangular cross-section, in centimetres?"
            unknown_name = "perpendicular height"
            # uses_slant_side: use slant instead of base to recover height
            # Correct: height_tri = volume * 2 / (base * length)
            # With slant: height_tri = volume * 2 / (slant * length)
            uses_slant_recovered = (volume * 2) // (slant * length)
            # wrong_inverse: forget halving: height_tri = volume / (base * length)
            wrong_inverse = volume // (base * length)
        else:  # hide_choice == "length"
            answer = length
            text = f"A triangular prism has a volume of {volume} cm³. Its triangular cross-section has a base of {base} cm and a perpendicular height of {height_tri} cm. A sloping side of the triangle is {slant} cm. What is the length of the prism, in centimetres?"
            unknown_name = "length"
            # uses_slant_side: use slant in area formula instead of height
            # Correct: length = volume / ((base * height_tri) / 2) = volume * 2 / (base * height_tri)
            # With slant: length = volume * 2 / (base * slant)
            uses_slant_recovered = (volume * 2) // (base * slant)
            # wrong_inverse: forget halving: length = volume / (base * height_tri)
            wrong_inverse = volume // (base * height_tri)

        answer_text = str(answer)
        working = ["Volume = (base × perpendicular height ÷ 2) × length"]
        if hide_choice == "length":
            area = base * height_tri // 2
            working.append(f"Cross-section area = {base} × {height_tri} ÷ 2 = {area} cm²")
            working.append(f"length = {volume} ÷ {area} = {answer}")
        else:
            area = volume // length
            known = height_tri if hide_choice == "base" else base
            working.append(f"Cross-section area = {volume} ÷ {length} = {area} cm²")
            working.append(
                f"{unknown_name} × {known} ÷ 2 = {area}, so {unknown_name} = {area} × 2 ÷ {known} = {answer}"
            )
        _verify_triangular_prism_inverse(base, height_tri, length, volume, answer, hide_choice)

        # Misconceptions:
        # - uses_slant_side: use slant in the formula instead of height_tri
        # - wrong_inverse_operation: forgets halving when rearranging
        check(wrong_inverse != answer, "wrong_inverse_operation should be distinct from answer")

        candidates = []
        # Add uses_slant only if it's distinct from answer
        if uses_slant_recovered != answer:
            candidates.append(str(uses_slant_recovered))
        candidates.append(str(wrong_inverse))
        candidates.append(str(answer * 2))  # doubled where it should halve
        distractors = pick_distractors(answer_text, candidates)

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
