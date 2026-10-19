"""Scale factor for similar figures and similar solids.

Two generators:

- ``scale_factor_similar_areas`` (``senior-obtain-a-scale-factor-and-use-it-to``,
  ACMGM024) -- 2D scale factors applied to areas of similar figures.

- ``scale_factor_similar_solids`` (``senior-obtain-a-scale-factor-and-use-it-to-acmgm025``,
  ACMGM025) -- 3D scale factors applied to surface areas and volumes of similar
  solids.

Both skills use only exact integers; all answers are plain ints.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, check_distinct, check_equivalent, generator, pick_distractors

# --- scale_factor_similar_areas (ACMGM024) -----------------------------------

RATIO_PAIRS = [
    (2, 3),
    (3, 2),
    (2, 5),
    (5, 2),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 5),
    (5, 4),
    (5, 6),
    (6, 5),
]

RATIO_PAIRS_SOLID = [
    (2, 3),
    (3, 2),
    (2, 5),
    (5, 2),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 5),
    (5, 4),
]


def _verify_scale_factor_areas_hard(length_a: int, length_b: int, area_a: int, area_b: int, answer: int) -> None:
    """Independent check for hard tier using cross-multiplied similarity identity.
    For similar figures: (length_b / length_a)^2 == (area_b / area_a)
    Cross-multiplied: answer^2 * area_a == length_a^2 * area_b
    where answer = corresponding length in figure B.
    """
    check_equivalent(answer**2 * area_a, length_a**2 * area_b, "scale factor hard tier identity failed")


def _verify_scale_factor_solids_hard(volume_a: int, volume_b: int, area_a: int, area_b: int, answer: int) -> None:
    """Independent check for hard tier using cross-multiplied volume/area identity.
    For similar solids: (volume_b / volume_a)^2 == (area_b / area_a)^3
    Both sides equal (p/q)^6.
    Cross-multiplied: answer^2 * area_a^3 == volume_a^2 * area_b^3
    where answer is the computed quantity (volume_b or area_b).
    """
    check_equivalent(volume_b**2 * area_a**3, volume_a**2 * area_b**3, "scale factor solids hard tier identity failed")


@generator(
    "scale_factor_similar_areas",
    level=10,
    presets={
        "easy": {"mode": "direct_ratio"},
        "medium": {"mode": "derive_from_lengths"},
        "hard": {"mode": "derive_length_from_areas"},
    },
)
def scale_factor_similar_areas(rng, mode: str = "direct_ratio") -> dict:
    """Scale factor for similar 2D figures: areas scale by k^2."""

    p, q = rng.choice(RATIO_PAIRS)

    if mode == "direct_ratio":
        # Easy: ratio stated directly
        m = rng.randint(2, 10)
        area_a = q * q * m
        answer = p * p * m

        text = (
            f"Figure A and Figure B are similar. The scale factor from Figure A "
            f"to Figure B is {p}/{q}. Figure A has an area of {area_a} cm². "
            f"What is the area of Figure B, in cm²?"
        )

        # Independent check: uses the stated ratio directly
        factor = Fraction(p, q) ** 2
        check_equivalent(area_a * factor, answer, "easy tier check failed")

        # Distractors
        uses_linear = p * q * m
        confuses_ratio_term = q * q * q * q * m
        forgets_to_scale = area_a

        check_distinct(answer, uses_linear, confuses_ratio_term, forgets_to_scale)

        candidates = [uses_linear, confuses_ratio_term, forgets_to_scale]

        working = [
            f"Scale factor = {p}/{q}, so areas are multiplied by ({p}/{q})² = {p * p}/{q * q}.",
            f"Area of B = {area_a} × {p * p}/{q * q} = {answer} cm².",
        ]

    elif mode == "derive_from_lengths":
        # Medium: ratio derived from raw lengths
        t = rng.randint(2, 4)
        length_a = q * t
        length_b = p * t
        m = rng.randint(2, 10)
        area_a = q * q * m
        answer = p * p * m

        text = (
            f"Figure A and Figure B are similar. A side of Figure A measures "
            f"{length_a} cm; the corresponding side of Figure B measures {length_b} cm. "
            f"Figure A has an area of {area_a} cm². What is the area of Figure B, in cm²?"
        )

        # Independent check: uses raw lengths, derives ratio
        ratio = Fraction(length_b, length_a)
        check_equivalent(area_a * ratio**2, answer, "medium tier check failed")

        # Distractors (same three formulas as easy)
        uses_linear = p * q * m
        confuses_ratio_term = q * q * q * q * m
        forgets_to_scale = area_a

        check_distinct(answer, uses_linear, confuses_ratio_term, forgets_to_scale)

        candidates = [uses_linear, confuses_ratio_term, forgets_to_scale]

        working = [
            f"Scale factor = {length_b}/{length_a} = {p}/{q}.",
            f"Areas are multiplied by ({p}/{q})² = {p * p}/{q * q}.",
            f"Area of B = {area_a} × {p * p}/{q * q} = {answer} cm².",
        ]

    else:  # mode == "derive_length_from_areas"
        # Hard: derive linear ratio from area ratio, apply to length
        s_prime = rng.choice([3, 5, 7, 9])
        m = rng.choice([4, 6, 8, 10, 12])

        length_a = p * q * s_prime
        answer = p * p * s_prime
        area_a = q * q * m
        area_b = p * p * m

        text = (
            f"Figure A and Figure B are similar. Figure A has an area of "
            f"{area_a} cm² and Figure B has an area of {area_b} cm². A side "
            f"of Figure A measures {length_a} cm. What is the length of the "
            f"corresponding side of Figure B, in cm?"
        )

        # Independent check: cross-multiplied identity (no square root)
        # answer^2 * area_a == length_a^2 * area_b
        _verify_scale_factor_areas_hard(length_a, answer, area_a, area_b, answer)

        # Distractors
        swaps_p_and_q = q * q * s_prime
        forgets_to_scale = length_a
        confuses_area_for_length = area_b  # p*p*m (guaranteed != answer because m is even, s_prime is odd)

        check_distinct(answer, swaps_p_and_q, forgets_to_scale, confuses_area_for_length)

        candidates = [swaps_p_and_q, forgets_to_scale, confuses_area_for_length]

        working = [
            f"Area scale factor = {area_b}/{area_a} = {p * p}/{q * q}, so the length scale factor is {p}/{q}.",
            f"Side of B = {length_a} × {p}/{q} = {answer} cm.",
        ]

    answer_text = str(answer)
    return {
        "text": text,
        "latex": text,  # For now, plain text suffices
        "answer": answer,
        "answer_text": answer_text,
        "working": working,
        "distractors": pick_distractors(answer_text, candidates),
    }


@generator(
    "scale_factor_similar_solids",
    level=10,
    presets={
        "easy": {"mode": "direct_ratio"},
        "medium": {"mode": "derive_from_lengths"},
        "hard": {"mode": "cross_exponent"},
    },
)
def scale_factor_similar_solids(rng, mode: str = "direct_ratio") -> dict:
    """Scale factor for similar 3D solids: surface areas scale by k^2, volumes by k^3."""

    p, q = rng.choice(RATIO_PAIRS_SOLID)
    kind = rng.choice(["volume", "surface_area"])
    exponent = 3 if kind == "volume" else 2
    unit = "cm³" if kind == "volume" else "cm²"
    kind_label = "volume" if kind == "volume" else "surface area"

    if mode == "direct_ratio":
        # Easy: ratio stated directly
        m = rng.randint(2, 8)
        base = q**exponent * m
        answer = p**exponent * m

        text = (
            f"Solid A and Solid B are similar. The scale factor from Solid A to "
            f"Solid B is {p}/{q}. Solid A has a {kind_label} of {base} {unit}. "
            f"What is Solid B's {kind_label}, in {unit}?"
        )

        # Independent check
        factor = Fraction(p, q) ** exponent
        check_equivalent(base * factor, answer, "easy tier check failed")

        # Distractors
        uses_wrong_exponent_val = p ** (2 if exponent == 3 else 3) * m
        forgets_to_scale = base
        uses_linear_only = p * q ** (exponent - 1) * m

        check_distinct(answer, uses_wrong_exponent_val, forgets_to_scale, uses_linear_only)

        candidates = [uses_wrong_exponent_val, forgets_to_scale, uses_linear_only]

        working = [
            f"Scale factor = {p}/{q}, so the {kind_label} is multiplied by ({p}/{q}){'³' if exponent == 3 else '²'} "
            f"= {p**exponent}/{q**exponent}.",
            f"{kind_label.capitalize()} of B = {base} × {p**exponent}/{q**exponent} = {answer} {unit}.",
        ]

    elif mode == "derive_from_lengths":
        # Medium: ratio derived from raw lengths
        t = rng.randint(2, 4)
        length_a = q * t
        length_b = p * t
        m = rng.randint(2, 8)
        base = q**exponent * m
        answer = p**exponent * m

        text = (
            f"Solid A and Solid B are similar. A side of Solid A measures "
            f"{length_a} cm; the corresponding side of Solid B measures {length_b} cm. "
            f"Solid A has a {kind_label} of {base} {unit}. What is Solid B's "
            f"{kind_label}, in {unit}?"
        )

        # Independent check
        ratio = Fraction(length_b, length_a)
        check_equivalent(base * ratio**exponent, answer, "medium tier check failed")

        # Distractors (same three as easy)
        uses_wrong_exponent_val = p ** (2 if exponent == 3 else 3) * m
        forgets_to_scale = base
        uses_linear_only = p * q ** (exponent - 1) * m

        check_distinct(answer, uses_wrong_exponent_val, forgets_to_scale, uses_linear_only)

        candidates = [uses_wrong_exponent_val, forgets_to_scale, uses_linear_only]

        working = [
            f"Scale factor = {length_b}/{length_a} = {p}/{q}.",
            f"The {kind_label} is multiplied by ({p}/{q}){'³' if exponent == 3 else '²'} = {p**exponent}/{q**exponent}.",
            f"{kind_label.capitalize()} of B = {base} × {p**exponent}/{q**exponent} = {answer} {unit}.",
        ]

    else:  # mode == "cross_exponent"
        # Hard: given one quantity pair, derive linear ratio, apply other exponent
        direction = rng.choice(["areas_to_volume", "volumes_to_area"])
        n = rng.choice([3, 5, 7, 9])
        m = rng.choice([4, 6, 8, 10])

        # Guard: resample if collision occurs
        while (p * n == m) or (p * m == n):
            n = rng.choice([3, 5, 7, 9])

        volume_a = q**3 * n
        volume_b = p**3 * n
        area_a = q**2 * m
        area_b = p**2 * m

        if direction == "areas_to_volume":
            answer = volume_b
            text = (
                f"Solid A and Solid B are similar. Solid A has a surface area of "
                f"{area_a} cm² and Solid B has a surface area of {area_b} cm². "
                f"Solid A has a volume of {volume_a} cm³. What is Solid B's volume, in cm³?"
            )

            # Independent check
            _verify_scale_factor_solids_hard(volume_a, volume_b, area_a, area_b, answer)

            # Distractors
            uses_area_exponent = p * p * n
            forgets_to_scale = volume_a
            confuses_area_for_volume = area_b

            check_distinct(answer, uses_area_exponent, forgets_to_scale, confuses_area_for_volume)

            candidates = [uses_area_exponent, forgets_to_scale, confuses_area_for_volume]

            working = [
                f"Surface area ratio = {area_b}/{area_a} = {p * p}/{q * q}, so the scale factor is {p}/{q}.",
                f"Volumes are multiplied by ({p}/{q})³ = {p**3}/{q**3}.",
                f"Volume of B = {volume_a} × {p**3}/{q**3} = {answer} cm³.",
            ]

        else:  # direction == "volumes_to_area"
            answer = area_b
            text = (
                f"Solid A and Solid B are similar. Solid A has a volume of "
                f"{volume_a} cm³ and Solid B has a volume of {volume_b} cm³. "
                f"Solid A has a surface area of {area_a} cm². What is Solid B's "
                f"surface area, in cm²?"
            )

            # Independent check
            _verify_scale_factor_solids_hard(volume_a, volume_b, area_a, area_b, answer)

            # Distractors
            uses_volume_exponent = p * p * p * m
            forgets_to_scale = area_a
            confuses_volume_for_area = volume_b

            check_distinct(answer, uses_volume_exponent, forgets_to_scale, confuses_volume_for_area)

            candidates = [uses_volume_exponent, forgets_to_scale, confuses_volume_for_area]

            working = [
                f"Volume ratio = {volume_b}/{volume_a} = {p**3}/{q**3}, so the scale factor is {p}/{q}.",
                f"Surface areas are multiplied by ({p}/{q})² = {p * p}/{q * q}.",
                f"Surface area of B = {area_a} × {p * p}/{q * q} = {answer} cm².",
            ]

    answer_text = str(answer)
    return {
        "text": text,
        "latex": text,  # For now, plain text suffices
        "answer": answer,
        "answer_text": answer_text,
        "working": working,
        "distractors": pick_distractors(answer_text, candidates),
    }
