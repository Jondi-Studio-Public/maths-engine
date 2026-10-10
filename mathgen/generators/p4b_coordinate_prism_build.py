"""Two genuinely new Foundation-Year 10 skills.

- ``coordinate_movement`` (AC9M6SP02, prerequisites ``coordinate-plane-readiness``,
  non-virgin area) -- given a point's coordinates before and after a supplied movement
  in the Cartesian plane, state the change in each coordinate. Response family:
  ``ordered_structures`` (``answer: pair``).

- ``prism_cross_sections`` (AC9M6SP01, prerequisites ``rectangular-prism-volume``,
  virgin area) -- given a description of how a solid's cross-sections parallel to a
  stated direction compare, classify the solid as a right prism or not. Response family:
  ``spatial_classification`` (``answer: mcq``).

Shared vocabulary and structure.
"""

from __future__ import annotations

from ..core import check, generator, latex_answer, pick_distractors


def _signed_paren(n: int) -> str:
    """Wrap a negative number in parentheses for use right after a literal
    "-" in a worked-steps subtraction expression, e.g. "0 - (-6)" instead of
    "0 - -6" (a "- -" double-minus the shared text-quality test
    `test_no_bad_sign_patterns` rejects on every topic). Non-negative values
    are returned unchanged."""
    return f"({n})" if n < 0 else str(n)


# -- Skill 1: coordinate_movement (pair answer) ----------------------------------


def _verify_coordinate_change(x1, y1, x2, y2, dx, dy):
    """Independent check via addition (invert the subtraction operation):
    reconstruct the claimed end point by adding the claimed (dx, dy) back
    onto the start point and assert it equals the given end point exactly."""
    check(
        x1 + dx == x2 and y1 + dy == y2,
        f"reconstructed endpoint ({x1 + dx}, {y1 + dy}) does not match given endpoint ({x2}, {y2})",
    )


@generator(
    "coordinate_movement",
    level=6,
    presets={
        "easy": dict(coord_lo=1, coord_hi=10),
        "medium": dict(coord_lo=-10, coord_hi=10),
        "hard": dict(coord_lo=-10, coord_hi=10, sequence_length=3),
    },
)
def coordinate_movement(rng, coord_lo=-10, coord_hi=10, sequence_length=2):
    """Given a point's coordinates before and after a movement (or sequence of
    movements), state the total change in each coordinate."""
    if coord_hi <= coord_lo:
        raise ValueError("coord_hi must be greater than coord_lo")
    if sequence_length not in (2, 3):
        raise ValueError("sequence_length must be 2 (easy/medium) or 3 (hard)")

    if sequence_length == 2:
        # Easy/medium: single movement
        x1 = rng.randint(coord_lo, coord_hi)
        y1 = rng.randint(coord_lo, coord_hi)
        x2 = rng.randint(coord_lo, coord_hi)
        y2 = rng.randint(coord_lo, coord_hi)
        # Ensure the two points are different
        while (x1, y1) == (x2, y2):
            x2 = rng.randint(coord_lo, coord_hi)
            y2 = rng.randint(coord_lo, coord_hi)

        dx = x2 - x1
        dy = y2 - y1
        _verify_coordinate_change(x1, y1, x2, y2, dx, dy)

        text = f"A point moves from ({x1}, {y1}) to ({x2}, {y2}). What is the change in each coordinate?"
        answer_text = f"dx = {dx}, dy = {dy}"

        # Misconception distractors for easy/medium:
        # - swapped-coordinates: dx and dy swapped (distractor distinct only if dx != dy)
        # - wrong-direction: negated change (always distinct)
        # - omits-subtraction: reports the end coordinates instead of the change
        candidates = [
            f"dx = {dy}, dy = {dx}",  # swapped
            f"dx = {-dx}, dy = {-dy}",  # wrong-direction
            f"dx = {x2}, dy = {y2}",  # omits-subtraction (reports end coords)
        ]

    else:  # sequence_length == 3
        # Hard: two sequential movements
        x1 = rng.randint(coord_lo, coord_hi)
        y1 = rng.randint(coord_lo, coord_hi)
        x2 = rng.randint(coord_lo, coord_hi)
        y2 = rng.randint(coord_lo, coord_hi)
        # Ensure first and second points are different
        while (x1, y1) == (x2, y2):
            x2 = rng.randint(coord_lo, coord_hi)
            y2 = rng.randint(coord_lo, coord_hi)

        x3 = rng.randint(coord_lo, coord_hi)
        y3 = rng.randint(coord_lo, coord_hi)
        # Ensure second and third points are different
        while (x2, y2) == (x3, y3):
            x3 = rng.randint(coord_lo, coord_hi)
            y3 = rng.randint(coord_lo, coord_hi)

        # Total change from start to end
        dx = x3 - x1
        dy = y3 - y1
        _verify_coordinate_change(x1, y1, x3, y3, dx, dy)

        text = (
            f"A point starts at ({x1}, {y1}), moves to ({x2}, {y2}), "
            f"then moves again to ({x3}, {y3}). What is the total change "
            f"in each coordinate from the start to the end?"
        )
        answer_text = f"dx = {dx}, dy = {dy}"

        # Misconception distractors for hard:
        # - swapped-coordinates: dx and dy swapped (distinct if dx != dy)
        # - wrong-direction: negated change (always distinct)
        # - ignores-first-leg: reports only the second leg's change (always distinct since A != B)
        first_leg_dx = x2 - x1
        first_leg_dy = y2 - y1
        second_leg_dx = x3 - x2
        second_leg_dy = y3 - y2

        candidates = [
            f"dx = {dy}, dy = {dx}",  # swapped
            f"dx = {-dx}, dy = {-dy}",  # wrong-direction
            f"dx = {second_leg_dx}, dy = {second_leg_dy}",  # ignores-first-leg
        ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"dx": dx, "dy": dy},
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=[
            f"Change in x: Δx = x_end - x_start = {_signed_paren(x2 if sequence_length == 2 else x3)} - {_signed_paren(x1)} = {dx}",
            f"Change in y: Δy = y_end - y_start = {_signed_paren(y2 if sequence_length == 2 else y3)} - {_signed_paren(y1)} = {dy}",
            f"Therefore: dx = {dx}, dy = {dy}",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


# -- Skill 2: prism_cross_sections (mcq answer) ----------------------------------


SHAPES = ["rectangular", "triangular", "hexagonal", "pentagonal", "square"]


def _verify_prism_classification_measurements(measurements, is_constant):
    """Independent check using set-based equality (genuinely different route
    from the == comparison used during construction). Only applies to medium/hard
    tiers where measurements are actually used."""
    check(
        (len(set(measurements)) == 1) == is_constant,
        f"classification mismatch: measurements {measurements}, is_constant {is_constant}",
    )


@generator(
    "prism_cross_sections",
    level=6,
    presets={
        "easy": dict(shape_pool=SHAPES, num_measurements=1),
        "medium": dict(shape_pool=SHAPES, num_measurements=2),
        "hard": dict(shape_pool=SHAPES, num_measurements=3),
    },
)
def prism_cross_sections(rng, shape_pool=SHAPES, num_measurements=2):
    """Given a description of how a solid's cross-sections compare, classify
    the solid as a right prism or not."""
    if not shape_pool:
        raise ValueError("shape_pool must be non-empty")
    unknown = [s for s in shape_pool if s not in SHAPES]
    if unknown:
        raise ValueError(f"unknown shapes {unknown!r}; expected subset of {SHAPES}")
    if num_measurements not in (1, 2, 3):
        raise ValueError("num_measurements must be 1 (easy), 2 (medium), or 3 (hard)")

    shape = rng.choice(shape_pool)

    if num_measurements == 1:
        # Easy: direct keyword (constant vs tapering)
        is_constant = rng.random() < 0.5
        if is_constant:
            text = (
                f"A solid's cross-sections, taken perpendicular to its length, "
                f"are always the same {shape} shape and the same size, from one end "
                f"to the other. Is this solid a right prism or not?"
            )
            answer = "a right prism"
        else:
            text = (
                f"A solid's cross-sections, taken perpendicular to its length, "
                f"are {shape} in shape but get smaller and smaller until the solid "
                f"comes to a point at one end. Is this solid a right prism or not?"
            )
            answer = "not a right prism"

    elif num_measurements == 2:
        # Medium: numeric two-point comparison (no keyword)
        m1 = rng.randint(2, 12)
        is_constant = rng.random() < 0.5
        if is_constant:
            m2 = m1
        else:
            m2 = rng.randint(2, 12)
            while m2 == m1:
                m2 = rng.randint(2, 12)

        text = (
            f"A solid's cross-sections, taken perpendicular to its length, "
            f"are always {shape} in shape. They measure {m1} cm across at one end "
            f"and {m2} cm across at the other end. Is this solid a right prism or not?"
        )
        answer = "a right prism" if is_constant else "not a right prism"
        measurements = [m1, m2]
        _verify_prism_classification_measurements(measurements, is_constant)

    else:  # num_measurements == 3
        # Hard: three-point comparison
        m1 = rng.randint(2, 12)
        m2 = rng.randint(2, 12)
        m3 = rng.randint(2, 12)

        is_constant = rng.random() < 0.5
        if is_constant:
            # All three equal
            m2 = m1
            m3 = m1
        else:
            # Exactly one pairwise equality holds, at least one differs
            # Pick one of: (m1==m2!=m3), (m1!=m2==m3), (m1==m3!=m2)
            parity_choice = rng.choice([0, 1, 2])
            if parity_choice == 0:
                # m1 == m2 != m3
                m2 = m1
                while m3 == m1:
                    m3 = rng.randint(2, 12)
            elif parity_choice == 1:
                # m1 != m2 == m3
                m3 = m2
                while m1 == m2:
                    m1 = rng.randint(2, 12)
            else:
                # m1 == m3 != m2
                m3 = m1
                while m2 == m1:
                    m2 = rng.randint(2, 12)

        text = (
            f"A solid's cross-sections, taken perpendicular to its length, "
            f"are always {shape} in shape. They measure {m1} cm across at the start, "
            f"{m2} cm across in the middle, and {m3} cm across at the end. "
            f"Is this solid a right prism or not?"
        )
        answer = "a right prism" if is_constant else "not a right prism"
        measurements = [m1, m2, m3]
        _verify_prism_classification_measurements(measurements, is_constant)

    # Fixed three MCQ options: exactly the two classification labels plus the foil
    fixed_options = ["a right prism", "not a right prism", "there is not enough information to tell"]
    # The two distractors are the non-answer options from the fixed 3
    candidates = [opt for opt in fixed_options if opt != answer]
    check(len(candidates) == 2, f"expected exactly 2 distractors, got {len(candidates)}")

    working = [
        f"Based on the description: {text}",
        f"Answer: {answer}",
    ]

    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=pick_distractors(answer, candidates, n=2),
    )
