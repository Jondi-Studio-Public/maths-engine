"""Parallelogram area and composite area/perimeter (f10-parallelogram-area,
f10-regular-irregular-area-perimeter).

- f10-parallelogram-area: Given a parallelogram's base and perpendicular
  height, calculate its area. Easy tier uses base and height directly. Medium
  tier adds a distracting slant side. Hard tier is the inverse: given area and
  one dimension, solve for the other.

- f10-regular-irregular-area-perimeter: Given rectilinear shapes (rectangles
  or L-shaped corner-notch decompositions), calculate perimeter or area using
  stated part dimensions. Easy tier is a plain rectangle. Medium tier is an
  L-shape asking for area (subtract-the-notch logic). Hard tier is the same
  L-shape asking for perimeter (counter-intuitive: the notch does not change
  the perimeter).
"""

from __future__ import annotations

from ..core import check, generator, pick_distractors


# ============================================================================
# Skill 1: f10-parallelogram-area
# ============================================================================


def _verify_parallelogram_area(b: int, h: int, answer: int) -> None:
    """Independent check: recompute area via repeated addition (manual loop)
    instead of multiplication. This is a genuinely different computational
    route from the b * h construction."""
    total = 0
    for _ in range(h):
        total += b
    check(total == answer, f"repeated addition {total} != {answer}")


def _verify_parallelogram_inverse(area: int, known_side: int, answer: int) -> None:
    """Independent check for hard tier: multiply the recovered value back
    instead of re-running the division."""
    check(answer * known_side == area, f"recovered {answer} * {known_side} != {area}")


@generator(
    "parallelogram_area",
    level=1,
    presets={
        "easy": dict(mode="direct"),
        "medium": dict(mode="with_slant"),
        "hard": dict(mode="inverse"),
    },
)
def parallelogram_area(rng, mode="direct"):
    """Calculate the area of a parallelogram from base and perpendicular height.
    Hard tier is inverse: given area and one dimension, find the other."""
    if mode not in ("direct", "with_slant", "inverse"):
        raise ValueError(f"unknown mode {mode!r}")

    # Sample base and height; exclude b=h=2 because b+h would equal b*h
    for _ in range(1000):
        b = rng.randint(2, 12)
        h = rng.randint(2, 12)
        if not (b == 2 and h == 2):
            break
    area = b * h

    if mode == "direct":
        # Easy: just base and height stated directly
        text = f"A parallelogram has a base of {b} cm and a perpendicular height of {h} cm. What is its area, in square centimetres?"
        answer = area
        answer_text = str(answer)
        working = [
            f"Area of a parallelogram = base × perpendicular height",
            f"Area = {b} × {h} = {answer} square centimetres",
        ]
        _verify_parallelogram_area(b, h, answer)

        # Misconceptions:
        # - adds_instead_of_multiplies: b + h
        # - uses only base or only height: b * 2 or h * 2 (double one dimension)
        adds_result = b + h
        b_doubled = b * 2
        h_doubled = h * 2

        candidates = [str(adds_result), str(b_doubled), str(h_doubled)]
        distractors = pick_distractors(answer_text, candidates)

    elif mode == "with_slant":
        # Medium: base, height, and a slant side (distractor)
        s = h + rng.randint(1, 5)
        check(s != h, "slant side should never equal height")

        text = f"A parallelogram has a base of {b} cm, a slanted side of {s} cm, and a perpendicular height of {h} cm. What is its area, in square centimetres?"
        answer = area
        answer_text = str(answer)
        working = [
            f"Area of a parallelogram = base × perpendicular height",
            f"The slanted side is not used in the area formula.",
            f"Area = {b} × {h} = {answer} square centimetres",
        ]
        _verify_parallelogram_area(b, h, answer)

        # Misconceptions:
        # - uses_slant_side: b * s
        # - adds_instead_of_multiplies: b + h
        uses_slant = b * s
        check(uses_slant != answer, "uses_slant_side should be distinct from answer")

        adds_result = b + h
        check(adds_result != answer, "adds_instead_of_multiplies should be distinct from answer")

        candidates = [str(uses_slant), str(adds_result)]
        distractors = pick_distractors(answer_text, candidates)

    else:  # mode == "inverse"
        # Hard: given area and one of b/h, solve for the other
        # Decide randomly which to hide
        hide_base = rng.choice([True, False])

        if hide_base:
            # Hide b, ask for it
            text = f"A parallelogram has an area of {area} square centimetres and a perpendicular height of {h} cm. What is its base, in centimetres?"
            answer = b
            answer_text = str(answer)
            unknown_name = "base"
            known_name = "perpendicular height"
            known_value = h
        else:
            # Hide h, ask for it
            text = f"A parallelogram has an area of {area} square centimetres and a base of {b} cm. What is its perpendicular height, in centimetres?"
            answer = h
            answer_text = str(answer)
            unknown_name = "perpendicular height"
            known_name = "base"
            known_value = b

        # Also include the slant-side distractor as in medium
        s = known_value + rng.randint(1, 5)
        check(s != known_value, "slant side should never equal the known side")

        working = [
            f"Area = {unknown_name} × {known_name}",
            f"{area} = {unknown_name} × {known_value}",
            f"{unknown_name} = {area} ÷ {known_value} = {answer}",
        ]
        _verify_parallelogram_inverse(area, known_value, answer)

        # Misconceptions:
        # - uses_slant_side: s (the slant distractor)
        # - wrong_dimension_relation: area - known_side (subtraction instead of division)
        wrong_subtract = area - known_value

        # Guard: if wrong_subtract equals answer, resample b and h
        if wrong_subtract == answer:
            for _ in range(1000):
                b = rng.randint(2, 12)
                h = rng.randint(2, 12)
                if not (b == 2 and h == 2):
                    if hide_base:
                        area = b * h
                        answer = b
                        known_value = h
                    else:
                        area = b * h
                        answer = h
                        known_value = b
                    wrong_subtract = area - known_value
                    if wrong_subtract != answer:
                        break

        candidates = [str(s), str(wrong_subtract)]
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
# Skill 2: f10-regular-irregular-area-perimeter
# ============================================================================


def _verify_lshape_area_via_tiling(L: int, W: int, a: int, b: int, answer: int) -> None:
    """Independent check for medium (area): recompute via additive tiling
    instead of subtraction. Split the L-shape into two rectangles:
    rect1 = L * (W - b) and rect2 = (L - a) * b.
    Their sum equals L*W - a*b."""
    rect1 = L * (W - b)
    rect2 = (L - a) * b
    total = rect1 + rect2
    check(total == answer, f"additive tiling {total} != {answer}")


def _verify_lshape_perimeter_via_edge_sum(L: int, W: int, a: int, b: int, answer: int) -> None:
    """Independent check for hard (perimeter): recompute by summing all
    six edges of the actual L-shape boundary instead of using the
    2*(L+W) shortcut. With notch cut from top-right, the six edges are:
    L, W-b, a, b, L-a, W."""
    edge_sum = L + (W - b) + a + b + (L - a) + W
    check(edge_sum == answer, f"six-edge sum {edge_sum} != {answer}")


@generator(
    "regular_irregular_area_perimeter",
    level=1,
    presets={
        "easy": dict(mode="rectangle"),
        "medium": dict(mode="lshape_area"),
        "hard": dict(mode="lshape_perimeter"),
    },
)
def regular_irregular_area_perimeter(rng, mode="rectangle"):
    """Calculate perimeter or area of rectilinear shapes (rectangles or
    L-shaped corner-notch decompositions)."""
    if mode not in ("rectangle", "lshape_area", "lshape_perimeter"):
        raise ValueError(f"unknown mode {mode!r}")

    if mode == "rectangle":
        # Easy: plain rectangle, randomly ask for area or perimeter
        L = rng.randint(6, 20)
        W = rng.randint(6, 20)

        ask_area = rng.choice([True, False])
        if ask_area:
            text = f"A rectangular sheet is {L} cm by {W} cm. What is its area, in square centimetres?"
            answer = L * W
            answer_text = str(answer)
            working = [
                f"Area of a rectangle = length × width",
                f"Area = {L} × {W} = {answer} square centimetres",
            ]
            # Distractors:
            # - uses_perimeter_for_area: 2*(L+W)
            # - adds instead of multiplies: L + W
            # - only one dimension doubled: L*2 or W*2
            perimeter = 2 * (L + W)
            adds = L + W
            l_only = L * 2
            w_only = W * 2
            candidates = [str(perimeter), str(adds), str(l_only), str(w_only)]
        else:
            text = f"A rectangular sheet is {L} cm by {W} cm. What is its perimeter, in centimetres?"
            answer = 2 * (L + W)
            answer_text = str(answer)
            working = [
                f"Perimeter of a rectangle = 2 × (length + width)",
                f"Perimeter = 2 × ({L} + {W}) = 2 × {L + W} = {answer} centimetres",
            ]
            # Distractors:
            # - uses_area_for_perimeter: L*W
            # - only adds two sides: L + W
            # - only multiplies two sides by 2: 2*L or 2*W
            area = L * W
            adds = L + W
            l_times_two = 2 * L
            w_times_two = 2 * W
            candidates = [str(area), str(adds), str(l_times_two), str(w_times_two)]

        distractors = pick_distractors(answer_text, candidates)

    elif mode == "lshape_area":
        # Medium: L-shape (bounding box with notch), ask for area
        L = rng.randint(6, 20)
        W = rng.randint(6, 20)
        a = rng.randint(2, L - 2)
        b = rng.randint(2, W - 2)

        text = f"A rectangular sheet {L} cm by {W} cm has a rectangular notch {a} cm by {b} cm cut from one corner. What is the area of the remaining shape, in square centimetres?"
        answer = L * W - a * b
        answer_text = str(answer)
        working = [
            f"The L-shaped area = big rectangle − notch",
            f"Area = {L} × {W} − {a} × {b}",
            f"Area = {L * W} − {a * b} = {answer} square centimetres",
        ]
        _verify_lshape_area_via_tiling(L, W, a, b, answer)

        # Misconceptions:
        # - adds_notch_instead_of_subtracts: L*W + a*b
        # - ignores_notch: L*W
        adds_notch = L * W + a * b
        ignores_notch = L * W
        candidates = [str(adds_notch), str(ignores_notch)]
        distractors = pick_distractors(answer_text, candidates)

    else:  # mode == "lshape_perimeter"
        # Hard: L-shape (same construction), ask for perimeter
        # Guard against uses_area_formula collision
        for attempt in range(20):
            L = rng.randint(6, 20)
            W = rng.randint(6, 20)
            a = rng.randint(2, L - 2)
            b = rng.randint(2, W - 2)

            area_value = L * W - a * b
            perimeter_value = 2 * (L + W)

            # Check for collision
            if area_value != perimeter_value:
                break
        else:
            # If still colliding after 20 attempts, use the last sampled values anyway
            # (collision guard fires for this seed)
            pass

        text = f"A rectangular sheet {L} cm by {W} cm has a rectangular notch {a} cm by {b} cm cut from one corner. What is the perimeter of the remaining L-shaped outline, in centimetres?"
        answer = 2 * (L + W)
        answer_text = str(answer)
        working = [
            f"The L-shaped perimeter = the bounding rectangle's perimeter (notch does not change it)",
            f"Perimeter = 2 × ({L} + {W}) = {answer} centimetres",
        ]
        _verify_lshape_perimeter_via_edge_sum(L, W, a, b, answer)

        # Misconceptions:
        # - subtracts_notch_perimeter: 2*(L+W) - 2*(a+b)
        # - uses_area_formula: L*W - a*b (and guard against collision)
        subtracts_notch_perim = 2 * (L + W) - 2 * (a + b)
        uses_area = L * W - a * b

        candidates = [str(subtracts_notch_perim)]

        # Add uses_area_formula only if it's not equal to the answer
        if uses_area != answer:
            candidates.append(str(uses_area))

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
