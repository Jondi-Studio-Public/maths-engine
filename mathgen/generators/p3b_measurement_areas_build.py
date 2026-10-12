"""Area formulas and irregular decomposition, both senior ``acara-senior-v8-4``, domain ``measurement``,
``answer: number``.

Two generators:

- ``use_formulas_to_calculate_areas`` (``senior-use-formulas-to-calculate-areas-
  of-regular-shapes``, ACMEM093) -- calculate the area of a rectangle, square,
  triangle, parallelogram, or trapezium using the shape's own area formula.
  **Circles and sectors are explicitly excluded** from this graded set because
  their formulas produce irrational (pi-based) results; the five exact-arithmetic
  shapes (rectangles, squares, triangles, parallelograms, trapeziums) provide a
  rich enough difficulty ladder and keep every canonical answer exact.

- ``find_the_area_of_irregular_figures`` (``senior-find-the-area-of-irregular-
  figures-by-decomposition``, ACMEM094) -- find the area of an irregular figure
  by decomposing it into regular components (rectangles and/or right triangles)
  and adding or subtracting their areas. The figure is always described
  numerically (named components and stated dimensions), never as a drawn diagram.

Both generators use exact ``Fraction`` arithmetic and format ``answer_text`` from
the result with explicit, bounded fixed-point decimal places (never
``str(float(...))``, which can silently switch to scientific notation below
``1e-4``). Every dimension is an integer by construction, so answers are integers
or simple decimals (half-integers from the `1/2` factors) terminating within
1 decimal place.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    draw_until,
    exact_decimal_string,
    generator,
    numeric_answer,
    pick_distractors,
    rounded_decimal_string,
    stated_precision_string,
)


def _shaped_distractors(answer_text: str, candidates: list[str]) -> list[str]:
    """Misconception distractors shaped like the key: a whole-number key keeps
    only whole-number options, a half-integer key shows every option to one
    decimal place ("36.0" beside "63.5"), so the decimal point never gives the
    key away. Doubling/halving/off-by-one slips top the list up when the shape
    filter drops a candidate."""
    answer = Fraction(answer_text)
    places = 1 if "." in answer_text else 0
    scale = 10**places
    pool = [Fraction(c) for c in candidates] + [answer * 2, answer / 2, answer + 1, answer - 1]
    return pick_distractors(
        answer_text,
        [stated_precision_string(v, places) for v in pool if v > 0 and (v * scale).denominator == 1],
    )


# --- use_formulas_to_calculate_areas -----------------------------------------


def _verify_rectangle_or_square(length: Fraction, width: Fraction, expected: Fraction) -> None:
    """Independent check for rectangle/square: recompute via width*length
    (swapped order from construction's length*width)."""
    recomputed = width * length
    check(recomputed == expected, "rectangle/square area mismatch")


def _verify_triangle_or_parallelogram(base: Fraction, height: Fraction, is_triangle: bool, expected: Fraction) -> None:
    """Independent check for triangle/parallelogram: recompute via different
    grouping. For triangle, compute as (base / 2) * height instead of
    (base * height) / 2."""
    if is_triangle:
        recomputed = Fraction(base, 2) * height
    else:
        recomputed = height * base
    check(recomputed == expected, "triangle/parallelogram area mismatch")


def _verify_trapezium(a: Fraction, b: Fraction, height: Fraction, expected: Fraction) -> None:
    """Independent check for trapezium: recompute as the sum of two half-rectangles.
    Construction may compute (a + b) / 2 * h; verifier computes a/2 * h + b/2 * h."""
    recomputed = Fraction(a, 2) * height + Fraction(b, 2) * height
    check(recomputed == expected, "trapezium area mismatch")


def _build_easy_rectangle_square(rng, side_hi: int) -> dict:
    """Easy: rectangle or square area (l x w or s x s) -- the single most
    familiar formula."""
    shape = rng.choice(["rectangle", "square"])
    if shape == "square":
        side = rng.randint(2, side_hi)
        length = width = Fraction(side)
        text = f"A square has sides of {side} cm. What is its area, in cm²?"
    else:
        length = Fraction(rng.randint(2, side_hi))
        width = draw_until(
            rng,
            lambda r: Fraction(r.randint(2, side_hi)),
            lambda candidate: candidate != length,
            what="a rectangle width different from the length",
        )
        text = f"A rectangle is {int(length)} cm long and {int(width)} cm wide. What is its area, in cm²?"

    answer = length * width
    _verify_rectangle_or_square(length, width, answer)
    answer_text = exact_decimal_string(answer, 1)

    # Misconceptions:
    #  - uses-perimeter: sums the sides instead of applying the area formula.
    #    For a square: perimeter = 4*side; for rectangle: perimeter = 2*(l+w).
    #  - a doubled-answer filler to guarantee >=2 distractors (not a named
    #    misconception, but ensures >=2 on most seeds).
    if shape == "square":
        uses_perimeter = 4 * length
    else:
        uses_perimeter = 2 * (length + width)
    doubled_area = answer * 2

    candidates = [
        rounded_decimal_string(uses_perimeter, 1),
        rounded_decimal_string(doubled_area, 1),
    ]

    return dict(
        text=text,
        latex=rf"{int(length)}\,\mathrm{{cm}} \times {int(width)}\,\mathrm{{cm}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Area = length × width = {int(length)} × {int(width)} = {answer_text} cm²."],
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_medium_triangle_parallelogram(rng, dim_hi: int) -> dict:
    """Medium: triangle or parallelogram area (1/2 x b x h or b x h) -- a
    genuinely different formula the learner must select based on the shape named
    in the question, and, for the triangle case, apply the 1/2 factor."""
    shape = rng.choice(["triangle", "parallelogram"])
    base = Fraction(rng.randint(2, dim_hi))
    height = Fraction(rng.randint(2, dim_hi))

    if shape == "triangle":
        answer = Fraction(base * height, 2)
        text = f"A triangle has a base of {int(base)} cm and a height of {int(height)} cm. What is its area, in cm²?"
        # Misconceptions:
        #  - uses-perimeter: sums base + height instead of using area formula.
        #  - omits-half-factor: applies base * height without the 1/2 factor.
        #  - filler to ensure >=2 distractors.
        uses_perimeter = base + height
        omits_half_factor = base * height
        filler = (base + height) * 2  # perimeter * 2
    else:
        answer = base * height
        text = (
            f"A parallelogram has a base of {int(base)} cm and a height of {int(height)} cm. What is its area, in cm²?"
        )
        # Misconceptions:
        #  - uses-perimeter: sums base + height.
        #  - filler candidates.
        uses_perimeter = base + height
        filler = (base + height) * 2
        omits_half_factor = None  # not applicable for parallelogram

    _verify_triangle_or_parallelogram(base, height, is_triangle=(shape == "triangle"), expected=answer)
    answer_text = exact_decimal_string(answer, 1)

    if omits_half_factor is not None:
        candidates = [
            rounded_decimal_string(uses_perimeter, 1),
            rounded_decimal_string(omits_half_factor, 1),
            rounded_decimal_string(filler, 1),
        ]
    else:
        candidates = [
            rounded_decimal_string(uses_perimeter, 1),
            rounded_decimal_string(filler, 1),
        ]

    return dict(
        text=text,
        latex=rf"{int(base)}\,\mathrm{{cm}} \times {int(height)}\,\mathrm{{cm}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Area = base × height {'÷ 2' if shape == 'triangle' else ''} = {answer_text} cm²."],
        distractors=_shaped_distractors(answer_text, candidates),
    )


def _build_hard_trapezium(rng, side_hi: int) -> dict:
    """Hard: trapezium area (1/2 x (a + b) x h) -- two independently stated
    parallel side lengths plus a height. This requires the learner to combine
    two side lengths before applying the height and the 1/2 factor, a genuine
    extra reasoning step over medium."""
    a = Fraction(rng.randint(2, side_hi))
    b = Fraction(rng.randint(2, side_hi))
    # Force a != b so that the "uses-one-parallel-side" misconception is always
    # a genuinely different value from the correct answer.
    while b == a:
        b = Fraction(rng.randint(2, side_hi))
    height = Fraction(rng.randint(2, side_hi))

    answer = Fraction(a + b, 2) * height
    _verify_trapezium(a, b, height, answer)
    answer_text = exact_decimal_string(answer, 1)

    text = (
        f"A trapezium has parallel sides of {int(a)} cm and {int(b)} cm, "
        f"and a height of {int(height)} cm. What is its area, in cm²?"
    )

    # Misconceptions:
    #  - uses-perimeter: sums the three stated dimensions.
    #  - omits-half-factor: applies (a + b) * h without the 1/2 factor.
    #  - uses-one-parallel-side: uses only one of the two parallel sides (e.g., a * h).
    uses_perimeter = a + b + height
    omits_half_factor = (a + b) * height
    uses_one_parallel_side = a * height

    candidates = [
        rounded_decimal_string(uses_perimeter, 1),
        rounded_decimal_string(omits_half_factor, 1),
        rounded_decimal_string(uses_one_parallel_side, 1),
    ]

    return dict(
        text=text,
        latex=rf"\frac{{1}}{{2}}({int(a)} + {int(b)}) \times {int(height)}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Area = 1/2 × (a + b) × height = 1/2 × ({int(a)} + {int(b)}) × {int(height)} = {answer_text} cm²."],
        distractors=_shaped_distractors(answer_text, candidates),
    )


@generator(
    "use_formulas_to_calculate_areas",
    level=11,
    presets={
        "easy": dict(mode="rectangle_square"),
        "medium": dict(mode="triangle_parallelogram"),
        "hard": dict(mode="trapezium"),
    },
)
def use_formulas_to_calculate_areas(
    rng,
    mode="rectangle_square",
    side_hi=30,
    dim_hi=25,
):
    """Calculate the area of a rectangle, square, triangle, parallelogram, or
    trapezium using the shape's own area formula. Circles and sectors are not
    assessed."""
    if mode == "rectangle_square":
        if side_hi < 3:
            raise ValueError(
                "use_formulas_to_calculate_areas needs side_hi >= 3 for mode='rectangle_square': "
                "a rectangle's two sides are different whole numbers from 2..side_hi"
            )
        return _build_easy_rectangle_square(rng, side_hi)
    if mode == "triangle_parallelogram":
        return _build_medium_triangle_parallelogram(rng, dim_hi)
    if mode == "trapezium":
        return _build_hard_trapezium(rng, dim_hi)
    raise ValueError(f"unknown mode: {mode!r}")


# --- find_the_area_of_irregular_figures --------------------------------------


def _verify_easy_decomposition(area_a: Fraction, area_b: Fraction, expected: Fraction) -> None:
    """Independent check for easy-tier two-rectangle sum: recompute in reverse
    order of construction."""
    recomputed = area_b + area_a
    check(recomputed == expected, "easy decomposition mismatch")


def _verify_medium_decomposition(rect_area: Fraction, tri_area: Fraction, expected: Fraction) -> None:
    """Independent check for medium-tier rectangle+triangle sum: recompute as
    separate operation from raw values."""
    recomputed = tri_area + rect_area
    check(recomputed == expected, "medium decomposition mismatch")


def _verify_hard_decomposition(big_area: Fraction, cut_area: Fraction, expected: Fraction) -> None:
    """Independent check for hard-tier subtractive decomposition: recompute as
    a fresh subtraction from raw areas, not by replaying construction's
    accumulation."""
    recomputed = big_area - cut_area
    check(recomputed == expected, "hard decomposition mismatch")


def _build_easy_two_rectangles(rng, rect_w_hi: int, rect_h_hi: int) -> dict:
    """Easy: additive decomposition of two rectangles (an "L-shape", described
    as two named rectangular sections with stated dimensions) -- sum of two
    rectangle areas."""
    rect_a_w = Fraction(rng.randint(2, rect_w_hi))
    rect_a_h = Fraction(rng.randint(2, rect_h_hi))
    rect_b_w = Fraction(rng.randint(2, rect_w_hi))
    rect_b_h = Fraction(rng.randint(2, rect_h_hi))

    area_a = rect_a_w * rect_a_h
    area_b = rect_b_w * rect_b_h
    answer = area_a + area_b
    _verify_easy_decomposition(area_a, area_b, answer)
    answer_text = exact_decimal_string(answer, 1)

    text = (
        f"An L-shaped figure consists of two rectangular sections: "
        f"Section A is {int(rect_a_w)} cm by {int(rect_a_h)} cm, and "
        f"Section B is {int(rect_b_w)} cm by {int(rect_b_h)} cm. "
        f"What is the total area, in cm²?"
    )

    # Misconceptions:
    #  - omits-a-component: drops one of the two rectangles from the sum.
    #    We need area_a != area_b so these are distinct distractors.
    #  - double-counts-shared-region: NOT exercisable here without an explicit
    #    overlap region defined; for a plain L-shape sum, there is no
    #    well-defined "shared region" concept. We omit this misconception.
    #  - filler to ensure >=2 distractors.
    omits_a = area_b
    omits_b = area_a
    filler = answer * 2

    candidates = [
        rounded_decimal_string(omits_a if area_a != area_b else filler, 1),
        rounded_decimal_string(omits_b if area_a != area_b else filler, 1),
        rounded_decimal_string(filler, 1),
    ]

    return dict(
        text=text,
        latex=rf"\text{{L-shaped figure: }} {int(rect_a_w)}\,\mathrm{{cm}} \times {int(rect_a_h)}\,\mathrm{{cm}} + {int(rect_b_w)}\,\mathrm{{cm}} \times {int(rect_b_h)}\,\mathrm{{cm}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Section A area = {int(rect_a_w)} × {int(rect_a_h)} = {int(area_a)} cm²",
            f"Section B area = {int(rect_b_w)} × {int(rect_b_h)} = {int(area_b)} cm²",
            f"Total = {int(area_a)} + {int(area_b)} = {answer_text} cm²",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_medium_rectangle_triangle(rng, rect_w_hi: int, rect_h_hi: int, tri_w_hi: int, tri_h_hi: int) -> dict:
    """Medium: additive decomposition of a rectangle and a right triangle
    (e.g. "a rectangular section topped by a triangular section") -- genuinely
    different reasoning from easy: two *different* area formulas must be
    applied then summed."""
    rect_w = Fraction(rng.randint(2, rect_w_hi))
    rect_h = Fraction(rng.randint(2, rect_h_hi))
    tri_base = Fraction(rng.randint(2, tri_w_hi))
    tri_height = Fraction(rng.randint(2, tri_h_hi))

    rect_area = rect_w * rect_h
    tri_area = Fraction(tri_base * tri_height, 2)
    answer = rect_area + tri_area
    _verify_medium_decomposition(rect_area, tri_area, answer)
    answer_text = exact_decimal_string(answer, 1)

    text = (
        f"An irregular figure consists of a rectangular base ({int(rect_w)} cm by {int(rect_h)} cm) "
        f"topped by a triangular point (base {int(tri_base)} cm, height {int(tri_height)} cm). "
        f"What is the total area, in cm²?"
    )

    # Misconceptions:
    #  - omits-a-component: drops the rectangle or the triangle from the sum.
    #  - filler candidates.
    omits_rect = tri_area
    omits_tri = rect_area
    filler = answer * 2

    candidates = [
        rounded_decimal_string(omits_rect, 1),
        rounded_decimal_string(omits_tri, 1),
        rounded_decimal_string(filler, 1),
    ]

    return dict(
        text=text,
        latex=rf"\text{{Rectangle + Triangle}}: {int(rect_w)}\,\mathrm{{cm}} \times {int(rect_h)}\,\mathrm{{cm}} + \frac{{1}}{{2}} \times {int(tri_base)}\,\mathrm{{cm}} \times {int(tri_height)}\,\mathrm{{cm}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Rectangle area = {int(rect_w)} × {int(rect_h)} = {int(rect_area)} cm²",
            f"Triangle area = 1/2 × {int(tri_base)} × {int(tri_height)} = {rounded_decimal_string(tri_area, 1)} cm²",
            f"Total = {int(rect_area)} + {rounded_decimal_string(tri_area, 1)} = {answer_text} cm²",
        ],
        distractors=_shaped_distractors(answer_text, candidates),
    )


def _build_hard_subtractive_notch(rng, big_w_hi: int, big_h_hi: int, cut_w_hi: int, cut_h_hi: int) -> dict:
    """Hard: subtractive decomposition -- a large rectangle with a smaller
    rectangle or right triangle cut out (a "notch"), the total area is the
    large area minus the removed piece's area -- a genuine extra reasoning
    step (subtract rather than add), the common error being to add every
    stated piece regardless of whether it is being removed or included."""
    big_w = Fraction(rng.randint(4, big_w_hi))
    big_h = Fraction(rng.randint(4, big_h_hi))
    # Force cut to be strictly inside big so answer > 0.
    cut_w = Fraction(rng.randint(1, min(int(big_w) - 1, cut_w_hi)))
    cut_h = Fraction(rng.randint(1, min(int(big_h) - 1, cut_h_hi)))

    cut_is_triangle = rng.choice([False, True])
    big_area = big_w * big_h
    if cut_is_triangle:
        cut_area = Fraction(cut_w * cut_h, 2)
        cut_description = f"a triangular notch (base {int(cut_w)} cm, height {int(cut_h)} cm)"
    else:
        cut_area = cut_w * cut_h
        cut_description = f"a rectangular notch ({int(cut_w)} cm by {int(cut_h)} cm)"

    answer = big_area - cut_area
    _verify_hard_decomposition(big_area, cut_area, answer)
    answer_text = exact_decimal_string(answer, 1)

    text = (
        f"An irregular figure is a {int(big_w)} cm by {int(big_h)} cm rectangle "
        f"with {cut_description} removed from one corner. "
        f"What is the remaining area, in cm²?"
    )

    # Misconceptions:
    #  - wrong-operation: adds the cut-out piece instead of subtracting.
    #  - omits-a-component: calculates only the big rectangle or only the cut-out.
    wrong_operation = big_area + cut_area
    omits_big = cut_area
    omits_cut = big_area

    candidates = [
        rounded_decimal_string(wrong_operation, 1),
        rounded_decimal_string(omits_big, 1),
        rounded_decimal_string(omits_cut, 1),
    ]

    cut_label = "Triangle" if cut_is_triangle else "Notch"
    return dict(
        text=text,
        latex=rf"{int(big_w)}\,\mathrm{{cm}} \times {int(big_h)}\,\mathrm{{cm}} - \text{{{cut_label} }}{int(cut_w)} \times {int(cut_h)}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Big rectangle area = {int(big_w)} × {int(big_h)} = {int(big_area)} cm²",
            f"Cut-out area = {rounded_decimal_string(cut_area, 1)} cm²",
            f"Remaining area = {int(big_area)} - {rounded_decimal_string(cut_area, 1)} = {answer_text} cm²",
        ],
        distractors=_shaped_distractors(answer_text, candidates),
    )


@generator(
    "find_the_area_of_irregular_figures",
    level=11,
    presets={
        "easy": dict(mode="two_rectangles"),
        "medium": dict(mode="rectangle_triangle"),
        "hard": dict(mode="notch"),
    },
)
def find_the_area_of_irregular_figures(
    rng,
    mode="two_rectangles",
    rect_w_hi=20,
    rect_h_hi=20,
    tri_w_hi=15,
    tri_h_hi=15,
    big_w_hi=25,
    big_h_hi=25,
    cut_w_hi=15,
    cut_h_hi=15,
):
    """Find the area of an irregular figure by decomposing it into regular
    shapes (rectangles and/or right triangles) and adding or subtracting their
    areas."""
    if mode == "two_rectangles":
        return _build_easy_two_rectangles(rng, rect_w_hi, rect_h_hi)
    if mode == "rectangle_triangle":
        return _build_medium_rectangle_triangle(rng, rect_w_hi, rect_h_hi, tri_w_hi, tri_h_hi)
    if mode == "notch":
        return _build_hard_subtractive_notch(rng, big_w_hi, big_h_hi, cut_w_hi, cut_h_hi)
    raise ValueError(f"unknown mode: {mode!r}")
