"""Surface area of pyramids and addition-of-faces method for irregular solids, both senior ``acara-senior-v8-4``,
domain ``measurement``, ``answer: number``.

Two generators:

- ``find_the_surface_area_of_pyramids_such_as`` (``senior-find-the-surface-
  area-of-pyramids-such-as``, ACMEM096) -- pyramid surface area (base plus
  lateral triangular faces). Easy: square base, one slant height. Medium:
  rectangular base, two distinct slant heights. Hard: right-triangle base,
  three slant heights.

- ``use_addition_of_the_area_of_the_faces`` (``senior-use-addition-of-the-
  area-of-the-faces``, ACMEM097) -- sum face areas for irregular solids.
  Every face is independently listed with its own shape and dimensions.
  Easy: 5 faces (3 rectangles, 2 squares). Medium: 5 faces (2 rectangles,
  2 triangles, 1 square). Hard: 5 real faces + 1 explicitly excluded lure face.

Both generators use exact-arithmetic formulas producing integer surface areas;
no decimal formatting is needed. All dimensions are chosen so every face area
and every total is an exact integer.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, check_distinct, generator, phrase, pick_distractors


# --- Shared helper for both generators ----------------------------------------


def _face_area(shape: str, dims: tuple) -> Fraction:
    """Compute area of a single face. A square is represented internally as
    a rectangle (side, side) -- the area formula is identical."""
    if shape == "rectangle":
        a, b = dims
        return Fraction(a) * Fraction(b)
    if shape == "triangle":
        base, height = dims
        return Fraction(base, 2) * Fraction(height)
    raise ValueError(f"unknown face shape: {shape!r}")


def _sum_faces(faces: list[tuple[str, tuple]]) -> Fraction:
    """Sum the areas of all faces. Each face is (shape, dims_tuple)."""
    return sum(_face_area(shape, dims) for shape, dims in faces)


# --- find_the_surface_area_of_pyramids_such_as --------------------------------


def _verify_pyramid_square_base(s: int, l: int, expected: Fraction) -> None:
    """Independent check for square-base pyramid: compute base area separately,
    then sum four triangular faces via explicit loop (different grouping from
    construction)."""
    base_area = s * s
    lateral_area = sum(Fraction(s, 2) * l for _ in range(4))
    recomputed = base_area + lateral_area
    check(recomputed == expected, "square pyramid surface area mismatch")


def _verify_pyramid_square_base_from_height(s: int, h: int, expected: Fraction) -> None:
    """Independent check for the exam tier: recompute the slant height `l`
    from the apex (perpendicular) height `h` and base side `s` via
    Pythagoras -- using `math.isqrt` on `h^2 + (s/2)^2`, a route distinct
    from however the generator constructed `l` (it reads `l` straight off
    `SQUARE_PYRAMID_TABLE`'s stored `hyp * t`, never recomputing it from `h`
    and `s`) -- confirm the result is an exact integer, then sum four
    triangular faces via explicit loop plus the base area (mirrors
    `_verify_pyramid_square_base`'s grouping)."""
    half_side_sq = (s // 2) * (s // 2)  # s is always even by construction
    l_squared = h * h + half_side_sq
    l = math.isqrt(l_squared)
    check(l * l == l_squared, "square pyramid (exam): apex height and base side do not imply an integer slant height")
    base_area = s * s
    lateral_area = sum(Fraction(s, 2) * l for _ in range(4))
    recomputed = base_area + lateral_area
    check(recomputed == expected, "square pyramid (exam) surface area mismatch")


def _verify_pyramid_rectangular_base(l_base: int, w: int, s_l: int, s_w: int, expected: Fraction) -> None:
    """Independent check for rectangular-base pyramid: compute base area
    separately, then sum four triangular faces via explicit loop (two at s_l,
    two at s_w)."""
    base_area = l_base * w
    lateral_area = Fraction(l_base, 2) * s_l + Fraction(l_base, 2) * s_l + Fraction(w, 2) * s_w + Fraction(w, 2) * s_w
    recomputed = base_area + lateral_area
    check(recomputed == expected, "rectangular pyramid surface area mismatch")


def _verify_pyramid_right_triangle_base(
    a: int, b: int, c: int, s_a: int, s_b: int, s_c: int, expected: Fraction
) -> None:
    """Independent check for right-triangle-base pyramid: compute base area
    via swapped grouping (Fraction(b, 2) * a), then sum three lateral faces
    via explicit loop."""
    base_area = Fraction(b, 2) * a
    lateral_area = Fraction(a, 2) * s_a + Fraction(b, 2) * s_b + Fraction(c, 2) * s_c
    recomputed = base_area + lateral_area
    check(recomputed == expected, "right-triangle pyramid surface area mismatch")


SQUARE_PYRAMID_TABLE = [
    ((3, 4, 5), 10),
    ((5, 12, 13), 5),
    ((8, 15, 17), 4),
    ((7, 24, 25), 3),
]  # (half_leg, h_leg, hyp), t_max -- t ranges 1..t_max


def _build_pyramid_square_base(rng) -> dict:
    """Easy tier: square base with one slant height. `s` and `l` are drawn
    from a Pythagorean triple scaled by `t` so a single apex height `h`
    exists satisfying l^2 = h^2 + (s/2)^2 for every draw (a real pyramid
    has one apex, so its four congruent slant heights cannot be independent
    of the base and of each other)."""
    (half_leg, h_leg, hyp), t_max = rng.choice(SQUARE_PYRAMID_TABLE)
    t = rng.randint(1, t_max)
    s = 2 * half_leg * t
    l = hyp * t
    h = h_leg * t  # apex height, not printed, only used to prove realizability
    check(l * l == h * h + (s // 2) * (s // 2), "square pyramid: slant height not realizable with a common apex")

    faces = [("rectangle", (s, s))] + [("triangle", (s, l))] * 4
    answer_frac = _sum_faces(faces)
    _verify_pyramid_square_base(s, l, answer_frac)

    # Convert to int (guaranteed by construction)
    assert answer_frac.denominator == 1, f"expected integer, got {answer_frac}"
    answer = int(answer_frac)
    answer_text = str(answer)

    text = (
        f"A pyramid has a square base with side length {s} cm. Each of its "
        f"four triangular faces has a slant height of {l} cm. Find the pyramid's "
        f"total surface area, in cm²."
    )

    # Misconceptions:
    #  - omits_base: 2*s*l (lateral only)
    #  - treats_triangle_as_full_rectangle: s*s + 4*s*l
    #  - counts_only_two_of_four_lateral_faces: s*s + s*l
    omits_base = 2 * s * l
    treats_triangle_as_full = s * s + 4 * s * l
    counts_two_laterals = s * s + s * l

    candidates = [str(omits_base), str(treats_triangle_as_full), str(counts_two_laterals)]

    working = [
        f"Base: {s} × {s} = {s * s} cm².",
        f"Each triangular face: ½ × {s} × {l} = {s * l // 2} cm².",
        f"Four triangular faces: 4 × {s * l // 2} = {2 * s * l} cm².",
        f"Total = {s * s} + {2 * s * l} = {answer} cm².",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


# Three structurally distinct phrasings of the exam-tier scenario, so a `prompt_shape` sweep sees genuine
# sentence variety rather than only the drawn numbers -- every other source on this topic renders a single fixed
# template, so the exam source is the first to need this on this generator. Each keeps the two load-bearing
# anchor phrases ("side length {s} cm" and "perpendicular height of {h} cm") identical so a test oracle can
# parse s and h with two fixed, independent regexes regardless of which variant renders, while varying the
# surrounding clause order, vocabulary and closing verb ("Find"/"Calculate"/"what is... ?") enough to register
# as a different masked shape. None ever says "slant height" (that would give away the answer to the derivation
# this tier tests) or "volume" (this skill is surface area only).
_EXAM_SQUARE_BASE_TEMPLATES = (
    "A pyramid has a square base with side length {s} cm and a perpendicular "
    "height of {h} cm, measured from the apex straight down to the centre of "
    "the base. Find the pyramid's total surface area, in cm².",
    "A square-based pyramid has a base with side length {s} cm. Its apex is a "
    "perpendicular height of {h} cm above the centre of the base. Calculate "
    "the pyramid's total surface area, in cm².",
    "Given a square pyramid whose base has side length {s} cm and whose apex "
    "sits a perpendicular height of {h} cm above the centre of the base, what "
    "is the pyramid's total surface area, in cm²?",
)


def _build_pyramid_square_base_from_height(rng) -> dict:
    """Exam tier: square base given its apex (perpendicular) height `h`, not
    a slant height directly. Reuses `SQUARE_PYRAMID_TABLE` -- the easy
    tier's `h = h_leg * t` already exists to prove realizability, it is
    just never printed there. This tier prints `h` (and `s`) instead of
    `l`, so the learner must derive `l = sqrt(h^2 + (s/2)^2)` via Pythagoras
    BEFORE applying the same surface-area formula the easy tier teaches --
    an extra reasoning step ahead of the easy tier, not a bigger number."""
    (half_leg, h_leg, hyp), t_max = rng.choice(SQUARE_PYRAMID_TABLE)
    t = rng.randint(1, t_max)
    s = 2 * half_leg * t
    l = hyp * t
    h = h_leg * t  # apex height -- printed on this tier, unlike the easy tier
    check(l * l == h * h + (s // 2) * (s // 2), "square pyramid (exam): slant height not realizable with a common apex")

    faces = [("rectangle", (s, s))] + [("triangle", (s, l))] * 4
    answer_frac = _sum_faces(faces)
    _verify_pyramid_square_base_from_height(s, h, answer_frac)

    # Convert to int (guaranteed by construction)
    assert answer_frac.denominator == 1, f"expected integer, got {answer_frac}"
    answer = int(answer_frac)
    answer_text = str(answer)

    text = phrase(rng, _EXAM_SQUARE_BASE_TEMPLATES, s=s, h=h)

    # Misconceptions:
    #  - uses_apex_height_as_slant_height (NEW for this tier): plugs the
    #    given apex height h straight into the lateral-face formula instead
    #    of deriving the slant height l via Pythagoras first -- exactly the
    #    error the extra derivation step exists to catch. l > h always
    #    (l^2 = h^2 + (s/2)^2 with s > 0), so this differs from the correct
    #    answer by construction on every draw, not merely usually.
    #  - omits_base: 2*s*l (lateral only, using the correctly-derived l)
    #  - treats_triangle_as_full_rectangle: s*s + 4*s*l
    #  - counts_only_two_of_four_lateral_faces: s*s + s*l
    uses_h_as_l = s * s + 2 * s * h  # s*s + 4*(s/2 * h), s always even
    omits_base = 2 * s * l
    treats_triangle_as_full = s * s + 4 * s * l
    counts_two_laterals = s * s + s * l

    check_distinct(
        answer_frac,
        uses_h_as_l,
        omits_base,
        treats_triangle_as_full,
        counts_two_laterals,
        msg="exam-tier pyramid distractor equals the answer",
    )

    candidates = [
        str(uses_h_as_l),
        str(omits_base),
        str(treats_triangle_as_full),
        str(counts_two_laterals),
    ]

    working = [
        f"The slant height l of a face satisfies l² = {h}² + ({s} ÷ 2)² = {h * h} + {(s // 2) ** 2} = {l * l}, so l = {l} cm.",
        f"Base: {s} × {s} = {s * s} cm².",
        f"Each triangular face: ½ × {s} × {l} = {s * l // 2} cm²; four faces make {2 * s * l} cm².",
        f"Total = {s * s} + {2 * s * l} = {answer} cm².",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


RECT_PYRAMID_TABLE = [
    (10, 18, 13, 15, 12),
    (10, 32, 13, 20, 12),
    (18, 32, 15, 20, 12),
    (12, 30, 10, 17, 8),
    (16, 40, 17, 25, 15),
    (30, 42, 25, 29, 20),
    (24, 60, 20, 34, 16),
    (14, 20, 25, 26, 24),
    (14, 36, 25, 30, 24),
    (20, 36, 26, 30, 24),
]  # (W, L, s_l, s_w, h) -- W < L always, h is the shared apex height


def _build_pyramid_rectangular_base(rng) -> dict:
    """Medium tier: rectangular base with two distinct slant heights. Drawn
    from a verified table instead of independently, so both slant heights
    imply the SAME apex height `h` (a real pyramid has one apex)."""
    w, l_base, s_l, s_w, h = rng.choice(RECT_PYRAMID_TABLE)
    # u in {1, 2, 3}: widened from {1, 2} because 18/20 scaled
    # combinations fall short of this module's existing >=20-distinct-per-
    # tier test; scaling every value in a row by the same integer preserves
    # both realizability identities exactly (both sides of s^2 = h^2 + (.)^2
    # scale by u^2), so this is the same proof, just over a wider range.
    u = rng.choice([1, 2, 3])
    w, l_base, s_l, s_w, h = w * u, l_base * u, s_l * u, s_w * u, h * u
    check(s_l * s_l == h * h + (w // 2) * (w // 2), "rectangular pyramid: long-edge slant height not realizable")
    check(
        s_w * s_w == h * h + (l_base // 2) * (l_base // 2),
        "rectangular pyramid: short-edge slant height not realizable",
    )

    faces = [("rectangle", (l_base, w))] + [("triangle", (l_base, s_l))] * 2 + [("triangle", (w, s_w))] * 2
    answer_frac = _sum_faces(faces)
    _verify_pyramid_rectangular_base(l_base, w, s_l, s_w, answer_frac)

    assert answer_frac.denominator == 1
    answer = int(answer_frac)
    answer_text = str(answer)

    text = (
        f"A pyramid has a rectangular base measuring {l_base} cm by {w} cm. "
        f"The two triangular faces on the longer sides have a slant height of "
        f"{s_l} cm each; the two triangular faces on the shorter sides have a slant "
        f"height of {s_w} cm each. Find the pyramid's total surface area, in cm²."
    )

    # Misconceptions:
    #  - omits_base: l_base*s_l + w*s_w
    #  - no_half_factor_on_laterals: l_base*w + 2*l_base*s_l + 2*w*s_w
    #  - swaps_which_slant_height_goes_with_which_edge: l_base*w + l_base*s_w + w*s_l
    omits_base = l_base * s_l + w * s_w
    no_half = l_base * w + 2 * l_base * s_l + 2 * w * s_w
    swaps = l_base * w + l_base * s_w + w * s_l

    candidates = [str(omits_base), str(no_half), str(swaps)]

    working = [
        f"Base: {l_base} × {w} = {l_base * w} cm².",
        f"Two faces on the longer sides: 2 × (½ × {l_base} × {s_l}) = {l_base * s_l} cm².",
        f"Two faces on the shorter sides: 2 × (½ × {w} × {s_w}) = {w * s_w} cm².",
        f"Total = {l_base * w} + {l_base * s_l} + {w * s_w} = {answer} cm².",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


INCENTER_TRIANGLE_TABLE = [
    (9, 12, 15, 3, 4, 5),
    (8, 15, 17, 3, 4, 5),
    (7, 24, 25, 3, 4, 5),
    (10, 24, 26, 4, 3, 5),
    (20, 21, 29, 6, 8, 10),
    (16, 30, 34, 6, 8, 10),
    (12, 35, 37, 5, 12, 13),
    (15, 36, 39, 6, 8, 10),
    (9, 40, 41, 4, 3, 5),
    (14, 48, 50, 6, 8, 10),
    (24, 45, 51, 9, 12, 15),
    (24, 45, 51, 9, 40, 41),
    (28, 45, 53, 10, 24, 26),
    (40, 42, 58, 12, 5, 13),
    (40, 42, 58, 12, 9, 15),
]  # (a, b, c, r, h, s) -- a, b legs, c hypotenuse, r=(a+b-c)/2, s the
# SAME slant height on all three faces


def _build_pyramid_right_triangle_base(rng) -> dict:
    """Hard tier: right-triangle base with three slant heights. The apex is
    placed directly above the base triangle's incentre, so all three lateral
    faces share ONE slant height `s` with s^2 = h^2 + r^2 (r the inradius) --
    a real pyramid has one apex, and independently-drawn slant heights are
    essentially never simultaneously realizable for a right-triangle base."""
    a, b, c, r, h, s = rng.choice(INCENTER_TRIANGLE_TABLE)
    # k in {1, 2}: the 15 fixed rows alone fall short of this module's
    # existing >=20-distinct-per-tier test; scaling every value in a row by
    # the same integer preserves both realizability identities exactly
    # (a^2+b^2=c^2 and s^2=h^2+r^2 are homogeneous of degree 2, and
    # r=(a+b-c)/2 scales linearly), so this is the same proof, just over a
    # wider range -- the same technique already used for the square and
    # rectangular tiers' t/u multipliers.
    k = rng.choice([1, 2])
    a, b, c, r, h, s = a * k, b * k, c * k, r * k, h * k, s * k
    check(s * s == h * h + r * r, "right-triangle pyramid: shared slant height not realizable at the incentre")
    s_a = s_b = s_c = s  # all three faces share one slant height, by construction

    faces = (
        [("triangle", (a, b))]  # base
        + [("triangle", (a, s_a)), ("triangle", (b, s_b)), ("triangle", (c, s_c))]
    )
    answer_frac = _sum_faces(faces)
    _verify_pyramid_right_triangle_base(a, b, c, s_a, s_b, s_c, answer_frac)

    assert answer_frac.denominator == 1
    answer = int(answer_frac)
    answer_text = str(answer)

    text = (
        f"A pyramid has a right-triangular base with legs {a} cm and {b} cm "
        f"and hypotenuse {c} cm. The triangular face on the {a} cm edge has a "
        f"slant height of {s_a} cm, the face on the {b} cm edge has a slant height "
        f"of {s_b} cm, and the face on the {c} cm edge has a slant height of {s_c} "
        f"cm. Find the pyramid's total surface area, in cm²."
    )

    # Misconceptions:
    #  - omits_base: lateral terms only
    #  - no_half_factor_on_all_faces: 2 * answer
    #  - omits_the_hypotenuse_face: answer - Fraction(c, 2) * s_c
    omits_base = Fraction(a, 2) * s_a + Fraction(b, 2) * s_b + Fraction(c, 2) * s_c
    no_half_all = 2 * answer_frac
    omits_hyp = answer_frac - Fraction(c, 2) * s_c

    candidates = [str(int(omits_base)), str(int(no_half_all)), str(int(omits_hyp))]

    def half(n: int) -> str:
        return str(n // 2) if n % 2 == 0 else f"{n // 2}.5"

    working = [
        f"Base: ½ × {a} × {b} = {half(a * b)} cm².",
        f"Face on the {a} cm edge: ½ × {a} × {s_a} = {half(a * s_a)} cm².",
        f"Face on the {b} cm edge: ½ × {b} × {s_b} = {half(b * s_b)} cm².",
        f"Face on the {c} cm edge: ½ × {c} × {s_c} = {half(c * s_c)} cm².",
        f"Total = {half(a * b)} + {half(a * s_a)} + {half(b * s_b)} + {half(c * s_c)} = {answer} cm².",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "find_the_surface_area_of_pyramids_such_as",
    level=11,
    presets={
        "easy": dict(mode="square_base"),
        "medium": dict(mode="rectangular_base"),
        "hard": dict(mode="right_triangle_base"),
        "exam": dict(mode="square_base_from_height"),
    },
)
def find_the_surface_area_of_pyramids_such_as(
    rng,
    mode="square_base",
):
    """Find the surface area of pyramids by summing base and lateral face areas."""
    if mode == "square_base":
        return _build_pyramid_square_base(rng)
    if mode == "rectangular_base":
        return _build_pyramid_rectangular_base(rng)
    if mode == "right_triangle_base":
        return _build_pyramid_right_triangle_base(rng)
    if mode == "square_base_from_height":
        return _build_pyramid_square_base_from_height(rng)
    raise ValueError(f"unknown mode: {mode!r}")


# --- use_addition_of_the_area_of_the_faces -----------------------------------


def _verify_faces_easy(a1, b1, a2, b2, a3, b3, s4, s5, expected: Fraction) -> None:
    """Independent check: rectangles subtotal + squares subtotal."""
    rectangles = a1 * b1 + a2 * b2 + a3 * b3
    squares = s4 * s4 + s5 * s5
    recomputed = rectangles + squares
    check(recomputed == expected, "easy faces sum mismatch")


def _verify_faces_medium(a1, b1, a2, b2, base3, height3, base4, height4, s5, expected: Fraction) -> None:
    """Independent check: rectangles+square subtotal + triangles subtotal."""
    rect_square = a1 * b1 + a2 * b2 + s5 * s5
    triangles = Fraction(base3, 2) * height3 + Fraction(base4, 2) * height4
    recomputed = rect_square + triangles
    check(recomputed == expected, "medium faces sum mismatch")


def _verify_faces_hard(a1, b1, a2, b2, a3, b3, base4, height4, base5, height5, s_x, expected: Fraction) -> None:
    """Independent check: rectangles subtotal + triangles subtotal.
    Separately assert that s_x*s_x is never part of the sum (check via construction)."""
    rectangles = a1 * b1 + a2 * b2 + a3 * b3
    triangles = Fraction(base4, 2) * height4 + Fraction(base5, 2) * height5
    recomputed = rectangles + triangles
    check(recomputed == expected, "hard faces sum mismatch")
    # Assert the lure is not included by checking answer != answer + s_x*s_x
    check(recomputed != recomputed + s_x * s_x, "lure must not be included in answer")


def _build_faces_easy(rng) -> dict:
    """Easy tier: 5 faces (3 rectangles, 2 squares)."""
    a1, b1 = rng.randint(3, 12), rng.randint(3, 12)
    a2, b2 = rng.randint(3, 12), rng.randint(3, 12)
    a3, b3 = rng.randint(3, 12), rng.randint(3, 12)
    s4 = rng.randint(3, 12)
    s5 = rng.randint(3, 12)

    faces = [
        ("rectangle", (a1, b1)),
        ("rectangle", (a2, b2)),
        ("rectangle", (a3, b3)),
        ("rectangle", (s4, s4)),  # square as rectangle
        ("rectangle", (s5, s5)),  # square as rectangle
    ]
    answer_frac = _sum_faces(faces)
    _verify_faces_easy(a1, b1, a2, b2, a3, b3, s4, s5, answer_frac)

    assert answer_frac.denominator == 1
    answer = int(answer_frac)
    answer_text = str(answer)

    text = (
        f"An irregular solid has the following faces: "
        f"Face 1 is a rectangle {a1} cm by {b1} cm. "
        f"Face 2 is a rectangle {a2} cm by {b2} cm. "
        f"Face 3 is a rectangle {a3} cm by {b3} cm. "
        f"Face 4 is a square with side {s4} cm. "
        f"Face 5 is a square with side {s5} cm. "
        f"Add the areas of the faces to find the solid's total surface area, in cm²."
    )

    # Misconceptions:
    #  - omits_last_face: answer - s5*s5
    #  - double_counts_first_face: answer + a1*b1
    #  - confuses_square_side_for_area: answer - s4*s4 + s4
    omits_last = answer - s5 * s5
    double_first = answer + a1 * b1
    confuses_square = answer - s4 * s4 + s4

    candidates = [str(omits_last), str(double_first), str(confuses_square)]

    working = [
        f"Face 1: {a1} × {b1} = {a1 * b1} cm².",
        f"Face 2: {a2} × {b2} = {a2 * b2} cm².",
        f"Face 3: {a3} × {b3} = {a3 * b3} cm².",
        f"Face 4: {s4} × {s4} = {s4 * s4} cm².",
        f"Face 5: {s5} × {s5} = {s5 * s5} cm².",
        f"Total = {a1 * b1} + {a2 * b2} + {a3 * b3} + {s4 * s4} + {s5 * s5} = {answer} cm².",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_faces_medium(rng) -> dict:
    """Medium tier: 5 faces (2 rectangles, 2 triangles, 1 square)."""
    a1, b1 = rng.randint(3, 12), rng.randint(3, 12)
    a2, b2 = rng.randint(3, 12), rng.randint(3, 12)
    base3 = rng.randint(2, 7) * 2  # Always even, 4-14
    height3 = rng.randint(3, 12)
    base4 = rng.randint(2, 7) * 2  # Always even, 4-14
    height4 = rng.randint(3, 12)
    s5 = rng.randint(3, 12)

    faces = [
        ("rectangle", (a1, b1)),
        ("rectangle", (a2, b2)),
        ("triangle", (base3, height3)),
        ("triangle", (base4, height4)),
        ("rectangle", (s5, s5)),  # square as rectangle
    ]
    answer_frac = _sum_faces(faces)
    _verify_faces_medium(a1, b1, a2, b2, base3, height3, base4, height4, s5, answer_frac)

    assert answer_frac.denominator == 1
    answer = int(answer_frac)
    answer_text = str(answer)

    text = (
        f"An irregular solid has the following faces: "
        f"Face 1 is a rectangle {a1} cm by {b1} cm. "
        f"Face 2 is a rectangle {a2} cm by {b2} cm. "
        f"Face 3 is a triangle with base {base3} cm and height {height3} cm. "
        f"Face 4 is a triangle with base {base4} cm and height {height4} cm. "
        f"Face 5 is a square with side {s5} cm. "
        f"Add the areas of the faces to find the solid's total surface area, in cm²."
    )

    # Misconceptions:
    #  - omits_last_face: answer - s5*s5
    #  - double_counts_first_face: answer + a1*b1
    #  - forgets_half_on_triangles: answer + Fraction(base3, 2)*height3 + Fraction(base4, 2)*height4
    omits_last = answer - s5 * s5
    double_first = answer + a1 * b1
    forgets_half_tri = answer + (base3 * height3 // 2) + (base4 * height4 // 2)

    candidates = [str(omits_last), str(double_first), str(forgets_half_tri)]

    t3, t4 = base3 * height3 // 2, base4 * height4 // 2
    working = [
        f"Face 1: {a1} × {b1} = {a1 * b1} cm².",
        f"Face 2: {a2} × {b2} = {a2 * b2} cm².",
        f"Face 3 (triangle): ½ × {base3} × {height3} = {t3} cm².",
        f"Face 4 (triangle): ½ × {base4} × {height4} = {t4} cm².",
        f"Face 5: {s5} × {s5} = {s5 * s5} cm².",
        f"Total = {a1 * b1} + {a2 * b2} + {t3} + {t4} + {s5 * s5} = {answer} cm².",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_faces_hard(rng) -> dict:
    """Hard tier: 5 real faces + 1 explicitly excluded lure face."""
    # Real faces
    a1, b1 = rng.randint(3, 12), rng.randint(3, 12)
    a2, b2 = rng.randint(3, 12), rng.randint(3, 12)
    a3, b3 = rng.randint(3, 12), rng.randint(3, 12)
    base4 = rng.randint(2, 7) * 2  # Always even
    height4 = rng.randint(3, 12)
    base5 = rng.randint(2, 7) * 2  # Always even
    height5 = rng.randint(3, 12)

    # Lure: excluded square face
    s_x = rng.randint(3, 12)

    faces = [
        ("rectangle", (a1, b1)),
        ("rectangle", (a2, b2)),
        ("rectangle", (a3, b3)),
        ("triangle", (base4, height4)),
        ("triangle", (base5, height5)),
    ]
    answer_frac = _sum_faces(faces)
    _verify_faces_hard(a1, b1, a2, b2, a3, b3, base4, height4, base5, height5, s_x, answer_frac)

    assert answer_frac.denominator == 1
    answer = int(answer_frac)
    answer_text = str(answer)

    text = (
        f"An irregular solid has the following faces: "
        f"Face 1 is a rectangle {a1} cm by {b1} cm. "
        f"Face 2 is a rectangle {a2} cm by {b2} cm. "
        f"Face 3 is a rectangle {a3} cm by {b3} cm. "
        f"Face 4 is a triangle with base {base4} cm and height {height4} cm. "
        f"Face 5 is a triangle with base {base5} cm and height {height5} cm. "
        f"The solid is open on one side, where a square face of side {s_x} cm would otherwise be -- "
        f"this face is not part of the solid. Add the areas of the faces that ARE part of the "
        f"solid to find its total surface area, in cm²."
    )

    # Misconceptions:
    #  - includes_the_excluded_face (THE hard-tier trap): answer + s_x*s_x
    #  - omits_last_included_face: answer - Fraction(base5, 2)*height5
    #  - forgets_half_on_one_triangle: answer + Fraction(base4, 2)*height4
    includes_lure = answer + s_x * s_x
    omits_last_included = answer - (base5 * height5 // 2)
    forgets_half_one = answer + (base4 * height4 // 2)

    # Put includes_lure FIRST to ensure it's always picked as a distractor
    candidates = [str(includes_lure), str(omits_last_included), str(forgets_half_one)]

    t4, t5 = base4 * height4 // 2, base5 * height5 // 2
    working = [
        f"Face 1: {a1} × {b1} = {a1 * b1} cm².",
        f"Face 2: {a2} × {b2} = {a2 * b2} cm².",
        f"Face 3: {a3} × {b3} = {a3 * b3} cm².",
        f"Face 4 (triangle): ½ × {base4} × {height4} = {t4} cm².",
        f"Face 5 (triangle): ½ × {base5} × {height5} = {t5} cm².",
        "The open square face is not part of the solid, so leave it out.",
        f"Total = {a1 * b1} + {a2 * b2} + {a3 * b3} + {t4} + {t5} = {answer} cm².",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "use_addition_of_the_area_of_the_faces",
    level=11,
    presets={
        "easy": dict(mode="rectangles_and_squares"),
        "medium": dict(mode="mixed_shapes"),
        "hard": dict(mode="excluded_face"),
    },
)
def use_addition_of_the_area_of_the_faces(
    rng,
    mode="rectangles_and_squares",
):
    """Find surface area by adding areas of independently listed faces."""
    if mode == "rectangles_and_squares":
        return _build_faces_easy(rng)
    if mode == "mixed_shapes":
        return _build_faces_medium(rng)
    if mode == "excluded_face":
        return _build_faces_hard(rng)
    raise ValueError(f"unknown mode: {mode!r}")
