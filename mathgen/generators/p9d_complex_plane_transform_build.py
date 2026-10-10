"""Generates `complex_mult_linear_transform` and `complex_addition_vector` questions: images of points under
complex multiplication and addition.

Complex-plane transformation skills.

Skill A: senior-examine-and-use-multiplication-as-a-linear-transformation (ACMSM085)
- topic ``complex_mult_linear_transform``, three tiers by multiplier family:
  real -> purely imaginary -> general complex. Every question fixes a
  multiplier ``w`` and asks for the image of a supplied point ``z`` under
  multiplication by ``w`` -- the geometric (linear-transformation) reading of
  multiplication, never the abstract "compute this product" framing that
  ``multiply-complex-numbers`` already assesses.
- Answer is always the image coordinate pair (x, y), built via
  ``mathgen.core.xy_solution``.
- Independent check via Python's native ``complex`` type, mirroring
  ``p5b_complex_plane_build.py``'s own established construction/check split.

Skill B: senior-examine-addition-of-complex-numbers-as-vector-addition (ACMSM072, ACMSM084)
- topic ``complex_addition_vector``, three tiers: two-vector sum ->
  three-vector chained sum -> inverse (solve for the missing vector given the
  resultant and one addend). Every question is framed in vector language
  ("vector", "resultant") throughout, which
  ``senior-consider-complex-numbers-as-points-in-a-plane``'s own ``sum``
  variant never uses, and reaches two difficulty shapes (chained addition,
  inverse solve) that variant never reaches either.
- Answer is always a coordinate pair (x, y), built via
  ``mathgen.core.xy_solution``.
- Independent check via Python's native ``complex`` type; the inverse tier's
  check forward-adds the computed answer back to the given vector and
  confirms it reproduces the given resultant -- a genuinely different route
  from the subtraction used to construct it.
"""

from __future__ import annotations

from ..core import check, generator, join_terms, pick_distractors, xy_solution


def _fmt_complex(a: int, b: int) -> str:
    """Format ``a + bi`` the way a student would write it: "3 - 2i", "i", "-6i", "-7". `mathgen.core.join_terms`
    already handles the zero coefficient (dropping it) and the unit coefficient (``fmt_signed`` renders ``1``
    as nothing, so ``b = 1`` gives "i", not the "1i" that a hand-rolled formatter tends to produce) -- reused
    rather than hand-rolled.py's helpers, never hand-roll." This is also the fix for
    `p5b_complex_plane_build.py`'s own `format_complex` (read-only, not edited here), which drops the sign on
    a purely-imaginary negative value (``a=0, b=-6`` renders "6i", losing the minus): `join_terms` never has
    that bug because it formats each term's sign independently.
    """
    return join_terms([(a, ""), (b, "i")])


def _sample_nonzero(rng, lo: int = -9, hi: int = 9) -> tuple[int, int]:
    """(x, y) each independently drawn from ``lo..hi``, resampled until not
    the origin -- a nonzero point/vector."""
    x, y = rng.randint(lo, hi), rng.randint(lo, hi)
    while (x, y) == (0, 0):
        x, y = rng.randint(lo, hi), rng.randint(lo, hi)
    return x, y


def _filter_candidates(answer_xy: tuple[int, int], candidates: list[tuple[int, int]]) -> list[str]:
    """Drop any candidate tuple equal to the answer, then format the rest as
    ``xy_solution``-shaped text ("x = 3, y = -5") for ``pick_distractors``."""
    return [f"x = {cx}, y = {cy}" for cx, cy in candidates if (cx, cy) != answer_xy]


def _has_enough_distractors(filtered: list[str], minimum: int = 2) -> bool:
    """Whether at least ``minimum`` of the filtered candidates are distinct
    from EACH OTHER, not just from the answer. Two misconceptions can
    coincide with each other on a symmetric draw without either equalling
    the answer -- e.g. the "sign-drop" and "swapped" candidates for a
    purely-imaginary multiplier collapse to the same tuple whenever the
    sampled point has y = 0, and the "swapped-formula" candidate for a
    general multiplier collapses onto "real-part-only" whenever a == b and
    y = 0. `pick_distractors` already dedupes by exact string, so this is
    the same criterion applied *before* offering the draw, so every tier's
    generator can resample instead of silently shipping a single-distractor
    question."""
    return len(set(filtered)) >= minimum


def _check_complex_product(w: complex, z: complex, claimed_x: int, claimed_y: int) -> None:
    """Independent check for Skill A: multiply ``w`` by ``z`` via Python's
    native ``complex`` type and confirm it matches the claimed image point.
    Factored out (rather than inlined per tier) so a test can call it
    directly with a deliberately wrong ``claimed_x``/``claimed_y`` and prove
    it raises."""
    result = w * z
    check(
        int(result.real) == claimed_x and int(result.imag) == claimed_y,
        "native complex multiplication disagrees with the direct formula",
    )


def _check_complex_sum(parts: list[complex], claimed_x: int, claimed_y: int) -> None:
    """Independent check for Skill B's two_term/chained tiers: sum the given
    complex parts via Python's native ``complex`` type and confirm it
    matches the claimed resultant. Factored out so a test can call it
    directly with a deliberately wrong claimed resultant."""
    total = sum(parts, complex(0, 0))
    check(
        int(total.real) == claimed_x and int(total.imag) == claimed_y,
        "native complex addition disagrees with the direct formula",
    )


def _check_forward_sum(z1: complex, z2: complex, claimed_r_x: int, claimed_r_y: int) -> None:
    """Independent check for the inverse tier: forward-add ``z1`` and the
    computed ``z2`` and confirm it reproduces the given resultant R -- a
    genuinely different route from the subtraction used to construct z2.
    Factored out so a test can call it directly with a deliberately wrong
    claimed resultant."""
    total = z1 + z2
    check(
        int(total.real) == claimed_r_x and int(total.imag) == claimed_r_y,
        "forward addition of the computed vector does not reproduce the given resultant",
    )


# --------------------------------------------------------------------------
# Skill A: multiplication as a linear transformation
# --------------------------------------------------------------------------

_K_VALUES = (-6, -5, -4, -3, -2, 2, 3, 4, 5, 6)
_AB_VALUES = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)


@generator(
    "complex_mult_linear_transform",
    level=12,
    presets={
        "easy": dict(variant="real"),
        "medium": dict(variant="imaginary"),
        "hard": dict(variant="general"),
    },
)
def complex_mult_linear_transform(rng, variant="general"):
    """Multiplication by a fixed multiplier w as a linear transformation of
    the complex plane; find the image of a supplied point z.

    Boundary against multiply-complex-numbers: that skill grades the raw
    Cartesian product of two SUPPLIED complex numbers via MCQ. This skill's
    kernel is the GEOMETRIC reading of multiplication -- w is always a fixed
    "multiplier" drawn from an increasingly general family (real -> purely
    imaginary -> general complex), and every question is framed as finding
    the image point of z under multiplication by w, never as "compute the
    product" in the abstract.

    Variants:
    - real: w = k, k real, k in {-6..-2, 2..6}
    - imaginary: w = mi, m in {-6..-2, 2..6}
    - general: w = a + bi, a, b each in {-5..-1, 1..5}
    """
    if variant == "real":
        return _mult_real(rng)
    elif variant == "imaginary":
        return _mult_imaginary(rng)
    elif variant == "general":
        return _mult_general(rng)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def _mult_real(rng):
    while True:
        x, y = _sample_nonzero(rng)
        k = rng.choice(_K_VALUES)
        candidates = [
            (x + k, y + k),  # added-not-multiplied
            (k * x, y),  # scaled-x-only
            (k * y, k * x),  # swapped
        ]
        filtered = _filter_candidates((k * x, k * y), candidates)
        if _has_enough_distractors(filtered):
            break

    image_x, image_y = k * x, k * y

    # Independent check via Python's native complex type.
    _check_complex_product(complex(k, 0), complex(x, y), image_x, image_y)

    z_text = _fmt_complex(x, y)

    text = (
        f"Multiplication by the real number w = {k} is a linear transformation "
        f"of the complex plane. Find the image of z = {z_text} under this "
        f"transformation, as an ordered pair (x, y)."
    )
    latex = (
        rf"\text{{Multiplication by }} w = {k} \text{{ (real). Find the image of }} "
        rf"z = {z_text} \text{{ under this transformation, as an ordered pair.}}"
    )
    working = [
        "w z = k(x + yi) = (kx) + (ky)i",
        f"w z = {k}({join_terms([(x, ''), (y, 'i')])}) = {join_terms([(k * x, ''), (k * y, 'i')])}",
        f"Image point: ({image_x}, {image_y}).",
    ]
    result_dict = xy_solution(image_x, image_y)
    result_dict["distractors"] = pick_distractors(result_dict["answer_text"], filtered)
    result_dict.update(text=text, latex=latex, working=working)
    return result_dict


def _mult_imaginary(rng):
    while True:
        x, y = _sample_nonzero(rng)
        m = rng.choice(_K_VALUES)
        candidates = [
            (m * y, m * x),  # sign-drop (forgot i^2 = -1)
            (m * x, m * y),  # treated-as-real
            (m * x, -m * y),  # swapped
        ]
        filtered = _filter_candidates((-m * y, m * x), candidates)
        if _has_enough_distractors(filtered):
            break

    image_x, image_y = -m * y, m * x

    # Independent check via Python's native complex type.
    _check_complex_product(complex(0, m), complex(x, y), image_x, image_y)

    z_text = _fmt_complex(x, y)

    text = (
        f"Multiplication by the purely imaginary number w = {m}i is a linear "
        f"transformation of the complex plane. Find the image of z = {z_text} "
        f"under this transformation, as an ordered pair (x, y)."
    )
    latex = (
        rf"\text{{Multiplication by }} w = {m}i \text{{ (purely imaginary). Find the "
        rf"image of }} z = {z_text} \text{{ under this transformation, as an ordered pair.}}"
    )
    working = [
        "w z = mi(x + yi) = -my + (mx)i, since i^2 = -1",
        f"w z = {m}i({join_terms([(x, ''), (y, 'i')])}) = {join_terms([(-m * y, ''), (m * x, 'i')])}",
        f"Image point: ({image_x}, {image_y}).",
    ]
    result_dict = xy_solution(image_x, image_y)
    result_dict["distractors"] = pick_distractors(result_dict["answer_text"], filtered)
    result_dict.update(text=text, latex=latex, working=working)
    return result_dict


def _mult_general(rng):
    while True:
        x, y = _sample_nonzero(rng)
        a = rng.choice(_AB_VALUES)
        b = rng.choice(_AB_VALUES)
        image_x, image_y = a * x - b * y, a * y + b * x
        candidates = [
            (a * x + b * y, a * y + b * x),  # sign-error-cross-term
            (a * x, a * y),  # real-part-only
            (a * y + b * x, a * x - b * y),  # swapped-formula
        ]
        filtered = _filter_candidates((image_x, image_y), candidates)
        if _has_enough_distractors(filtered):
            break

    # Independent check via Python's native complex type.
    _check_complex_product(complex(a, b), complex(x, y), image_x, image_y)

    z_text = _fmt_complex(x, y)
    w_text = _fmt_complex(a, b)

    text = (
        f"Multiplication by the complex number w = {w_text} is a linear "
        f"transformation of the complex plane. Find the image of z = {z_text} "
        f"under this transformation, as an ordered pair (x, y)."
    )
    latex = (
        rf"\text{{Multiplication by }} w = {w_text} \text{{. Find the image of }} "
        rf"z = {z_text} \text{{ under this transformation, as an ordered pair.}}"
    )
    # Each raw factor is parenthesised on its own, so a negative value never
    # sits directly after a literal "+"/"-" operator (the pattern
    # `mathgen.text_quality`'s `signed_after_operator` check forbids).
    working = [
        "w z = (a + bi)(x + yi) = (ax - by) + (ay + bx)i, since i^2 = -1",
        (f"w z = ({a})({x}) - ({b})({y}) + [({a})({y}) + ({b})({x})]i"),
        f"w z = {join_terms([(a * x - b * y, ''), (a * y + b * x, 'i')])}",
        f"Image point: ({image_x}, {image_y}).",
    ]
    result_dict = xy_solution(image_x, image_y)
    result_dict["distractors"] = pick_distractors(result_dict["answer_text"], filtered)
    result_dict.update(text=text, latex=latex, working=working)
    return result_dict


# --------------------------------------------------------------------------
# Skill B: addition of complex numbers as vector addition
# --------------------------------------------------------------------------


@generator(
    "complex_addition_vector",
    level=12,
    presets={
        "easy": dict(variant="two_term"),
        "medium": dict(variant="chained"),
        "hard": dict(variant="inverse"),
    },
)
def complex_addition_vector(rng, variant="two_term"):
    """Addition of complex numbers, framed and computed as vector addition
    in the complex plane.

    Boundary against senior-consider-complex-numbers-as-points-in-a-plane:
    that skill's own ``sum`` variant is a bounded 1/3 slice of its own three
    tiers (two-term add/subtract only, no chaining, no inverse-solve) and is
    never framed in vector language -- no "vector", "resultant" or
    "displacement" wording anywhere in its question text. This skill's kept
    kernel is the explicit VECTOR-ADDITION framing (every question says
    "vector"/"resultant") plus a genuinely different difficulty shape:
    three-term chained addition at medium, and inverse/solve-for-the-other-
    vector reasoning at hard, neither of which the existing skill reaches at
    any tier.

    Variants:
    - two_term: z1 + z2, resultant vector as a coordinate pair
    - chained: z1 + z2 + z3, resultant vector as a coordinate pair
    - inverse: given z1 and the resultant R = z1 + z2, solve for z2
    """
    if variant == "two_term":
        return _addition_two_term(rng)
    elif variant == "chained":
        return _addition_chained(rng)
    elif variant == "inverse":
        return _addition_inverse(rng)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def _addition_two_term(rng):
    while True:
        x1, y1 = rng.randint(-9, 9), rng.randint(-9, 9)
        x2, y2 = _sample_nonzero(rng)
        answer_x, answer_y = x1 + x2, y1 + y2
        candidates = [
            (x1 - x2, y1 - y2),  # subtracted-not-added
            (y1 + y2, x1 + x2),  # swapped-result
            (x1, y1),  # only-first-vector (forgot to add z2)
        ]
        filtered = _filter_candidates((answer_x, answer_y), candidates)
        if _has_enough_distractors(filtered):
            break

    # Independent check via Python's native complex type.
    _check_complex_sum([complex(x1, y1), complex(x2, y2)], answer_x, answer_y)

    z1_text, z2_text = _fmt_complex(x1, y1), _fmt_complex(x2, y2)

    text = (
        f"The complex number z1 = {z1_text} is represented by the vector from the "
        f"origin to ({x1}, {y1}), and z2 = {z2_text} by the vector to ({x2}, {y2}). "
        f"Find the resultant vector z1 + z2, as a coordinate pair."
    )
    latex = (
        rf"\text{{Vector to }} z_1 = {z1_text} \text{{ and vector to }} z_2 = {z2_text}"
        rf"\text{{. Find the resultant vector }} z_1 + z_2 \text{{, as a coordinate pair.}}"
    )
    working = [
        f"z1 + z2 = ({join_terms([(x1, ''), (x2, '')])}) + ({join_terms([(y1, ''), (y2, '')])})i",
        f"z1 + z2 = {join_terms([(answer_x, ''), (answer_y, 'i')])}",
        f"Resultant vector: ({answer_x}, {answer_y}).",
    ]
    result_dict = xy_solution(answer_x, answer_y)
    result_dict["distractors"] = pick_distractors(result_dict["answer_text"], filtered)
    result_dict.update(text=text, latex=latex, working=working)
    return result_dict


def _addition_chained(rng):
    while True:
        x1, y1 = _sample_nonzero(rng)
        x2, y2 = _sample_nonzero(rng)
        x3, y3 = _sample_nonzero(rng)
        answer_x, answer_y = x1 + x2 + x3, y1 + y2 + y3
        candidates = [
            (x1 + x2, y1 + y2),  # omitted-third
            (x2 + x3, y2 + y3),  # omitted-first
            (x1 + x2 - x3, y1 + y2 - y3),  # sign-error-third
        ]
        filtered = _filter_candidates((answer_x, answer_y), candidates)
        if _has_enough_distractors(filtered):
            break

    # Independent check via Python's native complex type.
    _check_complex_sum([complex(x1, y1), complex(x2, y2), complex(x3, y3)], answer_x, answer_y)

    z1_text, z2_text, z3_text = _fmt_complex(x1, y1), _fmt_complex(x2, y2), _fmt_complex(x3, y3)

    text = (
        f"Three vectors represent the complex numbers z1 = {z1_text}, z2 = {z2_text} "
        f"and z3 = {z3_text}, each drawn from the origin to its own point. Find the "
        f"resultant vector z1 + z2 + z3, as a coordinate pair."
    )
    latex = (
        rf"\text{{Vectors for }} z_1 = {z1_text}, z_2 = {z2_text}, z_3 = {z3_text}"
        rf"\text{{. Find the resultant vector }} z_1 + z_2 + z_3 \text{{, as a coordinate pair.}}"
    )
    working = [
        f"z1 + z2 + z3 = ({join_terms([(x1, ''), (x2, ''), (x3, '')])}) + "
        f"({join_terms([(y1, ''), (y2, ''), (y3, '')])})i",
        f"z1 + z2 + z3 = {join_terms([(answer_x, ''), (answer_y, 'i')])}",
        f"Resultant vector: ({answer_x}, {answer_y}).",
    ]
    result_dict = xy_solution(answer_x, answer_y)
    result_dict["distractors"] = pick_distractors(result_dict["answer_text"], filtered)
    result_dict.update(text=text, latex=latex, working=working)
    return result_dict


def _addition_inverse(rng):
    while True:
        x1, y1 = _sample_nonzero(rng)
        rx, ry = _sample_nonzero(rng)
        while (rx, ry) == (x1, y1):
            rx, ry = _sample_nonzero(rng)
        answer_x, answer_y = rx - x1, ry - y1
        candidates = [
            (rx + x1, ry + y1),  # added-not-subtracted
            (answer_y, answer_x),  # swapped-result
            (x1, y1),  # reports-z1
        ]
        filtered = _filter_candidates((answer_x, answer_y), candidates)
        if _has_enough_distractors(filtered):
            break

    # Independent check: forward-add the computed z2 back to z1 and confirm
    # it reproduces the given resultant R -- a genuinely different route
    # from the subtraction used to construct z2.
    _check_forward_sum(complex(x1, y1), complex(answer_x, answer_y), rx, ry)

    z1_text, r_text = _fmt_complex(x1, y1), _fmt_complex(rx, ry)

    text = (
        f"The vector from the origin to ({x1}, {y1}) represents z1 = {z1_text}. "
        f"The resultant of z1 and an unknown second vector z2 is the vector to "
        f"({rx}, {ry}), representing R = {r_text}, where z1 + z2 = R. Find z2, "
        f"as a coordinate pair."
    )
    latex = (
        rf"\text{{Vector to }} z_1 = {z1_text}. \text{{ Resultant }} R = {r_text} = z_1 + z_2."
        rf"\text{{ Find }} z_2 \text{{, as a coordinate pair.}}"
    )
    working = [
        "z1 + z2 = R, so z2 = R - z1",
        f"z2 = ({join_terms([(rx, ''), (-x1, '')])}) + ({join_terms([(ry, ''), (-y1, '')])})i",
        f"z2 = {join_terms([(answer_x, ''), (answer_y, 'i')])}, as a coordinate pair: ({answer_x}, {answer_y}).",
    ]
    result_dict = xy_solution(answer_x, answer_y)
    result_dict["distractors"] = pick_distractors(result_dict["answer_text"], filtered)
    result_dict.update(text=text, latex=latex, working=working)
    return result_dict
