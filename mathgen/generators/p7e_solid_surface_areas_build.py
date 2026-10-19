"""Senior surface area, curved-vs-flat split.

Two generators:

- ``calculate_surface_areas_of_standard_solids``
  (``senior-calculate-the-surface-areas-of-standard-three-dimensional``,
  ACMGM020) -- CURVED-SURFACE solids only: spheres and cones (plus a
  hemisphere-on-cone composite for the hard tier). Answer is an exact
  integer multiple of pi (``answer: expression``). Deliberately excludes
  rectangular prisms, cylinders and pyramids: cylinders would collide in
  spirit with ``f10-volume-surface-area-prisms-cylinders``, and rectangular
  prisms/pyramids belong to Skill B / the existing
  ``senior-find-the-surface-area-of-pyramids-such-as`` skill.

- ``find_surface_area_of_familiar_solids``
  (``senior-find-the-surface-area-of-familiar-solids-including``, ACMEM095)
  -- FLAT-FACED polyhedra only: cubes, rectangular prisms, triangular
  prisms. Plain integer answer (``answer: number``). Deliberately excludes
  spheres and cylinders: both are assigned to Skill A / already covered by
  the existing F-10 cylinder skill.

Both use exact-integer-by-construction arithmetic. Skill A's pi coefficient
is always an exact integer by construction (every radius/slant height is a
plain integer and the formulas involve no division), so no rounding
question ever arises; Skill B needs no pi at all.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from ..core import check, check_distinct, generator, pick_distractors


# --- Shared: exact multiple-of-pi formatting (Skill A only) -----------------


def _format_pi(k: int) -> str:
    """Format an exact integer multiple of pi with no bare '*':
    ``_format_pi(1) -> "pi"``, ``_format_pi(36) -> "36pi"``. Mirrors
    ``p5f_tan_shift_build.py``'s own ``_format_pi_multiple``, simplified to
    the integer-coefficient case this module always produces."""
    return "pi" if k == 1 else f"{k}pi"


def _grader_form(text: str) -> str:
    """The internal ``answer`` (sympifiable) form of a ``_format_pi`` string:
    insert the explicit ``*`` that ``answer_text``/distractors never carry --
    exactly the ``p5f_tan_shift_build.py`` idiom
    (``re.sub(r'(\\d)pi', r'\\1*pi', ...)``). ``_parse_expression`` in
    ``mathgen/grading/typed.py`` accepts a learner's bare "36pi" via implicit
    multiplication, but ``sympy.sympify`` (used on the internal ``answer``
    field) does not, hence this second form."""
    return re.sub(r"(\d)pi", r"\1*pi", text)


# ==============================================================================
# Skill A -- calculate_surface_areas_of_standard_solids (curved solids)
# ==============================================================================


def _verify_sphere(r: int, expected_k: int) -> None:
    """Independent check: sum four separate r*r contributions via an
    explicit loop, a different grouping from the direct 4*r*r construction."""
    recomputed = sum(r * r for _ in range(4))
    check(recomputed == expected_k, "sphere surface area coefficient mismatch")


def _verify_cone(r: int, l: int, expected_k: int) -> None:
    """Independent check: factored form r*(r+l), a different grouping from
    the construction's distributed r*r + r*l."""
    recomputed = r * (r + l)
    check(recomputed == expected_k, "cone surface area coefficient mismatch")


def _verify_composite(r: int, l: int, expected_k: int) -> None:
    """Independent check: the hemisphere's curved surface derived as HALF of
    a full sphere's 4*r*r (a different route from the construction's direct
    2*r*r), plus the cone's lateral term."""
    check(4 * r * r % 2 == 0, "hemisphere curved-surface halving not exact")
    hemisphere_curved = (4 * r * r) // 2
    recomputed = hemisphere_curved + r * l
    check(recomputed == expected_k, "composite surface area coefficient mismatch")


def _build_sphere(rng) -> dict:
    """Easy tier: sphere, SA = 4*pi*r^2. Single formula, single given
    quantity."""
    r = rng.randint(2, 30)
    k = 4 * r * r
    _verify_sphere(r, k)

    answer_text = _format_pi(k)
    answer = _grader_form(answer_text)

    text = (
        f"A sphere has a radius of {r} cm. Find its total surface area, in "
        f"cm². Give your answer as an exact value in terms of pi."
    )

    # Misconceptions:
    #  - radius_treated_as_diameter: halves r before squaring -> pi*r^2
    #  - uses_hemisphere_formula: computes only half the sphere -> 2*pi*r^2
    #  - omits_squaring: uses 4*pi*r (linear) instead of 4*pi*r^2
    radius_as_diameter = r * r
    hemisphere_only = 2 * r * r
    omits_squaring = 4 * r

    candidates = [
        _format_pi(radius_as_diameter),
        _format_pi(hemisphere_only),
        _format_pi(omits_squaring),
    ]
    check_distinct(
        k * sympy.pi,
        *(sympy.sympify(_grader_form(c)) for c in candidates),
        msg="sphere distractor collides with the answer",
    )

    working = [
        f"Surface area of a sphere = 4πr² = 4π × {r}² = 4π × {r * r}.",
        f"= {_format_pi(k).replace('pi', 'π')} cm².",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_cone(rng) -> dict:
    """Medium tier: cone, SA = pi*r^2 + pi*r*l (base + lateral), radius and
    slant height both given directly -- the escalation over easy is
    correctly combining two area terms, not deriving the slant height."""
    r = rng.randint(2, 15)
    l = rng.randint(r + 2, r + 25)  # l > r always: geometrically realizable
    k = r * r + r * l
    _verify_cone(r, l, k)

    answer_text = _format_pi(k)
    answer = _grader_form(answer_text)

    text = (
        f"A cone has base radius {r} cm and slant height {l} cm. Find its "
        f"total surface area (base circle plus curved lateral surface), in "
        f"cm². Give your answer as an exact value in terms of pi."
    )

    # Misconceptions:
    #  - omits_base_circle: lateral surface only
    #  - omits_lateral_surface: base circle only
    #  - swaps_radius_and_slant_height: uses l as the radius and r as the
    #    slant height
    omits_base = r * l
    omits_lateral = r * r
    swaps = l * l + l * r

    candidates = [_format_pi(omits_base), _format_pi(omits_lateral), _format_pi(swaps)]
    check_distinct(
        k * sympy.pi,
        *(sympy.sympify(_grader_form(c)) for c in candidates),
        msg="cone distractor collides with the answer",
    )

    working = [
        f"Base circle: πr² = π × {r}² = {r * r}π cm².",
        f"Curved surface: πrl = π × {r} × {l} = {r * l}π cm².",
        f"Total = {r * r}π + {r * l}π = {_format_pi(k).replace('pi', 'π')} cm².",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_composite(rng) -> dict:
    """Hard tier: a hemisphere capping a cone, sharing one circular face. The
    escalation is recognising that the shared interface circle is internal
    and must be excluded once: total = cone's lateral surface + hemisphere's
    curved surface (the interface circle never enters the sum either way)."""
    r = rng.randint(2, 12)
    l = rng.randint(r + 2, r + 25)  # cone slant height > r, realizable
    k = r * l + 2 * r * r
    _verify_composite(r, l, k)

    answer_text = _format_pi(k)
    answer = _grader_form(answer_text)

    text = (
        f"A solid is formed by joining a hemisphere to the flat circular "
        f"base of a cone, so their two circular cross-sections coincide "
        f"exactly and that shared circle sits inside the solid. The cone "
        f"has base radius {r} cm and slant height {l} cm. Find the total "
        f"EXTERNAL surface area of the solid, in cm². Give your answer as "
        f"an exact value in terms of pi."
    )

    # Misconceptions:
    #  - includes_the_shared_interface_circle (THE hard-tier trap): adds the
    #    internal circle as if it were still an external face
    #  - uses_full_sphere_instead_of_hemisphere: doubles the curved-cap term
    #  - confuses_hemisphere_curved_with_its_flat_circle: uses pi*r^2 instead
    #    of 2*pi*r^2 for the curved cap
    includes_interface = k + r * r
    full_sphere = r * l + 4 * r * r
    confuses_curved_flat = r * l + r * r

    candidates = [
        _format_pi(includes_interface),
        _format_pi(full_sphere),
        _format_pi(confuses_curved_flat),
    ]
    check_distinct(
        k * sympy.pi,
        *(sympy.sympify(_grader_form(c)) for c in candidates),
        msg="composite distractor collides with the answer",
    )

    working = [
        "The shared circle is inside the solid, so it is not part of the surface.",
        f"Cone curved surface: πrl = π × {r} × {l} = {r * l}π cm².",
        f"Hemisphere curved surface: 2πr² = 2π × {r}² = {2 * r * r}π cm².",
        f"Total = {r * l}π + {2 * r * r}π = {_format_pi(k).replace('pi', 'π')} cm².",
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
    "calculate_surface_areas_of_standard_solids",
    level=11,
    presets={
        "easy": dict(mode="sphere"),
        "medium": dict(mode="cone"),
        "hard": dict(mode="composite"),
    },
)
def calculate_surface_areas_of_standard_solids(rng, mode="sphere"):
    """Calculate the surface area of curved-surface solids: spheres, cones
    and a hemisphere-on-cone composite. Answer is an exact multiple of pi."""
    if mode == "sphere":
        return _build_sphere(rng)
    if mode == "cone":
        return _build_cone(rng)
    if mode == "composite":
        return _build_composite(rng)
    raise ValueError(f"unknown mode: {mode!r}")


# ==============================================================================
# Skill B -- find_surface_area_of_familiar_solids (flat-faced solids)
# ==============================================================================


def _verify_cube(s: int, expected: int) -> None:
    """Independent check: sum six separate s*s terms via an explicit loop."""
    recomputed = sum(s * s for _ in range(6))
    check(recomputed == expected, "cube surface area mismatch")


def _verify_rect_prism(l: int, w: int, h: int, expected: int) -> None:
    """Independent check: sum three face-pair terms individually rather than
    the construction's single distributed 2*(l*w + l*h + w*h)."""
    recomputed = 2 * l * w + 2 * l * h + 2 * w * h
    check(recomputed == expected, "rectangular prism surface area mismatch")


def _verify_tri_prism(a: int, b: int, c: int, length: int, expected: int) -> None:
    """Independent check: the two triangular ends computed via a swapped
    grouping (Fraction(a, 2) * b), plus three lateral rectangles summed
    individually rather than via the construction's perimeter*length."""
    ends = 2 * (Fraction(a, 2) * b)
    laterals = a * length + b * length + c * length
    recomputed = ends + laterals
    check(recomputed == expected, "triangular prism surface area mismatch")


def _build_cube(rng) -> dict:
    """Easy tier: cube, SA = 6*s^2."""
    s = rng.randint(2, 26)
    answer = 6 * s * s
    _verify_cube(s, answer)
    answer_text = str(answer)

    text = f"A cube has side length {s} cm. Find its total surface area, in cm²."

    # Misconceptions:
    #  - omits_a_face: 5 faces instead of 6 (THE most common surface-area
    #    error)
    #  - uses_volume_formula: wrong-quantity, s^3 instead of 6*s^2
    #  - treats_each_faces_area_as_its_perimeter: 6*4*s instead of 6*s^2
    omits_face = 5 * s * s
    uses_volume = s**3
    uses_perimeter = 6 * 4 * s

    candidates = [str(omits_face), str(uses_volume), str(uses_perimeter)]

    working = [
        f"A cube has 6 equal square faces, each {s} × {s} = {s * s} cm².",
        f"Total = 6 × {s * s} = {answer} cm².",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_rect_prism(rng) -> dict:
    """Medium tier: rectangular prism, SA = 2*(l*w + l*h + w*h)."""
    l = rng.randint(3, 20)
    w = rng.randint(3, 20)
    h = rng.randint(3, 20)
    answer = 2 * (l * w + l * h + w * h)
    _verify_rect_prism(l, w, h, answer)
    answer_text = str(answer)

    text = f"A rectangular prism measures {l} cm by {w} cm by {h} cm. Find its total surface area, in cm²."

    # Misconceptions:
    #  - omits_a_pair_of_faces: drops the w*h pair entirely
    #  - forgets_factor_of_2: no doubling of the three face-pair areas
    #  - wrong_shape_formula: misapplies the cube formula 6*l^2 (another
    #    family member's formula applied to the wrong solid)
    omits_pair = 2 * (l * w + l * h)
    forgets_factor2 = l * w + l * h + w * h
    wrong_shape = 6 * l * l

    candidates = [str(omits_pair), str(forgets_factor2), str(wrong_shape)]

    working = [
        f"Surface area = 2 × (lw + lh + wh) = 2 × ({l} × {w} + {l} × {h} + {w} × {h}).",
        f"= 2 × ({l * w} + {l * h} + {w * h}) = 2 × {l * w + l * h + w * h} = {answer} cm².",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


TRIANGLE_TABLE = [
    (4, 3, 5),
    (12, 5, 13),
    (8, 15, 17),
    (24, 7, 25),
    (20, 21, 29),
    (40, 9, 41),
    (12, 35, 37),
    (28, 45, 53),
    (60, 11, 61),
    (16, 63, 65),
    (56, 33, 65),
    (48, 55, 73),
    (84, 13, 85),
    (36, 77, 85),
    (80, 39, 89),
]  # (even_leg, odd_leg, hyp) -- the even leg is always listed first so
# (even_leg * odd_leg) // 2, the triangular end's area, is always an
# exact integer regardless of which member of the pair is even.


def _build_tri_prism(rng) -> dict:
    """Hard tier: triangular prism with a right-triangle Pythagorean-triple
    cross-section. SA = 2*(triangle area) + (triangle perimeter)*(prism
    length) -- the escalation is combining two triangular ends with three
    independently-dimensioned rectangular lateral faces."""
    a, b, c = rng.choice(TRIANGLE_TABLE)
    length = rng.randint(5, 30)
    triangle_area = a * b // 2
    perimeter = a + b + c
    answer = 2 * triangle_area + perimeter * length
    _verify_tri_prism(a, b, c, length, answer)
    answer_text = str(answer)

    text = (
        f"A triangular prism has a right-triangular cross-section with legs "
        f"{a} cm and {b} cm and hypotenuse {c} cm, and a length of {length} cm. "
        f"Find its total surface area, in cm²."
    )

    # Misconceptions:
    #  - forgets_half_on_triangular_ends (REQUIRED trap): doubles both
    #    triangular ends instead of halving base*height
    #  - omits_one_triangular_end: counts only one of the two ends
    #  - wrong_shape_formula: misapplies the rectangular-prism formula to the
    #    prism's three linear dimensions (a, b, length), ignoring c entirely
    forgets_half = 2 * a * b + perimeter * length
    omits_one_end = triangle_area + perimeter * length
    wrong_shape = 2 * (a * b + a * length + b * length)

    candidates = [str(forgets_half), str(omits_one_end), str(wrong_shape)]

    working = [
        f"Each triangular end: ½ × {a} × {b} = {triangle_area} cm², so the two ends give {2 * triangle_area} cm².",
        f"The three rectangles together have width {a} + {b} + {c} = {perimeter} cm and length {length} cm: "
        f"{perimeter} × {length} = {perimeter * length} cm².",
        f"Total = {2 * triangle_area} + {perimeter * length} = {answer} cm².",
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
    "find_surface_area_of_familiar_solids",
    level=11,
    presets={
        "easy": dict(mode="cube"),
        "medium": dict(mode="rectangular_prism"),
        "hard": dict(mode="triangular_prism"),
    },
)
def find_surface_area_of_familiar_solids(rng, mode="cube"):
    """Find the surface area of flat-faced solids: cubes, rectangular prisms
    and triangular prisms. Plain integer answer."""
    if mode == "cube":
        return _build_cube(rng)
    if mode == "rectangular_prism":
        return _build_rect_prism(rng)
    if mode == "triangular_prism":
        return _build_tri_prism(rng)
    raise ValueError(f"unknown mode: {mode!r}")
