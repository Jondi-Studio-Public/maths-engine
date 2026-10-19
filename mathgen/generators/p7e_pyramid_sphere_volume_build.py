"""Generates `volume_of_pyramids_and_spheres` questions: the volume of a pyramid or a sphere from stated
dimensions.

Senior volume of pyramids and spheres.

One generator, ``volume_of_pyramids_and_spheres``
(``senior-use-formulas-to-find-the-volume-of-pyramids``, ACMEM104):

- easy: square-based pyramid, ``V = (1/3) * s^2 * h``, both the base edge
  and the height given directly.
- medium: rectangular-based pyramid, ``V = (1/3) * l * w * h`` -- the
  escalation is tracking three independently stated dimensions instead of
  two.
- hard: sphere, ``V = (4/3) * pi * r^3`` -- a genuinely different formula
  family (the 4/3 coefficient and cubing rather than squaring). The
  DIAMETER is given rather than the radius, so the learner must also take
  the standard radius = diameter/2 step; no other spatial derivation is
  ever required.
- exam: sphere again, but a genuine escalation in KIND over ``hard``, not
  merely bigger numbers. Either the diameter OR the radius is given (stated
  explicitly each time), the dimension pool excludes multiples of 6
  (diameter) / 3 (radius) so the exact volume is never a clean integer
  multiple of pi, and the answer is a stated 2 d.p. DECIMAL rather than an
  exact ``k*pi`` expression -- rounding a genuinely irrational value is a
  different reasoning demand from reading off an exact coefficient.

Grader-kind resolution: the pyramid tiers' natural answer is a plain
integer and the sphere tier's natural answer is an exact ``k*pi`` expression, but a skill declares exactly one
grader kind for every tier. A plain integer is a valid sympy expression, so ALL THREE tiers are graded
``expression`` -- the pyramid tiers' ``answer``/``answer_text`` are plain digit strings (e.g. ``"240"``),
which ``grade_expression`` parses and compares by symbolic equality exactly like any other expression. This
keeps the single planning label (``numeric_or_symbolic``) mapped to a single real grader kind.

Every seed in the first three tiers is constructed so the answer is an exact
value with no rounding: the pyramid tiers fix the height as a multiple of 3
(so ``(1/3)*base_area*h`` is always an integer); the sphere (hard) tier fixes
the diameter as a multiple of 6, so the radius is always a multiple of 3 and
``(4/3)*pi*r^3`` always has an integer pi-coefficient. The exam tier inverts
this deliberately -- its dimension pool EXCLUDES those multiples, so the
exact volume is never a clean pi-coefficient and a stated 2 d.p. decimal
answer is the only sensible way to state it (true
``math.pi``/``sympy.pi`` carried at full precision throughout, rounded once
at the very end).
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import sympy

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    pick_distractors,
    rounds_unambiguously,
    stated_precision_tolerance,
)


#: Resample budget for the exam sphere tier's rounding-boundary guard --
#: mirrors ``p7a_trig_ratio_elevation_build.py``'s ``_RESAMPLE_LIMIT``.
_SPHERE_DECIMAL_RESAMPLE_LIMIT = 80


# --- Shared: exact multiple-of-pi formatting (sphere tier only) -------------
# Mirrors the local idiom in ``p5f_tan_shift_build.py`` and
# ``p7e_solid_surface_areas_build.py`` (not centralised in ``core.py``, so
# reproduced locally here rather than imported across sibling modules).


def _format_pi(k: int) -> str:
    """Format an exact integer multiple of pi with no bare '*':
    ``_format_pi(1) -> "pi"``, ``_format_pi(36) -> "36pi"``."""
    return "pi" if k == 1 else f"{k}pi"


def _grader_form(text: str) -> str:
    """The internal ``answer`` (sympifiable) form of a ``_format_pi`` string:
    insert the explicit ``*`` that ``answer_text``/distractors never carry."""
    return re.sub(r"(\d)pi", r"\1*pi", text)


# --- Independent checks (a different route/grouping from construction) -----


def _verify_square_pyramid(s: int, h: int, expected: int) -> None:
    """Independent check: divide h by 3 FIRST, then multiply by s*s -- the
    opposite order from the construction's multiply-then-floor-divide."""
    check(h % 3 == 0, "square pyramid: height not a multiple of 3")
    recomputed = (h // 3) * s * s
    check(recomputed == expected, "square pyramid volume mismatch")


def _verify_rect_pyramid(l: int, w: int, h: int, expected: int) -> None:
    """Independent check: exact Fraction arithmetic dividing h by 3 first,
    a different route from the construction's integer floor-division of the
    full product -- a fractional remainder here would raise rather than
    silently truncate."""
    recomputed = Fraction(l) * Fraction(w) * Fraction(h, 3)
    check(recomputed.denominator == 1, "rectangular pyramid volume not exact")
    check(int(recomputed) == expected, "rectangular pyramid volume mismatch")


def _verify_sphere(d: int, expected_k: int) -> None:
    """Independent check: factor the radius as 3*third and expand
    (4/3)*(3*third)^3 = 36*third^3 algebraically -- a different route from
    the construction's direct r**3 cubing."""
    check(d % 6 == 0, "sphere: diameter not a multiple of 6")
    r = d // 2
    check(r % 3 == 0, "sphere: radius not a multiple of 3")
    third = r // 3
    recomputed = 4 * 9 * third**3
    check(recomputed == expected_k, "sphere volume coefficient mismatch")


# --- Tier builders -----------------------------------------------------------


def _build_square_pyramid(rng) -> dict:
    """Easy tier: square base, V = (1/3)*s^2*h. Single formula, single
    computation once base area and height are known."""
    h = 3 * rng.randint(2, 15)  # multiple of 3: 6..45
    s = rng.randint(2, 25)
    # s == 4 makes the perimeter-as-area distractor (4*s*h/3) collide with
    # the answer (4*s == s*s only at s=4); s == h makes the squares-height
    # distractor (s*h*h/3) collide with the answer (s*h*h == s*s*h at
    # s == h). Resample away from both, by construction.
    while s == 4 or s == h:
        s = rng.randint(2, 25)
    answer = (s * s * h) // 3
    _verify_square_pyramid(s, h, answer)
    answer_text = str(answer)

    text = f"A pyramid has a square base with side length {s} cm and a height of {h} cm. Find its volume, in cm³."

    # Misconceptions:
    #  - omits_one_third_factor: treats the pyramid as a prism, s*s*h
    #  - confuses_perimeter_with_base_area: uses the base's perimeter (4*s)
    #    instead of its area (s*s)
    #  - squares_wrong_dimension: squares the height instead of the base
    #    edge, (1/3)*s*h*h
    omits_third = s * s * h
    perimeter_as_area = (4 * s * h) // 3
    squares_height = (s * h * h) // 3

    candidates = [str(omits_third), str(perimeter_as_area), str(squares_height)]
    check_distinct(
        answer,
        omits_third,
        perimeter_as_area,
        squares_height,
        msg="square pyramid distractor collides with the answer",
    )

    working = [
        f"Base area = {s} × {s} = {s * s} cm².",
        f"Volume = ⅓ × base area × height = ⅓ × {s * s} × {h} = {answer} cm³.",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer_text,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_rect_pyramid(rng) -> dict:
    """Medium tier: rectangular base, V = (1/3)*l*w*h -- the escalation is
    correctly tracking and combining three independently stated dimensions
    instead of two."""
    h = 3 * rng.randint(2, 15)  # multiple of 3: 6..45
    l = rng.randint(3, 20)
    w = 2 * rng.randint(2, 10)  # always even: 4..20, so l*w*h is always even
    answer = (l * w * h) // 3
    _verify_rect_pyramid(l, w, h, answer)
    answer_text = str(answer)

    text = (
        f"A pyramid has a rectangular base measuring {l} cm by {w} cm and a height of {h} cm. Find its volume, in cm³."
    )

    # Misconceptions:
    #  - omits_one_third_factor: treats the pyramid as a prism, l*w*h
    #  - uses_prism_half_factor (wrong-shape-formula): applies the
    #    triangular prism's 1/2 factor instead of the pyramid's 1/3 factor
    #  - omits_a_dimension: uses only two of the three stated dimensions,
    #    dropping the width entirely
    omits_third = l * w * h
    prism_half_factor = (l * w * h) // 2
    omits_width = (h // 3) * l

    candidates = [str(omits_third), str(prism_half_factor), str(omits_width)]
    check_distinct(
        answer,
        omits_third,
        prism_half_factor,
        omits_width,
        msg="rectangular pyramid distractor collides with the answer",
    )

    working = [
        f"Base area = {l} × {w} = {l * w} cm².",
        f"Volume = ⅓ × base area × height = ⅓ × {l * w} × {h} = {answer} cm³.",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer_text,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_sphere(rng) -> dict:
    """Hard tier: sphere, V = (4/3)*pi*r^3 -- a genuinely different formula
    family from the pyramid tiers (the 4/3 coefficient and cubing rather
    than squaring). The DIAMETER is given, adding the standard
    radius = diameter/2 step over the pyramid tiers' single-step formula
    application."""
    third = rng.randint(1, 30)
    r = 3 * third  # multiple of 3: 3..90
    d = 2 * r  # multiple of 6: 6..180
    k = 4 * 9 * third**3  # (4/3) * r^3 = (4/3) * (3*third)^3 = 36*third^3
    _verify_sphere(d, k)

    answer_text = _format_pi(k)
    answer = _grader_form(answer_text)

    text = (
        f"A sphere has a diameter of {d} cm. Find its volume, in cm³. "
        f"Give your answer as an exact value in terms of pi."
    )

    # Misconceptions:
    #  - uses_diameter_as_radius: cubes the diameter directly instead of
    #    halving it first, (4/3)*pi*d^3
    #  - uses_4_instead_of_4_thirds: omits the /3, 4*pi*r^3
    #  - confuses_sphere_with_circle_area: squares the radius instead of
    #    cubing it, (4/3)*pi*r^2
    uses_diameter = 4 * d**3 // 3
    uses_four = 4 * r**3
    squares_not_cubes = (4 * r * r) // 3

    candidates = [
        _format_pi(uses_diameter),
        _format_pi(uses_four),
        _format_pi(squares_not_cubes),
    ]
    check_distinct(
        k * sympy.pi,
        *(sympy.sympify(_grader_form(c)) for c in candidates),
        msg="sphere distractor collides with the answer",
    )

    working = [
        f"Radius = {d} ÷ 2 = {r} cm.",
        f"Volume = (4/3)πr³ = (4/3) × π × {r}³ = (4/3) × {r**3}π = {k}π cm³.",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


def _verify_sphere_decimal(r: Fraction, given_kind: str, given_value: int, expected_text: str) -> None:
    """Independent check: re-derive the radius from the given quantity, then
    recompute the volume via sympy's exact ``Rational`` radius and a
    high-precision (mpmath-backed) evaluation of pi -- a different numeric
    route from the construction's ``math.pi`` float cubing -- and round via
    a second, independent formatting call to confirm it matches the
    constructed ``answer_text`` exactly."""
    if given_kind == "diameter":
        check_equivalent(Fraction(given_value, 2), r, "sphere decimal: radius does not match given diameter/2")
    else:
        check_equivalent(Fraction(given_value), r, "sphere decimal: radius does not match given radius")
    exact = sympy.Rational(4, 3) * sympy.pi * sympy.Rational(r.numerator, r.denominator) ** 3
    value = float(sympy.N(exact, 30))
    check(rounds_unambiguously(value, 2), "sphere decimal volume sits on a rounding boundary")
    recomputed_text = f"{value:.2f}"
    check(recomputed_text == expected_text, "sphere decimal volume mismatch")


def _build_sphere_decimal(rng) -> dict:
    """Exam tier: sphere volume from EITHER a diameter or a radius, answered
    as a stated 2 d.p. decimal -- a genuine escalation in KIND over the hard
    tier's exact ``k*pi`` expression, not merely bigger numbers. The
    dimension pool excludes multiples of 6 (diameter) / 3 (radius) by
    construction, so the exact volume is never a clean integer multiple of
    pi the way the hard tier's always is. Resamples (mirroring
    ``p7a_trig_ratio_elevation_build.py``'s ``_easy_tier_elevation``) away
    from the rare draw whose rounded volume sits on the ``.xx5`` boundary,
    rather than failing the question outright."""
    _tol = float(stated_precision_tolerance(2))
    given_kind = r = given_value = phrasing = volume = answer_text = None
    diameter_confusion = uses_four = circle_area = None
    for _ in range(_SPHERE_DECIMAL_RESAMPLE_LIMIT):
        given_kind = rng.choice(["diameter", "radius"])
        phrasing = rng.choice([0, 1])
        if given_kind == "diameter":
            d = rng.randint(7, 44)
            while d % 6 == 0:
                d = rng.randint(7, 44)
            r = Fraction(d, 2)
            given_value = d
        else:
            r_int = rng.randint(4, 27)
            while r_int % 3 == 0:
                r_int = rng.randint(4, 27)
            r = Fraction(r_int)
            given_value = r_int

        r_float = float(r)
        volume = (4 / 3) * math.pi * r_float**3
        if not rounds_unambiguously(volume, 2):
            continue

        # Misconceptions (reused unchanged from the hard tier's catalogue):
        #  - uses_diameter_as_radius: when the DIAMETER is given, cubes it
        #    directly instead of halving it first; when the RADIUS is given
        #    directly (already correct as-is), the mirrored error is
        #    halving it anyway, assuming every sphere problem states a
        #    diameter.
        #  - uses_4_instead_of_4_thirds: omits the /3, 4*pi*r^3.
        #  - confuses_sphere_with_circle_area: squares the radius instead
        #    of cubing it, (4/3)*pi*r^2.
        if given_kind == "diameter":
            wrong_r = float(Fraction(given_value))
        else:
            wrong_r = float(r / 2)
        diameter_confusion = (4 / 3) * math.pi * wrong_r**3
        uses_four = 4 * math.pi * r_float**3
        circle_area = (4 / 3) * math.pi * r_float**2

        # Guard the rare coincidence where two of the four values (answer
        # plus three misconception distractors) land within the stated
        # precision of each other -- e.g. radius=8 makes (r/2)^3 == r^2
        # exactly, colliding uses_diameter_as_radius with
        # confuses_sphere_with_circle_area. Resample rather than silently
        # dropping a distractor and shipping only two options.
        values = [volume, diameter_confusion, uses_four, circle_area]
        collision = any(
            abs(values[i] - values[j]) <= _tol for i in range(len(values)) for j in range(i + 1, len(values))
        )
        if collision:
            continue

        answer_text = f"{volume:.2f}"
        break
    else:
        check(False, "sphere decimal: no non-degenerate draw found")
    answer = answer_text
    _verify_sphere_decimal(r, given_kind, given_value, answer_text)

    if phrasing == 0:
        text = f"A sphere has a {given_kind} of {given_value} cm. Find its volume, in cm³, correct to 2 decimal places."
    else:
        text = (
            f"A sphere's {given_kind} is {given_value} cm. Calculate its volume, in cm³, correct to 2 decimal places."
        )

    check_distinct(
        volume,
        diameter_confusion,
        uses_four,
        circle_area,
        msg="sphere decimal distractor collides with the answer",
    )
    candidates = [
        f"{diameter_confusion:.2f}",
        f"{uses_four:.2f}",
        f"{circle_area:.2f}",
    ]
    distractors = pick_distractors(
        answer_text,
        candidates,
        tolerance=stated_precision_tolerance(2),
    )

    r_text = f"{float(r):g}"
    radius_line = f"Radius = {given_value} ÷ 2 = {r_text} cm." if given_kind == "diameter" else f"Radius = {r_text} cm."
    working = [
        radius_line,
        f"Volume = (4/3)πr³ = (4/3) × π × {r_text}³ = {answer_text} cm³ (2 d.p.).",
    ]

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        working=working,
        distractors=distractors,
    )


@generator(
    "volume_of_pyramids_and_spheres",
    level=11,
    presets={
        "easy": dict(mode="square_pyramid"),
        "medium": dict(mode="rectangular_pyramid"),
        "hard": dict(mode="sphere"),
        "exam": dict(mode="sphere_decimal"),
    },
)
def volume_of_pyramids_and_spheres(rng, mode="square_pyramid"):
    """Use formulas to find the volume of pyramids and spheres. Pyramid
    tiers answer with a plain integer; the hard sphere tier answers with an
    exact integer multiple of pi; the exam sphere tier answers with a
    stated 2 d.p. decimal -- all four graded as ``expression``."""
    if mode == "square_pyramid":
        return _build_square_pyramid(rng)
    if mode == "rectangular_pyramid":
        return _build_rect_pyramid(rng)
    if mode == "sphere":
        return _build_sphere(rng)
    if mode == "sphere_decimal":
        return _build_sphere_decimal(rng)
    raise ValueError(f"unknown mode: {mode!r}")
