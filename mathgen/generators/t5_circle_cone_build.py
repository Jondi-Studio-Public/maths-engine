"""Generates `circle_reverse_from_area` and `cone_volume` questions: a circle's radius, diameter or circumference
from its area, and the volume of a cone.

Circle-reverse-from-area
+ cone-volume.
YAMLs are the whole contract.

- ``circle_reverse_from_area`` (``skills/measurement/circle-reverse-from-area.yaml``):
  the inverse of ``f10-circle-measures`` -- given a circle's AREA, find its
  radius (easy), diameter (medium) or circumference (hard).

  **Critical design point.** The area stated in the question is itself a
  rounded (2dp) value -- a learner never sees the true, unrounded internal
  radius used to construct it. Every reverse quantity here is therefore
  recomputed from that STATED area alone (``_draw_area_stated`` returns it
  and nothing else); the internal radius used to build the area is
  discarded immediately and never touched again. Because the inverse
  involves a square root, ``medium``/``hard`` each have two defensible
  learner routes (round-then-continue vs continue-then-round) that must
  agree at the stated 1dp precision on every seed shipped -- enforced by a
  bounded resample loop, the same shape as
  ``p8d_quadrilateral_theorems_build.py``'s own resample loops. No float
  sqrt anywhere: ``_newton_sqrt`` is a ``Fraction``-based Newton's method,
  cross-checked against ``_bisection_sqrt``, a structurally different
  algorithm, via ``_verify_sqrt``.

- ``cone_volume`` (``skills/measurement/cone-volume.yaml``):
  ``V = (1/3) * pi * r**2 * h``. Not covered by
  ``senior-use-formulas-to-find-the-volume-of-pyramids`` (square pyramid,
  rectangular pyramid and sphere only). ``hard`` gives the slant height
  instead of the height, so the height must be derived via Pythagoras --
  drawn from a scaled Pythagorean triple so that derivation always lands on
  an exact integer.

``cone_volume`` rounds its final volume to 1 decimal place (not 2 -- the error budget: at 2dp the ~2.65e-6 relative error between 3.14159 and true pi already eats
too much of a 0.005 tolerance on a volume in the hundreds/thousands). ``circle_reverse_from_area`` uses its
own, narrower precision scheme (area to 2dp, every reverse quantity to 1dp) described above.
"""

from __future__ import annotations

import math
from fractions import Fraction

from sympy import sqrt as sympy_sqrt

from ..core import check, check_distinct, check_equivalent, generator, pick_distractors, round_half_up

PI = Fraction(314159, 100000)

_HALF = Fraction(1, 2)
_NOT_ON_BOUNDARY = Fraction(1, 1000)
_DISTRACTOR_TOLERANCE = Fraction(1, 20)


def _boundary_margin(value: Fraction, places: int) -> Fraction:
    """Distance from `value` to the nearest `places`-decimal-place rounding
    boundary (an ``x.x5``-style midpoint), in the same units as `value`.
    Small means "sitting on the boundary"; the generator resamples whenever
    this is not comfortably positive."""
    scale = 10**places
    scaled = value * scale
    frac = scaled - (scaled.numerator // scaled.denominator)
    return abs(frac - _HALF) / scale


def _fmt1(value: Fraction) -> str:
    return f"{float(value):.1f}"


# ---------------------------------------------------------------------------
# Fraction-based square root -- no float sqrt anywhere in this module.
# ---------------------------------------------------------------------------


def _newton_sqrt(n: Fraction, iterations: int = 6, cap: int = 10**24) -> Fraction:
    """Newton's-method square root in exact ``Fraction`` arithmetic. The
    seed comes from ``math.isqrt`` (an exact integer floor-sqrt, not a
    float approximation) so the whole computation stays exact-rational and
    reproducible bit-for-bit across machines. ``limit_denominator(cap)``
    after every step is a performance cap only, not a precision one: it
    still leaves ~24 decimal digits, orders of magnitude past both the
    1e-6 cross-check below and the 1dp this module ever rounds to."""
    check(n >= 0, "_newton_sqrt: cannot take the square root of a negative Fraction")
    if n == 0:
        return Fraction(0)
    seed = math.isqrt(n.numerator * n.denominator)
    guess = Fraction(max(seed, 1), n.denominator)
    for _ in range(iterations):
        guess = ((guess + n / guess) / 2).limit_denominator(cap)
    return guess


def _bisection_sqrt(n: Fraction, iterations: int = 40) -> Fraction:
    """Independent square root by bisection -- a structurally different
    algorithm from ``_newton_sqrt``, used only to cross-check it."""
    check(n >= 0, "_bisection_sqrt: cannot take the square root of a negative Fraction")
    if n == 0:
        return Fraction(0)
    lo, hi = Fraction(0), n if n >= 1 else Fraction(1)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if mid * mid <= n:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _verify_sqrt(n: Fraction, claimed: Fraction) -> None:
    """The independent check: re-derive sqrt(n) via bisection (a
    structurally different route from Newton's method) and confirm it
    lands within Fraction(1, 10**6) of the value construction claims."""
    bisected = _bisection_sqrt(n)
    check(
        abs(bisected - claimed) < Fraction(1, 10**6),
        f"sqrt verification failed: bisection={bisected} claimed={claimed} n={n}",
    )


# ============================================================================
# Skill 1: circle_reverse_from_area
# ============================================================================

_MIN_TENTHS = 20  # internal radius >= 2.0 cm
_MAX_TENTHS = 150  # internal radius <= 15.0 cm


def _draw_area_stated(rng) -> Fraction:
    """Draw a clean-tenth internal radius purely to build a plausible
    stated area, then discard the radius immediately. Every downstream
    computation in this module works from the returned STATED (rounded to
    2dp) area alone -- never from this internal radius (the module
    docstring's "critical design point": a learner only ever reads the
    rounded area)."""
    k = rng.randint(_MIN_TENTHS, _MAX_TENTHS)
    r_internal = Fraction(k, 10)
    area = PI * r_internal**2
    check(area >= 0, "cone/circle measurement must not be negative")
    return round_half_up(area, 2)


def _route_a_diameter(area_stated: Fraction) -> Fraction:
    """Route A: round the radius to 1dp first, then double."""
    r_exact = _newton_sqrt(area_stated / PI)
    check(r_exact >= 0, "cone/circle measurement must not be negative")
    return 2 * round_half_up(r_exact, 1)


def _route_b_diameter(area_stated: Fraction) -> Fraction:
    """Route B: 2*sqrt(area/pi) computed directly, rounded once."""
    r_exact = _newton_sqrt(area_stated / PI)
    doubled = 2 * r_exact
    check(doubled >= 0, "cone/circle measurement must not be negative")
    return round_half_up(doubled, 1)


def _route_a_circumference(area_stated: Fraction) -> Fraction:
    """Route A: round the radius to 1dp first, then C = 2*pi*r, rounded again."""
    r_exact = _newton_sqrt(area_stated / PI)
    check(r_exact >= 0, "cone/circle measurement must not be negative")
    r_rounded = round_half_up(r_exact, 1)
    circumference = 2 * PI * r_rounded
    check(circumference >= 0, "cone/circle measurement must not be negative")
    return round_half_up(circumference, 1)


def _route_b_circumference(area_stated: Fraction) -> Fraction:
    """Route B: 2*sqrt(pi*area) computed directly, rounded once."""
    circumference = 2 * _newton_sqrt(PI * area_stated)
    check(circumference >= 0, "cone/circle measurement must not be negative")
    return round_half_up(circumference, 1)


def _area_text(area_stated: Fraction) -> str:
    text = f"{float(area_stated):.2f}"
    check(
        Fraction(text) == area_stated,
        "circle_reverse_from_area: rendered area text does not round-trip to the stated area",
    )
    return text


def _circle_reverse_easy(rng):
    found = False
    area_stated = r_exact = None
    for _ in range(1000):
        area_stated = _draw_area_stated(rng)
        r_exact = _newton_sqrt(area_stated / PI)
        _verify_sqrt(area_stated / PI, r_exact)
        if _boundary_margin(r_exact, 1) <= _NOT_ON_BOUNDARY:
            continue
        found = True
        break
    check(found, "circle_reverse_from_area/easy: no non-boundary draw found in 1000 attempts")

    check(r_exact >= 0, "cone/circle measurement must not be negative")
    r_answer = round_half_up(r_exact, 1)
    answer_text = _fmt1(r_answer)
    answer = float(r_answer)
    area_text = _area_text(area_stated)

    question_text = (
        f"A circle has an area of {area_text} cm². Using π ≈ 3.14159, find "
        f"the radius of the circle, in cm, to 1 decimal place."
    )

    bad_forgets_sqrt = area_stated / PI
    bad_omits_pi = _newton_sqrt(area_stated)
    _verify_sqrt(area_stated, bad_omits_pi)

    check_distinct(
        r_exact, bad_forgets_sqrt, msg="circle_reverse_from_area/easy: forgets-square-root collides with the answer"
    )
    check_distinct(
        r_exact, bad_omits_pi, msg="circle_reverse_from_area/easy: omits-pi-division collides with the answer"
    )

    check(bad_forgets_sqrt >= 0, "cone/circle measurement must not be negative")
    check(bad_omits_pi >= 0, "cone/circle measurement must not be negative")
    distractors = pick_distractors(
        answer_text,
        [_fmt1(round_half_up(bad_forgets_sqrt, 1)), _fmt1(round_half_up(bad_omits_pi, 1))],
        tolerance=_DISTRACTOR_TOLERANCE,
    )

    working = [
        "Area = π × r², so r = sqrt(area ÷ π).",
        f"r = sqrt({area_text} ÷ 3.14159) = {answer_text} cm (to 1 decimal place).",
    ]
    return dict(
        text=question_text,
        latex=question_text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=working,
    )


def _circle_reverse_medium(rng):
    found = False
    area_stated = r_exact = diameter_b = None
    for _ in range(1000):
        area_stated = _draw_area_stated(rng)
        r_exact = _newton_sqrt(area_stated / PI)
        _verify_sqrt(area_stated / PI, r_exact)
        if _boundary_margin(r_exact, 1) <= _NOT_ON_BOUNDARY:
            continue
        diameter_b_exact = 2 * r_exact
        if _boundary_margin(diameter_b_exact, 1) <= _NOT_ON_BOUNDARY:
            continue
        diameter_a = _route_a_diameter(area_stated)
        diameter_b = _route_b_diameter(area_stated)
        if diameter_a != diameter_b:
            continue
        found = True
        break
    check(found, "circle_reverse_from_area/medium: no seed with agreeing, non-boundary routes found in 1000 attempts")

    diameter_answer = diameter_b
    answer_text = _fmt1(diameter_answer)
    answer = float(diameter_answer)
    area_text = _area_text(area_stated)
    check(r_exact >= 0, "cone/circle measurement must not be negative")
    r_rounded = round_half_up(r_exact, 1)

    question_text = (
        f"A circle has an area of {area_text} cm². Using π ≈ 3.14159, find "
        f"the diameter of the circle, in cm, to 1 decimal place."
    )

    bad_forgets_sqrt_r = area_stated / PI
    bad_forgets_sqrt = 2 * bad_forgets_sqrt_r
    bad_omits_pi_r = _newton_sqrt(area_stated)
    _verify_sqrt(area_stated, bad_omits_pi_r)
    bad_omits_pi = 2 * bad_omits_pi_r

    diameter_exact = 2 * r_exact
    check_distinct(
        diameter_exact,
        bad_forgets_sqrt,
        msg="circle_reverse_from_area/medium: forgets-square-root collides with the answer",
    )
    check_distinct(
        diameter_exact, bad_omits_pi, msg="circle_reverse_from_area/medium: omits-pi-division collides with the answer"
    )

    check(bad_forgets_sqrt >= 0, "cone/circle measurement must not be negative")
    check(bad_omits_pi >= 0, "cone/circle measurement must not be negative")
    distractors = pick_distractors(
        answer_text,
        [_fmt1(round_half_up(bad_forgets_sqrt, 1)), _fmt1(round_half_up(bad_omits_pi, 1))],
        tolerance=_DISTRACTOR_TOLERANCE,
    )

    working = [
        "Area = π × r², so r = sqrt(area ÷ π); diameter = 2 × r.",
        f"r = sqrt({area_text} ÷ 3.14159) = {_fmt1(r_rounded)} cm.",
        f"diameter = 2 × {_fmt1(r_rounded)} = {answer_text} cm (to 1 decimal place).",
    ]
    return dict(
        text=question_text,
        latex=question_text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=working,
    )


def _circle_reverse_hard(rng):
    found = False
    area_stated = r_exact = circumference_b = None
    for _ in range(1000):
        area_stated = _draw_area_stated(rng)
        r_exact = _newton_sqrt(area_stated / PI)
        _verify_sqrt(area_stated / PI, r_exact)
        if _boundary_margin(r_exact, 1) <= _NOT_ON_BOUNDARY:
            continue
        check(r_exact >= 0, "cone/circle measurement must not be negative")
        r_rounded = round_half_up(r_exact, 1)
        circumference_a_exact = 2 * PI * r_rounded
        if _boundary_margin(circumference_a_exact, 1) <= _NOT_ON_BOUNDARY:
            continue
        circumference_b_exact = 2 * _newton_sqrt(PI * area_stated)
        _verify_sqrt(PI * area_stated, circumference_b_exact / 2)
        if _boundary_margin(circumference_b_exact, 1) <= _NOT_ON_BOUNDARY:
            continue
        circumference_a = _route_a_circumference(area_stated)
        circumference_b = _route_b_circumference(area_stated)
        if circumference_a != circumference_b:
            continue
        found = True
        break
    check(found, "circle_reverse_from_area/hard: no seed with agreeing, non-boundary routes found in 1000 attempts")

    circumference_answer = circumference_b
    answer_text = _fmt1(circumference_answer)
    answer = float(circumference_answer)
    area_text = _area_text(area_stated)
    check(r_exact >= 0, "cone/circle measurement must not be negative")
    r_rounded = round_half_up(r_exact, 1)

    question_text = (
        f"A circle has an area of {area_text} cm². Using π ≈ 3.14159, find "
        f"the circumference of the circle, in cm, to 1 decimal place."
    )

    bad_forgets_sqrt_r = area_stated / PI
    bad_forgets_sqrt = 2 * PI * bad_forgets_sqrt_r
    bad_omits_pi_r = _newton_sqrt(area_stated)
    _verify_sqrt(area_stated, bad_omits_pi_r)
    bad_omits_pi = 2 * PI * bad_omits_pi_r
    bad_reports_diameter = 2 * r_exact

    circumference_exact = 2 * PI * r_exact
    check_distinct(
        circumference_exact,
        bad_forgets_sqrt,
        msg="circle_reverse_from_area/hard: forgets-square-root collides with the answer",
    )
    check_distinct(
        circumference_exact,
        bad_omits_pi,
        msg="circle_reverse_from_area/hard: omits-pi-division collides with the answer",
    )
    check_distinct(
        circumference_exact,
        bad_reports_diameter,
        msg="circle_reverse_from_area/hard: reports-diameter-as-circumference collides with the answer",
    )

    check(bad_forgets_sqrt >= 0, "cone/circle measurement must not be negative")
    check(bad_omits_pi >= 0, "cone/circle measurement must not be negative")
    check(bad_reports_diameter >= 0, "cone/circle measurement must not be negative")
    distractors = pick_distractors(
        answer_text,
        [
            _fmt1(round_half_up(bad_forgets_sqrt, 1)),
            _fmt1(round_half_up(bad_omits_pi, 1)),
            _fmt1(round_half_up(bad_reports_diameter, 1)),
        ],
        tolerance=_DISTRACTOR_TOLERANCE,
    )

    working = [
        "Area = π × r², so r = sqrt(area ÷ π); circumference = 2 × π × r.",
        f"r = sqrt({area_text} ÷ 3.14159) = {_fmt1(r_rounded)} cm.",
        f"circumference = 2 × 3.14159 × {_fmt1(r_rounded)} = {answer_text} cm (to 1 decimal place).",
    ]
    return dict(
        text=question_text,
        latex=question_text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=working,
    )


@generator(
    "circle_reverse_from_area",
    level=12,
    presets={
        "easy": dict(variant="radius"),
        "medium": dict(variant="diameter"),
        "hard": dict(variant="circumference"),
    },
)
def circle_reverse_from_area(rng, variant="radius"):
    """Inverse of f10-circle-measures: given a circle's stated (rounded)
    area, find its radius / diameter / circumference. See the module
    docstring's "critical design point" -- every reverse quantity is
    recomputed from the STATED area, never from the internal radius used
    to construct it."""
    if variant == "radius":
        return _circle_reverse_easy(rng)
    if variant == "diameter":
        return _circle_reverse_medium(rng)
    if variant == "circumference":
        return _circle_reverse_hard(rng)
    raise ValueError(f"unknown variant {variant!r}")


# ============================================================================
# Skill 2: cone_volume
# ============================================================================

# Copied verbatim from p8d_quadrilateral_theorems_build.py's own _TRIPLES.
_TRIPLES = [
    (3, 4, 5),
    (6, 8, 10),
    (5, 12, 13),
    (8, 15, 17),
    (7, 24, 25),
    (9, 12, 15),
    (12, 16, 20),
    (20, 21, 29),
    (9, 40, 41),
]

# Plausibility band. A cone whose height is under a quarter of its radius (near-flat disc) or over
# four times its radius (near-spike) is technically a cone but not what this skill is teaching.
# Applied to every tier so the drawn figure is always one a learner could believe.
_MIN_ASPECT = Fraction(1, 4)
_MAX_ASPECT = Fraction(4, 1)

_MIN_DIM = 3
_MAX_DIM = 20

# A volume cap, not just a dimension cap, because the hard tier's Pythagorean
# triples can pair a large leg with a small one (aspect ratio still inside
# the band) while V ~ r^2*h still grows unboundedly with either dimension.
# Found empirically: with the triple/scale draw unconstrained, r=80, h=84
# gives V~563000 cm^3, where 3.14159's ~8.4e-7 relative error against true
# pi shifts the volume enough that the 1dp-rounded answer using a
# calculator's built-in pi DISAGREES with the one using the stated
# 3.14159 -- by 0.5, ten times the 0.05 grading tolerance. This cap keeps
# every shipped volume at a magnitude comparable to easy/medium (the
# difficulty progression is meant to add a reasoning step, not bigger
# numbers); `_pi_precision_safe` below is the direct, load-bearing proof
# that the specific disagreement cannot happen.
_MAX_CONE_VOLUME = Fraction(12000)

# Used only as a one-sided safety cross-check (never in the actual construction or grading of a
# question): confirms that the volume a learner computes with the STATED pi and the volume a learner
# would get from their calculator's own (far more precise) pi round to the SAME 1dp answer, for
# every (r, h) this module ever ships.
_PI_HIGH_PRECISION = Fraction(31415926535897932, 10**16)


def _pi_precision_safe(r, h) -> bool:
    """Whether the STATED-pi volume and a much-higher-precision-pi volume
    round to the identical 1dp answer -- the direct test of the finding
    above (an indirect boundary-margin proxy either rejected too much or
    too little; this compares the two pi values the way a real learner's
    calculator would diverge from the question's stated constant)."""
    v_stated = _volume(r, h)
    v_precise = _PI_HIGH_PRECISION * Fraction(r) ** 2 * Fraction(h) * Fraction(1, 3)
    check(v_stated >= 0, "cone/circle measurement must not be negative")
    check(v_precise >= 0, "cone/circle measurement must not be negative")
    return round_half_up(v_stated, 1) == round_half_up(v_precise, 1)


def _aspect_ok(r, h) -> bool:
    ratio = Fraction(h) / Fraction(r)
    return _MIN_ASPECT <= ratio <= _MAX_ASPECT


def _volume(r, h) -> Fraction:
    """Primary computation: PI first, then r squared, then h, then the
    1/3 factor last."""
    return PI * Fraction(r) ** 2 * Fraction(h) * Fraction(1, 3)


def _volume_check(r, h) -> Fraction:
    """Independent cross-check: a differently-grouped computation -- r
    squared times h times PI first, dividing by 3 last (construction
    applies the 1/3 factor last too, but as a multiplication by
    Fraction(1, 3) rather than a division, and in a different left-to-right
    order)."""
    return (Fraction(r) ** 2 * Fraction(h) * PI) / 3


def _cone_easy(rng):
    found = False
    r = h = v_exact = None
    for _ in range(1000):
        r = rng.randint(_MIN_DIM, _MAX_DIM)
        h = rng.randint(_MIN_DIM, _MAX_DIM)
        if r == h:
            # squares-wrong-dimension (swap r/h) would collide with the
            # real answer when the two dimensions are equal.
            continue
        if h == 3 * r:
            # omits-one-third-factor (pi*r^2*h) and squares-wrong-dimension
            # ((1/3)*pi*h^2*r) collide exactly when h == 3*r: both then equal
            # pi*r^2*h -- a deterministic property of (r, h), not a rare
            # resample-away collision.
            continue
        if not _aspect_ok(r, h):
            continue
        v_exact = _volume(r, h)
        if v_exact > _MAX_CONE_VOLUME:
            continue
        if _boundary_margin(v_exact, 1) <= _NOT_ON_BOUNDARY:
            continue
        if not _pi_precision_safe(r, h):
            continue
        found = True
        break
    check(found, "cone_volume/easy: no plausible, non-boundary (r, h) draw found in 1000 attempts")

    v_check = _volume_check(r, h)
    check_equivalent(v_exact, v_check, "cone_volume/easy: independent volume cross-check disagrees")

    check(v_exact >= 0, "cone/circle measurement must not be negative")
    v_answer = round_half_up(v_exact, 1)
    answer_text = _fmt1(v_answer)
    answer = float(v_answer)

    question_text = (
        f"A cone has a radius of {r} cm and a height of {h} cm. Using "
        f"π ≈ 3.14159, find its volume, in cm³, to 1 decimal place."
    )

    bad_no_third = PI * Fraction(r) ** 2 * Fraction(h)
    bad_wrong_dim = PI * Fraction(h) ** 2 * Fraction(r) * Fraction(1, 3)

    check_distinct(v_exact, bad_no_third, msg="cone_volume/easy: omits-one-third-factor collides with the answer")
    check_distinct(v_exact, bad_wrong_dim, msg="cone_volume/easy: squares-wrong-dimension collides with the answer")

    check(bad_no_third >= 0, "cone/circle measurement must not be negative")
    check(bad_wrong_dim >= 0, "cone/circle measurement must not be negative")
    distractors = pick_distractors(
        answer_text,
        [_fmt1(round_half_up(bad_no_third, 1)), _fmt1(round_half_up(bad_wrong_dim, 1))],
        tolerance=_DISTRACTOR_TOLERANCE,
    )

    working = [
        "Volume = (1/3) × π × r² × h.",
        f"Volume = (1/3) × 3.14159 × {r}² × {h} = {answer_text} cm³ (to 1 decimal place).",
    ]
    return dict(
        text=question_text,
        latex=question_text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=working,
    )


def _cone_medium(rng):
    found = False
    r = h = v_exact = None
    for _ in range(1000):
        r = rng.randint(_MIN_DIM, _MAX_DIM)
        h = rng.randint(_MIN_DIM, _MAX_DIM)
        if r == h:
            continue
        if h == 3 * r:
            # omits-one-third-factor and squares-wrong-dimension collide
            # when h == 3*r (see _cone_easy).
            continue
        if h == 4 * r:
            # squares-wrong-dimension ((1/3)*pi*h^2*r) and
            # uses-diameter-as-radius ((4/3)*pi*r^2*h, d=2r) collide exactly
            # when h == 4*r: both then equal (4/3)*pi*r^2*h.
            continue
        if not _aspect_ok(r, h):
            continue
        v_exact = _volume(r, h)
        if v_exact > _MAX_CONE_VOLUME:
            continue
        if _boundary_margin(v_exact, 1) <= _NOT_ON_BOUNDARY:
            continue
        if not _pi_precision_safe(r, h):
            continue
        found = True
        break
    check(found, "cone_volume/medium: no plausible, non-boundary (r, h) draw found in 1000 attempts")
    d = 2 * r

    v_check = _volume_check(r, h)
    check_equivalent(v_exact, v_check, "cone_volume/medium: independent volume cross-check disagrees")

    check(v_exact >= 0, "cone/circle measurement must not be negative")
    v_answer = round_half_up(v_exact, 1)
    answer_text = _fmt1(v_answer)
    answer = float(v_answer)

    question_text = (
        f"A cone has a diameter of {d} cm and a height of {h} cm. Using "
        f"π ≈ 3.14159, find its volume, in cm³, to 1 decimal place."
    )

    bad_no_third = PI * Fraction(r) ** 2 * Fraction(h)
    bad_wrong_dim = PI * Fraction(h) ** 2 * Fraction(r) * Fraction(1, 3)
    bad_diameter_as_radius = PI * Fraction(d) ** 2 * Fraction(h) * Fraction(1, 3)

    check_distinct(v_exact, bad_no_third, msg="cone_volume/medium: omits-one-third-factor collides with the answer")
    check_distinct(v_exact, bad_wrong_dim, msg="cone_volume/medium: squares-wrong-dimension collides with the answer")
    check_distinct(
        v_exact, bad_diameter_as_radius, msg="cone_volume/medium: uses-diameter-as-radius collides with the answer"
    )

    check(bad_no_third >= 0, "cone/circle measurement must not be negative")
    check(bad_wrong_dim >= 0, "cone/circle measurement must not be negative")
    check(bad_diameter_as_radius >= 0, "cone/circle measurement must not be negative")
    distractors = pick_distractors(
        answer_text,
        [
            _fmt1(round_half_up(bad_no_third, 1)),
            _fmt1(round_half_up(bad_wrong_dim, 1)),
            _fmt1(round_half_up(bad_diameter_as_radius, 1)),
        ],
        tolerance=_DISTRACTOR_TOLERANCE,
    )

    working = [
        "Radius = diameter ÷ 2, then Volume = (1/3) × π × r² × h.",
        f"r = {d} ÷ 2 = {r} cm.",
        f"Volume = (1/3) × 3.14159 × {r}² × {h} = {answer_text} cm³ (to 1 decimal place).",
    ]
    return dict(
        text=question_text,
        latex=question_text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=working,
    )


def _cone_hard(rng):
    found = False
    r = h = l = v_exact = None
    for _ in range(1000):
        leg1, leg2, hyp = rng.choice(_TRIPLES)
        if rng.random() < 0.5:
            leg1, leg2 = leg2, leg1
        scale = rng.randint(1, 9)
        candidate_r, candidate_h, hyp_scaled = leg1 * scale, leg2 * scale, hyp * scale
        # The leg order is randomised (not tried both ways with the first
        # valid one kept) so both leg-to-(r,h) assignments of a triple are
        # reachable across seeds -- trying (a,b) then (b,a) and keeping
        # whichever came first halved the realised (r,h) pool to 14 distinct
        # questions over 60 seeds, short of the required 20.
        if candidate_r < _MIN_DIM or candidate_h < _MIN_DIM:
            continue
        if not _aspect_ok(candidate_r, candidate_h):
            continue
        v_candidate = _volume(candidate_r, candidate_h)
        # Volume cap, not a dimension cap: a Pythagorean triple can pair a
        # large leg with a small one (aspect ratio still inside the band)
        # while V ~ r^2*h still grows unboundedly with either dimension.
        # Found empirically that an unconstrained scale on the largest
        # triples can reach r=80, h=84, V~563000 cm^3, where 3.14159's
        # ~8e-7 relative error against true pi shifts the 1dp-rounded
        # volume enough that a calculator's built-in pi and the question's
        # stated 3.14159 round DIFFERENTLY (by 0.5, ten times the 0.05
        # grading tolerance) -- a real learner who used their calculator's
        # pi button would be marked wrong. The cap keeps magnitudes
        # comparable to easy/medium; `_pi_precision_safe` below is the
        # direct proof the specific disagreement cannot happen.
        if v_candidate > _MAX_CONE_VOLUME:
            continue
        if _boundary_margin(v_candidate, 1) <= _NOT_ON_BOUNDARY:
            continue
        if not _pi_precision_safe(candidate_r, candidate_h):
            continue
        r, h, l, v_exact = candidate_r, candidate_h, hyp_scaled, v_candidate
        found = True
        break
    check(found, "cone_volume/hard: no plausible, non-boundary Pythagorean-triple draw found in 1000 attempts")

    check(r**2 + h**2 == l**2, "cone_volume/hard: (r, h, l) is not a Pythagorean triple")

    # Independent re-derivation of h from l and r via sympy -- a different
    # route from however the triple assignment produced h.
    h_check = sympy_sqrt(l**2 - r**2)
    check_equivalent(h_check, h, "cone_volume/hard: sympy Pythagoras disagrees with the drawn h")

    v_check = _volume_check(r, h)
    check_equivalent(v_exact, v_check, "cone_volume/hard: independent volume cross-check disagrees")

    check(v_exact >= 0, "cone/circle measurement must not be negative")
    v_answer = round_half_up(v_exact, 1)
    answer_text = _fmt1(v_answer)
    answer = float(v_answer)

    question_text = (
        f"A cone has a radius of {r} cm and a slant height of {l} cm. The "
        f"radius, height and slant height form a right triangle. Using "
        f"π ≈ 3.14159, find the cone's volume, in cm³, to 1 decimal place."
    )

    bad_no_third = PI * Fraction(r) ** 2 * Fraction(h)
    bad_wrong_dim = PI * Fraction(h) ** 2 * Fraction(r) * Fraction(1, 3)
    # Not applicable: uses-diameter-as-radius has nothing to bite on here --
    # this tier gives the radius directly, never a diameter (honest
    # omission, same discipline the easy tier's own "not applicable" uses).
    # Added instead: the tier's own natural slip -- using the given slant
    # height as if it were the height.
    bad_uses_slant_as_height = PI * Fraction(r) ** 2 * Fraction(l) * Fraction(1, 3)

    check_distinct(v_exact, bad_no_third, msg="cone_volume/hard: omits-one-third-factor collides with the answer")
    check_distinct(v_exact, bad_wrong_dim, msg="cone_volume/hard: squares-wrong-dimension collides with the answer")
    check_distinct(
        v_exact, bad_uses_slant_as_height, msg="cone_volume/hard: uses-slant-height-as-height collides with the answer"
    )

    check(bad_no_third >= 0, "cone/circle measurement must not be negative")
    check(bad_wrong_dim >= 0, "cone/circle measurement must not be negative")
    check(bad_uses_slant_as_height >= 0, "cone/circle measurement must not be negative")
    distractors = pick_distractors(
        answer_text,
        [
            _fmt1(round_half_up(bad_no_third, 1)),
            _fmt1(round_half_up(bad_wrong_dim, 1)),
            _fmt1(round_half_up(bad_uses_slant_as_height, 1)),
        ],
        tolerance=_DISTRACTOR_TOLERANCE,
    )

    working = [
        "First find the height from the right triangle formed by the radius, height and slant height: h² = l² - r².",
        f"h² = {l}² - {r}² = {h * h}, so h = {h} cm.",
        "Volume = (1/3) × π × r² × h.",
        f"Volume = (1/3) × 3.14159 × {r}² × {h} = {answer_text} cm³ (to 1 decimal place).",
    ]
    return dict(
        text=question_text,
        latex=question_text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=working,
    )


@generator(
    "cone_volume",
    level=12,
    presets={
        "easy": dict(variant="radius_height"),
        "medium": dict(variant="diameter_height"),
        "hard": dict(variant="radius_slant_height"),
    },
)
def cone_volume(rng, variant="radius_height"):
    """V = (1/3) * pi * r**2 * h. Not covered by
    senior-use-formulas-to-find-the-volume-of-pyramids (square pyramid,
    rectangular pyramid and sphere only)."""
    if variant == "radius_height":
        return _cone_easy(rng)
    if variant == "diameter_height":
        return _cone_medium(rng)
    if variant == "radius_slant_height":
        return _cone_hard(rng)
    raise ValueError(f"unknown variant {variant!r}")
