"""Generates `applied_pythagoras` questions: Pythagoras word problems such as a ladder against a wall, a walking
displacement, an equilateral triangle's altitude and a box's space diagonal.

Applied_pythagoras.

``applied-pythagoras-problems`` (domain trigonometry, answer: number) --
Pythagoras word problems drawn from four real scenarios named in the
an exam-style item gap plan, rather than the bare "triangle has legs 5
and 12" recognition the existing ``pythagorean-hypotenuse``/
``pythagorean-leg`` skills only ever ask: a ladder against a wall, a
walking displacement inferred from directional language, the altitude of
an equilateral triangle, and the space diagonal of a rectangular solid.

Every question states its own required rounding ("to 1 decimal place") and
every canonical answer is displayed to exactly 1 decimal place with a
non-zero tenths digit -- see ``_bad_draw`` for the two reject-and-redraw
numeric traps this closes (a rounding-boundary landing, and a trailing-zero/
exact-integer landing).

**Distractor-collision checking deliberately does NOT use
``core.check_distinct``/``core.symbolically_equal``.** Those route through ``sympy.simplify(left - right) ==
0``, and sympy >=1.13 changed ``Float == Integer``/``Float == int`` to compare *type* as well as value
("exact" and "inexact" numbers are no longer ``==``), so ``simplify(18.4 - 18.4) == 0`` is ``False`` even
though the two values are identical (confirmed interactively: ``.is_zero``/``.equals(0)`` are both ``True``
where ``== 0`` is ``False``). Every candidate here is instead compared as a plain 1-decimal-place *string*
against ``answer_text``: two different 1dp values differ by at least 0.1, which is already more than
`GRADER_TOLERANCE` (~0.051), so a string match is both necessary and sufficient proof of a collision for this
skill's precision.
"""

from __future__ import annotations

import math

from ..core import GRADER_TOLERANCE, check, generator, pick_distractors

# Bounded retry budget for the reject-and-redraw loops below -- cheap but bounded so a bad range can
# never spin forever.
_MAX_ATTEMPTS = 200

#: Tight tolerance for the rounding-boundary check below: this proves a
#: genuine boundary landing, not floating-point noise (the same discipline
#: as ``geometry.py``'s ``cosine_rule`` branch, ``check(abs(sym - exact) <
#: 1e-6, ...)``).
_ROUNDING_BOUNDARY_TOL = 1e-6


def _bad_draw(v: float) -> bool:
    """True if a 1-decimal-place display of ``v`` would be an ambiguous or
    degenerate answer.

    Two traps, both reject-and-redraw, never
    post-hoc clamping:

    1. Rounding-boundary landing -- ``v``'s first decimal digit is not
       cleanly resolved (``v * 10`` sits within ``1e-6`` of an exact
       ``x.5`` boundary), i.e. a question whose last digit the learner's
       calculator decides.
    2. Trailing-zero / exact-integer landing -- ``round(v, 1)``'s tenths
       digit is exactly 0, i.e. the displayed answer would read like
       ``"12.0"`` instead of a genuine one-decimal value.
    """
    scaled = v * 10
    if abs(scaled - math.floor(scaled) - 0.5) < _ROUNDING_BOUNDARY_TOL:
        return True
    tenths = int(round(round(v, 1) * 10)) % 10
    return tenths == 0


def _fmt(v: float) -> str:
    """The learner-facing display of a value: exactly 1 decimal place."""
    return f"{round(v, 1):.1f}"


@generator(
    "applied_pythagoras",
    level=11,
    presets={
        "easy": dict(variant="ladder"),
        "medium": dict(variant="displacement"),
        "hard": dict(variant="hard_mixed"),
    },
)
def applied_pythagoras(rng, variant="ladder"):
    """Four scenarios, three tiers: `easy` hands you the triangle (`ladder`),
    `medium` makes you build the triangle from directional words
    (`displacement`), and `hard` makes you derive a value the triangle
    itself depends on before you can even set the triangle up
    (`equilateral_altitude` or `rect_solid_diagonal`, chosen uniformly at
    random from the still-seeded ``rng``)."""
    if variant == "hard_mixed":
        variant = rng.choice(["equilateral_altitude", "rect_solid_diagonal"])

    if variant == "ladder":
        return _ladder(rng)
    if variant == "displacement":
        return _displacement(rng)
    if variant == "equilateral_altitude":
        return _equilateral_altitude(rng)
    if variant == "rect_solid_diagonal":
        return _rect_solid_diagonal(rng)
    raise ValueError(f"unknown variant {variant!r}")


# ============================================================================
# easy: ladder
# ============================================================================


def _ladder(rng):
    """A right triangle given explicitly: ladder length (hypotenuse) and
    ONE of {height up the wall, distance from the wall} given; find the
    other.

    Constructibility: ladder length in [2.5, 9.0] m
    (a believable domestic/trade ladder range), the given leg strictly
    between 15% and 90% of the ladder length -- both bounds chosen so
    neither the found leg nor the implied foot angle is a physically silly
    sliver (no near-vertical-against-a-wall degenerate case, no leg so
    short the triangle is barely a triangle). This also guarantees ladder
    length > given leg (0.90 < 1), asserted explicitly below anyway since
    that invariant is load-bearing.
    """
    kind = rng.choice(["height", "distance"])
    for _ in range(_MAX_ATTEMPTS):
        ladder = round(rng.uniform(2.5, 9.0), 1)
        lo = ladder * 0.15
        hi = ladder * 0.90
        given = round(rng.uniform(lo, hi), 1)
        if not (lo < given < hi):
            continue
        other = math.sqrt(ladder**2 - given**2)
        if _bad_draw(other):
            continue
        answer_text = _fmt(other)
        # Misconceptions: added-not-subtracted (treats the given leg as if
        # it were also hypotenuse-independent and adds it); forgot-sqrt
        # (reports the un-rooted difference of squares). Reject-and-redraw
        # the WHOLE draw (not just the distractor list) if either
        # misconception's own answer happens to collide with the correct
        # one -- a colliding misconception is a question with two correct
        # options, not merely a thinner options list.
        added_val = math.sqrt(ladder**2 + given**2)
        forgot_sqrt_val = ladder**2 - given**2
        if _fmt(added_val) == answer_text or _fmt(forgot_sqrt_val) == answer_text:
            continue
        break
    else:
        check(False, "applied_pythagoras/ladder: no valid draw in 200 attempts")

    check(ladder > given, "applied_pythagoras/ladder: ladder length is not strictly greater than the given leg")

    if kind == "height":
        question_text = (
            f"A ladder of length {ladder} m leans against a vertical wall, "
            f"reaching {given} m up the wall. Find the horizontal distance "
            f"from the foot of the ladder to the base of the wall, to 1 "
            f"decimal place (in m)."
        )
    else:
        question_text = (
            f"A ladder of length {ladder} m leans against a vertical wall. "
            f"The foot of the ladder is {given} m from the base of the "
            f"wall. Find how high up the wall the ladder reaches, to 1 "
            f"decimal place (in m)."
        )

    # Independent re-verification: confirm the Pythagorean identity holds
    # for (given, other, ladder) directly, rather than trusting the
    # subtraction already used to derive `other`.
    check(
        abs((given**2 + other**2) - ladder**2) < 1e-6, "applied_pythagoras/ladder: Pythagorean identity does not hold"
    )

    # By-construction guard, checked unconditionally on every seed (see the
    # module docstring for why this is a plain string comparison and not
    # `core.check_distinct`): neither misconception distractor may equal
    # the answer.
    check(
        _fmt(added_val) != answer_text, "applied_pythagoras/ladder: added-not-subtracted distractor equals the answer"
    )
    check(_fmt(forgot_sqrt_val) != answer_text, "applied_pythagoras/ladder: forgot-sqrt distractor equals the answer")

    distractors = pick_distractors(
        answer_text,
        [_fmt(added_val), _fmt(forgot_sqrt_val)],
        tolerance=GRADER_TOLERANCE,
    )
    check(
        len(distractors) == 2,
        "applied_pythagoras/ladder: expected exactly 2 distractors after the by-construction collision guard",
    )

    working = [
        "Pythagoras: (ladder length)^2 = (given leg)^2 + (unknown leg)^2, "
        "so unknown leg = sqrt((ladder length)^2 - (given leg)^2).",
        f"unknown leg = sqrt({ladder}^2 - {given}^2) = {answer_text} m.",
    ]

    return dict(
        text=question_text,
        latex=question_text,
        answer=other,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=working,
    )


# ============================================================================
# medium: displacement
# ============================================================================


def _displacement(rng):
    """The right triangle must be inferred from directional language ("walks
    p km north, then q km east") rather than being handed to the learner
    pre-labelled as a triangle -- an extra reasoning step (translate words
    into legs) over `ladder`, even though the final arithmetic (given both
    legs, find the hypotenuse) is no harder in isolation. Always solves for
    the resultant straight-line distance.

    Constructibility: both legs strictly positive, within a believable
    walking-distance range (1-30 km) -- no zero-distance leg (degenerates
    to a 1-D walk, not a right triangle) and no absurd distance.
    """
    dir1 = rng.choice(["north", "south"])
    dir2 = rng.choice(["east", "west"])
    for _ in range(_MAX_ATTEMPTS):
        p = round(rng.uniform(1.0, 30.0), 1)
        q = round(rng.uniform(1.0, 30.0), 1)
        distance = math.sqrt(p**2 + q**2)
        if _bad_draw(distance):
            continue
        answer_text = _fmt(distance)
        # Misconceptions: linear-sum (p + q instead of the hypotenuse);
        # one-leg-only (reports just one walked distance as the resultant).
        # Reject-and-redraw the whole draw if either collides with the
        # answer (see `_ladder`'s comment: found live at seed 24 when one
        # leg was tiny relative to the other, making sqrt(p^2+q^2) ~= p).
        linear_sum = p + q
        one_leg_only = p
        if _fmt(linear_sum) == answer_text or _fmt(one_leg_only) == answer_text:
            continue
        break
    else:
        check(False, "applied_pythagoras/displacement: no valid draw in 200 attempts")

    check(p > 0 and q > 0, "applied_pythagoras/displacement: a walked leg is not strictly positive")

    question_text = (
        f"A hiker walks {p} km {dir1}, then {q} km {dir2}. Find the "
        f"straight-line distance from the starting point to the hiker's "
        f"current position, to 1 decimal place (in km)."
    )

    # Independent re-verification: confirm the Pythagorean identity holds
    # for (p, q, distance) directly, rather than trusting the sqrt already
    # used to derive `distance`.
    check(
        abs((p**2 + q**2) - distance**2) < 1e-6, "applied_pythagoras/displacement: Pythagorean identity does not hold"
    )

    check(_fmt(linear_sum) != answer_text, "applied_pythagoras/displacement: linear-sum distractor equals the answer")
    check(
        _fmt(one_leg_only) != answer_text, "applied_pythagoras/displacement: one-leg-only distractor equals the answer"
    )

    distractors = pick_distractors(
        answer_text,
        [_fmt(linear_sum), _fmt(one_leg_only)],
        tolerance=GRADER_TOLERANCE,
    )
    check(
        len(distractors) == 2,
        "applied_pythagoras/displacement: expected exactly 2 distractors after the by-construction collision guard",
    )

    working = [
        f"The {dir1}/{dir2} legs of the walk are perpendicular, so the "
        f"resultant straight-line distance is the hypotenuse of a right "
        f"triangle with those two legs.",
        f"distance = sqrt({p}^2 + {q}^2) = {answer_text} km.",
    ]

    return dict(
        text=question_text,
        latex=question_text,
        answer=distance,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=working,
    )


# ============================================================================
# hard: equilateral_altitude
# ============================================================================


def _equilateral_altitude(rng):
    """An extra *geometric* derivation step before Pythagoras even applies:
    halve the base to find the foot of the altitude (the altitude of an
    equilateral triangle bisects the opposite side at a right angle), THEN
    apply Pythagoras on the resulting half-triangle.

    Constructibility: side length strictly positive, within a believable
    range (3-25 units) -- every side length trivially produces a real
    equilateral triangle, so the only "constructibility" content here is
    the sensible-magnitude bound.
    """
    for _ in range(_MAX_ATTEMPTS):
        side = round(rng.uniform(3.0, 25.0), 1)
        half_base = side / 2
        altitude = math.sqrt(side**2 - half_base**2)
        if _bad_draw(altitude):
            continue
        answer_text = _fmt(altitude)
        # Misconceptions: area-formula-confusion (computes the area
        # formula side^2 * sqrt(3) / 4 instead of the altitude formula
        # side * sqrt(3) / 2); stopped-at-half-base (reports the half-base
        # used partway through the correct method). Reject-and-redraw the
        # whole draw if either collides with the answer.
        area_formula_confusion = (side**2) * math.sqrt(3) / 4
        stopped_at_half_base = half_base
        if _fmt(area_formula_confusion) == answer_text or _fmt(stopped_at_half_base) == answer_text:
            continue
        break
    else:
        check(False, "applied_pythagoras/equilateral_altitude: no valid draw in 200 attempts")

    check(side > 0, "applied_pythagoras/equilateral_altitude: side length is not strictly positive")

    # Independent re-verification: the closed-form altitude = side *
    # sqrt(3) / 2 must agree with the halve-the-base-then-Pythagoras route
    # used to construct `altitude`, within tight tolerance (the same
    # verify-then-round shape as geometry.py's cosine_rule).
    closed_form = side * math.sqrt(3) / 2
    check(
        abs(closed_form - altitude) < 1e-6,
        "applied_pythagoras/equilateral_altitude: closed-form and halve-the-base routes disagree",
    )

    question_text = (
        f"An equilateral triangle has side length {side} units. Find the "
        f"length of the altitude from one vertex to the opposite side, to "
        f"1 decimal place (in units)."
    )

    check(
        _fmt(area_formula_confusion) != answer_text,
        "applied_pythagoras/equilateral_altitude: area-formula-confusion distractor equals the answer",
    )
    check(
        _fmt(stopped_at_half_base) != answer_text,
        "applied_pythagoras/equilateral_altitude: stopped-at-half-base distractor equals the answer",
    )

    distractors = pick_distractors(
        answer_text,
        [_fmt(area_formula_confusion), _fmt(stopped_at_half_base)],
        tolerance=GRADER_TOLERANCE,
    )
    check(
        len(distractors) == 2,
        "applied_pythagoras/equilateral_altitude: expected exactly 2 distractors after the by-construction collision guard",
    )

    working = [
        f"The altitude of an equilateral triangle bisects the opposite "
        f"side at a right angle: half-base = {side} / 2 = {_fmt(half_base)}.",
        f"Pythagoras on the half-triangle: altitude = sqrt({side}^2 - {_fmt(half_base)}^2) = {answer_text} units.",
    ]

    return dict(
        text=question_text,
        latex=question_text,
        answer=altitude,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=working,
    )


# ============================================================================
# hard: rect_solid_diagonal
# ============================================================================


def _rect_solid_diagonal(rng):
    """An extra *geometric* derivation step before Pythagoras even applies:
    extend from a 2-D face diagonal to the 3-D space diagonal. Implemented
    as the direct 3-D formula d = sqrt(l^2 + w^2 + h^2), independently
    re-verified via the two-stage route (face diagonal sqrt(l^2+w^2), then
    sqrt(that^2 + h^2)).

    Constructibility: length/width/height all strictly positive, within a
    believable range (2-20 units) -- always geometrically valid for any
    positive dimensions, so the bound is the sensible-magnitude
    requirement.
    """
    for _ in range(_MAX_ATTEMPTS):
        length = round(rng.uniform(2.0, 20.0), 1)
        width = round(rng.uniform(2.0, 20.0), 1)
        height = round(rng.uniform(2.0, 20.0), 1)
        diagonal = math.sqrt(length**2 + width**2 + height**2)
        if _bad_draw(diagonal):
            continue
        answer_text = _fmt(diagonal)
        # Misconceptions: forgot-third-dimension (reports the 2-D face
        # diagonal, ignoring height entirely); summed-edges (l + w + h).
        # Reject-and-redraw the whole draw if either collides with the
        # answer -- the face-diagonal gap can be as small as ~h^2/(2*face)
        # when h is at its minimum and l/w are both at their maximum, so
        # this guard earns its keep here too, not just on `displacement`.
        face_diagonal = math.sqrt(length**2 + width**2)
        summed_edges = length + width + height
        if _fmt(face_diagonal) == answer_text or _fmt(summed_edges) == answer_text:
            continue
        break
    else:
        check(False, "applied_pythagoras/rect_solid_diagonal: no valid draw in 200 attempts")

    check(
        length > 0 and width > 0 and height > 0,
        "applied_pythagoras/rect_solid_diagonal: a dimension is not strictly positive",
    )

    # Independent re-verification: the two-stage route (face diagonal, then
    # extend to the space diagonal) must agree with the direct 3-D formula
    # used to construct `diagonal`, within tight tolerance -- if they do
    # not, that is a real bug, not tolerance noise (matching geometry.py's cosine_rule check(abs(sym - exact) < 1e-6,
    # ...) pattern).
    two_stage = math.sqrt(face_diagonal**2 + height**2)
    check(
        abs(two_stage - diagonal) < 1e-6, "applied_pythagoras/rect_solid_diagonal: two-stage and direct routes disagree"
    )

    question_text = (
        f"A rectangular box has length {length} m, width {width} m and "
        f"height {height} m. Find the length of the space diagonal (the "
        f"longest straight line between two opposite corners of the box), "
        f"to 1 decimal place (in m)."
    )

    check(
        _fmt(face_diagonal) != answer_text,
        "applied_pythagoras/rect_solid_diagonal: forgot-third-dimension distractor equals the answer",
    )
    check(
        _fmt(summed_edges) != answer_text,
        "applied_pythagoras/rect_solid_diagonal: summed-edges distractor equals the answer",
    )

    distractors = pick_distractors(
        answer_text,
        [_fmt(face_diagonal), _fmt(summed_edges)],
        tolerance=GRADER_TOLERANCE,
    )
    check(
        len(distractors) == 2,
        "applied_pythagoras/rect_solid_diagonal: expected exactly 2 distractors after the by-construction collision guard",
    )

    working = [
        f"Face diagonal: sqrt({length}^2 + {width}^2) = {_fmt(face_diagonal)}.",
        f"Space diagonal: sqrt({_fmt(face_diagonal)}^2 + {height}^2) = {answer_text} m.",
    ]

    return dict(
        text=question_text,
        latex=question_text,
        answer=diagonal,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
        working=working,
    )
