"""Coordinate-geometry pilot generators (AC9M9A03): exact midpoint and exact
Euclidean distance between two Cartesian points.

Both reuse foundations already active in the catalogue (``coordinate-plane-readiness``,
``integer-arithmetic``, ``pythagorean-hypotenuse``) and follow the same shape as ``acara_gap_build.py``:
seed-deterministic sampling, an independent SymPy check of the derived answer, worked steps, and
misconception-based distractors.
"""

from __future__ import annotations

import re

from sympy import Rational, simplify, sqrt

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    latex_answer,
    paren_negative,
    phrase,
    pick_distractors,
    rational_text,
)

# A handful of primitive Pythagorean triples used to deliberately mix
# rational (integer) distances in among the surd distances a generic random
# (dx, dy) pair produces. Each preset only offers the triples whose legs fit
# inside its own leg_hi, so "easy" only ever gets (3, 4, 5) while "hard" also
# reaches the larger ones.
_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (9, 40, 41)]


def _verify_midpoint(x1, y1, x2, y2, mx, my):
    """Independent check, two routes, neither of which re-derives the
    midpoint the same way it was constructed (a plain coordinate average):
    (1) equidistance from both endpoints via the distance formula -- goes
    through squares and sums of squares instead of an average; (2) the
    endpoint-sum identity -- catches a construction bug (wrong axis, wrong
    sign, an off-by-factor) that (1) alone would miss, since equidistance is
    satisfied by every point on the perpendicular bisector, not only the
    midpoint."""
    d1_sq = (mx - x1) ** 2 + (my - y1) ** 2
    d2_sq = (mx - x2) ** 2 + (my - y2) ** 2
    check_equivalent(d1_sq, d2_sq, "midpoint is not equidistant from both endpoints")
    check(2 * mx == x1 + x2 and 2 * my == y1 + y2, "endpoint sum does not equal twice the midpoint")


def _verify_distance(dx, dy, distance):
    """Independent check: square the claimed distance and compare with
    dx*dx + dy*dy computed directly from the coordinate differences -- a
    different route from constructing the answer as sqrt(dx**2 + dy**2) in
    the first place, since it never takes a square root at all."""
    check_equivalent(distance**2, dx * dx + dy * dy, "distance squared does not equal dx^2 + dy^2")
    check(distance >= 0, "distance must be non-negative")


def _draw_leg_pair(rng, leg_hi, allow_negative):
    """One (dx, dy) displacement: a deliberate 50/50 split between a Pythagorean-triple leg pair (a rational,
    integer distance) and a generic random leg pair (almost always an irrational surd, since dx**2 + dy**2 is
    a perfect square only by coincidence). Factored out of `distance_two_points`'s `kind="standard"` body so
    `kind="perimeter"` can draw two independent displacements with the IDENTICAL distribution -- called from
    the exact point in the rng stream the inline code used to occupy, so `kind="standard"`'s output is
    unchanged for the same seed."""
    usable_triples = [t for t in _TRIPLES if t[0] <= leg_hi and t[1] <= leg_hi]
    use_triple = bool(usable_triples) and rng.random() < 0.5
    if use_triple:
        a, b, _c = rng.choice(usable_triples)
        dx, dy = (a, b) if rng.random() < 0.5 else (b, a)
    else:
        dx = rng.randint(1, leg_hi)
        dy = rng.randint(1, leg_hi)

    if allow_negative:
        if rng.random() < 0.5:
            dx = -dx
        if rng.random() < 0.5:
            dy = -dy
    return dx, dy


@generator(
    "midpoint_two_points",
    level=8,
    presets={
        "easy": dict(coord_lo=0, coord_hi=6),
        "medium": dict(coord_lo=-8, coord_hi=8),
        "hard": dict(coord_lo=-14, coord_hi=14),
    },
)
def midpoint_two_points(rng, coord_lo=-8, coord_hi=8, kind="standard"):
    """Calculate the exact midpoint of two supplied Cartesian points.

    `kind="find_endpoint"` is the inverse problem: given ONE
    endpoint and the midpoint, find the OTHER endpoint. `kind="standard"`
    (the default, unchanged) is this function's own original body below.
    """
    if kind == "find_endpoint":
        return _midpoint_two_points_find_endpoint(rng, coord_lo, coord_hi)
    if coord_hi <= coord_lo:
        raise ValueError("coord_hi must be greater than coord_lo")

    x1 = rng.randint(coord_lo, coord_hi)
    y1 = rng.randint(coord_lo, coord_hi)
    x2 = rng.randint(coord_lo, coord_hi)
    y2 = rng.randint(coord_lo, coord_hi)
    # The two points must not coincide -- reroll the second point until it
    # differs. coord_hi > coord_lo guarantees at least two distinct integer
    # values exist, so this terminates.
    while (x1, y1) == (x2, y2):
        x2 = rng.randint(coord_lo, coord_hi)
        y2 = rng.randint(coord_lo, coord_hi)

    # No parity correction: x1+x2 and y1+y2 are each odd about half the time
    # under uniform sampling, so half-integer midpoints occur naturally and
    # are never filtered out, alongside plain integer ones.
    mx = Rational(x1 + x2, 2)
    my = Rational(y1 + y2, 2)
    _verify_midpoint(x1, y1, x2, y2, mx, my)

    answer_text = f"x = {rational_text(mx)}, y = {rational_text(my)}"

    # Misconception distractors:
    #  - subtract-instead-of-average: subtracts the coordinates instead of
    #    averaging them.
    #  - halve-one-coordinate: halves one coordinate but forgets to halve the
    #    other (offered on both axes for collision resistance -- both encode
    #    the same misconception).
    #  - omits-division: sums the coordinates and stops, never dividing by 2.
    sub_x, sub_y = Rational(x1 - x2, 2), Rational(y1 - y2, 2)
    halve_x_only = f"x = {rational_text(mx)}, y = {y1 + y2}"
    halve_y_only = f"x = {x1 + x2}, y = {rational_text(my)}"
    candidates = [
        f"x = {rational_text(sub_x)}, y = {rational_text(sub_y)}",
        halve_x_only,
        halve_y_only,
        f"x = {x1 + x2}, y = {y1 + y2}",
    ]

    return dict(
        text=(f"Calculate the exact midpoint of the points ({x1}, {y1}) and ({x2}, {y2})."),
        latex=rf"\text{{Exact midpoint of }}({x1},{y1})\text{{ and }}({x2},{y2})",
        # float(), not the SymPy Rational used internally for the exact
        # working/verification -- every value here has denominator 1 or 2,
        # which a binary float represents exactly (no rounding), so this
        # keeps Question.answer plain-JSON-serialisable (per the Question
        # dataclass's documented contract) without losing any precision.
        answer={"x": float(mx), "y": float(my)},
        answer_text=answer_text,
        answer_latex=rf"\left(\frac{{{x1}+{x2}}}{{2}}, \frac{{{y1}+{y2}}}{{2}}\right) "
        rf"= \left({rational_text(mx)}, {rational_text(my)}\right)",
        working=[
            f"Midpoint = ((x₁+x₂)/2, (y₁+y₂)/2) = (({x1} + {paren_negative(x2)})/2, ({y1} + {paren_negative(y2)})/2).",
            f"Midpoint = ({rational_text(mx)}, {rational_text(my)}).",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


# Wording variants for `kind="find_endpoint"`, all
# sharing the same slot set so `phrase()` can pick any one.
_FIND_ENDPOINT_TEMPLATES = [
    "The midpoint of A({x1}, {y1}) and B is M({mx}, {my}). Find the coordinates of B.",
    "M({mx}, {my}) is the midpoint of the segment from A({x1}, {y1}) to B. Find B.",
    "A({x1}, {y1}) and B are two points whose midpoint is M({mx}, {my}). Find the coordinates of B.",
    "The segment AB has midpoint M({mx}, {my}). Given A = ({x1}, {y1}), find B.",
]


def _midpoint_two_points_find_endpoint(rng, coord_lo, coord_hi):
    """`kind="find_endpoint"`: given ONE endpoint A and the
    MIDPOINT M, find the OTHER endpoint B = 2M - A. This is the INVERSE
    problem (an unknown coordinate on "the other side" of the relationship,
    not a bigger version of the same forward computation): construct
    backwards exactly like every other generator in this file -- draw A and
    B first (the SAME range and no-coincidence rules as `kind="standard"`,
    including its "no half-integer filtering" comment: A and B are always
    integer-coordinate points, only M may land on a half-integer), compute M
    from them, then present A and M as the GIVEN quantities and B as the
    unknown.
    """
    if coord_hi <= coord_lo:
        raise ValueError("coord_hi must be greater than coord_lo")

    x1 = rng.randint(coord_lo, coord_hi)
    y1 = rng.randint(coord_lo, coord_hi)
    x2 = rng.randint(coord_lo, coord_hi)
    y2 = rng.randint(coord_lo, coord_hi)
    # A and B must not coincide -- reroll B until it differs, the same rule
    # `kind="standard"` uses for its own two points.
    while (x1, y1) == (x2, y2):
        x2 = rng.randint(coord_lo, coord_hi)
        y2 = rng.randint(coord_lo, coord_hi)

    mx = Rational(x1 + x2, 2)
    my = Rational(y1 + y2, 2)
    # First independent check: `_verify_midpoint` proves the three-point relationship symmetrically
    # (equidistance + endpoint-sum identity), so it is exactly the right check for either direction
    # -- reused unchanged
    _verify_midpoint(x1, y1, x2, y2, mx, my)
    # Second, explicit re-derivation via the algebraic identity B = 2M - A,
    # computed fresh from mx/my/x1/y1 -- NOT the x2/y2 this branch already
    # drew -- so a transcription slip inside THIS branch (using the wrong
    # variable, swapping an axis) is caught rather than confirmed by its own
    # construction. Mirrors `_distance_two_points_perimeter`'s own
    # cross-check of its derived third side against the actual points.
    bx_check = 2 * mx - x1
    by_check = 2 * my - y1
    check(bx_check == x2 and by_check == y2, "2*M - A does not equal the other endpoint")

    answer_text = f"x = {x2}, y = {y2}"

    # Misconception distractors:
    #  - averages-known-point-and-midpoint: computes the midpoint of A and M
    #    instead of reflecting M through A -- a very common "averaged the
    #    two given points" slip when the operation itself (find the OTHER
    #    endpoint) is unfamiliar and the two given quantities look just like
    #    the forward problem's two given points.
    #  - subtracted-instead-of-doubling: B = M - A instead of B = 2M - A --
    #    forgets to double the midpoint before subtracting.
    #  - reported-known-point-back: reports A, or M, back as if it were the
    #    answer -- swapped which point is being solved for.
    avg_x, avg_y = (x1 + mx) / 2, (y1 + my) / 2
    sub_x, sub_y = mx - x1, my - y1
    # check_distinct covers the three candidates PROVABLY always distinct
    # from B = 2M - A given A != B (enforced by the reroll above):
    # avg == B and (x1, y1) == B both algebraically reduce to A == B, and
    # M == B reduces to A == M which in turn reduces to A == B. `sub` is
    # deliberately excluded from this unconditional check: sub = M - A
    # equals B = 2M - A exactly when M = (0, 0) -- a legitimate draw, not a
    # bug -- so that one coincidence is left to `pick_distractors`'s own
    # per-question dedup below rather than hard-failing generation.
    check_distinct(
        (x2, y2),
        (avg_x, avg_y),
        (x1, y1),
        (mx, my),
        msg="a find_endpoint misconception distractor equals the correct endpoint",
    )
    candidates = [
        f"x = {rational_text(avg_x)}, y = {rational_text(avg_y)}",
        f"x = {rational_text(sub_x)}, y = {rational_text(sub_y)}",
        f"x = {x1}, y = {y1}",
        f"x = {rational_text(mx)}, y = {rational_text(my)}",
    ]

    mx_text, my_text = rational_text(mx), rational_text(my)
    text = phrase(rng, _FIND_ENDPOINT_TEMPLATES, x1=x1, y1=y1, mx=mx_text, my=my_text)

    return dict(
        text=text,
        latex=rf"\text{{Midpoint }}M({mx_text},{my_text})\text{{ of }}A({x1},{y1})"
        rf"\text{{ and }}B\text{{ -- find }}B",
        # float(), matching `kind="standard"`'s own float-safety comment --
        # x2, y2 are always plain integers here (drawn directly), so this is
        # exact, no rounding.
        answer={"x": float(x2), "y": float(y2)},
        answer_text=answer_text,
        answer_latex=rf"B = (2M_x - x_1, 2M_y - y_1) = (2({mx_text}) - {paren_negative(x1)}, "
        rf"2({my_text}) - {paren_negative(y1)}) = ({x2}, {y2})",
        working=[
            f"B = (2Mx − x₁, 2My − y₁) = (2({mx_text}) − {paren_negative(x1)}, 2({my_text}) − {paren_negative(y1)}).",
            f"B = ({x2}, {y2}).",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


@generator(
    "distance_two_points",
    level=8,
    presets={
        "easy": dict(coord_hi=6, leg_hi=6, allow_negative=False),
        "medium": dict(coord_hi=10, leg_hi=12, allow_negative=True),
        "hard": dict(coord_hi=16, leg_hi=24, allow_negative=True),
    },
)
def distance_two_points(rng, coord_hi=10, leg_hi=12, allow_negative=True, kind="standard"):
    """Calculate the exact Euclidean distance between two supplied
    Cartesian points."""
    if kind == "perimeter":
        return _distance_two_points_perimeter(rng, coord_hi, leg_hi, allow_negative)
    if coord_hi < 1 or leg_hi < 1:
        raise ValueError("coord_hi and leg_hi must be at least 1")

    lo = -coord_hi if allow_negative else 0
    x1 = rng.randint(lo, coord_hi)
    y1 = rng.randint(lo, coord_hi)

    # Deliberate 50/50 split between a Pythagorean-triple leg pair (a rational, integer distance)
    # and a generic random leg pair (almost always an irrational surd, since dx**2 + dy**2 is a
    # perfect square only by coincidence). `_draw_leg_pair` is the exact same code that used to be
    # inline here.
    dx, dy = _draw_leg_pair(rng, leg_hi, allow_negative)

    x2, y2 = x1 + dx, y1 + dy
    distance = sqrt(dx * dx + dy * dy)
    _verify_distance(dx, dy, distance)

    answer_text = str(distance)

    # Misconception distractors, in priority order (pick_distractors keeps
    # the first 3 that survive deduping against the answer and each other --
    # forgot-square-root and summed-differences coincide exactly when
    # |dx| = |dy| = 1, which is when the 4th, ignores-one-dimension, steps in
    # to keep at least two distinct distractors):
    #  - forgot-square-root: reports dx^2 + dy^2 without ever taking the
    #    square root.
    #  - summed-differences: adds the horizontal and vertical differences
    #    (Manhattan distance) instead of applying Pythagoras.
    #  - ignores-one-dimension: uses only one coordinate's difference,
    #    ignoring the other axis entirely.
    #  - sqrt-without-squaring: remembers the square root but forgets to
    #    square dx and dy first, so takes sqrt(dx + dy) instead.
    dx_abs, dy_abs = abs(dx), abs(dy)
    candidates = [
        str(dx * dx + dy * dy),
        str(dx_abs + dy_abs),
        str(dx_abs),
        str(sqrt(dx_abs + dy_abs)),
    ]

    dx_diff_text = join_terms([(x2, ""), (-x1, "")])
    dy_diff_text = join_terms([(y2, ""), (-y1, "")])
    # Display-only rendering of the answer for the worked-solution line:
    # str(distance) is SymPy's default Mul notation ("6*sqrt(5)", "sqrt(5)"
    # or a plain integer for the rational-triple branch), fine for
    # answer/answer_text/latex (which the grader and renderer's latex path
    # consume) but a bare "*" and "sqrt(...)" read as prose is the wrong
    # notation for a line the learner reads. Rewrite "N*sqrt(R)"/"sqrt(R)" as
    # "N√R"/"√R" here only, matching the √ notation the line already uses
    # twice; the value is unchanged.
    answer_display = re.sub(r"(\d*)\*?sqrt\((\d+)\)", r"\1√\2", answer_text)
    return dict(
        text=(f"Calculate the exact distance between the points ({x1}, {y1}) and ({x2}, {y2})."),
        latex=rf"\text{{Exact distance from }}({x1},{y1})\text{{ to }}({x2},{y2})",
        # str(), matching the existing expression-kind generators in
        # mathgen/generators/calculus.py -- grade_expression sympifies this
        # back, and it keeps Question.answer plain-JSON-serialisable rather
        # than relying on Question.to_json()'s default=str fallback.
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=rf"\sqrt{{({x2}-{x1})^2+({y2}-{y1})^2}} = {latex_answer(answer_text)}",
        working=[
            f"d = √((x₂-x₁)² + (y₂-y₁)²) = √(({dx_diff_text})² + ({dy_diff_text})²).",
            # dx and dy are squared, so their magnitude alone (not sign)
            # determines the result -- shown here as |dx| and |dy| to avoid
            # a confusing "+ -4²" on the page.
            f"d = √({dx_abs}² + {dy_abs}²) = √({dx * dx} + {dy * dy}) = {answer_display}.",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


# The `kind="perimeter"` escalation over the single-pair distance
# above: three points P, Q, R -- find the perimeter of triangle PQR, the sum
# of the three pairwise distances |PQ| + |QR| + |RP|. Four alternative
# wordings (at least 3 distinct rendered shapes per 60 seeds for the hard
# tier alone), all sharing the same slot set so `phrase()` can pick any one.
_PERIMETER_TEMPLATES = [
    "Calculate the exact perimeter of the triangle with vertices P({px}, {py}), Q({qx}, {qy}) and R({rx}, {ry}).",
    "Find the exact perimeter of triangle PQR, where P = ({px}, {py}), Q = ({qx}, {qy}) and R = ({rx}, {ry}).",
    "A triangle has vertices P({px}, {py}), Q({qx}, {qy}) and R({rx}, {ry}). Calculate its exact perimeter.",
    "Triangle PQR has vertices P({px}, {py}), Q({qx}, {qy}) and R({rx}, {ry}). Find its exact perimeter.",
]


def _distance_two_points_perimeter(rng, coord_hi, leg_hi, allow_negative):
    """`kind="perimeter"`: three points P, Q, R -- draw P, then two
    INDEPENDENT displacements (dx1, dy1) and (dx2, dy2) with
    `_draw_leg_pair`'s own triple/generic-leg-pair 50/50 split, giving
    Q = P + (dx1, dy1) and R = Q + (dx2, dy2). The third side's
    displacement, P->R, is DERIVED as (dx1+dx2, dy1+dy2) -- never
    independently drawn -- so it lands wherever that sum happens to be,
    rational or surd, and is verified the same way regardless. A genuine
    escalation over the single-pair kind: three independent distance
    computations (each checked the same way `_verify_distance` always has,
    reused once per side) combined into one exact total, not one bigger
    computation.

    Degeneracy guard: reroll the second displacement while it is a scalar
    multiple of the first (`dx1*dy2 == dy1*dx2`), i.e. P, Q, R collinear --
    this also catches the special case R == P (dx2 = -dx1, dy2 = -dy1, the
    cross product is zero) without a second condition. `_draw_leg_pair`
    never returns (0, 0) on either axis, so this always terminates.
    """
    if coord_hi < 1 or leg_hi < 1:
        raise ValueError("coord_hi and leg_hi must be at least 1")

    lo = -coord_hi if allow_negative else 0
    px = rng.randint(lo, coord_hi)
    py = rng.randint(lo, coord_hi)

    dx1, dy1 = _draw_leg_pair(rng, leg_hi, allow_negative)
    dx2, dy2 = _draw_leg_pair(rng, leg_hi, allow_negative)
    while dx1 * dy2 == dy1 * dx2:
        dx2, dy2 = _draw_leg_pair(rng, leg_hi, allow_negative)

    qx, qy = px + dx1, py + dy1
    rx, ry = qx + dx2, qy + dy2
    dx3, dy3 = dx1 + dx2, dy1 + dy2  # P->R, derived, not independently drawn

    d_pq = sqrt(dx1 * dx1 + dy1 * dy1)
    d_qr = sqrt(dx2 * dx2 + dy2 * dy2)
    d_rp = sqrt(dx3 * dx3 + dy3 * dy3)
    _verify_distance(dx1, dy1, d_pq)
    _verify_distance(dx2, dy2, d_qr)
    _verify_distance(dx3, dy3, d_rp)
    # `_verify_distance(dx3, dy3, d_rp)` above only checks d_rp against
    # WHATEVER dx3/dy3 it is handed -- it cannot see a slip in the
    # derivation dx3 = dx1 + dx2 itself. Close that gap by cross-checking
    # dx3/dy3 against the actual constructed points (mirrors
    # `_verify_midpoint`'s own second, differently-routed check).
    check(dx3 == rx - px and dy3 == ry - py, "third side's displacement does not match R - P")

    perimeter = simplify(d_pq + d_qr + d_rp)

    # Independent check: recompute the same total a second way, summing the
    # three _verify_distance-checked distances again directly from the
    # displacement vectors -- never by re-adding the already-built
    # answer_text -- and compare the total by value.
    recomputed = sqrt(dx1 * dx1 + dy1 * dy1) + sqrt(dx2 * dx2 + dy2 * dy2) + sqrt(dx3 * dx3 + dy3 * dy3)
    check_equivalent(perimeter, recomputed, "perimeter does not equal the independently re-summed side lengths")
    check(perimeter > 0, "perimeter must be positive")

    answer_text = str(perimeter)

    # Misconception distractors:
    #  - missed-a-side: sums only |PQ| + |QR|, forgetting to close the
    #    triangle back to P -- a "forgot a side" slip a 2-point distance
    #    question never offers.
    #  - one side's Manhattan distance (|dx1| + |dy1| for |PQ| only, keeping
    #    |QR| and |RP| exact) instead of Euclidean -- the existing
    #    `summed-differences` misconception's own idea, scaled up to one
    #    side of three.
    #  - the three SQUARED distances summed instead of the three square
    #    roots -- the existing `forgot-square-root` misconception's own
    #    idea, scaled up to three sides.
    missed_a_side = simplify(d_pq + d_qr)
    manhattan_one_side = simplify((abs(dx1) + abs(dy1)) + d_qr + d_rp)
    summed_squares = (dx1 * dx1 + dy1 * dy1) + (dx2 * dx2 + dy2 * dy2) + (dx3 * dx3 + dy3 * dy3)
    check_distinct(
        perimeter,
        missed_a_side,
        manhattan_one_side,
        summed_squares,
        msg="a perimeter misconception distractor equals the correct perimeter",
    )
    candidates = [str(missed_a_side), str(manhattan_one_side), str(summed_squares)]

    dx1_abs, dy1_abs = abs(dx1), abs(dy1)
    dx2_abs, dy2_abs = abs(dx2), abs(dy2)
    dx3_abs, dy3_abs = abs(dx3), abs(dy3)
    # Display-only √ notation for the worked-solution lines -- see
    # `distance_two_points`'s own identical comment; the value is unchanged.
    pq_display = re.sub(r"(\d*)\*?sqrt\((\d+)\)", r"\1√\2", str(d_pq))
    qr_display = re.sub(r"(\d*)\*?sqrt\((\d+)\)", r"\1√\2", str(d_qr))
    rp_display = re.sub(r"(\d*)\*?sqrt\((\d+)\)", r"\1√\2", str(d_rp))
    answer_display = re.sub(r"(\d*)\*?sqrt\((\d+)\)", r"\1√\2", answer_text)

    text = phrase(rng, _PERIMETER_TEMPLATES, px=px, py=py, qx=qx, qy=qy, rx=rx, ry=ry)

    return dict(
        text=text,
        latex=rf"\text{{Exact perimeter of the triangle }}P({px},{py})Q({qx},{qy})R({rx},{ry})",
        # str(), matching `distance_two_points`'s own kind="standard" answer
        # -- grade_expression sympifies this back.
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=(
            rf"|PQ|+|QR|+|RP| = \sqrt{{({qx}-{px})^2+({qy}-{py})^2}}"
            rf"+\sqrt{{({rx}-{qx})^2+({ry}-{qy})^2}}"
            rf"+\sqrt{{({px}-{rx})^2+({py}-{ry})^2}} = {latex_answer(answer_text)}"
        ),
        working=[
            f"|PQ| = √({dx1_abs}² + {dy1_abs}²) = √({dx1 * dx1} + {dy1 * dy1}) = {pq_display}.",
            f"|QR| = √({dx2_abs}² + {dy2_abs}²) = √({dx2 * dx2} + {dy2 * dy2}) = {qr_display}.",
            f"|RP| = √({dx3_abs}² + {dy3_abs}²) = √({dx3 * dx3} + {dy3 * dy3}) = {rp_display}.",
            f"Perimeter = |PQ| + |QR| + |RP| = {pq_display} + {qr_display} + {rp_display} = {answer_display}.",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )
