"""Generates `proof_circle_theorems` items: find unknown angles and lengths with the circle theorems, and prove
further results from them. The configuration is described in words (points on a circle with centre O), so no
figure is needed.

Every question names the theorems the learner may use and asks for a justified solution or a proof. The work is
marked against a rubric, so `answer` is a marking dict (see `_free_proof_kit`), not a value.

Ladder:

- easy: one theorem applied to find an angle (angle at the centre, angle in a semicircle, tangent and radius).
- medium: one theorem with a second step (cyclic quadrilateral, the alternate segment, equal tangents).
- hard: prove a standard result from an earlier one (angle at the centre, same segment, cyclic quadrilateral,
  intersecting chords, equal chords).
- exam: prove the tangent-chord theorem, the tangent-secant and chord-product results, an angle bisection in a
  cyclic quadrilateral, or a numeric problem that needs three theorems.

Every configuration is built from coordinates on a circle and every statement is checked on that concrete
figure before the question is returned, so no false statement can be asked.
"""

from __future__ import annotations

import math

from ..core import check, generator
from ._free_proof_kit import Prompt, TAILS, attempts, finish, marking

LETTERS = "ABCDEFGHKLMNPQRSTUVWXYZ"


def _tail(rng) -> str:
    return rng.choice(TAILS)


def _labels(rng, k: int) -> list[str]:
    return rng.sample(LETTERS, k)


def _pol(deg: float) -> tuple[float, float]:
    r = math.radians(deg)
    return (math.cos(r), math.sin(r))


def _d(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _ang(a, b, c) -> float:
    """The angle at b in degrees."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    cosv = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


def _eq(x: float, y: float, tol: float = 1e-6) -> bool:
    return abs(x - y) < tol


O = (0.0, 0.0)


def _cyclic_arcs(rng):
    """Four arcs (degrees, even, summing to 360) for a cyclic quadrilateral with points in order A, B, C, D."""
    for _ in attempts(500, "four arcs for a cyclic quadrilateral"):
        cuts = sorted(rng.sample(range(10, 175), 3))
        arcs = [2 * cuts[0], 2 * (cuts[1] - cuts[0]), 2 * (cuts[2] - cuts[1]), 360 - 2 * cuts[2]]
        # no arc of 180 or more keeps every inscribed angle below 180 and the figure convex
        if min(arcs) >= 20 and max(arcs) < 180:
            return arcs
    raise AssertionError("unreachable")


def _on_circle(arcs):
    t = [0, arcs[0], arcs[0] + arcs[1], arcs[0] + arcs[1] + arcs[2]]
    return [_pol(x) for x in t]


# ---------------------------------------------------------------------------
# Easy
# ---------------------------------------------------------------------------
def _c_centre_angle(rng) -> dict:
    a_, b_, c_ = _labels(rng, 3)
    x = rng.randint(20, 80)
    forward = rng.random() < 0.5
    A, B = _pol(0), _pol(2 * x)
    C = _pol(2 * x + rng.randint(20, 360 - 2 * x - 20))
    check(
        _eq(_ang(A, C, B), x) and _eq(_ang(A, O, B), 2 * x),
        "angle at the centre is twice the angle at the circumference",
    )
    p = Prompt()
    p.t(f"{a_}, {b_} and {c_} are points on a circle with centre O, and {c_} lies on the major arc {a_}{b_}. ")
    if forward:
        p.t(f"∠{a_}{c_}{b_} = {x}°. Find ∠{a_}O{b_}. ")
        claim = f"∠{a_}O{b_} = {2 * x}°"
        ref = [
            f"∠{a_}O{b_} is the angle at the centre and ∠{a_}{c_}{b_} is the angle at the circumference, both standing on arc {a_}{b_}.",
            f"The angle at the centre is twice the angle at the circumference, so ∠{a_}O{b_} = 2 × {x}° = {2 * x}°.",
        ]
        step = f"finds ∠{a_}O{b_} = {2 * x}° and names the theorem"
    else:
        p.t(f"∠{a_}O{b_} = {2 * x}°. Find ∠{a_}{c_}{b_}. ")
        claim = f"∠{a_}{c_}{b_} = {x}°"
        ref = [
            f"∠{a_}O{b_} is the angle at the centre and ∠{a_}{c_}{b_} the angle at the circumference, both on arc {a_}{b_}.",
            f"The angle at the circumference is half the angle at the centre, so ∠{a_}{c_}{b_} = {2 * x}° ÷ 2 = {x}°.",
        ]
        step = f"finds ∠{a_}{c_}{b_} = {x}° and names the theorem"
    p.t("Give the circle theorem that justifies your answer. ").t(_tail(rng))
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", step),
            ("logical_completeness", "identifies the arc that both angles stand on"),
            (
                "definitions_assumptions",
                "states that the angle at the centre is twice the angle at the circumference on the same arc",
            ),
            ("communication", "gives the reason beside the answer"),
        ],
        traps=[
            (
                "halves_and_doubles_swapped",
                "does it halve the angle at the circumference or double the angle at the centre?",
            ),
            ("no_theorem_named", "does the answer say which theorem is used, not only the number?"),
        ],
    )
    return finish(p, claim, mark)


def _c_semicircle(rng) -> dict:
    a_, b_, c_ = _labels(rng, 3)
    a = rng.randint(15, 75)
    A, B = (-1.0, 0.0), (1.0, 0.0)
    # C on the circle with angle CAB = a: central angle of arc BC is 2a
    C = _pol(2 * a)
    check(_eq(_ang(B, A, C), a) and _eq(_ang(A, C, B), 90) and _eq(_ang(A, B, C), 90 - a), "angle in a semicircle")
    p = Prompt()
    p.t(
        f"{a_}{b_} is a diameter of a circle and {c_} is another point on the circle. ∠{b_}{a_}{c_} = {a}°. Find ∠{a_}{c_}{b_} and ∠{a_}{b_}{c_}, giving a reason for each. "
    ).t(_tail(rng))
    ref = [
        f"{a_}{b_} is a diameter, so the angle in a semicircle ∠{a_}{c_}{b_} = 90°.",
        f"The angles of triangle {a_}{b_}{c_} sum to 180°, so ∠{a_}{b_}{c_} = 180° - 90° - {a}° = {90 - a}°.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"finds ∠{a_}{c_}{b_} = 90° and ∠{a_}{b_}{c_} = {90 - a}°"),
            ("logical_completeness", "uses the angle sum of the triangle for the second angle"),
            ("definitions_assumptions", "names the angle in a semicircle theorem and says the chord is a diameter"),
            ("communication", "gives a reason with each angle"),
        ],
        traps=[
            ("right_angle_at_wrong_vertex", f"does it put the right angle at {a_} or {b_} instead of {c_}?"),
            ("no_reason", "does each angle come with a reason?"),
        ],
    )
    return finish(p, f"∠{a_}{c_}{b_} = 90° and ∠{a_}{b_}{c_} = {90 - a}°", mark)


def _c_tangent_radius(rng) -> dict:
    p_, t_ = _labels(rng, 2)
    a = rng.randint(15, 75)
    T = (1.0, 0.0)
    P = (1.0, math.tan(math.radians(90 - a)))
    check(
        _eq(_ang(P, T, O), 90) and _eq(_ang(T, P, O), a) and _eq(_ang(P, O, T), 90 - a),
        "tangent meets radius at 90 degrees",
    )
    p = Prompt()
    p.t(
        f"The line {p_}{t_} touches a circle with centre O at {t_}, where {p_} is outside the circle. ∠{t_}{p_}O = {a}°. Find ∠{p_}O{t_}, giving a reason for each step. "
    ).t(_tail(rng))
    ref = [
        f"The tangent {p_}{t_} is perpendicular to the radius O{t_} at the point of contact, so ∠O{t_}{p_} = 90°.",
        f"The angles of triangle O{t_}{p_} sum to 180°, so ∠{p_}O{t_} = 180° - 90° - {a}° = {90 - a}°.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"finds ∠{p_}O{t_} = {90 - a}°"),
            ("logical_completeness", f"uses the right angle at {t_} and the angle sum of triangle O{t_}{p_}"),
            ("definitions_assumptions", "states that a tangent is perpendicular to the radius at the point of contact"),
            ("communication", "gives a reason for each step"),
        ],
        traps=[
            ("right_angle_missing", "does it use the 90° between tangent and radius?"),
            ("wrong_angle_found", f"does it find ∠{p_}O{t_} and not ∠O{p_}{t_}?"),
        ],
    )
    return finish(p, f"∠{p_}O{t_} = {90 - a}°", mark)


# ---------------------------------------------------------------------------
# Medium
# ---------------------------------------------------------------------------
def _c_cyclic_quad(rng) -> dict:
    a_, b_, c_, d_ = _labels(rng, 4)
    arcs = _cyclic_arcs(rng)
    A, B, C, D = _on_circle(arcs)
    angle_b = (arcs[2] + arcs[3]) // 2  # angle ABC stands on arc CDA
    angle_a = (arcs[1] + arcs[2]) // 2  # angle DAB stands on arc BCD
    angle_c = (arcs[3] + arcs[0]) // 2
    angle_d = (arcs[0] + arcs[1]) // 2
    check(
        _eq(_ang(A, B, C), angle_b)
        and _eq(_ang(D, A, B), angle_a)
        and _eq(_ang(B, C, D), angle_c)
        and _eq(_ang(C, D, A), angle_d),
        "cyclic quadrilateral angles",
    )
    check(angle_b + angle_d == 180 and angle_a + angle_c == 180, "opposite angles are supplementary")
    p = Prompt()
    p.t(
        f"{a_}, {b_}, {c_} and {d_} lie on a circle in that order, forming the cyclic quadrilateral {a_}{b_}{c_}{d_}. ∠{a_}{b_}{c_} = {angle_b}° and ∠{d_}{a_}{b_} = {angle_a}°. Find ∠{c_}{d_}{a_} and ∠{b_}{c_}{d_}, giving the theorem you use. "
    ).t(_tail(rng))
    ref = [
        f"The opposite angles of a cyclic quadrilateral are supplementary (sum to 180°).",
        f"∠{c_}{d_}{a_} = 180° - ∠{a_}{b_}{c_} = 180° - {angle_b}° = {angle_d}°.",
        f"∠{b_}{c_}{d_} = 180° - ∠{d_}{a_}{b_} = 180° - {angle_a}° = {angle_c}°.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"finds ∠{c_}{d_}{a_} = {angle_d}° and ∠{b_}{c_}{d_} = {angle_c}°"),
            ("logical_completeness", "pairs each angle with its opposite angle correctly"),
            ("definitions_assumptions", "names the cyclic quadrilateral theorem"),
            ("communication", "gives the reason with each answer"),
        ],
        traps=[
            (
                "adjacent_angles_paired",
                "does it subtract from 180° using an adjacent angle instead of the opposite one?",
            ),
            ("angle_set_equal", "does it say opposite angles are equal?"),
        ],
    )
    return finish(p, f"∠{c_}{d_}{a_} = {angle_d}° and ∠{b_}{c_}{d_} = {angle_c}°", mark)


def _c_alternate_segment(rng) -> dict:
    a_, b_, c_, t_ = _labels(rng, 4)
    t = rng.randint(25, 80)
    A, T = (1.0, 0.0), (1.0, 1.0)
    B = _pol(2 * t)
    C = _pol(2 * t + rng.randint(30, 360 - 2 * t - 30))
    check(_eq(_ang(T, A, B), t) and _eq(_ang(A, C, B), t), "alternate segment theorem")
    p = Prompt()
    p.t(
        f"The line {a_}{t_} touches a circle at {a_}. {b_} and {c_} are points on the circle, with {c_} on the opposite side of chord {a_}{b_} from {t_}. ∠{t_}{a_}{b_} = {t}°. Find ∠{a_}{c_}{b_} and name the theorem you use. "
    ).t(_tail(rng))
    ref = [
        f"The angle between the tangent {a_}{t_} and the chord {a_}{b_} equals the angle in the alternate segment.",
        f"{c_} is in the alternate segment, so ∠{a_}{c_}{b_} = ∠{t_}{a_}{b_} = {t}°.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"finds ∠{a_}{c_}{b_} = {t}°"),
            (
                "logical_completeness",
                f"explains that {c_} lies in the segment on the other side of the chord from the tangent angle",
            ),
            ("definitions_assumptions", "names the alternate segment (tangent-chord) theorem"),
            ("communication", "states the theorem as the reason"),
        ],
        traps=[
            ("same_side_segment", f"does it use an angle in the segment on the same side as {t_}?"),
            ("doubles_angle", "does it double the angle as if it were an angle at the centre?"),
        ],
    )
    return finish(p, f"∠{a_}{c_}{b_} = {t}°", mark)


def _c_equal_tangents(rng) -> dict:
    p_, a_, b_ = _labels(rng, 3)
    ang_p = 2 * rng.randint(20, 70)
    phi = 90 - ang_p // 2
    A, B = _pol(phi), _pol(-phi)
    P = (1 / math.cos(math.radians(phi)), 0.0)
    check(
        _eq(_d(P, A), _d(P, B)) and _eq(_ang(A, P, B), ang_p) and _eq(_ang(P, A, B), (180 - ang_p) / 2),
        "equal tangents",
    )
    base = (180 - ang_p) // 2
    p = Prompt()
    p.t(
        f"From a point {p_} outside a circle with centre O, the tangents {p_}{a_} and {p_}{b_} are drawn, touching the circle at {a_} and {b_}. ∠{a_}{p_}{b_} = {ang_p}°. Prove that {p_}{a_} = {p_}{b_} and find ∠{p_}{a_}{b_}. "
    ).t(_tail(rng))
    ref = [
        f"Join O to {a_}, O to {b_} and O to {p_}. Tangents are perpendicular to radii, so ∠O{a_}{p_} = ∠O{b_}{p_} = 90°.",
        f"In triangles O{a_}{p_} and O{b_}{p_}: O{a_} = O{b_} (radii), O{p_} is common, and the angles at {a_} and {b_} are 90°. They are congruent (RHS).",
        f"So {p_}{a_} = {p_}{b_}, corresponding sides of congruent triangles.",
        f"Triangle {p_}{a_}{b_} is isosceles with apex angle {ang_p}°, so ∠{p_}{a_}{b_} = (180° - {ang_p}°) ÷ 2 = {base}°.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"proves {p_}{a_} = {p_}{b_} and finds ∠{p_}{a_}{b_} = {base}°"),
            ("logical_completeness", "proves the congruence with all three conditions, including the right angles"),
            (
                "definitions_assumptions",
                "states that a tangent is perpendicular to the radius, and that the radii are equal",
            ),
            ("communication", "names the congruence test and the isosceles step"),
        ],
        traps=[
            ("assumes_equal_tangents", "does it assume the two tangents are equal instead of proving it?"),
            ("wrong_isosceles_angle", f"does it use {ang_p}° as a base angle rather than the apex angle?"),
        ],
    )
    return finish(p, f"{p_}{a_} = {p_}{b_} and ∠{p_}{a_}{b_} = {base}°", mark)


# ---------------------------------------------------------------------------
# Hard
# ---------------------------------------------------------------------------
def _c_prove_centre(rng) -> dict:
    a_, b_, c_, d_ = _labels(rng, 4)
    # acute triangle so that O is inside
    for _ in attempts(500, "an acute triangle on a circle"):
        ta, tb, tc = sorted(rng.sample(range(0, 360), 3))
        arcs = (tb - ta, tc - tb, 360 - (tc - ta))
        if max(arcs) < 175 and min(arcs) > 25:
            break
    A, B, C = _pol(ta), _pol(tb), _pol(tc)
    # O is inside the triangle when every arc is below 180 degrees; angle AOB = 2 angle ACB
    check(_eq(_ang(A, O, B), 2 * _ang(A, C, B)), "the angle at the centre is twice the angle at the circumference")
    p = Prompt()
    p.t(
        f"{a_}, {b_} and {c_} are points on a circle with centre O, and O lies inside triangle {a_}{b_}{c_}. Prove that ∠{a_}O{b_} = 2∠{a_}{c_}{b_}. "
    )
    p.t(
        f"Hint: join {c_}O and extend it to meet the circle again at {d_}. You may use that the base angles of an isosceles triangle are equal and that an exterior angle of a triangle equals the sum of the two opposite interior angles. "
    ).t(_tail(rng))
    ref = [
        f"Join {c_}O and extend it to {d_} on the circle. O{a_} = O{b_} = O{c_} (radii).",
        f"Triangle O{c_}{a_} is isosceles, so ∠O{c_}{a_} = ∠O{a_}{c_} = x, say. ∠{a_}O{d_} is an exterior angle of this triangle, so ∠{a_}O{d_} = 2x.",
        f"Similarly triangle O{c_}{b_} is isosceles with ∠O{c_}{b_} = ∠O{b_}{c_} = y, and ∠{b_}O{d_} = 2y.",
        f"Because O is inside the triangle, ∠{a_}O{b_} = ∠{a_}O{d_} + ∠{b_}O{d_} = 2x + 2y = 2(x + y).",
        f"And ∠{a_}{c_}{b_} = ∠O{c_}{a_} + ∠O{c_}{b_} = x + y, so ∠{a_}O{b_} = 2∠{a_}{c_}{b_}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "reaches ∠AOB = 2(x + y) and ∠ACB = x + y"),
            ("logical_completeness", "uses both isosceles triangles and adds the two exterior angles"),
            (
                "definitions_assumptions",
                "states that OA, OB and OC are radii, and that O is inside the triangle so the angles add",
            ),
            ("communication", "introduces x and y and the extra line"),
        ],
        traps=[
            ("assumes_the_theorem", "does it quote the angle at the centre theorem as a reason?"),
            ("one_isosceles_only", "does it deal with only one of the two isosceles triangles?"),
        ],
    )
    return finish(p, f"∠{a_}O{b_} = 2∠{a_}{c_}{b_}", mark)


def _c_prove_same_segment(rng) -> dict:
    a_, b_, c_, d_ = _labels(rng, 4)
    arcs = _cyclic_arcs(rng)
    pts = _on_circle(arcs)
    A, B, C, D = pts  # C and D lie on the arc BA that does not hold... C, D both on arc from B round to A
    check(_eq(_ang(A, C, B), _ang(A, D, B)) and _eq(_ang(A, C, B), (arcs[0]) / 2), "angles in the same segment")
    p = Prompt()
    p.t(
        f"{a_}, {b_}, {c_} and {d_} are points on a circle with centre O. {c_} and {d_} lie on the same side of the chord {a_}{b_}. Prove that ∠{a_}{c_}{b_} = ∠{a_}{d_}{b_}. "
    )
    p.t(
        "You may use the theorem that the angle at the centre is twice the angle at the circumference on the same arc. "
    ).t(_tail(rng))
    ref = [
        f"Let ∠{a_}O{b_} be the angle at the centre standing on the arc {a_}{b_} that is on the opposite side of the chord from {c_} and {d_}.",
        f"By the angle at the centre theorem, ∠{a_}{c_}{b_} = ½∠{a_}O{b_}.",
        f"By the same theorem, ∠{a_}{d_}{b_} = ½∠{a_}O{b_}, since {d_} also stands on the same arc {a_}{b_}.",
        f"Both angles equal ½∠{a_}O{b_}, so ∠{a_}{c_}{b_} = ∠{a_}{d_}{b_}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "shows both angles equal half the same angle at the centre"),
            ("logical_completeness", "applies the theorem twice, once for each point, to the same arc"),
            (
                "definitions_assumptions",
                "identifies the common arc, and that C and D are on the same side of the chord",
            ),
            ("communication", "names the common angle at the centre"),
        ],
        traps=[
            ("different_arcs", "does it use a different arc for each of the two angles?"),
            ("no_common_value", "does it state that the two angles are each half the same angle?"),
        ],
    )
    return finish(p, f"∠{a_}{c_}{b_} = ∠{a_}{d_}{b_}", mark)


def _c_prove_cyclic_quad(rng) -> dict:
    a_, b_, c_, d_ = _labels(rng, 4)
    arcs = _cyclic_arcs(rng)
    A, B, C, D = _on_circle(arcs)
    check(_eq(_ang(A, B, C) + _ang(C, D, A), 180), "opposite angles of a cyclic quadrilateral")
    p = Prompt()
    p.t(
        f"{a_}, {b_}, {c_} and {d_} lie on a circle with centre O, in that order round the circle. Prove that ∠{a_}{b_}{c_} + ∠{c_}{d_}{a_} = 180°. "
    )
    p.t(
        "You may use the theorem that the angle at the centre is twice the angle at the circumference on the same arc, and that angles round a point sum to 360°. "
    ).t(_tail(rng))
    ref = [
        f"Let the angle at the centre on arc {a_}{d_}{c_} be x. Then the angle at the centre on arc {a_}{b_}{c_} (the other way round) is 360° - x.",
        f"{b_} is on the circle, so ∠{a_}{b_}{c_} stands on arc {a_}{d_}{c_}: by the angle at the centre theorem, ∠{a_}{b_}{c_} = x ÷ 2.",
        f"{d_} is on the other arc, so ∠{c_}{d_}{a_} stands on arc {a_}{b_}{c_}, with angle at the centre 360° - x: ∠{c_}{d_}{a_} = (360° - x) ÷ 2.",
        f"Adding, ∠{a_}{b_}{c_} + ∠{c_}{d_}{a_} = x ÷ 2 + (360° - x) ÷ 2 = 180°.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", "obtains x/2 and (360° - x)/2 and adds to 180°"),
            ("logical_completeness", "uses the angles round the centre, and the correct arc for each angle"),
            (
                "definitions_assumptions",
                "states the angle at the centre theorem and that the two arcs make a full turn",
            ),
            ("communication", "introduces x and keeps track of which arc each angle stands on"),
        ],
        traps=[
            ("assumes_result", "does it quote the cyclic quadrilateral theorem as a reason?"),
            ("both_on_same_arc", "does it use the same arc for both angles?"),
        ],
    )
    return finish(p, f"∠{a_}{b_}{c_} + ∠{c_}{d_}{a_} = 180°", mark)


def _c_intersecting_chords(rng) -> dict:
    a_, b_, c_, d_, p_ = _labels(rng, 5)
    for _ in attempts(500, "a factor product with two factorisations"):
        prod = rng.choice((12, 16, 18, 20, 24, 30, 36, 40, 48))
        pairs = [(i, prod // i) for i in range(1, prod + 1) if prod % i == 0]
        pa, pb = rng.choice(pairs)
        pc, pd = rng.choice(pairs)
        if len({pa, pb, pc, pd}) == 4 and min(pa, pb, pc, pd) >= 2:
            break
    # concyclic by the converse of the chord product theorem; confirm with a circumcircle through three points
    A, B = (-float(pa), 0.0), (float(pb), 0.0)
    C, D = (0.0, float(pc)), (0.0, -float(pd))
    ax, ay, bx, by, cx, cy = A[0], A[1], B[0], B[1], C[0], C[1]
    den = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / den
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / den
    check(_eq(math.hypot(D[0] - ux, D[1] - uy), math.hypot(ax - ux, ay - uy)), "the four points are concyclic")
    check(pa * pb == pc * pd, "intersecting chords")
    p = Prompt()
    p.t(
        f"The chords {a_}{b_} and {c_}{d_} of a circle meet at the point {p_} inside the circle. {p_}{a_} = {pa}, {p_}{b_} = {pb} and {p_}{c_} = {pc}. "
    )
    p.t(
        f"Prove that {p_}{a_} × {p_}{b_} = {p_}{c_} × {p_}{d_} and hence find {p_}{d_}. You may use the theorem that angles in the same segment are equal and that vertically opposite angles are equal. "
    ).t(_tail(rng))
    ref = [
        f"Join {a_}{c_} and {b_}{d_}. In triangles {p_}{a_}{c_} and {p_}{d_}{b_}: ∠{a_}{p_}{c_} = ∠{d_}{p_}{b_} (vertically opposite).",
        f"∠{p_}{a_}{c_} = ∠{p_}{d_}{b_} (angles in the same segment, both standing on arc {c_}{b_}).",
        f"So the triangles are similar (AA), and {p_}{a_} ÷ {p_}{d_} = {p_}{c_} ÷ {p_}{b_}, which gives {p_}{a_} × {p_}{b_} = {p_}{c_} × {p_}{d_}.",
        f"{pa} × {pb} = {pa * pb}, so {pc} × {p_}{d_} = {pa * pb} and {p_}{d_} = {pa * pb} ÷ {pc} = {pd}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"derives the product relation and finds {p_}{d_} = {pd}"),
            (
                "logical_completeness",
                "proves the two triangles similar with two equal angles and writes the ratios in the matching order",
            ),
            (
                "definitions_assumptions",
                "names vertically opposite angles and angles in the same segment as the reasons",
            ),
            ("communication", "names the similar triangles with their vertices in matching order"),
        ],
        traps=[
            ("wrong_ratio_pairing", "does it pair the sides of the similar triangles in the wrong order?"),
            ("assumes_the_product", "does it use the chord product result as a reason in its own proof?"),
        ],
    )
    return finish(p, f"{p_}{a_} × {p_}{b_} = {p_}{c_} × {p_}{d_} and {p_}{d_} = {pd}", mark)


def _c_equal_chords(rng) -> dict:
    a_, b_, c_, d_, m_, n_ = _labels(rng, 6)
    t1 = rng.randint(20, 150)
    t2 = rng.randint(200, 340)
    half = rng.randint(20, 80)
    A, B = _pol(t1 - half), _pol(t1 + half)
    C, D = _pol(t2 - half), _pol(t2 + half)
    check(_eq(_d(A, B), _d(C, D)), "equal chords")
    mid1 = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)
    mid2 = ((C[0] + D[0]) / 2, (C[1] + D[1]) / 2)
    check(_eq(_d(O, mid1), _d(O, mid2)), "equal chords are equidistant from the centre")
    p = Prompt()
    p.t(
        f"{a_}{b_} and {c_}{d_} are equal chords of a circle with centre O. {m_} and {n_} are the midpoints of {a_}{b_} and {c_}{d_}. Prove that O{m_} = O{n_}. "
    )
    p.t(
        "You may use that the line from the centre to the midpoint of a chord is perpendicular to the chord, and the RHS congruence test. "
    ).t(_tail(rng))
    ref = [
        f"The line from the centre to the midpoint of a chord is perpendicular to it, so ∠O{m_}{a_} = ∠O{n_}{c_} = 90°.",
        f"{a_}{m_} = ½{a_}{b_} and {c_}{n_} = ½{c_}{d_}. The chords are equal, so {a_}{m_} = {c_}{n_}.",
        f"O{a_} = O{c_} (radii).",
        f"Triangles O{m_}{a_} and O{n_}{c_} have a right angle, equal hypotenuses O{a_} = O{c_} and equal sides {a_}{m_} = {c_}{n_}. They are congruent (RHS).",
        f"So O{m_} = O{n_}, corresponding sides of congruent triangles.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"proves triangles O{m_}{a_} and O{n_}{c_} congruent and concludes O{m_} = O{n_}"),
            ("logical_completeness", "establishes all three RHS conditions, including half-chords equal"),
            (
                "definitions_assumptions",
                "states that the radii are equal and that the midpoint line is perpendicular to the chord",
            ),
            ("communication", "names the congruence test and the corresponding sides"),
        ],
        traps=[
            (
                "half_chord_not_justified",
                "does it say the half-chords are equal without using that the chords are equal and the points are midpoints?",
            ),
            ("circular", "does it assume the chords are equidistant from the centre?"),
        ],
    )
    return finish(p, f"O{m_} = O{n_}", mark)


# ---------------------------------------------------------------------------
# Exam
# ---------------------------------------------------------------------------
def _c_prove_alternate_segment(rng) -> dict:
    a_, b_, c_, t_, e_ = _labels(rng, 5)
    t = rng.randint(25, 80)
    A, T = (1.0, 0.0), (1.0, 1.0)
    B = _pol(2 * t)
    C = _pol(2 * t + rng.randint(30, 360 - 2 * t - 30))
    check(_eq(_ang(T, A, B), _ang(A, C, B)), "alternate segment theorem")
    p = Prompt()
    p.t(
        f"The line {a_}{t_} touches a circle with centre O at {a_}. {b_} and {c_} are points on the circle, with {c_} on the opposite side of the chord {a_}{b_} from {t_}. Prove that ∠{t_}{a_}{b_} = ∠{a_}{c_}{b_}. "
    )
    p.t(
        f"Hint: draw the diameter {a_}{e_}. You may use that a tangent is perpendicular to the radius at the point of contact, that the angle in a semicircle is 90°, and that angles in the same segment are equal. "
    ).t(_tail(rng))
    ref = [
        f"Draw the diameter {a_}{e_}, so the tangent is perpendicular to it: ∠{t_}{a_}{e_} = 90° (tangent and radius).",
        f"Take {e_} on the same side of {a_}{b_} as {c_}, so ∠{t_}{a_}{b_} = 90° - ∠{b_}{a_}{e_}.",
        f"∠{a_}{b_}{e_} = 90° (angle in a semicircle on the diameter {a_}{e_}). In triangle {a_}{b_}{e_}, ∠{a_}{e_}{b_} = 90° - ∠{b_}{a_}{e_}.",
        f"So ∠{t_}{a_}{b_} = ∠{a_}{e_}{b_}.",
        f"∠{a_}{e_}{b_} = ∠{a_}{c_}{b_} (angles in the same segment, both standing on chord {a_}{b_}). So ∠{t_}{a_}{b_} = ∠{a_}{c_}{b_}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"shows ∠{t_}{a_}{b_} = ∠{a_}{e_}{b_} = ∠{a_}{c_}{b_}"),
            ("logical_completeness", "uses all three allowed theorems and links the angle at E to the angle at C"),
            ("definitions_assumptions", "states each theorem used, and places E on the same side of the chord as C"),
            ("communication", "labels the diameter and gives a reason on each line"),
        ],
        traps=[
            ("quotes_alternate_segment", "does it quote the alternate segment theorem as a reason for the result?"),
            ("e_wrong_side", "does it ignore which side of the chord E is on?"),
        ],
    )
    return finish(p, f"∠{t_}{a_}{b_} = ∠{a_}{c_}{b_}", mark)


def _c_cyclic_bisect(rng) -> dict:
    a_, b_, c_, d_ = _labels(rng, 4)
    arcs = []
    for _ in attempts(500, "arcs with AB = AD"):
        w = 2 * rng.randint(15, 80)
        rest = 360 - 2 * w
        x = 2 * rng.randint(10, rest // 2 - 10)
        if 20 <= x < 180 and 20 <= rest - x < 180 and w < 180:
            arcs = [w, x, rest - x, w]
            break
    # points in order A, B, C, D: A at angle 0; arc DA is the last arc
    A, B, C, D = _on_circle(arcs)
    check(_eq(_d(A, B), _d(A, D)) and _eq(_ang(B, C, A), _ang(A, C, D)), "CA bisects angle BCD when AB = AD")
    p = Prompt()
    p.t(
        f"{a_}, {b_}, {c_} and {d_} lie on a circle in that order, and {a_}{b_} = {a_}{d_}. Prove that {c_}{a_} bisects ∠{b_}{c_}{d_}. "
    )
    p.t("You may use that equal chords of a circle subtend equal angles at the circumference, on the same side. ").t(
        _tail(rng)
    )
    ref = [
        f"The chords {a_}{b_} and {a_}{d_} are equal (given), so they cut off arcs of equal size.",
        f"∠{a_}{c_}{b_} stands on the chord {a_}{b_} and ∠{a_}{c_}{d_} stands on the chord {a_}{d_}, with {c_} on the circle on the same side of both chords.",
        f"Equal chords subtend equal angles at the circumference, so ∠{a_}{c_}{b_} = ∠{a_}{c_}{d_}.",
        f"{a_} lies between {b_} and {d_} on the arc not containing {c_}, so the ray {c_}{a_} lies inside ∠{b_}{c_}{d_}.",
        f"Since ∠{b_}{c_}{a_} = ∠{a_}{c_}{d_}, the line {c_}{a_} bisects ∠{b_}{c_}{d_}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"shows ∠{b_}{c_}{a_} = ∠{a_}{c_}{d_}"),
            (
                "logical_completeness",
                f"links the equal chords to equal angles at {c_}, and notes the ray {c_}{a_} lies between {c_}{b_} and {c_}{d_}",
            ),
            ("definitions_assumptions", "states the theorem on equal chords and equal angles"),
            ("communication", "concludes with the bisector statement"),
        ],
        traps=[
            ("angles_at_wrong_point", f"does it compare angles at {b_} or {d_} instead of at {c_}?"),
            ("triangle_ABD_only", f"does it only show that triangle {a_}{b_}{d_} is isosceles?"),
        ],
    )
    return finish(p, f"{c_}{a_} bisects ∠{b_}{c_}{d_}", mark)


def _c_tangent_secant(rng) -> dict:
    p_, t_, a_, b_ = _labels(rng, 4)
    # T = (1, 0) on the unit circle; tangent x = 1; P = (1, h). A secant from P cuts the circle at A then B.
    for _ in attempts(500, "a secant from the external point"):
        h = rng.uniform(1.2, 3.0)
        P = (1.0, h)
        direction = _pol(rng.uniform(190, 260))
        # intersect the line P + s*direction with the unit circle
        bq = P[0] * direction[0] + P[1] * direction[1]
        cq = P[0] ** 2 + P[1] ** 2 - 1
        disc = bq * bq - cq
        if disc > 0.05:
            s1, s2 = -bq - math.sqrt(disc), -bq + math.sqrt(disc)
            if 0 < s1 < s2:
                break
    PA, PB, PT = s1, s2, h
    check(_eq(PT * PT, PA * PB), "tangent-secant")
    p = Prompt()
    p.t(
        f"From a point {p_} outside a circle, the tangent {p_}{t_} touches the circle at {t_}, and a straight line from {p_} cuts the circle at {a_} and then at {b_} (so {a_} is between {p_} and {b_}). Prove that {p_}{t_}² = {p_}{a_} × {p_}{b_}. "
    )
    p.t(
        f"You may use the alternate segment theorem and that two triangles with two equal angles are similar (AA). Hint: join {t_}{a_} and {t_}{b_}. "
    ).t(_tail(rng))
    ref = [
        f"Consider triangles {p_}{t_}{a_} and {p_}{b_}{t_}. They share the angle at {p_}: ∠{t_}{p_}{a_} = ∠{b_}{p_}{t_}.",
        f"By the alternate segment theorem, ∠{p_}{t_}{a_} = ∠{t_}{b_}{a_}, that is ∠{p_}{t_}{a_} = ∠{p_}{b_}{t_}.",
        f"Two angles are equal, so triangles {p_}{t_}{a_} and {p_}{b_}{t_} are similar (AA), with {t_} matching {b_} and {a_} matching {t_}.",
        f"Corresponding sides are in proportion: {p_}{t_} ÷ {p_}{b_} = {p_}{a_} ÷ {p_}{t_}.",
        f"Cross-multiplying, {p_}{t_}² = {p_}{a_} × {p_}{b_}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"reaches {p_}{t_} ÷ {p_}{b_} = {p_}{a_} ÷ {p_}{t_} and cross-multiplies"),
            (
                "logical_completeness",
                "proves the triangles similar with two angles, matching the vertices in the right order",
            ),
            ("definitions_assumptions", "names the alternate segment theorem and the shared angle at P"),
            ("communication", "lists the corresponding vertices"),
        ],
        traps=[
            ("ratios_mismatched", "does it write the ratios in a non-corresponding order?"),
            ("assumes_result", "does it quote the tangent-secant result?"),
        ],
    )
    return finish(p, f"{p_}{t_}² = {p_}{a_} × {p_}{b_}", mark)


def _c_chord_tangent_product(rng) -> dict:
    a_, b_, c_, d_ = _labels(rng, 4)
    t = rng.randint(20, 160)
    A, B = (-1.0, 0.0), (1.0, 0.0)
    C = _pol(t)
    # line AC meets the tangent at B (x = 1) at D
    s = (1 - A[0]) / (C[0] - A[0])
    D = (1.0, A[1] + s * (C[1] - A[1]))
    check(_eq(_ang(A, B, D), 90) and _eq(_d(A, C) * _d(A, D), _d(A, B) ** 2), "AC x AD = AB^2")
    p = Prompt()
    p.t(
        f"{a_}{b_} is a diameter of a circle and {c_} is another point on the circle. The tangent at {b_} meets the line {a_}{c_} extended at {d_}. Prove that {a_}{c_} × {a_}{d_} = {a_}{b_}². "
    )
    p.t(
        "You may use the angle in a semicircle, that a tangent is perpendicular to the radius at the point of contact, and AA similarity. "
    ).t(_tail(rng))
    ref = [
        f"∠{a_}{c_}{b_} = 90° (angle in a semicircle) and ∠{a_}{b_}{d_} = 90° (tangent perpendicular to the radius, {a_}{b_} being a diameter).",
        f"Triangles {a_}{c_}{b_} and {a_}{b_}{d_} share the angle at {a_}.",
        f"Two angles equal, so they are similar (AA), with {a_} matching {a_}, {c_} matching {b_} and {b_} matching {d_}.",
        f"Corresponding sides: {a_}{c_} ÷ {a_}{b_} = {a_}{b_} ÷ {a_}{d_}.",
        f"Cross-multiplying, {a_}{c_} × {a_}{d_} = {a_}{b_}².",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"reaches {a_}{c_} ÷ {a_}{b_} = {a_}{b_} ÷ {a_}{d_} and then the product"),
            ("logical_completeness", "justifies both right angles and the shared angle"),
            ("definitions_assumptions", "names the angle in a semicircle and tangent-radius theorems"),
            ("communication", "matches the vertices of the similar triangles"),
        ],
        traps=[
            ("wrong_vertex_order", "does it match the triangles' vertices in the wrong order, giving a wrong ratio?"),
            ("right_angle_at_a", f"does it put a right angle at {a_}?"),
        ],
    )
    return finish(p, f"{a_}{c_} × {a_}{d_} = {a_}{b_}²", mark)


def _c_three_theorems(rng) -> dict:
    p_, a_, b_, c_, d_ = _labels(rng, 5)
    ang_p = 2 * rng.randint(20, 70)
    phi = 90 - ang_p // 2
    A, B = _pol(phi), _pol(-phi)
    P = (1 / math.cos(math.radians(phi)), 0.0)
    C = _pol(180 + rng.randint(-40, 40))  # on the major arc AB (the side away from P)
    D = _pol(rng.randint(-phi + 10, phi - 10))  # on the minor arc AB (the side near P)
    check(_eq(_ang(A, P, B), ang_p), "apex angle")
    aob = 180 - ang_p
    acb = aob // 2
    adb = 180 - acb
    check(
        _eq(_ang(A, O, B), aob) and _eq(_ang(A, C, B), acb) and _eq(_ang(A, D, B), adb),
        "angles at centre and circumference",
    )
    p = Prompt()
    p.t(
        f"From a point {p_} outside a circle with centre O, the tangents {p_}{a_} and {p_}{b_} touch the circle at {a_} and {b_}. ∠{a_}{p_}{b_} = {ang_p}°. {c_} is a point on the major arc {a_}{b_} and {d_} is a point on the minor arc {a_}{b_}. "
    )
    p.t(f"Find ∠{a_}O{b_}, ∠{a_}{c_}{b_} and ∠{a_}{d_}{b_}, giving a circle theorem as the reason for each. ").t(
        _tail(rng)
    )
    ref = [
        f"The tangents are perpendicular to the radii: ∠O{a_}{p_} = ∠O{b_}{p_} = 90°.",
        f"The angles of the quadrilateral {p_}{a_}O{b_} sum to 360°, so ∠{a_}O{b_} = 360° - 90° - 90° - {ang_p}° = {aob}°.",
        f"{c_} is on the major arc, so ∠{a_}{c_}{b_} is the angle at the circumference on the same arc as ∠{a_}O{b_}: ∠{a_}{c_}{b_} = {aob}° ÷ 2 = {acb}°.",
        f"{a_}{c_}{b_}{d_} is a cyclic quadrilateral, so opposite angles are supplementary: ∠{a_}{d_}{b_} = 180° - {acb}° = {adb}°.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"finds ∠{a_}O{b_} = {aob}°, ∠{a_}{c_}{b_} = {acb}° and ∠{a_}{d_}{b_} = {adb}°"),
            (
                "logical_completeness",
                "uses the quadrilateral angle sum, the angle at the centre and the cyclic quadrilateral in a connected chain",
            ),
            (
                "definitions_assumptions",
                "names the tangent-radius, angle at the centre and cyclic quadrilateral theorems",
            ),
            ("communication", "gives a reason with each of the three angles"),
        ],
        traps=[
            (
                "minor_arc_halved",
                f"does it halve the angle at the centre to get ∠{a_}{d_}{b_} even though {d_} is on the minor arc?",
            ),
            ("angle_sum_of_triangle_used", "does it treat the figure with the two right angles as a triangle?"),
        ],
    )
    return finish(p, f"∠{a_}O{b_} = {aob}°, ∠{a_}{c_}{b_} = {acb}°, ∠{a_}{d_}{b_} = {adb}°", mark)


FAMILIES = {
    "easy": (_c_centre_angle, _c_semicircle, _c_tangent_radius),
    "medium": (_c_cyclic_quad, _c_alternate_segment, _c_equal_tangents),
    "hard": (_c_prove_centre, _c_prove_same_segment, _c_prove_cyclic_quad, _c_intersecting_chords, _c_equal_chords),
    "exam": (
        _c_prove_alternate_segment,
        _c_cyclic_bisect,
        _c_tangent_secant,
        _c_chord_tangent_product,
        _c_three_theorems,
    ),
}

_PRESETS = {t: dict(variant=t) for t in ("easy", "medium", "hard", "exam")}


@generator("proof_circle_theorems", level=14, presets=_PRESETS)
def proof_circle_theorems(rng, variant="easy"):
    """`senior-solve-problems-finding-unknown-angles-and-lengths-and`. Rubric-assessed (`answer: rubric`): every
    question returned carries a per-item marking dict (`Question.answer`), never a plain value."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    return rng.choice(FAMILIES[variant])(rng)
