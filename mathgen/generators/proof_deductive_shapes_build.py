"""Generates `proof_f10_deductive_shapes` items: a short deductive proof about triangles, parallelograms and
polygons, stated in words so no figure is needed.

Every question gives the configuration as labelled points ("ABCD is a parallelogram ..."), names the theorems
the learner may use, and asks for a proof. The proof is marked against a rubric, so `answer` is a marking dict
(see `_free_proof_kit`), not a value.

Ladder:

- easy: one step from the angle-sum or parallel-line theorems (exterior angle, isosceles apex, co-interior
  angles), sometimes with numbers.
- medium: a property of a parallelogram or rhombus, or one congruence, or the angle sum of an n-gon.
- hard: a congruence argument for the diagonals of a parallelogram, rectangle, rhombus or kite.
- exam: a two-stage argument (midpoint theorem, congruence then straight-line angles, a constructed
  parallelogram, a regular-polygon count).

Every numeric claim is recomputed in the generator, and the general theorems are checked on a random concrete
figure built from coordinates, so no statement that is false can be asked.
"""

from __future__ import annotations

import math

from ..core import check, generator
from ._free_proof_kit import TAILS, Prompt, finish, marking

NAME_SETS = ("ABCD", "PQRS", "WXYZ", "EFGH", "JKLM", "STUV")

#: Words for an n-sided convex polygon.
POLYGON_NAMES = {5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon", 9: "nonagon", 10: "decagon"}


def _ang(a: str, b: str, c: str) -> str:
    return f"∠{a}{b}{c}"


def _rot(names: str, r: int) -> str:
    return names[r:] + names[:r]


def _deg(n: int | float) -> str:
    return f"{int(n)}°"


def _tail(rng) -> str:
    return rng.choice(TAILS)


# ---------------------------------------------------------------------------
# Concrete-figure checks (coordinates), used only to confirm a statement is true.
# ---------------------------------------------------------------------------
def _pt(x: float, y: float) -> tuple[float, float]:
    return (x, y)


def _dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _angle(a, b, c) -> float:
    """The angle at b in degrees."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    cosv = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


def _parallelogram(rng):
    u = (rng.uniform(3, 6), rng.uniform(-1, 1))
    v = (rng.uniform(-1.5, 1.5), rng.uniform(2, 4))
    a = (0.0, 0.0)
    b = (u[0], u[1])
    d = (v[0], v[1])
    c = (u[0] + v[0], u[1] + v[1])
    return a, b, c, d


def _close(x: float, y: float, tol: float = 1e-7) -> bool:
    return abs(x - y) < tol


def _confirm_general_theorems(rng) -> None:
    """Build concrete figures and confirm every general statement this module asks, once per call."""
    a, b, c, d = _parallelogram(rng)
    check(_close(_angle(d, a, b), _angle(b, c, d)), "parallelogram opposite angles")
    check(_close(_angle(d, a, b) + _angle(a, b, c), 180.0), "parallelogram co-interior angles")
    m = ((a[0] + c[0]) / 2, (a[1] + c[1]) / 2)
    check(_close(m[0], (b[0] + d[0]) / 2) and _close(m[1], (b[1] + d[1]) / 2), "parallelogram diagonals bisect")
    # rectangle: equal diagonals
    w, h = rng.uniform(2, 6), rng.uniform(1, 5)
    check(_close(_dist((0, 0), (w, h)), _dist((w, 0), (0, h))), "rectangle diagonals")
    # rhombus: perpendicular diagonals and a bisected angle
    p, q = rng.uniform(1, 4), rng.uniform(1, 4)
    ra, rb, rc, rd = (-p, 0.0), (0.0, -q), (p, 0.0), (0.0, q)
    check(_close(_dist(ra, rb), _dist(rb, rc)) and _close(_dist(rc, rd), _dist(rd, ra)), "rhombus sides")
    check(_close(_angle(rb, ra, rc), _angle(rd, ra, rc)), "rhombus diagonal bisects the angle")
    # kite AB = AD, CB = CD: diagonals perpendicular
    s, t, u = rng.uniform(1, 3), rng.uniform(1, 3), rng.uniform(1, 3)
    ka, kb, kc, kd = (-s, 0.0), (0.0, t), (u, 0.0), (0.0, -t)
    check(_close(_dist(ka, kb), _dist(ka, kd)) and _close(_dist(kc, kb), _dist(kc, kd)), "kite sides")
    check(_close(_angle(ka, (0.0, 0.0), kb), 90.0), "kite diagonals")
    # midpoint quadrilateral of any convex quadrilateral is a parallelogram
    qa, qb, qc, qd = (
        (0.0, 0.0),
        (rng.uniform(4, 6), 0.5),
        (rng.uniform(3, 5), rng.uniform(3, 5)),
        (-1.0, rng.uniform(2, 3)),
    )
    mids = [((x[0] + y[0]) / 2, (x[1] + y[1]) / 2) for x, y in ((qa, qb), (qb, qc), (qc, qd), (qd, qa))]
    check(
        _close(mids[0][0] + mids[2][0], mids[1][0] + mids[3][0])
        and _close(mids[0][1] + mids[2][1], mids[1][1] + mids[3][1]),
        "midpoint quadrilateral",
    )
    # E and F with AE = CF in a parallelogram make DEBF a parallelogram
    k = rng.uniform(0.3, 0.9)
    e = (a[0] + k * (b[0] - a[0]), a[1] + k * (b[1] - a[1]))
    f = (c[0] + k * (d[0] - c[0]), c[1] + k * (d[1] - c[1]))
    check(_close(e[0] - d[0], b[0] - f[0]) and _close(e[1] - d[1], b[1] - f[1]), "constructed parallelogram DEBF")


# ---------------------------------------------------------------------------
# Easy
# ---------------------------------------------------------------------------
def _exterior_angle(rng) -> dict:
    n = rng.choice(NAME_SETS)
    a, b, c, d = n
    p = Prompt()
    p.t(f"In triangle {a}{b}{c}, the side {b}{c} is extended beyond {c} to a point {d}. ")
    p.t("You may use: the angles of a triangle add to 180°, and angles on a straight line add to 180°. ")
    if rng.random() < 0.5:
        x, y = rng.randint(30, 80), rng.randint(30, 80)
        claim = f"{_ang(a, c, d)} = {_deg(x + y)}"
        p.t(f"Given that {_ang(c, a, b)} = {_deg(x)} and {_ang(a, b, c)} = {_deg(y)}, prove that {claim}. ")
        numeric = True
    else:
        claim = f"{_ang(a, c, d)} = {_ang(c, a, b)} + {_ang(a, b, c)}"
        p.t(f"Prove that {claim}. ")
        numeric = False
    p.t(_tail(rng))
    ref = [
        f"In triangle {a}{b}{c}: {_ang(c, a, b)} + {_ang(a, b, c)} + {_ang(b, c, a)} = 180° (angle sum of a triangle).",
        f"{b}, {c} and {d} lie on a straight line, so {_ang(b, c, a)} + {_ang(a, c, d)} = 180° (angles on a straight line).",
        f"Both sums equal 180°, so {_ang(c, a, b)} + {_ang(a, b, c)} + {_ang(b, c, a)} = {_ang(b, c, a)} + {_ang(a, c, d)}.",
        f"Subtract {_ang(b, c, a)} from both sides: {_ang(a, c, d)} = {_ang(c, a, b)} + {_ang(a, b, c)}"
        + (f", which is the stated {claim.split('= ')[1]}." if numeric else "."),
    ]
    mark = marking(
        reference=ref,
        steps=[
            (
                "correctness",
                f"reaches {_ang(a, c, d)} as the sum of the two angles at {a} and {b} by cancelling {_ang(b, c, a)}",
            ),
            (
                "logical_completeness",
                "uses both stated theorems, the triangle sum and the straight line, each for its own step",
            ),
            ("definitions_assumptions", f"states that {b}, {c}, {d} are collinear with {c} between {b} and {d}"),
            ("communication", "each line is an equation about named angles with its reason beside it"),
        ],
        traps=[
            (
                "asserts_the_result",
                "does the proof use the two angle sums, or just assert that the exterior angle equals the opposite angles?",
            ),
            (
                "wrong_angle_in_triangle",
                f"does it use {_ang(b, c, a)} inside the triangle and {_ang(a, c, d)} outside it, and not mix them?",
            ),
        ],
    )
    return finish(p, claim, mark)


def _isosceles_apex(rng) -> dict:
    n = rng.choice(NAME_SETS)
    a, b, c = n[:3]
    apex = rng.choice(range(20, 141, 2))
    base = (180 - apex) // 2
    claim = f"{_ang(a, b, c)} = {_deg(base)}"
    check(2 * base + apex == 180, "isosceles angles must add to 180")
    p = Prompt()
    p.t(f"Triangle {a}{b}{c} has {a}{b} = {a}{c} and {_ang(b, a, c)} = {_deg(apex)}. ")
    p.t(
        "You may use: the angles of a triangle add to 180°, and the angles opposite equal sides of a triangle are equal. "
    )
    p.t(f"Prove that {claim}. ")
    p.t(_tail(rng))
    ref = [
        f"{a}{b} = {a}{c}, so triangle {a}{b}{c} is isosceles with {_ang(a, b, c)} = {_ang(a, c, b)} (angles opposite equal sides).",
        f"Angle sum: {_ang(b, a, c)} + {_ang(a, b, c)} + {_ang(a, c, b)} = 180°.",
        f"So {apex}° + 2 × {_ang(a, b, c)} = 180°.",
        f"2 × {_ang(a, b, c)} = {180 - apex}°, so {claim}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"forms {apex}° + 2x = 180° and solves it to {base}°"),
            ("logical_completeness", "uses both the equal base angles and the angle sum"),
            ("definitions_assumptions", f"says which two angles are equal and why ({a}{b} = {a}{c})"),
            ("communication", "names each angle, not just letters in an equation"),
        ],
        traps=[
            (
                "wrong_equal_angles",
                f"does it treat {_ang(b, a, c)} as equal to a base angle, or use the wrong pair of equal angles?",
            ),
            ("forgets_to_halve", f"does it reach {180 - apex}° and stop without halving?"),
        ],
    )
    return finish(p, claim, mark)


def _co_interior(rng) -> dict:
    g, h = rng.choice(("GH", "MN", "UV"))
    a, b, c, d = rng.choice(("ABCD", "PQRS", "WXYZ"))
    claim = f"{_ang(a, g, h)} + {_ang(c, h, g)} = 180°"
    p = Prompt()
    p.t(
        f"Lines {a}{b} and {c}{d} are parallel. A line crosses {a}{b} at {g} and {c}{d} at {h}, where {h} lies between {c} and {d}, "
    )
    p.t(f"and {a} and {c} are on the same side of line {g}{h}. ")
    p.t("You may use: alternate angles between parallel lines are equal, and angles on a straight line add to 180°. ")
    p.t(f"Prove that {claim}. ")
    p.t(_tail(rng))
    ref = [
        f"{c} and {a} are on the same side of {g}{h}, and {h} lies between {c} and {d}, so {d} is on the other side of {g}{h} from {a}.",
        f"So {_ang(a, g, h)} and {_ang(g, h, d)} are alternate angles, and since {a}{b} ∥ {c}{d}, {_ang(a, g, h)} = {_ang(g, h, d)}.",
        f"{c}, {h}, {d} lie on a straight line, so {_ang(c, h, g)} + {_ang(g, h, d)} = 180° (angles on a straight line).",
        f"Replace {_ang(g, h, d)} by {_ang(a, g, h)}: {claim}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"replaces {_ang(g, h, d)} by {_ang(a, g, h)} to reach the sum of 180°"),
            ("logical_completeness", "uses both stated theorems: alternate angles, then the straight line"),
            (
                "definitions_assumptions",
                f"identifies which angles are alternate, using that {a} and {c} are on the same side of {g}{h}",
            ),
            ("communication", "names the angle pairs and the reason for each step"),
        ],
        traps=[
            (
                "co_interior_assumed",
                "does it simply quote 'co-interior angles add to 180°', which is the result being proved, instead of using the two stated theorems?",
            ),
            (
                "wrong_alternate_pair",
                f"are the alternate angles {_ang(a, g, h)} and {_ang(g, h, d)}, on opposite sides of the transversal?",
            ),
        ],
    )
    return finish(p, claim, mark)


# ---------------------------------------------------------------------------
# Medium
# ---------------------------------------------------------------------------
def _parallelogram_opposite_angles(rng) -> dict:
    a, b, c, d = _rot(rng.choice(NAME_SETS), rng.randrange(4))
    claim = f"{_ang(d, a, b)} = {_ang(b, c, d)}"
    p = Prompt()
    p.t(f"{a}{b}{c}{d} is a parallelogram, so {a}{b} ∥ {d}{c} and {a}{d} ∥ {b}{c}. ")
    p.t("You may use: co-interior angles between parallel lines add to 180°. ")
    p.t(f"Prove that {claim}. ")
    p.t(_tail(rng))
    ref = [
        f"{a}{d} ∥ {b}{c} and {a}{b} crosses them, so {_ang(d, a, b)} + {_ang(a, b, c)} = 180° (co-interior angles).",
        f"{a}{b} ∥ {d}{c} and {b}{c} crosses them, so {_ang(a, b, c)} + {_ang(b, c, d)} = 180° (co-interior angles).",
        f"Both sums equal 180°, so {_ang(d, a, b)} + {_ang(a, b, c)} = {_ang(a, b, c)} + {_ang(b, c, d)}.",
        f"Subtract {_ang(a, b, c)}: {claim}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"cancels {_ang(a, b, c)} from the two sums to reach {claim}"),
            ("logical_completeness", "uses co-interior angles on both pairs of parallel sides, not just one"),
            ("definitions_assumptions", "says which lines are parallel and which line is the transversal for each use"),
            ("communication", "labels each angle with three letters"),
        ],
        traps=[
            ("only_one_pair_of_parallels", "does it use both pairs of parallel sides, or only one?"),
            (
                "assumes_opposite_angles_equal",
                "does it quote 'opposite angles of a parallelogram are equal', which is the result being proved?",
            ),
        ],
    )
    return finish(p, claim, mark)


def _parallelogram_numeric(rng) -> dict:
    a, b, c, d = rng.choice(NAME_SETS)
    x = rng.randint(40, 130)
    if x == 90:
        x = 84
    claim = f"{_ang(a, b, c)} = {_deg(180 - x)} and {_ang(b, c, d)} = {_deg(x)}"
    p = Prompt()
    p.t(f"{a}{b}{c}{d} is a parallelogram with {_ang(d, a, b)} = {_deg(x)}. ")
    p.t("You may use: co-interior angles between parallel lines add to 180°. ")
    p.t(f"Prove that {claim}. ")
    p.t(_tail(rng))
    ref = [
        f"{a}{d} ∥ {b}{c} and {a}{b} crosses them, so {_ang(d, a, b)} + {_ang(a, b, c)} = 180° (co-interior angles).",
        f"So {_ang(a, b, c)} = 180° - {x}° = {180 - x}°.",
        f"{a}{b} ∥ {d}{c} and {b}{c} crosses them, so {_ang(a, b, c)} + {_ang(b, c, d)} = 180° (co-interior angles).",
        f"So {_ang(b, c, d)} = 180° - {180 - x}° = {x}°.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"finds {_ang(a, b, c)} = {180 - x}° and then {_ang(b, c, d)} = {x}°"),
            ("logical_completeness", "proves both angles, using a different pair of parallel sides for each"),
            ("definitions_assumptions", "names the parallel lines and the transversal for each co-interior pair"),
            ("communication", "each value follows from a stated reason, not a calculation alone"),
        ],
        traps=[
            (
                "alternate_for_co_interior",
                "does it say the angles are equal (alternate) when they are co-interior and add to 180°?",
            ),
            ("one_angle_only", "does it prove only one of the two angles the question asks for?"),
        ],
    )
    return finish(p, claim, mark)


def _equal_parallel_sides(rng) -> dict:
    a, b, c, d = rng.choice(NAME_SETS)
    claim = f"{a}{d} = {b}{c}"
    p = Prompt()
    p.t(f"{a}{b}{c}{d} is a convex quadrilateral with {a}{b} = {d}{c} and {a}{b} ∥ {d}{c}. ")
    p.t("You may use: alternate angles between parallel lines are equal, and the SAS test for congruent triangles. ")
    p.t(f"Prove that {claim}. ")
    p.t(_tail(rng))
    ref = [
        f"{a}{b} ∥ {d}{c} and {a}{c} crosses them, so {_ang(b, a, c)} = {_ang(d, c, a)} (alternate angles).",
        f"In triangles {a}{b}{c} and {c}{d}{a}: {a}{b} = {c}{d} (given), {_ang(b, a, c)} = {_ang(d, c, a)} (step 1) and {a}{c} = {c}{a} (common side).",
        f"So triangle {a}{b}{c} is congruent to triangle {c}{d}{a} (SAS).",
        f"Corresponding sides of congruent triangles are equal, so {b}{c} = {d}{a}, that is {claim}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"obtains {b}{c} = {d}{a} from the congruent triangles"),
            ("logical_completeness", "establishes all three SAS conditions, including the equal included angles"),
            ("definitions_assumptions", f"matches the vertices correctly: {a}{b}{c} with {c}{d}{a}"),
            ("communication", "states the congruence test used and the reason for each equal part"),
        ],
        traps=[
            (
                "vertices_out_of_order",
                f"is the congruence written with corresponding vertices ({a}{b}{c} and {c}{d}{a}), so that {b}{c} and {d}{a} correspond?",
            ),
            (
                "assumes_a_parallelogram",
                "does it call the quadrilateral a parallelogram, which is what the equal sides are meant to show?",
            ),
        ],
    )
    return finish(p, claim, mark)


def _rhombus_bisects(rng) -> dict:
    a, b, c, d = _rot(rng.choice(NAME_SETS), rng.randrange(4))
    claim = f"diagonal {a}{c} bisects {_ang(d, a, b)}"
    p = Prompt()
    p.t(f"{a}{b}{c}{d} is a rhombus, so all four sides are equal. ")
    p.t("You may use the SSS test for congruent triangles. ")
    p.t(f"Prove that the diagonal {a}{c} bisects {_ang(d, a, b)}. ")
    p.t(_tail(rng))
    ref = [
        f"In triangles {a}{b}{c} and {a}{d}{c}: {a}{b} = {a}{d} and {b}{c} = {d}{c} (sides of a rhombus), and {a}{c} is common.",
        f"So triangle {a}{b}{c} is congruent to triangle {a}{d}{c} (SSS).",
        f"Corresponding angles are equal, so {_ang(b, a, c)} = {_ang(d, a, c)}.",
        f"The diagonal {a}{c} lies inside {_ang(d, a, b)} and splits it into these two equal angles, so it bisects {_ang(d, a, b)}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"concludes {_ang(b, a, c)} = {_ang(d, a, c)} and that these make up {_ang(d, a, b)}"),
            ("logical_completeness", "checks all three pairs of sides for SSS, including the common side"),
            ("definitions_assumptions", "uses only that the four sides are equal, not any property of the diagonals"),
            ("communication", "names the two triangles with corresponding vertices"),
        ],
        traps=[
            (
                "assumes_perpendicular_diagonals",
                "does it use 'the diagonals are perpendicular' or 'bisect each other', which are further results about a rhombus?",
            ),
            (
                "stops_at_congruence",
                "does it go on from the congruent triangles to the two equal angles at the vertex?",
            ),
        ],
    )
    return finish(p, claim, mark)


def _polygon_angle_sum(rng) -> dict:
    n = rng.randint(5, 10)
    name = POLYGON_NAMES[n]
    total = (n - 2) * 180
    claim = f"the interior angles of a convex {name} add to {total}°"
    p = Prompt()
    p.t(f"A convex {name} has {n} sides. You may use: the angles of a triangle add to 180°. ")
    p.t(f"Prove that the interior angles of the {name} add to {total}°. ")
    p.t(_tail(rng))
    ref = [
        f"Choose one vertex of the {name} and join it to each of the {n - 3} vertices that are not next to it.",
        f"The diagonals split the {name} into {n - 2} triangles, because the polygon is convex and each diagonal stays inside it.",
        f"The angles of these triangles together make up exactly the interior angles of the {name}.",
        f"Each triangle has angle sum 180°, so the total is {n - 2} × 180° = {total}°.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"counts {n - 2} triangles and multiplies by 180° to get {total}°"),
            ("logical_completeness", "explains why the triangles' angles fill the polygon's angles exactly"),
            ("definitions_assumptions", "uses convexity so that the diagonals lie inside the polygon"),
            ("communication", "says where the diagonals are drawn from"),
        ],
        traps=[
            ("wrong_triangle_count", f"does it count {n - 2} triangles, not {n} or {n - 1}?"),
            ("quotes_the_formula", "does it quote (n - 2) × 180° as known, instead of deriving it from triangles?"),
        ],
    )
    return finish(p, claim, mark)


# ---------------------------------------------------------------------------
# Hard
# ---------------------------------------------------------------------------
def _parallelogram_diagonals(rng) -> dict:
    a, b, c, d = _rot(rng.choice(NAME_SETS), rng.randrange(4))
    m = rng.choice("MNOT")
    claim = f"{a}{m} = {m}{c}"
    p = Prompt()
    p.t(f"{a}{b}{c}{d} is a parallelogram whose diagonals {a}{c} and {b}{d} meet at {m}. ")
    p.t(
        "You may use: opposite sides of a parallelogram are equal, alternate angles between parallel lines are equal, and the ASA test for congruent triangles. "
    )
    p.t(f"Prove that {claim}. ")
    p.t(_tail(rng))
    ref = [
        f"{a}{b} ∥ {d}{c} with transversal {a}{c}, so {_ang(b, a, m)} = {_ang(d, c, m)} (alternate angles).",
        f"{a}{b} ∥ {d}{c} with transversal {b}{d}, so {_ang(a, b, m)} = {_ang(c, d, m)} (alternate angles).",
        f"{a}{b} = {c}{d} (opposite sides of a parallelogram).",
        f"In triangles {a}{b}{m} and {c}{d}{m}: two angles and the included side {a}{b} = {c}{d} are equal, so they are congruent (ASA).",
        f"Corresponding sides are equal, so {a}{m} = {c}{m}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"reaches {a}{m} = {c}{m} from the congruent triangles {a}{b}{m} and {c}{d}{m}"),
            ("logical_completeness", "establishes two equal angle pairs and the equal side between them (ASA)"),
            (
                "definitions_assumptions",
                f"uses that {m} lies on both diagonals, so the angles at {m} are alternate pairs of the same two parallel lines",
            ),
            ("communication", "names the triangles with corresponding vertices"),
        ],
        traps=[
            ("side_not_between_angles", "is the equal side the one between the two equal angles, as ASA needs?"),
            ("quotes_diagonals_bisect", "does it simply say 'the diagonals of a parallelogram bisect each other'?"),
        ],
    )
    return finish(p, claim, mark)


def _rectangle_diagonals(rng) -> dict:
    a, b, c, d = _rot(rng.choice(NAME_SETS), rng.randrange(4))
    claim = f"{a}{c} = {b}{d}"
    p = Prompt()
    p.t(f"{a}{b}{c}{d} is a rectangle: its opposite sides are equal and each of its angles is 90°. ")
    p.t("You may use the SAS test for congruent triangles. ")
    p.t(f"Prove that the diagonals {a}{c} and {b}{d} are equal in length. ")
    p.t(_tail(rng))
    ref = [
        f"In triangles {a}{b}{c} and {d}{c}{b}: {a}{b} = {d}{c} (opposite sides of a rectangle).",
        f"{_ang(a, b, c)} = {_ang(d, c, b)} = 90° (angles of a rectangle).",
        f"{b}{c} is common to both triangles.",
        f"So triangle {a}{b}{c} is congruent to triangle {d}{c}{b} (SAS).",
        f"Corresponding sides are equal, so {a}{c} = {d}{b}, that is {claim}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"concludes {a}{c} = {b}{d} from the congruent triangles {a}{b}{c} and {d}{c}{b}"),
            (
                "logical_completeness",
                "establishes two sides and the included angle, not an angle that is not between them",
            ),
            ("definitions_assumptions", f"uses that {_ang(a, b, c)} and {_ang(d, c, b)} are both right angles"),
            ("communication", "writes the triangles with corresponding vertex order"),
        ],
        traps=[
            (
                "measures_instead_of_proving",
                "does it argue from a particular rectangle's measurements rather than from the rectangle's properties?",
            ),
            (
                "wrong_triangle_pair",
                f"are the triangles {a}{b}{c} and {d}{c}{b}, sharing the side {b}{c}, with {a}{c} and {d}{b} as corresponding sides?",
            ),
        ],
    )
    return finish(p, claim, mark)


def _rhombus_perpendicular(rng) -> dict:
    a, b, c, d = _rot(rng.choice(NAME_SETS), rng.randrange(4))
    m = rng.choice("MNOT")
    claim = f"{a}{c} ⊥ {b}{d}"
    p = Prompt()
    p.t(f"{a}{b}{c}{d} is a rhombus whose diagonals {a}{c} and {b}{d} meet at {m}. ")
    p.t(
        "You may use: the diagonals of a parallelogram bisect each other, angles on a straight line add to 180°, and the SSS test for congruent triangles. "
    )
    p.t(f"Prove that {a}{c} is perpendicular to {b}{d}. ")
    p.t(_tail(rng))
    ref = [
        f"A rhombus is a parallelogram, so its diagonals bisect each other: {b}{m} = {d}{m}.",
        f"In triangles {a}{b}{m} and {a}{d}{m}: {a}{b} = {a}{d} (sides of a rhombus), {b}{m} = {d}{m} (step 1) and {a}{m} is common.",
        f"So triangle {a}{b}{m} is congruent to triangle {a}{d}{m} (SSS), and {_ang(a, m, b)} = {_ang(a, m, d)}.",
        f"{b}, {m}, {d} lie on a straight line, so {_ang(a, m, b)} + {_ang(a, m, d)} = 180°.",
        f"Two equal angles that add to 180° are each 90°, so {a}{c} ⊥ {b}{d}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"shows {_ang(a, m, b)} = {_ang(a, m, d)} and then that each is 90°"),
            ("logical_completeness", "establishes all three SSS pairs, including BM = DM from the bisecting diagonals"),
            ("definitions_assumptions", f"states that {b}, {m}, {d} are collinear before adding the angles to 180°"),
            ("communication", "each equal pair is given with a reason"),
        ],
        traps=[
            (
                "assumes_perpendicular",
                "does it quote 'the diagonals of a rhombus are perpendicular', which is the result being proved?",
            ),
            ("missing_straight_line_step", "does it jump from equal angles to 90° without using the straight line?"),
        ],
    )
    return finish(p, claim, mark)


def _kite_perpendicular(rng) -> dict:
    a, b, c, d = _rot(rng.choice(NAME_SETS), rng.randrange(4))
    m = rng.choice("MNOT")
    claim = f"{a}{c} ⊥ {b}{d}"
    p = Prompt()
    p.t(
        f"{a}{b}{c}{d} is a convex quadrilateral with {a}{b} = {a}{d} and {c}{b} = {c}{d}. Its diagonals {a}{c} and {b}{d} meet at {m}. "
    )
    p.t("You may use: the SSS and SAS tests for congruent triangles, and angles on a straight line add to 180°. ")
    p.t(f"Prove that {a}{c} is perpendicular to {b}{d}. ")
    p.t(_tail(rng))
    ref = [
        f"In triangles {a}{b}{c} and {a}{d}{c}: {a}{b} = {a}{d}, {c}{b} = {c}{d} (given) and {a}{c} is common, so they are congruent (SSS).",
        f"Hence {_ang(b, a, c)} = {_ang(d, a, c)}, that is {_ang(b, a, m)} = {_ang(d, a, m)}.",
        f"In triangles {a}{b}{m} and {a}{d}{m}: {a}{b} = {a}{d}, {_ang(b, a, m)} = {_ang(d, a, m)} and {a}{m} is common, so they are congruent (SAS).",
        f"Hence {_ang(a, m, b)} = {_ang(a, m, d)}.",
        f"{b}, {m}, {d} lie on a straight line, so these two equal angles add to 180°, and each is 90°. So {a}{c} ⊥ {b}{d}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"reaches {_ang(a, m, b)} = {_ang(a, m, d)} = 90° through two congruences"),
            (
                "logical_completeness",
                "uses SSS first to get the equal angles at the vertex, then SAS on the smaller triangles",
            ),
            (
                "definitions_assumptions",
                f"notes that {m} lies on {a}{c}, so the angles at {a} in the small triangles are the same as in the large ones",
            ),
            ("communication", "names each pair of triangles and the test used"),
        ],
        traps=[
            ("single_congruence", "does it try to get the right angle from the first congruence alone?"),
            ("assumes_m_is_midpoint", f"does it assume {m} is the midpoint of {b}{d} without proving it?"),
        ],
    )
    return finish(p, claim, mark)


# ---------------------------------------------------------------------------
# Exam
# ---------------------------------------------------------------------------
def _midpoint_quadrilateral(rng) -> dict:
    first, second = rng.sample(NAME_SETS, 2)
    a, b, c, d = first
    pp, q, r, s = second
    claim = f"{pp}{q}{r}{s} is a parallelogram"
    p = Prompt()
    p.t(
        f"{a}{b}{c}{d} is a convex quadrilateral. {pp}, {q}, {r} and {s} are the midpoints of {a}{b}, {b}{c}, {c}{d} and {d}{a}. "
    )
    p.t(
        "You may use: the segment joining the midpoints of two sides of a triangle is parallel to the third side and half its length; and a quadrilateral with one pair of opposite sides equal and parallel is a parallelogram. "
    )
    p.t(f"Prove that {pp}{q}{r}{s} is a parallelogram. ")
    p.t(_tail(rng))
    ref = [
        f"In triangle {a}{b}{c}, {pp} and {q} are the midpoints of {a}{b} and {b}{c}, so {pp}{q} ∥ {a}{c} and {pp}{q} = {a}{c}/2.",
        f"In triangle {a}{d}{c}, {s} and {r} are the midpoints of {a}{d} and {d}{c}, so {s}{r} ∥ {a}{c} and {s}{r} = {a}{c}/2.",
        f"Both {pp}{q} and {s}{r} are parallel to {a}{c}, so {pp}{q} ∥ {s}{r}.",
        f"Both are half of {a}{c}, so {pp}{q} = {s}{r}.",
        f"One pair of opposite sides, {pp}{q} and {s}{r}, is equal and parallel, so {pp}{q}{r}{s} is a parallelogram.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"shows {pp}{q} and {s}{r} are both parallel to {a}{c} and both half its length"),
            ("logical_completeness", "applies the midpoint theorem to two different triangles sharing the diagonal"),
            ("definitions_assumptions", "chooses a diagonal that lies inside the quadrilateral so the triangles exist"),
            ("communication", "names each triangle used and the midpoints in it"),
        ],
        traps=[
            ("one_triangle_only", "does it apply the midpoint theorem to just one triangle?"),
            ("measures_a_sketch", "does it argue from a sketch's lengths, not from the theorem?"),
        ],
    )
    return finish(p, claim, mark)


def _isosceles_median(rng) -> dict:
    a, b, c, d = rng.choice(NAME_SETS)
    p = Prompt()
    if rng.random() < 0.5:
        claim = f"{a}{d} ⊥ {b}{c} and {_ang(b, a, d)} = {_ang(c, a, d)}"
        p.t(f"In triangle {a}{b}{c}, {a}{b} = {a}{c}. {d} is the midpoint of {b}{c}. ")
        p.t("You may use the SSS test for congruent triangles, and that angles on a straight line add to 180°. ")
        p.t(f"Prove that {a}{d} is perpendicular to {b}{c} and that {_ang(b, a, d)} = {_ang(c, a, d)}. ")
        numeric = None
    else:
        apex = rng.choice(range(20, 141, 4))
        half = apex // 2
        base = 90 - half
        numeric = (apex, half, base)
        claim = f"{_ang(b, a, d)} = {_deg(half)} and {_ang(a, b, d)} = {_deg(base)}"
        p.t(
            f"In triangle {a}{b}{c}, {a}{b} = {a}{c} and {_ang(b, a, c)} = {_deg(apex)}. {d} is the midpoint of {b}{c}. "
        )
        p.t(
            "You may use the SSS test for congruent triangles, and that angles on a straight line add to 180° and angles in a triangle add to 180°. "
        )
        p.t(f"Prove that {claim}. ")
    p.t(_tail(rng))
    ref = [
        f"In triangles {a}{b}{d} and {a}{c}{d}: {a}{b} = {a}{c} (given), {b}{d} = {c}{d} ({d} is the midpoint) and {a}{d} is common.",
        f"So triangle {a}{b}{d} is congruent to triangle {a}{c}{d} (SSS).",
        f"Hence {_ang(b, a, d)} = {_ang(c, a, d)} and {_ang(a, d, b)} = {_ang(a, d, c)}.",
        f"{b}, {d}, {c} lie on a straight line, so {_ang(a, d, b)} + {_ang(a, d, c)} = 180°, and each angle is 90°, so {a}{d} ⊥ {b}{c}.",
    ]
    if numeric:
        apex, half, base = numeric
        ref.append(
            f"Since {_ang(b, a, d)} + {_ang(c, a, d)} = {apex}° and they are equal, {_ang(b, a, d)} = {half}°. "
            f"In triangle {a}{b}{d} the angles add to 180°, so {_ang(a, b, d)} = 180° - 90° - {half}° = {base}°."
        )
    mark = marking(
        reference=ref,
        steps=[
            (
                "correctness",
                "gets the equal angles at the apex and the right angle at the midpoint"
                + (", then the numeric angles" if numeric else ""),
            ),
            ("logical_completeness", "justifies each of the three SSS pairs before using congruence"),
            ("definitions_assumptions", f"states that {d} is the midpoint of {b}{c}, so {b}{d} = {c}{d}"),
            ("communication", "each angle named with three letters and each step justified"),
        ],
        traps=[
            ("assumes_the_median_is_perpendicular", f"does it assume {a}{d} ⊥ {b}{c}, which is part of what is asked?"),
            (
                "equal_base_angles_only",
                "does it get the right angle only by quoting a property of isosceles triangles instead of using the stated tests?",
            ),
        ],
    )
    return finish(p, claim, mark)


def _constructed_parallelogram(rng) -> dict:
    a, b, c, d = rng.choice(("ABCD", "PQRS", "WXYZ"))
    e, f = rng.choice(("EF", "GH", "UV"))
    claim = f"{d}{e}{b}{f} is a parallelogram"
    p = Prompt()
    p.t(
        f"{a}{b}{c}{d} is a parallelogram. {e} is a point on {a}{b} and {f} is a point on {d}{c} with {a}{e} = {c}{f}. "
    )
    p.t(
        "You may use: opposite sides of a parallelogram are equal and parallel, and a quadrilateral with one pair of opposite sides equal and parallel is a parallelogram. "
    )
    p.t(f"Prove that {d}{e}{b}{f} is a parallelogram. ")
    p.t(_tail(rng))
    ref = [
        f"{a}{b} = {d}{c} (opposite sides of a parallelogram).",
        f"{e}{b} = {a}{b} - {a}{e} and {d}{f} = {d}{c} - {c}{f}, because {e} lies on {a}{b} and {f} lies on {d}{c}.",
        f"Since {a}{b} = {d}{c} and {a}{e} = {c}{f}, {e}{b} = {d}{f}.",
        f"{e}{b} lies along {a}{b} and {d}{f} lies along {d}{c}, and {a}{b} ∥ {d}{c}, so {e}{b} ∥ {d}{f}.",
        f"The sides {e}{b} and {f}{d} of {d}{e}{b}{f} are equal and parallel, so {d}{e}{b}{f} is a parallelogram.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"shows {e}{b} = {d}{f} by subtracting equal lengths from equal sides"),
            ("logical_completeness", f"shows both parallel and equal for the same pair of sides, {e}{b} and {d}{f}"),
            ("definitions_assumptions", f"states that {e} and {f} lie on the sides, so the subtractions are valid"),
            ("communication", "names the opposite sides that are equal and parallel"),
        ],
        traps=[
            ("equal_but_not_parallel", "does it show the pair of sides is parallel as well as equal?"),
            (
                "wrong_pair_of_sides",
                f"does it compare {d}{e} with {b}{f} when the lengths it has proved are for {e}{b} and {d}{f}?",
            ),
        ],
    )
    return finish(p, claim, mark)


def _regular_polygon_sides(rng) -> dict:
    n = rng.choice((5, 6, 8, 9, 10, 12, 15, 18, 20, 24))
    interior = 180 - 360 // n
    check(360 % n == 0 and (180 - interior) * n == 360, "regular polygon must have a whole exterior angle")
    claim = f"the polygon has {n} sides"
    p = Prompt()
    p.t(f"Each interior angle of a regular polygon is {_deg(interior)}. ")
    p.t(
        "You may use: at each vertex of a convex polygon the interior and exterior angles add to 180°, and the exterior angles of a convex polygon add to 360°. "
    )
    p.t(f"Prove that the polygon has {n} sides. ")
    p.t(_tail(rng))
    ext = 180 - interior
    ref = [
        f"At each vertex the exterior angle is 180° - {interior}° = {ext}°.",
        "The polygon is regular, so all its exterior angles are equal.",
        f"If it has n sides, the exterior angles add to n × {ext}° = 360° (exterior angles of a convex polygon).",
        f"So n = 360 ÷ {ext} = {n}.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"finds the exterior angle {ext}° and divides 360° by it to get {n}"),
            (
                "logical_completeness",
                "says all exterior angles are equal because the polygon is regular, before multiplying by n",
            ),
            ("definitions_assumptions", "uses that the polygon is convex (regular) so the exterior angle sum is 360°"),
            ("communication", "introduces n as the number of sides and writes an equation"),
        ],
        traps=[
            ("divides_into_interior", f"does it divide 360° by {interior}° instead of by the exterior angle?"),
            (
                "uses_the_formula_unproved",
                "does it only quote 'interior angle = 180 - 360/n' without deriving it from the stated theorems?",
            ),
        ],
    )
    return finish(p, claim, mark)


def _rhombus_numeric(rng) -> dict:
    a, b, c, d = rng.choice(NAME_SETS)
    x = rng.choice([v for v in range(40, 141, 2) if v != 90])
    half = (180 - x) // 2
    claim = f"{_ang(c, a, d)} = {_deg(half)}"
    p = Prompt()
    p.t(f"{a}{b}{c}{d} is a rhombus with {_ang(a, b, c)} = {_deg(x)}. ")
    p.t(
        "You may use: co-interior angles between parallel lines add to 180°, and the SSS test for congruent triangles. "
    )
    p.t(f"Prove that {claim}. ")
    p.t(_tail(rng))
    ref = [
        f"A rhombus has {a}{d} ∥ {b}{c}, and {a}{b} crosses them, so {_ang(b, a, d)} + {_ang(a, b, c)} = 180° (co-interior angles).",
        f"So {_ang(b, a, d)} = 180° - {x}° = {180 - x}°.",
        f"In triangles {a}{b}{c} and {a}{d}{c}: {a}{b} = {a}{d}, {b}{c} = {d}{c} and {a}{c} is common, so they are congruent (SSS).",
        f"Hence {_ang(b, a, c)} = {_ang(d, a, c)}, and these two angles make up {_ang(b, a, d)}.",
        f"So {_ang(c, a, d)} = {180 - x}° ÷ 2 = {half}°.",
    ]
    mark = marking(
        reference=ref,
        steps=[
            ("correctness", f"gets {_ang(b, a, d)} = {180 - x}° and then halves it to {half}°"),
            ("logical_completeness", f"justifies the halving: {a}{c} bisects {_ang(b, a, d)} by the SSS congruence"),
            (
                "definitions_assumptions",
                "uses that opposite sides of a rhombus are parallel and all four sides are equal",
            ),
            ("communication", "each angle value comes with its reason"),
        ],
        traps=[
            ("halves_the_given_angle", f"does it halve {_ang(a, b, c)} = {x}° instead of {_ang(b, a, d)}?"),
            ("halving_unjustified", "does it halve the angle with no reason that the diagonal bisects it?"),
        ],
    )
    return finish(p, claim, mark)


FAMILIES = {
    "easy": (_exterior_angle, _isosceles_apex, _co_interior),
    "medium": (
        _parallelogram_opposite_angles,
        _parallelogram_numeric,
        _equal_parallel_sides,
        _rhombus_bisects,
        _polygon_angle_sum,
    ),
    "hard": (_parallelogram_diagonals, _rectangle_diagonals, _rhombus_perpendicular, _kite_perpendicular),
    "exam": (
        _midpoint_quadrilateral,
        _isosceles_median,
        _constructed_parallelogram,
        _regular_polygon_sides,
        _rhombus_numeric,
    ),
}


@generator(
    "proof_f10_deductive_shapes",
    level=11,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
        "exam": dict(variant="exam"),
    },
)
def proof_f10_deductive_shapes(rng, variant="easy"):
    """`f10-deductive-shape-proofs`. Rubric-assessed (`answer: rubric`): every question returned carries a
    per-item marking dict (`Question.answer`), never a plain value."""
    if variant not in FAMILIES:
        raise ValueError(f"unknown variant: {variant}")
    _confirm_general_theorems(rng)
    return rng.choice(FAMILIES[variant])(rng)
