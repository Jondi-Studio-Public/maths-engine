"""Generates `fy1ms_lines_planes` questions and the exam chain `exd_x_fy1ms_lines_planes`.

Skill: `ucv1-lines-and-planes-3d` (vectors). The plane through three points, where a line meets a plane, and the angle
between two planes or between a line and a plane.

Tiers (topic `fy1ms_lines_planes`; every answer is a number):

* easy   -- the constant d of the plane ax + by + cz = d through three points, with (a, b, c) = AB x AC;
* medium -- where a line r = A + t u meets a plane ax + by + cz = k: the value of t, or one coordinate of the point;
* hard   -- the acute angle between two planes given by their equations, or between a line and a plane (degrees,
            1 decimal place); the equation's constant must be left out of the normal vector.

Exam (topic `exd_x_fy1ms_lines_planes`): a laser beam and a flat mirror. (a) the normal AB x AC (a vector, 2 marks),
(b) the constant d of the mirror's plane (1 mark), (c) the value of t at which the beam meets the mirror (2 marks,
follow-through from (b)), (d) one coordinate of that point (1 mark, follow-through from (c)).

Every key is computed from the formulas and checked a second way with SymPy: the plane is built from a determinant
(a 3 x 3 SymPy `Matrix.det`), the intersection is found by solving the equation for t, and each angle is recomputed from
`acos`/`asin` of a normalised dot product.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator, join_terms
from ._fy1ms_common import (
    Fm,
    build_until,
    clear_of_tie,
    dp_text,
    frac_dist,
    num_dist,
    number_part,
    number_question,
    say,
)
from ._fy1ms_common import cross3 as _cross
from ._fy1ms_common import dot3 as _dot
from ._fy1ms_common import plane_text as _plane
from ._fy1ms_common import sub3 as _sub
from ._fy1ms_common import vec_text as _v


def _pt(name: str, vec) -> str:
    return f"{name}{_v(vec)}"


def _line(a, u) -> Fm:
    return Fm(f"r = {_v(a)} + t{_v(u)}", rf"\mathbf{{r}} = {_v(a)} + t{_v(u)}")


def _draw_points(rng: random.Random, span: int = 4):
    for _ in attempts(500, "three non-collinear points"):
        a, b, c = ([rng.randint(-span, span) for _ in range(3)] for _ in range(3))
        n = _cross(_sub(b, a), _sub(c, a))
        if any(n) and max(abs(x) for x in n) <= 30 and len({tuple(a), tuple(b), tuple(c)}) == 3:
            return a, b, c, n
    raise AssertionError("unreachable")  # pragma: no cover


def _plane_through(a, b, c, n):
    """d for ax + by + cz = d, with a second route: all three points satisfy the equation, and SymPy agrees on n."""
    d = _dot(n, a)
    check(_dot(n, b) == d and _dot(n, c) == d, "the three points are not on the plane")
    x, y, z = sympy.symbols("x y z")
    det = sympy.Matrix([[x - a[0], y - a[1], z - a[2]], list(_sub(b, a)), list(_sub(c, a))]).det()
    check(sympy.expand(det - (n[0] * x + n[1] * y + n[2] * z - d)) == 0, "determinant form disagrees with AB x AC")
    return d


# ---------------------------------------------------------------------------
# Easy: the constant of the plane through three points


def _easy(rng: random.Random) -> dict:
    a, b, c, n = _draw_points(rng)
    ab, ac = _sub(b, a), _sub(c, a)
    d = _plane_through(a, b, c, n)
    slip = [n[0], -n[1], n[2]]
    swapped = [n[0], n[2], n[1]]
    componentwise = [ab[0] * ac[0], ab[1] * ac[1], ab[2] * ac[2]]
    cands = [
        num_dist(-d, 0, "plane-constant-sign"),
        num_dist(_dot(slip, a), 0, "cross-product-middle-sign-slip"),
        num_dist(_dot(componentwise, a), 0, "cross-product-componentwise"),
        num_dist(_dot(swapped, a), 0, "cross-product-components-swapped"),
        num_dist(3 * d, 0, "plane-constant-sum-of-points"),
    ]
    text, latex = say(
        rng.choice(["The plane through ", "The plane passing through ", "A plane through "]),
        Fm(_pt("A", a)),
        ", ",
        Fm(_pt("B", b)),
        " and ",
        Fm(_pt("C", c)),
        " has equation ",
        Fm("ax + by + cz = d"),
        ", where ",
        Fm("(a, b, c) = AB × AC", r"(a, b, c) = \overrightarrow{AB} \times \overrightarrow{AC}"),
        ". Find ",
        Fm("d"),
        ".",
    )
    return number_question(
        text,
        latex,
        d,
        0,
        cands,
        working=[
            f"AB = {_v(ab)} and AC = {_v(ac)}, so (a, b, c) = AB × AC = {_v(n)}.",
            f"d = (a, b, c) · A = {_dot(n, a)}.",
        ],
    )


# ---------------------------------------------------------------------------
# Medium: where a line meets a plane


def _draw_intersection(rng: random.Random, *, t_values=(-3, -2, -1, 1, 2, 3)):
    for _ in attempts(3000, "a line that meets the plane at a whole-number t"):
        n = [rng.choice((-3, -2, -1, 1, 2, 3)) for _ in range(3)]
        a = [rng.randint(-4, 4) for _ in range(3)]
        u = [rng.choice((-3, -2, -1, 1, 2, 3)) for _ in range(3)]
        t0 = rng.choice(t_values)
        if _dot(n, u) == 0:
            continue
        point = [a[i] + t0 * u[i] for i in range(3)]
        k = _dot(n, point)
        if _dot(n, a) == k:
            continue
        if math.gcd(math.gcd(abs(n[0]), abs(n[1])), abs(n[2])) != 1:
            continue
        return n, a, u, t0, point, k
    raise AssertionError("unreachable")  # pragma: no cover


def _solve_t(n, a, u, k):
    """t by SymPy: substitute the line into the plane and solve."""
    t = sympy.symbols("t")
    sol = sympy.solve(sympy.Eq(_dot(n, [a[i] + t * u[i] for i in range(3)]), k), t)
    check(len(sol) == 1, "the line does not meet the plane in exactly one point")
    return sol[0]


def _medium(rng: random.Random) -> dict:
    n, a, u, t0, point, k = _draw_intersection(rng)
    check(_solve_t(n, a, u, k) == t0, "t disagrees with SymPy")
    na, nu = _dot(n, a), _dot(n, u)
    wrong_t = {
        "line-plane-sign-in-numerator": Fraction(k + na, nu),
        "line-plane-not-divided-by-direction-dot": Fraction(k - na),
        "line-plane-ratio-inverted": Fraction(nu, k - na),
        "line-plane-point-term-forgotten": Fraction(k, nu),
    }
    ask = rng.choice(("t", "t", "x", "y", "z"))
    if ask == "t":
        answer = t0
        cands = [frac_dist(v, tag) for tag, v in wrong_t.items()]
        question = "Find the value of t at the point where L meets the plane."
    else:
        axis = "xyz".index(ask)
        answer = point[axis]
        cands = [frac_dist(a[axis] + v * u[axis], tag) for tag, v in wrong_t.items()]
        question = f"Find the {ask}-coordinate of the point where L meets the plane."
    text, latex = say(
        "The line L has equation ",
        _line(a, u),
        ", and a plane has equation ",
        Fm(_plane(n, k)),
        ". ",
        question,
    )
    return number_question(
        text,
        latex,
        answer,
        0,
        cands,
        working=[
            f"On L, {join_terms([(n[0], 'x'), (n[1], 'y'), (n[2], 'z')])} = {join_terms([(na, ''), (nu, 't')])}, so {join_terms([(na, ''), (nu, 't')])} = {k} and t = {t0}.",
            f"The point of intersection is {_v(point)}.",
        ],
    )


# ---------------------------------------------------------------------------
# Hard: angles


def _norm(v) -> float:
    return math.sqrt(_dot(v, v))


def _deg(x: float) -> float:
    return math.degrees(x)


def _hard_planes(rng: random.Random) -> dict:
    for _ in attempts(3000, "two planes at a clear angle"):
        n1 = [rng.randint(-3, 3) for _ in range(3)]
        n2 = [rng.randint(-3, 3) for _ in range(3)]
        k1, k2 = rng.randint(-6, 6), rng.randint(-6, 6)
        if (
            not any(n1)
            or not any(n2)
            or _dot(n1, n2) == 0
            or any(n1[i] * n2[j] - n1[j] * n2[i] for i in range(3) for j in range(i + 1, 3)) == 0
        ):
            continue
        if not _cross(n1, n2) or not any(_cross(n1, n2)):
            continue
        c = abs(_dot(n1, n2)) / (_norm(n1) * _norm(n2))
        theta = _deg(math.acos(min(1.0, c)))
        if not (10 < theta < 80) or not clear_of_tie(theta, 1) or k1 == 0 or k2 == 0:
            continue
        break
    # second route: SymPy's angle between the normal vectors
    sym = sympy.Matrix(n1).dot(sympy.Matrix(n2)) / (sympy.Matrix(n1).norm() * sympy.Matrix(n2).norm())
    check(abs(float(sympy.acos(abs(sym))) * 180 / math.pi - theta) < 1e-9, "angle disagrees with SymPy")
    four = abs(_dot(n1, n2) + k1 * k2) / (math.sqrt(_dot(n1, n1) + k1 * k1) * math.sqrt(_dot(n2, n2) + k2 * k2))
    cands = [
        num_dist(180 - theta, 1, "angle-between-planes-supplement"),
        num_dist(90 - theta, 1, "angle-between-planes-sine-used"),
        num_dist(_deg(math.acos(min(1.0, four))), 1, "angle-between-planes-constants-included"),
        num_dist(math.radians(theta), 1, "angle-given-in-radians"),
    ]
    text, latex = say(
        "The planes ",
        Fm(_plane(n1, k1)),
        " and ",
        Fm(_plane(n2, k2)),
        " meet along a line. Find the acute angle between the planes, in degrees, correct to 1 decimal place.",
    )
    return number_question(
        text,
        latex,
        theta,
        1,
        cands,
        working=[
            f"The normals are {_v(n1)} and {_v(n2)}; cos θ = |{_dot(n1, n2)}|/(√{_dot(n1, n1)} × √{_dot(n2, n2)}).",
            f"θ = {dp_text(theta, 1)}°.",
        ],
    )


def _hard_line_plane(rng: random.Random) -> dict:
    for _ in attempts(3000, "a line at a clear angle to a plane"):
        n = [rng.randint(-3, 3) for _ in range(3)]
        u = [rng.randint(-3, 3) for _ in range(3)]
        a = [rng.randint(-4, 4) for _ in range(3)]
        k = rng.randint(-8, 8)
        if not any(n) or not any(u) or _dot(n, u) == 0 or k == 0:
            continue
        s = abs(_dot(n, u)) / (_norm(n) * _norm(u))
        phi = _deg(math.asin(min(1.0, s)))
        if not (10 < phi < 80) or not clear_of_tie(phi, 1):
            continue
        break
    sym = sympy.Matrix(n).dot(sympy.Matrix(u)) / (sympy.Matrix(n).norm() * sympy.Matrix(u).norm())
    check(abs(float(sympy.asin(abs(sym))) * 180 / math.pi - phi) < 1e-9, "angle disagrees with SymPy")
    cands = [
        num_dist(90 - phi, 1, "line-plane-angle-cosine-used"),
        num_dist(_deg(math.atan(s)), 1, "line-plane-angle-tangent-used"),
        num_dist(math.radians(phi), 1, "angle-given-in-radians"),
        num_dist(180 - phi, 1, "angle-between-planes-supplement"),
    ]
    text, latex = say(
        "The line ",
        _line(a, u),
        " meets the plane ",
        Fm(_plane(n, k)),
        ". Find the acute angle between the line and the plane, in degrees, correct to 1 decimal place.",
    )
    return number_question(
        text,
        latex,
        phi,
        1,
        cands,
        working=[
            f"With direction {_v(u)} and normal {_v(n)}: sin φ = |{_dot(n, u)}|/(√{_dot(n, n)} × √{_dot(u, u)}).",
            f"φ = {dp_text(phi, 1)}°.",
        ],
    )


@generator(
    "fy1ms_lines_planes",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1ms_lines_planes(rng: random.Random, tier: str = "easy") -> dict:
    """Lines and planes in 3D: the plane through three points (easy), a line meeting a plane (medium), an angle (hard)."""
    if tier == "easy":
        return build_until(rng, _easy, "an easy plane question")
    if tier == "medium":
        return build_until(rng, _medium, "a medium line-and-plane question")
    if tier == "hard":
        return build_until(
            rng,
            lambda r: _hard_planes(r) if r.random() < 0.5 else _hard_line_plane(r),
            "a hard line-and-plane question",
        )
    raise AssertionError(f"unknown tier {tier!r}")


# ---------------------------------------------------------------------------
# Exam chain: a laser beam and a mirror


@generator("exd_x_fy1ms_lines_planes", level=19, presets={"exam": dict(tier="exam")})
def exd_x_fy1ms_lines_planes(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the normal, the constant of the plane, the parameter where a line meets it, a coordinate (6 marks)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return build_until(rng, _exam_chain, "a lines-and-planes exam question")


def _exam_chain(rng: random.Random) -> dict:
    p, q, r, n = _draw_points(rng, span=3)
    pq, pr = _sub(q, p), _sub(r, p)
    d = _plane_through(p, q, r, n)
    for _ in attempts(3000, "a beam that meets the mirror at a whole-number parameter"):
        i, j = rng.randint(-2, 2), rng.randint(-2, 2)
        meeting = [p[m] + i * pq[m] + j * pr[m] for m in range(3)]  # a point of the plane
        u = [rng.choice((-3, -2, -1, 1, 2, 3)) for _ in range(3)]
        t0 = rng.choice((-3, -2, -1, 1, 2, 3, 4))
        a = [meeting[m] - t0 * u[m] for m in range(3)]
        nu, na = _dot(n, u), _dot(n, a)
        if nu != 0 and max(abs(c) for c in a) <= 12:
            break
    point = [a[m] + t0 * u[m] for m in range(3)]
    check(point == meeting and _dot(n, point) == d, "intersection is not on the plane")
    check(_solve_t(n, a, u, d) == t0, "t disagrees with SymPy")
    axis = rng.randrange(3)
    name = "xyz"[axis]
    # (a) the normal as a vector
    n_text = _v(n)
    a_wrong = [
        ([-c for c in n], "cross-product-reversed-order"),
        ([n[0], -n[1], n[2]], "cross-product-middle-sign-slip"),
        ([pq[i] * pr[i] for i in range(3)], "cross-product-componentwise"),
        ([n[0], n[2], n[1]], "cross-product-components-swapped"),
    ]
    a_dist, seen = [], {n_text}
    for w, tag in a_wrong:
        if _v(w) not in seen:
            seen.add(_v(w))
            a_dist.append(Distractor(_v(w), tag, list(w)))
    # (b) d
    b_cands = [
        num_dist(-d, 0, "plane-constant-sign"),
        num_dist(_dot(a_wrong[1][0], p), 0, "cross-product-middle-sign-slip"),
        num_dist(_dot(a_wrong[3][0], p), 0, "cross-product-components-swapped"),
        num_dist(3 * d, 0, "plane-constant-sum-of-points"),
    ]
    # (c) t
    c_cands = [
        frac_dist(Fraction(d + na, nu), "line-plane-sign-in-numerator"),
        frac_dist(Fraction(d - na), "line-plane-not-divided-by-direction-dot"),
        frac_dist(Fraction(nu, d - na), "line-plane-ratio-inverted"),
        frac_dist(Fraction(d, nu), "line-plane-point-term-forgotten"),
    ]
    # (d) a coordinate
    d_cands = [
        frac_dist(a[axis] + Fraction(wrong) * u[axis], tag)
        for wrong, tag in [
            (Fraction(d + na, nu), "line-plane-sign-in-numerator"),
            (Fraction(d - na), "line-plane-not-divided-by-direction-dot"),
            (Fraction(nu, d - na), "line-plane-ratio-inverted"),
            (Fraction(d, nu), "line-plane-point-term-forgotten"),
        ]
    ]
    stem, stem_latex = say(
        "A flat mirror lies in the plane through ",
        Fm(_pt("P", p)),
        ", ",
        Fm(_pt("Q", q)),
        " and ",
        Fm(_pt("R", r)),
        ". A laser beam travels along the line ",
        _line(a, u),
        ". All distances are in metres.",
    )
    a_part = dict(
        prompt="Find the vector n = PQ × PR.",
        prompt_latex=r"\text{Find the vector }\mathbf{n} = \overrightarrow{PQ} \times \overrightarrow{PR}\text{.}",
        answer_kind="ordered_list",
        answer=list(n),
        answer_text=n_text,
        marks=2,
        working=[f"PQ = {_v(pq)} and PR = {_v(pr)}, so n = {n_text}."],
        distractors=a_dist,
    )
    check(len(a_dist) >= 2, "too few normal-vector distractors")
    return dict(
        text=stem,
        latex=stem_latex,
        parts=[
            a_part,
            number_part(
                f"Hence find the value of d in the equation of the mirror's plane, {join_terms([(n[0], 'x'), (n[1], 'y'), (n[2], 'z')])} = d.",
                d,
                0,
                b_cands,
                marks=1,
                working=[f"d = n · P = {_dot(n, p)}."],
            ),
            number_part(
                "Find the value of t at which the laser beam meets the mirror.",
                t0,
                0,
                c_cands,
                marks=2,
                follow_through=f"(part_b - ({na}))/({nu})",
                working=[f"n · (A + tu) = {join_terms([(na, ''), (nu, 't')])} = {d}, so t = {t0}."],
            ),
            number_part(
                f"Hence find the {name}-coordinate of the point where the beam meets the mirror.",
                point[axis],
                0,
                d_cands,
                marks=1,
                follow_through=f"({a[axis]}) + part_c*({u[axis]})",
                working=[f"{name} = {a[axis]} + t × {u[axis]} with t = {t0}, so {name} = {point[axis]}."],
            ),
        ],
    )
