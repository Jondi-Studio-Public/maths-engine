"""Generates `s2b3_projection` questions (easy, medium, hard) and the exam chain `exd_x_s2b3_projection`.

``uc2-projection-onto-subspace`` (``linear-algebra``, answer ``matrix``): orthogonal projection onto a line or a plane
in R^3 or R^4, the projection matrix, and the distance from a point to the subspace.

* easy: the projection of a vector onto a line through the origin in R^3 or R^4.
* medium: the projection of a vector onto a plane spanned by two NON-orthogonal vectors (R^3 or R^4), so the two line
  projections cannot simply be added.
* hard: the projection matrix P, either of a plane spanned by two vectors of R^3, or of a hyperplane n . x = 0 of R^3 or
  R^4 given by its equation.
* exam (``exd_x_s2b3_projection``): a plane in R^3 and a vector v. (a) the projection p of v, (b) |v - p|^2, (c) the
  exact distance from v to the plane (follows through from (b)), (d) the exact sine of the angle between v and the plane
  (follows through from (b)).

Every answer is worked out in exact fractions, and re-derived with SymPy as A (A^T A)^-1 A^T applied to the vector.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms, phrase
from ._s2b3_kit import (
    dot,
    etext,
    expression_answer,
    expression_distractors,
    fraction_answer,
    fraction_distractors,
    frac,
    mat_answer,
    mat_text,
    matrix_distractors,
    mat_vec,
    need,
    sizes_ok,
    to_sym,
    vec_answer,
    vec_text,
    vscale,
    vsub,
)
from ._ucv1_linalg import inverse, mmul, rank, transpose

_SPACE = {3: "ℝ³", 4: "ℝ⁴"}


def _fm(rows):
    return [[Fraction(v) for v in row] for row in rows]


def _vec(rng, dim, lo=-3, hi=3):
    return [rng.randint(lo, hi) for _ in range(dim)]


def _line_projection(v, a):
    return vscale(dot(v, a) / dot(a, a), a)


def _plane_projection(v, a1, a2):
    """The projection of v onto span{a1, a2} by the normal equations A^T A x = A^T v."""
    a = transpose(_fm([a1, a2]))  # columns a1, a2
    at = _fm([a1, a2])
    gram = mmul(at, a)
    x = mat_vec(inverse(gram), mat_vec(at, v))
    return [x[0] * p + x[1] * q for p, q in zip(a1, a2)], x


def _sympy_plane_projection(v, a1, a2):
    a = to_sym([a1, a2]).T
    p = a * (a.T * a).inv() * a.T * to_sym(list(v))
    return [frac(e) for e in p]


def _independent(a1, a2):
    return rank(_fm([a1, a2])) == 2


# -- easy -------------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "Find the orthogonal projection of v = {v} onto the line through the origin in {space} spanned by a = {a}.",
    "The line L in {space} passes through the origin with direction a = {a}. Find the vector in L closest to v = {v}.",
)


def _easy(rng):
    dim = rng.choice((3, 3, 4))
    for _ in attempts(3000, "a vector and a line with a readable projection"):
        a, v = _vec(rng, dim), _vec(rng, dim)
        if sum(1 for x in a if x) < 2 or sum(1 for x in v if x) < 2:
            continue
        p = _line_projection(v, a)
        if all(x == 0 for x in p) or p == [Fraction(x) for x in v] or not sizes_ok(p, 30, 30):
            continue
        break
    # Second route: SymPy projection by the matrix a a^T / (a^T a).
    sa = to_sym(a)
    check([frac(e) for e in sa * sa.T / (sa.T * sa)[0] * to_sym(v)] == p, "SymPy's projection differs")
    check(dot(vsub(v, p), a) == 0, "v - p is not orthogonal to a")
    c = dot(v, a) / dot(a, a)
    cands = matrix_distractors(
        [
            (_line_projection(a, v), "proj-onto-the-wrong-vector"),
            (vscale(dot(v, a), a), "proj-forgot-divide-by-norm-squared"),
            (vsub(v, p), "proj-gives-the-perpendicular-part"),
            (vscale(c, v), "proj-scales-v-not-a"),
        ],
        p,
        vector=True,
    )
    return dict(
        text=phrase(rng, _EASY_STEMS, v=vec_text(v), a=vec_text(a), space=_SPACE[dim]),
        distractors=need(cands),
        working=[
            f"The projection is (v·a / a·a) a = ({dot(v, a)}/{dot(a, a)}) a.",
            f"So the projection is {vec_text(p)}.",
        ],
        **vec_answer(p),
    )


# -- medium -----------------------------------------------------------------------------------------------------

_PLANE_STEMS = (
    "Let W be the plane in {space} spanned by a1 = {a1} and a2 = {a2}. Find the orthogonal projection of v = {v} onto "
    "W.",
    "Find the point of the plane spanned by a1 = {a1} and a2 = {a2} in {space} that is closest to v = {v}.",
)


def _draw_plane(rng, dim, den=30):
    for _ in attempts(4000, "a plane and a vector with a readable projection"):
        a1, a2, v = _vec(rng, dim), _vec(rng, dim), _vec(rng, dim)
        if not _independent(a1, a2) or dot(a1, a2) == 0 or dot(v, a1) == 0 or dot(v, a2) == 0:
            continue
        if any(sum(1 for x in w if x) < 2 for w in (a1, a2, v)):
            continue
        p, x = _plane_projection(v, a1, a2)
        if all(e == 0 for e in p) or p == [Fraction(e) for e in v] or not sizes_ok(p, 30, den):
            continue
        if x[0] == 0 or x[1] == 0 or any(a == 0 and b == 0 for a, b in zip(a1, a2)):
            continue
        return a1, a2, v, p
    raise AssertionError  # pragma: no cover


def _plane_errors(v, a1, a2, p):
    p1, p2 = _line_projection(v, a1), _line_projection(v, a2)
    return [
        ([x + y for x, y in zip(p1, p2)], "proj-adds-line-projections"),
        (vsub(v, p), "proj-gives-the-perpendicular-part"),
        (p1, "proj-onto-one-spanning-vector"),
        (p2, "proj-onto-one-spanning-vector"),
    ]


def _medium(rng):
    dim = rng.choice((3, 3, 4))
    a1, a2, v, p = _draw_plane(rng, dim)
    check(_sympy_plane_projection(v, a1, a2) == p, "SymPy's projection differs")
    check(dot(vsub(v, p), a1) == 0 and dot(vsub(v, p), a2) == 0, "v - p is not orthogonal to the plane")
    cands = matrix_distractors(_plane_errors(v, a1, a2, p), p, vector=True)
    return dict(
        text=phrase(rng, _PLANE_STEMS, a1=vec_text(a1), a2=vec_text(a2), v=vec_text(v), space=_SPACE[dim]),
        distractors=need(cands),
        working=[
            "a1 and a2 are not orthogonal, so solve the normal equations AᵀA x = Aᵀv for the coefficients of a1 and a2.",
            f"The projection is {vec_text(p)}.",
        ],
        **vec_answer(p),
    )


# -- hard: the projection matrix --------------------------------------------------------------------------------


def _identity(n):
    return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]


def _outer_over(u):
    """u u^T / (u . u)."""
    n2 = dot(u, u)
    return [[Fraction(x) * Fraction(y) / n2 for y in u] for x in u]


def _msub(a, b):
    return [[x - y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def _madd(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def _check_projection_matrix(p, dim, rank_wanted):
    sp = to_sym(p)
    check(sp * sp == sp and sp == sp.T, "P is not a symmetric idempotent matrix")
    check(sp.trace() == rank_wanted, "trace P is not the dimension of the subspace")


_MATRIX_STEMS = (
    "Find the matrix P of the orthogonal projection of {space} onto the plane spanned by a1 = {a1} and a2 = {a2}.",
    "Let W be the subspace of {space} spanned by {a1} and {a2}. Find the projection matrix P, so that Pv is the closest point of W to v.",
)
_HYPERPLANE_STEMS = (
    "Let W be the set of all points x in {space} with {equation} = 0. Find the matrix P of the orthogonal projection of {space} onto W.",
    "The subspace W of {space} is given by {equation} = 0. Find the matrix of the orthogonal projection onto W.",
)


def _hard_plane(rng):
    for _ in attempts(4000, "a plane in R^3 with a readable projection matrix"):
        a1, a2 = _vec(rng, 3), _vec(rng, 3)
        if not _independent(a1, a2) or dot(a1, a2) == 0 or any(sum(1 for x in w if x) < 2 for w in (a1, a2)):
            continue
        n = [a1[1] * a2[2] - a1[2] * a2[1], a1[2] * a2[0] - a1[0] * a2[2], a1[0] * a2[1] - a1[1] * a2[0]]
        pm = _msub(_identity(3), _outer_over(n))
        if sum(1 for x in n if x) < 3 or dot(n, n) > 30 or any(a == 0 and b == 0 for a, b in zip(a1, a2)):
            continue
        break
    # Second route: P = A (A^T A)^-1 A^T in SymPy.
    a = to_sym([a1, a2]).T
    sym_p = a * (a.T * a).inv() * a.T
    check([[frac(e) for e in sym_p.row(i)] for i in range(3)] == pm, "SymPy's projection matrix differs")
    _check_projection_matrix(pm, 3, 2)
    at = _fm([a1, a2])
    a_at = mmul(transpose(at), at)
    sum_of_lines = _madd(_outer_over(a1), _outer_over(a2))
    cands = matrix_distractors(
        [
            (_outer_over(n), "proj-matrix-onto-the-complement"),
            (a_at, "proj-matrix-forgot-the-inverse"),
            (sum_of_lines, "proj-adds-line-projections"),
        ],
        pm,
    )
    return dict(
        text=phrase(rng, _MATRIX_STEMS, a1=vec_text(a1), a2=vec_text(a2), space=_SPACE[3]),
        distractors=need(cands),
        working=["P = A(AᵀA)⁻¹Aᵀ with A the matrix whose columns are a1 and a2.", f"P = {mat_text(pm)}."],
        **mat_answer(pm),
    )


def _hard_hyperplane(rng):
    dim = rng.choice((3, 4))
    for _ in attempts(4000, "a hyperplane with a readable projection matrix"):
        n = _vec(rng, dim)
        if sum(1 for x in n if x) < 3 or all(abs(x) == 1 for x in n if x):
            continue
        pm = _msub(_identity(dim), _outer_over(n))
        if not sizes_ok(pm, 60, 40):
            continue
        break
    # Second route: the null space of n^T has dimension dim - 1; P is the symmetric idempotent that fixes it.
    sn = to_sym(n)
    basis = sn.T.nullspace()
    sp = to_sym(pm)
    check(all(sp * b == b for b in basis) and sp * sn == sympy.zeros(dim, 1), "P does not fix W or kill n")
    _check_projection_matrix(pm, dim, dim - 1)
    outer = [[Fraction(x * y) for y in n] for x in n]
    cands = matrix_distractors(
        [
            (_outer_over(n), "proj-matrix-onto-the-complement"),
            (outer, "proj-matrix-forgot-divide-by-norm-squared"),
            (_msub(_identity(dim), outer), "proj-matrix-forgot-divide-by-norm-squared"),
        ],
        pm,
    )
    names = "x1 x2 x3 x4".split()[:dim]
    equation = join_terms(list(zip(n, names)))
    return dict(
        text=phrase(rng, _HYPERPLANE_STEMS, space=_SPACE[dim], equation=equation),
        distractors=need(cands),
        working=[
            f"W is the set of vectors orthogonal to n = {vec_text(n)}, so P = I - nnᵀ/(n·n).",
            f"P = {mat_text(pm)}.",
        ],
        **mat_answer(pm),
    )


@generator(
    "s2b3_projection",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2b3_projection(rng, tier="easy"):
    """Orthogonal projection onto a line or plane, and the projection matrix; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice((_hard_plane, _hard_hyperplane))(rng)


_EXAM_STEMS = (
    "Let W be the plane in ℝ³ spanned by a1 = {a1} and a2 = {a2}, and let v = {v}.",
    "In ℝ³, the plane W is spanned by a1 = {a1} and a2 = {a2}. Let v = {v}.",
    "Consider the vector v = {v} and the plane W in ℝ³ that is spanned by a1 = {a1} and a2 = {a2}.",
)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_s2b3_projection", level=18, presets={"exam": dict(tier="exam")})
def exd_x_s2b3_projection(rng, tier="exam"):
    """A plane in R^3 and a vector v: the projection, the squared and exact distance, and the sine of the angle."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(4000, "a plane and a vector with a readable distance"):
        a1, a2, v, p = _draw_plane(rng, 3, den=24)
        n = [a1[1] * a2[2] - a1[2] * a2[1], a1[2] * a2[0] - a1[0] * a2[2], a1[0] * a2[1] - a1[1] * a2[0]]
        nv = dot(n, v)
        if nv == 0:
            continue
        dist_sq = nv * nv / dot(n, n)
        v_sq = dot(v, v)
        if dist_sq >= v_sq or dist_sq < 1:
            continue
        break
    check(_sympy_plane_projection(v, a1, a2) == p, "SymPy's projection differs")
    r = vsub(v, p)
    check(dot(r, r) == dist_sq, "the distance formula |n . v|/|n| disagrees with |v - p|")
    p_sq = dot(p, p)
    check(p_sq + dist_sq == v_sq, "Pythagoras fails")
    dist = sympy.sqrt(sympy.Rational(dist_sq.numerator, dist_sq.denominator))
    sine = sympy.sqrt(sympy.Rational(dist_sq.numerator, dist_sq.denominator) / sympy.Integer(int(v_sq)))
    cos_ = sympy.sqrt(sympy.Rational(p_sq.numerator, p_sq.denominator) / sympy.Integer(int(v_sq)))
    p1, p2 = _line_projection(v, a1), _line_projection(v, a2)
    wrong_p = [x + y for x, y in zip(p1, p2)]
    wrong_r = vsub(v, wrong_p)
    wrong_sq = dot(wrong_r, wrong_r)
    rat = lambda x: sympy.Rational(x.numerator, x.denominator)  # noqa: E731

    a_dist = matrix_distractors(_plane_errors(v, a1, a2, p), p, vector=True, limit=3)
    b_dist = fraction_distractors(
        dist_sq,
        [
            (p_sq, "proj-distance-is-the-projection-length"),
            (v_sq, "proj-forgot-to-subtract-the-projection"),
            (wrong_sq, "proj-adds-line-projections"),
            (v_sq + p_sq, "proj-pythagoras-with-the-wrong-sign"),
        ],
        positive=True,
    )
    c_dist = expression_distractors(
        dist,
        [
            (sympy.sqrt(rat(p_sq)), "proj-distance-is-the-projection-length"),
            (rat(dist_sq), "proj-forgot-the-square-root"),
            (sympy.sqrt(rat(wrong_sq)), "proj-adds-line-projections"),
            (sympy.sqrt(rat(v_sq)), "proj-forgot-to-subtract-the-projection"),
        ],
    )
    d_dist = expression_distractors(
        sine,
        [
            (cos_, "proj-cosine-for-sine"),
            (rat(dist_sq) / sympy.Integer(int(v_sq)), "proj-forgot-the-square-root"),
            (dist / sympy.sqrt(rat(p_sq)) if p_sq else None, "proj-wrong-ratio"),
            (dist, "proj-forgot-to-divide-by-the-length"),
        ],
    )
    stem = phrase(rng, _EXAM_STEMS, a1=vec_text(a1), a2=vec_text(a2), v=vec_text(v))
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the orthogonal projection p of v onto W.",
                marks=3,
                working=["a1 and a2 are not orthogonal, so use the normal equations.", f"p = {vec_text(p)}."],
                distractors=need(a_dist),
                **vec_answer(p),
            ),
            dict(
                prompt="Find |v - p|², the squared distance from v to W.",
                marks=2,
                sensible_range=(0, None),
                working=[f"v - p = {vec_text(r)}, so |v - p|² = {dist_sq}."],
                distractors=need(b_dist),
                **fraction_answer(dist_sq),
            ),
            dict(
                prompt="Hence find the exact distance from v to W.",
                marks=1,
                follow_through="sqrt(part_b)",
                working=[f"The distance is √({dist_sq}) = {etext(dist)}."],
                distractors=need(c_dist),
                **expression_answer(dist),
            ),
            dict(
                prompt="Find the exact sine of the angle between v and the plane W.",
                marks=2,
                follow_through=f"sqrt(part_b/{int(v_sq)})",
                working=[f"sin θ = (distance)/|v| = √({dist_sq})/√{int(v_sq)} = {etext(sine)}."],
                distractors=need(d_dist),
                **expression_answer(sine),
            ),
        ],
    )
