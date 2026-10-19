"""Generates `s2a2_surface_normal` questions (easy, medium, hard) and the exam chain `exd_x_s2a2_surface_normal`.

``uc2-surface-normal-parametrised`` (``multivariable-and-vector-calculus``, answer ``expression``): the normal vector
N = r_u × r_v to a surface r(u, v) = ⟨x(u, v), y(u, v), z(u, v)⟩, as a triple of expressions in u and v, at a point of
the surface, as a unit vector, and through the tangent plane.

* easy: a polynomial surface (a graph ⟨u, v, f(u, v)⟩, a mixed polynomial surface or a plane); N as a triple.
* medium: a surface with trigonometric components (cylinder, cone, paraboloid or helicoid, with the axes moved round);
  N as a triple of expressions in u and v.
* hard, one of
  (a) the unit normal at a given point of a polynomial surface, chosen so that |N| is a whole number;
  (b) the constant d in the equation of the tangent plane n · ⟨x, y, z⟩ = d at a point;
  (c) N for a sphere or a general surface of revolution ⟨g(u) cos v, g(u) sin v, h(u)⟩.
* exam (``exd_x_s2a2_surface_normal``): (a) the z-component of N as an expression, (b) its value at (u0, v0) (follows
  through from (a)), (c) the normal at that point as a triple, (d) where the tangent plane there meets the z-axis
  (follows through from (b)).

Every closed-form normal is confirmed against a numerical cross product of central differences of r at two points of
the (u, v) plane, and every point-valued answer is rebuilt from that numerical cross product as well.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from . import _s2a2_surf as S
from . import _s2a2_vec as V
from ._s2a2_vec import u, v

RU_RV = V.raw("r_u × r_v", r"\mathbf{r}_u \times \mathbf{r}_v")
N_VEC = V.raw("N", r"\mathbf{N}")
N_DEF = V.raw("N = r_u × r_v", r"\mathbf{N} = \mathbf{r}_u \times \mathbf{r}_v")


class _Redraw(Exception):
    """A draw that cannot give enough distinct wrong answers."""


def _tangents(surf):
    ru = tuple(sympy.diff(c, u) for c in surf.r)
    rv = tuple(sympy.diff(c, v) for c in surf.r)
    return ru, rv


def _candidates(surf):
    """The wrong normals a learner writes: the cross product in the wrong order, the j component with the wrong sign, a
    component-by-component product, a sum of the tangent vectors, or the position vector crossed with a tangent."""
    ru, rv = _tangents(surf)
    N = surf.N
    out = [
        V.dvec([-c for c in N], "vc2-surf-cross-order-reversed"),
        V.dvec([N[0], -N[1], N[2]], "vc2-surf-cross-j-sign"),
        V.dvec([p * q for p, q in zip(ru, rv)], "vc2-surf-cross-componentwise"),
        V.dvec([p + q for p, q in zip(ru, rv)], "vc2-surf-tangents-added"),
        V.dvec(V.cross(surf.r, rv), "vc2-surf-position-not-differentiated"),
    ]
    if surf.label == "graph":
        out.insert(1, V.dvec([-N[0], -N[1], N[2]], "vc2-surf-graph-normal-sign"))
    return out


def _intro(surf):
    return V.cat(V.R_OF_UV, V.mv(surf.r))


_STEMS = (
    "The surface S is given by {r}, where {region}. Find the normal vector {n} to S as a triple of expressions in u and v.",
    "A surface is parametrised by {r} for {region}. Calculate {n}, giving the answer as a triple of expressions in u and v.",
    "Let S be the surface {r}, where {region}. Find the normal vector {n} as a triple of expressions in u and v.",
)


def _stem(rng, surf):
    return V.fill(rng.choice(_STEMS), r=_intro(surf), region=S.region_phrase(surf), n=V.cat(N_VEC, V.raw(" = "), RU_RV))


# ---------------------------------------------------------------------------------------------------------------
# easy and medium


def _easy_surface(rng):
    roll = rng.random()
    if roll < 0.4:
        return S.graph(rng)
    if roll < 0.85:
        return S.mixed(rng)
    return S.linear(rng)


def _medium_surface(rng):
    shift = rng.randint(0, 2)
    pick = rng.random()
    if pick < 0.25:
        return S.cylinder(rng, shift)
    if pick < 0.55:
        return S.cone(rng, shift)
    if pick < 0.8:
        return S.paraboloid(rng, shift)
    return S.helicoid(rng, shift)


def _single_tuple(rng, surf, working):
    S.verify_surface(surf)
    text, latex = _stem(rng, surf)
    out = V.vec_answer(surf.N)
    dist = V.usable(out["answer"], out["answer_text"], _candidates(surf))
    if len(dist) < 3:
        raise _Redraw
    return dict(text=text, latex=latex, working=working, distractors=dist, **out)


# ---------------------------------------------------------------------------------------------------------------
# hard


_POINTS = ((1, 1), (1, 2), (2, 1), (-1, 1), (1, -1), (2, 2), (-1, 2), (2, -1))


def _integer_norm_point(rng):
    """A polynomial surface and a point (u0, v0) where N is a whole-number vector of whole-number length."""
    for _ in attempts(600, "a surface with a whole-number normal length"):
        r, N = S.mixed_draw(rng)
        for pt in _POINTS + ((0, 1), (1, 0)):
            n = [S._dval(c, pt) for c in N]
            norm2 = sum(c * c for c in n)
            root = math.isqrt(norm2)
            if all(n) and root * root == norm2 and 3 <= root <= 40:
                return S.from_draw(r, N), pt, [sympy.Integer(c) for c in n], sympy.Integer(root)
    raise AssertionError("unreachable")


def _hard_unit(rng):
    surf, (u0, v0), n, root = _integer_norm_point(rng)
    S.verify_surface(surf)
    unit = [c / root for c in n]
    ru, rv = _tangents(surf)
    num = V.cross([V.float_of(V.at(c, (u0, v0), S.UV)) for c in ru], [V.float_of(V.at(c, (u0, v0), S.UV)) for c in rv])
    length = sum(c * c for c in num) ** 0.5
    check(
        V.close_vec([c / length for c in num], [float(c) for c in unit], 1e-6),
        "unit normal disagrees with the numerical value",
    )
    text, latex = V.fill(
        "The surface S is given by {r}, where {region}. Find the unit normal vector, in the direction of {n}, at the "
        "point of S where u = {u0} and v = {v0}.",
        r=_intro(surf),
        region=S.region_phrase(surf),
        n=RU_RV,
        u0=V.m(sympy.Integer(u0)),
        v0=V.m(sympy.Integer(v0)),
    )
    out = V.vec_answer(unit)
    total = sum(n)
    cands = [
        V.dvec(n, "vc2-surf-normal-not-normalised"),
        V.dvec([c / root**2 for c in n], "vc2-surf-divided-by-length-squared"),
        V.dvec([-c for c in unit], "vc2-surf-cross-order-reversed"),
        V.dvec([c / sum(abs(c) for c in n) for c in n], "vc2-surf-divided-by-sum"),
    ]
    if total != 0:
        cands.append(V.dvec([c / total for c in n], "vc2-surf-divided-by-sum"))
    working = [
        f"At the point, {RU_RV} = {V.pt_plain(n)} with length {root}; divide each component by the length: {V.nice(out['answer_text'])}."
    ]
    dist = V.usable(out["answer"], out["answer_text"], cands)
    if len(dist) < 3:
        raise _Redraw
    return dict(text=text, latex=latex, working=working, distractors=dist, **out)


def _tangent_plane_d(rng):
    for _ in attempts(300, "a tangent plane constant"):
        r, N = S.mixed_draw(rng)
        pt = rng.choice(_POINTS)
        n = [S._dval(c, pt) for c in N]
        r0 = [S._dval(c, pt) for c in r]
        d = sum(p * q for p, q in zip(n, r0))
        if d != 0 and all(n) and abs(d) < 300:
            return (
                S.from_draw(r, N),
                pt,
                [sympy.Integer(c) for c in n],
                [sympy.Integer(c) for c in r0],
                sympy.Integer(d),
            )
    raise AssertionError("unreachable")


def _hard_plane(rng):
    surf, (u0, v0), n, r0, d = _tangent_plane_d(rng)
    S.verify_surface(surf)
    num_n = S._num_cross(surf.r, (u0, v0))
    num_r = [V.float_of(V.at(c, (u0, v0), S.UV)) for c in surf.r]
    check(V.close(V.dot(num_n, num_r), float(d), 1e-6), "plane constant disagrees with the numerical value")
    text, latex = V.fill(
        "The surface S is given by {r}, where {region}. The tangent plane to S at the point where u = {u0} and v = {v0} "
        "has the equation ax + by + cz = d, where (a, b, c) is the vector {n} at that point. Find d.",
        r=_intro(surf),
        region=S.region_phrase(surf),
        n=RU_RV,
        u0=V.m(sympy.Integer(u0)),
        v0=V.m(sympy.Integer(v0)),
    )
    out = V.expr_answer(d)
    swapped_n = [V.at(c, (v0, u0), S.UV) for c in surf.N]
    cands = [
        V.dexpr(-d, "vc2-surf-cross-order-reversed"),
        V.dexpr(V.dot(n, [u0, v0, 1]), "vc2-surf-parameters-as-coordinates"),
        V.dexpr(V.dot(swapped_n, r0), "vc2-surf-parameters-swapped"),
        V.dexpr(sum(n), "vc2-surf-plane-constant-normal-sum"),
        V.dexpr(V.dot(n, r0) - V.dot(n, [0, 0, r0[2]]), "vc2-surf-plane-constant-component-dropped"),
    ]
    working = [
        f"At the point, the normal is {V.pt_plain(n)} and the position is {V.pt_plain(r0)}; "
        f"the plane passes through the point, so d is their dot product: {V.nice(out['answer_text'])}."
    ]
    dist = V.usable(out["answer"], out["answer_text"], cands)
    if len(dist) < 3:
        raise _Redraw
    return dict(text=text, latex=latex, working=working, distractors=dist, **out)


def _hard_revolution(rng):
    shift = rng.randint(0, 2)
    if rng.random() < 0.5:
        surf = S.sphere(rng, shift)
    else:
        a, c = rng.randint(1, 3), rng.choice([1, 2, 3])
        g = u + a
        h = rng.choice([c * u**2, c * u, c * u**2 + u])
        surf = S.revolution(g, h, (0, rng.randint(1, 3)), (0, rng.choice([PI2, S.PI])), "revolution", shift)
    working = [
        "Differentiate r with respect to u and with respect to v, then expand the cross product; use cos² v + sin² v = 1 "
        "in the third component."
    ]
    return _single_tuple(rng, surf, working)


PI2 = 2 * S.PI


def _single(rng, tier):
    if tier == "easy":
        surf = _easy_surface(rng)
        working = [
            "Differentiate r with respect to u and with respect to v, then take the cross product "
            f"{RU_RV} = {V.pt_plain(surf.N)}."
        ]
        return _single_tuple(rng, surf, working)
    if tier == "medium":
        surf = _medium_surface(rng)
        working = [
            "Differentiate r with respect to u and with respect to v, then expand the cross product; "
            "use cos² + sin² = 1 where it simplifies a component."
        ]
        return _single_tuple(rng, surf, working)
    roll = rng.random()
    if roll < 0.35:
        return _hard_unit(rng)
    if roll < 0.65:
        return _hard_plane(rng)
    return _hard_revolution(rng)


@generator(
    "s2a2_surface_normal",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a2_surface_normal(rng: random.Random, tier="easy"):
    for _ in attempts(30, "a surface-normal question with enough distractors"):
        try:
            return _single(rng, tier)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_EXAM_STEMS = (
    "A curved panel of a sculpture is modelled by the surface S with position vector {r}, for {region}, where lengths are "
    "in metres. Let {n} be the normal vector to S.",
    "The surface S, parametrised by {r} for {region} (lengths in metres), is a glass panel in a building. Let {n} be the "
    "normal vector to S.",
)


def _exam_once(rng):
    for _ in attempts(400, "an exam surface"):
        r, N = S.mixed_draw(rng)
        u0, v0 = rng.choice(_POINTS)
        n0 = [S._dval(c, (u0, v0)) for c in N]
        r0 = [S._dval(c, (u0, v0)) for c in r]
        d0 = sum(p * q for p, q in zip(n0, r0))
        Nz = N[2]
        if len(Nz) >= 2 and set(Nz) != {(0, 0)} and all(n0) and d0 != 0 and d0 % n0[2] != 0 and abs(d0) < 400:
            break
    surf = S.from_draw(r, N)
    Nz = sympy.expand(surf.N[2])
    n0 = [sympy.Integer(c) for c in n0]
    r0 = [sympy.Integer(c) for c in r0]
    d0 = sympy.Integer(d0)
    S.verify_surface(surf)
    num_n = S._num_cross(surf.r, (u0, v0))
    num_r = [V.float_of(r_) for r_ in r0]
    check(
        V.close_vec(num_n, [float(c) for c in n0], 1e-6), "the normal at the point disagrees with the numerical value"
    )
    check(V.close(float(d0) / num_n[2], float(d0 / n0[2]), 1e-6), "the z-intercept disagrees with the numerical value")
    check(V.close(V.dot(num_n, num_r), float(d0), 1e-6), "the plane constant disagrees with the numerical value")

    ru, rv = _tangents(surf)
    zn = n0[2]
    intercept = d0 / zn

    # (a) the z-component
    a_out = V.expr_answer(Nz)
    a_dist = V.usable(
        a_out["answer"],
        a_out["answer_text"],
        [
            V.dexpr(-Nz, "vc2-surf-cross-order-reversed"),
            V.dexpr(ru[0] * rv[0], "vc2-surf-cross-componentwise"),
            V.dexpr(ru[0] * rv[1] + ru[1] * rv[0], "vc2-surf-cross-terms-added"),
            V.dexpr(ru[2] + rv[2], "vc2-surf-tangents-added"),
        ],
        n=4,
    )
    # (b) the value of N_z at the point
    b_out = V.frac_answer(zn)
    b_dist = V.usable(
        b_out["answer"],
        b_out["answer_text"],
        [
            V.dfrac(-zn, "vc2-surf-cross-order-reversed"),
            V.dfrac(V.at(Nz, (v0, u0), S.UV), "vc2-surf-parameters-swapped"),
            V.dfrac(V.at(ru[0] * rv[0], (u0, v0), S.UV), "vc2-surf-cross-componentwise"),
            V.dfrac(zn + 1, "vc2-surf-off-by-one"),
        ],
        kind="fraction",
    )
    # (c) the normal
    c_out = V.vec_answer(n0)
    c_dist = V.usable(
        c_out["answer"],
        c_out["answer_text"],
        [
            V.dvec([-c for c in n0], "vc2-surf-cross-order-reversed"),
            V.dvec([n0[0], -n0[1], n0[2]], "vc2-surf-cross-j-sign"),
            V.dvec([V.at(c, (v0, u0), S.UV) for c in surf.N], "vc2-surf-parameters-swapped"),
            V.dvec(
                [V.at(p, (u0, v0), S.UV) * V.at(q, (u0, v0), S.UV) for p, q in zip(ru, rv)],
                "vc2-surf-cross-componentwise",
            ),
            V.dvec(r0, "vc2-surf-position-not-differentiated"),
        ],
        n=4,
    )
    # (d) the z-intercept of the tangent plane
    d_out = V.frac_answer(intercept)
    d_dist = V.usable(
        d_out["answer"],
        d_out["answer_text"],
        [
            V.dfrac(-intercept, "vc2-surf-cross-order-reversed"),
            V.dfrac(d0, "vc2-surf-intercept-not-divided"),
            V.dfrac(zn / d0, "vc2-surf-intercept-inverted"),
            V.dfrac(r0[2], "vc2-surf-intercept-point-height"),
            V.dfrac(V.dot(n0, [u0, v0, 1]) / zn, "vc2-surf-parameters-as-coordinates"),
        ],
        kind="fraction",
        n=4,
    )
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    stem_text, stem_latex = V.fill(rng.choice(_EXAM_STEMS), r=_intro(surf), region=S.region_phrase(surf), n=N_DEF)

    def part(template, out, dist, marks, working, follow=None, **slots):
        text, latex = V.fill(template, **slots)
        d_ = dict(prompt=text, prompt_latex=latex, marks=marks, working=working, distractors=dist, **out)
        if follow:
            d_["follow_through"] = follow
        return d_

    nz = V.raw("N_z", r"N_z")
    parts = [
        part(
            "Find the z-component {nz} of {n} as an expression in u and v.",
            a_out,
            a_dist,
            2,
            [
                "Cross the two tangent vectors and keep the z-component: the x-part of r_u times the y-part of r_v, minus the y-part of r_u times the x-part of r_v."
            ],
            nz=nz,
            n=N_VEC,
        ),
        part(
            f"Hence find the value of {{nz}} at the point of S where u = {u0} and v = {v0}.",
            b_out,
            b_dist,
            1,
            [f"Substitute u = {u0} and v = {v0} into the expression from part (a): {V.nice(b_out['answer_text'])}."],
            follow=f"Subs(part_a, (u, v), ({u0}, {v0}))",
            nz=nz,
        ),
        part(
            f"Find the vector {{n}} at the point where u = {u0} and v = {v0}, as a triple.",
            c_out,
            c_dist,
            2,
            ["Evaluate all three components of the cross product at the point."],
            n=N_VEC,
        ),
        part(
            "The tangent plane to S at this point passes through the point of S and has the normal vector {n}. "
            "Find the z-coordinate of the point where this plane meets the z-axis.",
            d_out,
            d_dist,
            3,
            [
                f"The plane is {V.pt_plain(n0)} · (x, y, z) = {d0}; on the z-axis x = y = 0, so z = {d0} divided by "
                f"the z-component {zn}."
            ],
            follow=f"{d0}/part_b",
            n=N_VEC,
        ),
    ]
    return dict(text=stem_text, latex=stem_latex, parts=parts)


@generator("exd_x_s2a2_surface_normal", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a2_surface_normal(rng: random.Random, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam surface normal with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")
