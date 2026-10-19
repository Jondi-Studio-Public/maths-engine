"""Generates `s2a2_surface_area` questions (easy, medium, hard) and the exam chain `exd_x_s2a2_surface_area`.

``uc2-surface-area-parametric`` (``multivariable-and-vector-calculus``, answer ``expression``): the area of a surface
r(u, v) as the double integral of |r_u × r_v| du dv over the parameter region, as an exact expression.

* easy: a plane r(u, v) = u A + v B + C over a rectangle, where |r_u × r_v| is constant; the area is that length times
  the area of the rectangle.
* medium: a cylinder (the length is constant) or a cone (the length is c u, so u comes into the integral); the area of a
  part of it, with a parameter range given in radians.
* hard, one of
  (a) a zone of a sphere, |r_u × r_v| = a² sin u;
  (b) a paraboloid, where the integral needs a substitution;
  (c) a torus, |r_u × r_v| = r (R + r cos u);
  (d) a plane over a triangle 0 ≤ v ≤ k - u, where the limits are not constants.
* exam (``exd_x_s2a2_surface_area``): a conical roof r = ⟨a u cos v, a u sin v, h u⟩: (a) the constant c in the area
  element dS = c u du dv, (b) the exact area (follows through from (a)), (c) the cost of covering it at a price per square
  metre, to the nearest cent (follows through from (b)).

Every exact area is confirmed against Gauss-Legendre quadrature of |r_u × r_v|, with r differentiated and lambdified
on its own, over the same region.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import (
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)
from . import _s2a2_surf as S
from . import _s2a2_vec as V
from ._s2a2_vec import u, v

PI = sympy.pi
_TOL = stated_precision_tolerance(2)
RU_RV = V.raw("r_u × r_v", r"\mathbf{r}_u \times \mathbf{r}_v")
ABS_N = V.raw("|r_u × r_v|", r"\left|\mathbf{r}_u \times \mathbf{r}_v\right|")


class _Redraw(Exception):
    """A draw that cannot give enough distinct wrong answers."""


# ---------------------------------------------------------------------------------------------------------------
# the exact areas, and the wrong ones


def _area_plane(surf, U, W):
    s = surf.speed
    N = surf.N
    comps = sum(abs(int(c)) for c in N)
    area = s * U * W
    cands = [
        (s * s * U * W, "vc2-surf-area-magnitude-not-rooted"),
        (comps * U * W, "vc2-surf-area-sum-of-components"),
        (sympy.Integer(U * W), "vc2-surf-area-speed-dropped"),
        (s, "vc2-surf-area-range-dropped"),
        (s * (U + W), "vc2-surf-area-range-added"),
    ]
    return area, cands


def _area_cylinder(surf, v_len):
    a, c, U = surf.a, surf.c, surf.u_range[1]
    area = a * c * U * v_len
    cands = [
        (a * U * v_len, "vc2-surf-area-scale-dropped"),
        (a * c * v_len, "vc2-surf-area-range-dropped"),
        (a * c * U, "vc2-surf-area-angle-dropped"),
        (a * a * c * U * v_len, "vc2-surf-area-radius-squared"),
        (a * c * U * v_len / 2, "vc2-surf-area-halved"),
    ]
    return area, cands


def _area_cone(surf, v_len):
    a, h, U = surf.a, surf.slope, surf.u_range[1]
    c = a * sympy.sqrt(a * a + h * h)
    area = c * sympy.Integer(U) ** 2 / 2 * v_len
    cands = [
        (c * U**2 * v_len, "vc2-surf-area-integral-slip"),
        (c * U * v_len, "vc2-surf-area-jacobian-factor-dropped"),
        (sympy.sqrt(a * a + h * h) * sympy.Integer(U) ** 2 / 2 * v_len, "vc2-surf-area-scale-dropped"),
        (a * (a * a + h * h) * sympy.Integer(U) ** 2 / 2 * v_len, "vc2-surf-area-magnitude-not-rooted"),
        (c * sympy.Integer(U) ** 2 / 2, "vc2-surf-area-angle-dropped"),
    ]
    return area, cands


def _area_paraboloid(surf, v_len):
    c, U = surf.c, sympy.Integer(surf.u_range[1])
    base = 1 + 4 * c * c * U * U
    area = sympy.together((base ** sympy.Rational(3, 2) - 1) / (12 * c * c) * v_len)
    cands = [
        (base ** sympy.Rational(3, 2) / (12 * c * c) * v_len, "vc2-surf-area-lower-limit-dropped"),
        ((base ** sympy.Rational(3, 2) - 1) / (6 * c * c) * v_len, "vc2-surf-area-antiderivative-factor"),
        ((base ** sympy.Rational(3, 2) - 1) / (12 * c * c), "vc2-surf-area-angle-dropped"),
        (U * sympy.sqrt(base) * v_len / 2, "vc2-surf-area-integrand-not-integrated"),
        ((base ** sympy.Rational(3, 2) - 1) / (3 * c * c) * v_len, "vc2-surf-area-antiderivative-factor"),
    ]
    cands = [(sympy.together(x_), t) for x_, t in cands]
    return area, cands


def _area_sphere(surf):
    a, alpha = surf.a, surf.alpha
    k = 1 - sympy.cos(alpha)
    area = 2 * PI * a * a * k
    cands = [
        (2 * PI * a * a * sympy.cos(alpha), "vc2-surf-area-integral-slip"),
        (2 * PI * a * a * (1 + sympy.cos(alpha)), "vc2-surf-area-antiderivative-sign"),
        (2 * PI * a * k, "vc2-surf-area-scale-dropped"),
        (2 * PI * a * a * alpha, "vc2-surf-area-sine-dropped"),
        (PI * a * a * k, "vc2-surf-area-angle-dropped"),
    ]
    return area, cands


def _area_torus(surf):
    R, r, U = surf.R, surf.minor, surf.u_range[1]
    area = 2 * PI * r * R * U
    cands = [
        (2 * PI * R * U, "vc2-surf-area-scale-dropped"),
        (2 * PI * r * r * U, "vc2-surf-area-radii-swapped"),
        (2 * PI * r * R, "vc2-surf-area-range-dropped"),
        (PI * r * R * U, "vc2-surf-area-angle-dropped"),
        (2 * PI * r * (R + r) * U, "vc2-surf-area-term-kept"),
    ]
    return area, cands


def _area_triangle(surf, k):
    s = surf.speed
    area = s * sympy.Integer(k) ** 2 / 2
    cands = [
        (s * k * k, "vc2-surf-area-triangle-as-square"),
        (sympy.Integer(k * k) / 2, "vc2-surf-area-speed-dropped"),
        (s * s * sympy.Integer(k) ** 2 / 2, "vc2-surf-area-magnitude-not-rooted"),
        (s * k, "vc2-surf-area-range-dropped"),
        (sum(abs(int(c)) for c in surf.N) * sympy.Integer(k) ** 2 / 2, "vc2-surf-area-sum-of-components"),
    ]
    return area, cands


# ---------------------------------------------------------------------------------------------------------------
# building a question


_STEMS = (
    "Find the exact area of the surface S given by {r}, where {region}.",
    "A surface is parametrised by {r} for {region}. Find its exact surface area.",
    "Let S be the surface {r}, where {region}. Calculate the exact area of S.",
)

_STEMS_TRI = (
    "Find the exact area of the part of the plane given by {r} where u ≥ 0, v ≥ 0 and u + v ≤ {k}.",
    "The surface S is given by {r} for the triangle 0 ≤ u ≤ {k}, 0 ≤ v ≤ {k} - u. Find the exact area of S.",
)


def _finish(rng, surf, area, cands, working, text_latex, numeric):
    S.verify_surface(surf)
    check(V.close(numeric, V.float_of(area), 1e-7), "the exact area disagrees with the numerical integral")
    text, latex = text_latex
    out = V.expr_answer(area)
    dist = V.usable(out["answer"], out["answer_text"], [V.dexpr(c, t) for c, t in cands])
    if len(dist) < 3:
        raise _Redraw
    return dict(text=text, latex=latex, working=working, distractors=dist, **out)


def _plain_stem(rng, surf):
    return V.fill(rng.choice(_STEMS), r=S.rvec(surf), region=S.region_phrase(surf))


def _easy(rng):
    surf = S.linear(rng, whole=rng.random() < 0.6)
    U, W = rng.randint(1, 4), rng.randint(1, 4)
    surf.u_range, surf.v_range = (0, U), (0, W)
    area, cands = _area_plane(surf, U, W)
    working = [
        f"The tangent vectors are constant, so |r_u × r_v| = {V.plain(surf.speed)} everywhere; "
        f"multiply by the area {U * W} of the parameter rectangle."
    ]
    return _finish(rng, surf, area, cands, working, _plain_stem(rng, surf), S.numeric_area(surf))


def _medium(rng):
    if rng.random() < 0.4:
        surf = S.cylinder(rng)
        v_len = sympy.sympify(surf.v_range[1])
        area, cands = _area_cylinder(surf, v_len)
        working = [
            "The length |r_u × r_v| is constant for a cylinder, so the area is that constant times the parameter area."
        ]
    else:
        surf = S.cone(rng)
        v_len = sympy.sympify(surf.v_range[1])
        area, cands = _area_cone(surf, v_len)
        working = [
            "For the cone |r_u × r_v| is a constant multiple of u, so integrate that multiple of u from 0 to the upper limit, then multiply by the range of v."
        ]
    return _finish(rng, surf, area, cands, working, _plain_stem(rng, surf), S.numeric_area(surf))


def _hard(rng):
    roll = rng.random()
    if roll < 0.3:
        surf = S.sphere(rng)
        area, cands = _area_sphere(surf)
        working = ["For a sphere |r_u × r_v| = a² sin u, so integrate a² sin u over u and v."]
        return _finish(rng, surf, area, cands, working, _plain_stem(rng, surf), S.numeric_area(surf))
    if roll < 0.55:
        surf = S.paraboloid(rng)
        v_len = sympy.sympify(surf.v_range[1])
        area, cands = _area_paraboloid(surf, v_len)
        working = ["Integrate u √(1 + 4c²u²) from 0 with the substitution w = 1 + 4c²u²."]
        return _finish(rng, surf, area, cands, working, _plain_stem(rng, surf), S.numeric_area(surf, n=60))
    if roll < 0.8:
        surf = S.torus(rng)
        surf.u_range = (0, rng.choice([S.PI, 2 * S.PI]))
        area, cands = _area_torus(surf)
        working = [
            "For a torus |r_u × r_v| = r(R + r cos u); the cos u term integrates to a multiple of sin u, which is zero at the limits."
        ]
        return _finish(rng, surf, area, cands, working, _plain_stem(rng, surf), S.numeric_area(surf))
    surf = S.linear(rng, whole=rng.random() < 0.5)
    k = rng.randint(2, 5)
    surf.u_range, surf.v_range = (0, k), (0, k)
    area, cands = _area_triangle(surf, k)
    text, latex = V.fill(rng.choice(_STEMS_TRI), r=S.rvec(surf), k=V.m(sympy.Integer(k)))
    numeric = S.numeric_area(surf, (0.0, float(k)), (lambda uu: 0.0, lambda uu: k - uu))
    working = ["The length |r_u × r_v| is constant, so the area is that constant times the area k²/2 of the triangle."]
    return _finish(rng, surf, area, cands, working, (text, latex), numeric)


@generator(
    "s2a2_surface_area",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a2_surface_area(rng: random.Random, tier="easy"):
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(30, "a surface-area question with enough distractors"):
        try:
            return build(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_EXAM_STEMS = (
    "A conical roof is modelled by the surface S with position vector {r}, for {region}, where lengths are in metres.",
    "The roof of an observation hut is a cone, modelled by {r} for {region}, with lengths in metres.",
)
_PRICES = (14, 16, 18, 22, 24, 26, 28, 32, 36, 42)
_ANGLES = ((2 * PI, "2π"), (PI, "π"), (3 * PI / 2, "3π/2"))


def _cents(value: Fraction) -> Fraction:
    return round_half_up(value, 2)


def _txt(value: Fraction) -> str:
    return stated_precision_string(_cents(value), 2)


def _as_fraction(expr) -> Fraction:
    return Fraction(str(sympy.N(expr, 30)))


def _exam_once(rng):
    pairs = [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (5, 12), (5, 2), (4, 1), (2, 5)]
    a, h = rng.choice(pairs)
    U = rng.randint(2, 5)
    v_end, v_name = rng.choice(_ANGLES)
    price = rng.choice(_PRICES)
    surf = S.revolution(a * u, h * u, (0, U), (0, v_end), "cone", 0, a=a, slope=h)
    area, area_cands = _area_cone(surf, v_end)
    c = a * sympy.sqrt(a * a + h * h)
    check(sympy.simplify(area - c * U**2 / 2 * v_end) == 0, "the area does not match the area element")
    S.verify_surface(surf)
    check(V.close(S.numeric_area(surf), V.float_of(area), 1e-7), "the exact area disagrees with the numerical integral")
    cost = _cents(_as_fraction(area * price))
    # the cost must not sit on a rounding boundary
    if abs((_as_fraction(area * price) * 100) % 1 - Fraction(1, 2)) <= Fraction(1, 50):
        raise _Redraw  # the cost sits on a rounding boundary

    a_out = V.expr_answer(c)
    a_dist = V.usable(
        a_out["answer"],
        a_out["answer_text"],
        [
            V.dexpr(sympy.sqrt(a * a + h * h), "vc2-surf-area-scale-dropped"),
            V.dexpr(a * (a * a + h * h), "vc2-surf-area-magnitude-not-rooted"),
            V.dexpr(a * (a + h), "vc2-surf-area-sum-of-components"),
            V.dexpr(a * h, "vc2-surf-area-wrong-component"),
        ],
    )
    b_out = V.expr_answer(area)
    b_dist = V.usable(b_out["answer"], b_out["answer_text"], [V.dexpr(x_, t) for x_, t in area_cands])

    def num_dist(cands):
        shown, seen = [], {cost}
        for value, tag in cands:
            val = _cents(_as_fraction(value * price))
            if val not in seen:
                seen.add(val)
                shown.append(Distractor(_txt(val), tag, val))
        return pick_distractors(_txt(cost), shown, 3, tolerance=_TOL)

    c_cands = [(x_, t) for x_, t in area_cands]
    c_cands.append((area / price, "vc2-surf-area-cost-divided"))
    c_cands.append((area + price, "vc2-surf-area-cost-added"))
    c_dist = num_dist(c_cands)
    if min(len(a_dist), len(b_dist), len(c_dist)) < 2:
        raise _Redraw

    stem_text, stem_latex = V.fill(rng.choice(_EXAM_STEMS), r=S.rvec(surf), region=S.region_phrase(surf))

    def part(template, out, dist, marks, working, follow=None, **slots):
        text, latex = V.fill(template, **slots)
        d_ = dict(prompt=text, prompt_latex=latex, marks=marks, working=working, distractors=dist, **out)
        if follow:
            d_["follow_through"] = follow
        return d_

    c_out = dict(
        answer_kind="number",
        answer=numeric_answer(cost),
        answer_text=_txt(cost),
        answer_latex=latex_answer(_txt(cost)),
        sensible_range=(0, None),
    )
    parts = [
        part(
            "The area element is dS = c u du dv for a constant c. Find c.",
            a_out,
            a_dist,
            2,
            ["Cross r_u and r_v: the length of the result is a constant multiple of u, and that constant is c."],
        ),
        part(
            "Hence find the exact area of the roof.",
            b_out,
            b_dist,
            3,
            ["Area = the double integral of c u du dv = c × (upper limit of u)² / 2 × (range of v)."],
            follow=f"part_a*{U * U}/2*{sympy.sstr(v_end)}",
        ),
        part(
            f"The roof is covered in sheeting that costs ${price} per square metre. Find the cost of covering the roof, to the "
            "nearest cent.",
            c_out,
            c_dist,
            2,
            [f"Cost = area × {price} = {_txt(cost)} dollars."],
            follow=f"part_b*{price}",
        ),
    ]
    return dict(text=stem_text, latex=stem_latex, parts=parts)


@generator("exd_x_s2a2_surface_area", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a2_surface_area(rng: random.Random, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam surface area with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")
