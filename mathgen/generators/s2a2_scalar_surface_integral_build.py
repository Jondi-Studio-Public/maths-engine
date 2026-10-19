"""Generates `s2a2_scalar_surface` questions (easy, medium, hard) and the exam chain `exd_x_s2a2_scalar_surface`.

``uc2-scalar-surface-integral`` (``multivariable-and-vector-calculus``, answer ``expression``): the surface integral of
a scalar function, ∬ g dS = ∬ g(r(u, v)) |r_u × r_v| du dv, as an exact value.

* easy: a plane over a rectangle (|r_u × r_v| is a constant) and a quadratic g: substitute r into g, then integrate the
  polynomial over the rectangle.
* medium: a cylinder or a cone with a simple g; the angle parameter gives trigonometric integrals, and the cone's
  |r_u × r_v| brings a factor of u into the integral. Also worded as the mass of a thin sheet.
* hard, one of
  (a) a zone of a sphere with g built from z and x² + y², where the integral needs the substitution w = cos u;
  (b) a plane over a triangle 0 ≤ v ≤ k - u, with the limits of the inner integral depending on u;
  (c) a cone with a cubic g, mixing powers of u with sin and cos of v.
* exam (``exd_x_s2a2_scalar_surface``): a spherical shell r = ⟨a sin u cos v, a sin u sin v, a cos u⟩ with density σ = k z:
  (a) the constant c in dS = c sin u du dv, (b) the mass (follows through from (a)), (c) the surface area (follows through
  from (a)), (d) the average density, mass divided by area (follows through from (b) and (c)).

Every exact value is confirmed against Gauss-Legendre quadrature of g(r(u, v)) |r_u × r_v|, with r differentiated and
lambdified on its own, over the same region.
"""

from __future__ import annotations

import random

import sympy

from ..core import attempts, check, generator
from . import _s2a2_surf as S
from . import _s2a2_vec as V
from ._s2a2_vec import u, v, x, y, z

PI = sympy.pi
_c = sympy.Symbol("c_")


class _Redraw(Exception):
    """A draw that cannot give enough distinct wrong answers."""


# ---------------------------------------------------------------------------------------------------------------
# g as a list of monomials (coefficient, i, j, k) for coefficient x^i y^j z^k


def _g_expr(terms):
    return sympy.Add(*[sympy.Integer(c) * x**i * y**j * z**k for c, i, j, k in terms])


def _trig(i, j, length, slip=False):
    """The integral of cos(v)^i sin(v)^j from 0 to `length` for the pairs (i, j) used here."""
    if (i, j) == (0, 0):
        return length
    if (i, j) == (1, 0):
        return sympy.sin(length)
    if (i, j) == (0, 1):
        return 1 - sympy.cos(length)
    if (i, j) == (2, 0):
        return length if slip else length / 2 + sympy.sin(2 * length) / 4
    if (i, j) == (0, 2):
        return length if slip else length / 2 - sympy.sin(2 * length) / 4
    if (i, j) == (1, 1):
        return sympy.sin(length) ** 2 / 2
    raise AssertionError((i, j))


def _cylinder_value(terms, a, c, U, length, scale=True, trig_slip=False):
    speed = a * c if scale else a
    total = 0
    for coef, i, j, k in terms:
        total += coef * a ** (i + j) * c**k * sympy.Integer(U) ** (k + 1) / (k + 1) * _trig(i, j, length, trig_slip)
    return sympy.expand(total * speed)


def _cone_value(terms, a, h, U, length, scale=True, trig_slip=False, jacobian=True):
    speed = a * sympy.sqrt(a * a + h * h) if scale else sympy.sqrt(a * a + h * h)
    total = 0
    for coef, i, j, k in terms:
        n = i + j + k
        e = n + 1 if jacobian else n
        total += coef * a ** (i + j) * h**k * sympy.Integer(U) ** (e + 1) / (e + 1) * _trig(i, j, length, trig_slip)
    return sympy.expand(total * speed)


def _sphere_value(terms, a, alpha, scale=True, limits_slip=False):
    total = 0
    for coef, i, j, k in terms:
        p = i + j
        if p % 2:
            continue  # the integral over the full circle of v is zero
        m = p + 1
        poly = sympy.Poly(sympy.expand((1 - _c**2) ** ((m - 1) // 2) * _c**k), _c)
        F = poly.integrate().as_expr()
        A = (0 if limits_slip else F.subs(_c, 1)) - F.subs(_c, sympy.cos(alpha))
        total += coef * a ** (i + j + k + (2 if scale else 1)) * A * _trig(i, j, 2 * PI)
    return sympy.expand(total)


def _rect_integral(P, U, W, slip=False):
    P = sympy.expand(P)
    if slip:
        return sum(c * sympy.Integer(U) ** (i + 1) * sympy.Integer(W) ** (j + 1) for (i, j), c in _coeffs(P).items())
    F = V.pint(V.pint(P, u), v)
    return F.xreplace({u: U, v: W})


def _coeffs(P):
    return {m: c for m, c in sympy.Poly(P, u, v).terms()}


def _triangle_integral(P, k):
    inner = V.pint(sympy.expand(P), v).xreplace({v: k - u})
    return V.pint(sympy.expand(inner), u).xreplace({u: k})


# ---------------------------------------------------------------------------------------------------------------
# questions


def _pick_terms(rng, pool, n, lo=-3, hi=4):
    chosen = rng.sample(pool, n)
    out = []
    for _, i, j, k in chosen:
        out.append((rng.choice([n for n in range(lo, hi + 1) if n]), i, j, k))
    return out


_PLANE_POOL = [
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
    (1, 2, 0, 0),
    (1, 0, 2, 0),
    (1, 0, 0, 2),
    (1, 1, 1, 0),
    (1, 0, 1, 1),
    (1, 1, 0, 1),
    (1, 0, 0, 0),
]
_SIMPLE_POOL = [
    (1, 0, 0, 1),
    (1, 0, 0, 2),
    (1, 2, 0, 0),
    (1, 0, 2, 0),
    (1, 1, 0, 1),
    (1, 0, 1, 1),
    (1, 1, 1, 0),
    (1, 0, 0, 0),
]
_CUBIC_POOL = [
    (1, 0, 0, 1),
    (1, 0, 0, 2),
    (1, 0, 0, 3),
    (1, 2, 0, 0),
    (1, 0, 2, 0),
    (1, 2, 0, 1),
    (1, 0, 2, 1),
    (1, 1, 0, 1),
    (1, 0, 1, 1),
]
_SPHERE_POOL = [
    (1, 0, 0, 1),
    (1, 0, 0, 2),
    (1, 0, 0, 3),
    (1, 2, 0, 0),
    (1, 0, 2, 0),
    (1, 2, 0, 1),
    (1, 0, 2, 1),
    (1, 0, 0, 0),
]

_STEMS = (
    "Let S be the surface {r}, where {region}. Find the exact value of the surface integral of {g} over S.",
    "The surface S is given by {r} for {region}. Calculate the exact value of the integral of {g} over S with respect to surface area.",
)
_MASS_STEMS = (
    "A thin sheet has the shape of the surface S given by {r}, where {region} (lengths in metres). Its surface density is "
    "{g} (in kilograms per square metre). Find the exact total mass of the sheet.",
    "A thin metal plate occupies the surface {r} for {region}, with lengths in metres. The density of the metal per unit area is "
    "{g} (in kilograms per square metre). Find the exact mass of the plate.",
)
_STEMS_TRI = (
    "Find the exact value of the surface integral of {g} over the part of the plane {r} where u ≥ 0, v ≥ 0 and u + v ≤ {k}.",
    "Let S be the surface {r} for the triangle 0 ≤ u ≤ {k}, 0 ≤ v ≤ {k} - u. Calculate the exact value of the integral of {g} over S.",
)


def _g_text(g):
    return V.sdef("g", g)


def _non_negative(surf, g):
    """Whether g is at least zero over the whole parameter rectangle (so it can be a density)."""
    fn = sympy.lambdify(S.UV, g.xreplace(dict(zip((x, y, z), surf.r))), "math")
    (u0, u1), (v0, v1) = (tuple(float(sympy.sympify(c)) for c in rng_) for rng_ in (surf.u_range, surf.v_range))
    return all(fn(u0 + (u1 - u0) * i / 12, v0 + (v1 - v0) * j / 12) >= -1e-12 for i in range(13) for j in range(13))


def _usable(answer, answer_text, candidates, n=4):
    """`V.usable` without the zero distractors, which no learner would pick."""
    return V.usable(answer, answer_text, [c for c in candidates if c.text != "0"], n=n)


def _finish(rng, surf, terms, value, cands, working, numeric, stem=None):
    S.verify_surface(surf)
    g = _g_expr(terms)
    if value == 0:
        raise _Redraw  # a zero integral is a poor answer
    check(V.close(numeric, V.float_of(value), 1e-7), "the exact integral disagrees with the numerical integral")
    out = V.expr_answer(value)
    dist = V.usable(out["answer"], out["answer_text"], [V.dexpr(c, t) for c, t in cands if c != 0])
    if len(dist) < 3:
        raise _Redraw
    if stem is None:
        mass = rng.random() < 0.4 and _non_negative(surf, g)
        text, latex = V.fill(
            rng.choice(_MASS_STEMS if mass else _STEMS), r=S.rvec(surf), region=S.region_phrase(surf), g=_g_text(g)
        )
    else:
        text, latex = stem
    return dict(text=text, latex=latex, working=working, distractors=dist, **out)


def _plane_question(rng):
    surf = S.linear(rng, whole=rng.random() < 0.6)
    U, W = rng.randint(1, 3), rng.randint(1, 3)
    surf.u_range, surf.v_range = (0, U), (0, W)
    terms = _pick_terms(rng, _PLANE_POOL, rng.choice([2, 3]))
    g = _g_expr(terms)
    gr = sympy.expand(g.xreplace(dict(zip((x, y, z), surf.r))))
    s = surf.speed
    I = _rect_integral(gr, U, W)
    value = sympy.expand(s * I)
    check(gr.has(u) or gr.has(v), "g is constant on the plane")
    cands = [
        (I, "vc2-ssi-area-element-dropped"),
        (s * s * I, "vc2-ssi-magnitude-not-rooted"),
        (s * _rect_integral(gr, U, W, slip=True), "vc2-ssi-power-rule-division-dropped"),
        (s * U * W, "vc2-ssi-integrand-dropped"),
        (s * gr.xreplace({u: 0, v: 0}) * U * W, "vc2-ssi-integrand-at-corner"),
    ]
    working = [
        f"On the plane |r_u × r_v| = {V.plain(s)}. Substituting r(u, v) into g gives {V.plain(gr)}; integrate this over the "
        "rectangle and multiply by the constant."
    ]
    return _finish(rng, surf, terms, value, cands, working, S.numeric_area(surf, integrand=g))


def _cyl_cone_cands(terms, value, wrong):
    return [(val, tag) for val, tag in wrong if val != value]


def _medium(rng):
    terms = _pick_terms(rng, _SIMPLE_POOL, rng.choice([1, 2]))
    if rng.random() < 0.4:
        surf = S.cylinder(rng)
        a, c, U, L = surf.a, surf.c, surf.u_range[1], sympy.sympify(surf.v_range[1])
        value = _cylinder_value(terms, a, c, U, L)
        wrong = [
            (_cylinder_value(terms, a, c, U, L, scale=False), "vc2-ssi-scale-dropped"),
            (_cylinder_value(terms, a, c, U, L, trig_slip=True), "vc2-ssi-trig-average-slip"),
            (_cylinder_value([(1, 0, 0, 0)], a, c, U, L), "vc2-ssi-integrand-dropped"),
            (_cylinder_value(terms, 1, 1, U, L) * a * c, "vc2-ssi-radius-not-substituted"),
        ]
        working = [
            "For the cylinder |r_u × r_v| = a c is constant; write g in terms of u and v and integrate over the rectangle."
        ]
    else:
        surf = S.cone(rng)
        a, h, U, L = surf.a, surf.slope, surf.u_range[1], sympy.sympify(surf.v_range[1])
        value = _cone_value(terms, a, h, U, L)
        wrong = [
            (_cone_value(terms, a, h, U, L, scale=False), "vc2-ssi-scale-dropped"),
            (_cone_value(terms, a, h, U, L, trig_slip=True), "vc2-ssi-trig-average-slip"),
            (_cone_value(terms, a, h, U, L, jacobian=False), "vc2-ssi-jacobian-factor-dropped"),
            (_cone_value([(1, 0, 0, 0)], a, h, U, L), "vc2-ssi-integrand-dropped"),
        ]
        working = [
            "For the cone |r_u × r_v| = a √(a² + h²) u; write g in terms of u and v, multiply by that length and integrate "
            "over u and then v."
        ]
    g = _g_expr(terms)
    if not any(i or j or k for _, i, j, k in terms):
        raise _Redraw  # g is constant
    numeric = S.numeric_area(surf, integrand=g)
    return _finish(rng, surf, terms, value, _cyl_cone_cands(terms, value, wrong), working, numeric)


def _hard_sphere(rng):
    surf = S.sphere(rng)
    a, alpha = surf.a, surf.alpha
    terms = _pick_terms(rng, _SPHERE_POOL, rng.choice([1, 2]))
    value = _sphere_value(terms, a, alpha)
    if value == 0:
        raise _Redraw
    wrong = [
        (_sphere_value(terms, a, alpha, scale=False), "vc2-ssi-scale-dropped"),
        (_sphere_value(terms, a, alpha, limits_slip=True), "vc2-ssi-substitution-limits-slip"),
        (_sphere_value([(1, 0, 0, 0)], a, alpha), "vc2-ssi-integrand-dropped"),
        (value * 2, "vc2-ssi-trig-average-slip"),
    ]
    g = _g_expr(terms)
    working = [
        "On the sphere x = a sin u cos v, y = a sin u sin v, z = a cos u and |r_u × r_v| = a² sin u. Integrate over v first, "
        "then over u with the substitution w = cos u."
    ]
    return _finish(rng, surf, terms, value, wrong, working, S.numeric_area(surf, integrand=g))


def _hard_triangle(rng):
    surf = S.linear(rng, whole=rng.random() < 0.5)
    k = rng.randint(2, 4)
    surf.u_range, surf.v_range = (0, k), (0, k)
    terms = _pick_terms(rng, _PLANE_POOL, rng.choice([2, 3]))
    g = _g_expr(terms)
    gr = sympy.expand(g.xreplace(dict(zip((x, y, z), surf.r))))
    s = surf.speed
    I = _triangle_integral(gr, k)
    value = sympy.expand(s * I)
    if value == 0:
        raise _Redraw
    cands = [
        (s * _rect_integral(gr, k, k), "vc2-ssi-triangle-as-square"),
        (I, "vc2-ssi-area-element-dropped"),
        (s * s * I, "vc2-ssi-magnitude-not-rooted"),
        (s * sympy.Integer(k) ** 2 / 2, "vc2-ssi-integrand-dropped"),
        (s * V.pint(V.pint(sympy.expand(gr), v).xreplace({v: k}), u).xreplace({u: k}) / 2, "vc2-ssi-triangle-halved"),
    ]
    text, latex = V.fill(rng.choice(_STEMS_TRI), g=_g_text(g), r=S.rvec(surf), k=V.m(sympy.Integer(k)))
    numeric = S.numeric_area(surf, (0.0, float(k)), (lambda uu: 0.0, lambda uu: k - uu), integrand=g)
    working = [
        "The length |r_u × r_v| is constant. Integrate g(r(u, v)) with v from 0 to k - u first, then u from 0 to k."
    ]
    return _finish(rng, surf, terms, value, cands, working, numeric, stem=(text, latex))


def _hard_cone(rng):
    surf = S.cone(rng)
    a, h, U, L = surf.a, surf.slope, surf.u_range[1], sympy.sympify(surf.v_range[1])
    terms = _pick_terms(rng, _CUBIC_POOL, rng.choice([2, 3]))
    value = _cone_value(terms, a, h, U, L)
    if value == 0:
        raise _Redraw
    wrong = [
        (_cone_value(terms, a, h, U, L, scale=False), "vc2-ssi-scale-dropped"),
        (_cone_value(terms, a, h, U, L, trig_slip=True), "vc2-ssi-trig-average-slip"),
        (_cone_value(terms, a, h, U, L, jacobian=False), "vc2-ssi-jacobian-factor-dropped"),
        (_cone_value([(1, 0, 0, 0)], a, h, U, L), "vc2-ssi-integrand-dropped"),
    ]
    g = _g_expr(terms)
    working = [
        "Write x, y and z in terms of u and v, multiply g by a √(a² + h²) u and integrate: powers of u with cos v and sin v."
    ]
    return _finish(
        rng, surf, terms, value, _cyl_cone_cands(terms, value, wrong), working, S.numeric_area(surf, integrand=g)
    )


def _hard(rng):
    roll = rng.random()
    if roll < 0.4:
        return _hard_sphere(rng)
    if roll < 0.7:
        return _hard_triangle(rng)
    return _hard_cone(rng)


@generator(
    "s2a2_scalar_surface",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a2_scalar_surface(rng: random.Random, tier="easy"):
    build = {"easy": _plane_question, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(40, "a scalar surface integral question with enough distractors"):
        try:
            return build(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_EXAM_STEMS = (
    "A thin dome-shaped shell is modelled by the surface S with position vector {r}, for {region}, where lengths are in "
    "metres. At the point (x, y, z) the shell has surface density {k}z kilograms per square metre.",
    "A thin spherical cap is modelled by {r} for {region}, with lengths in metres. Its surface density at the point (x, y, z) "
    "is {k}z kilograms per square metre, so it is denser higher up.",
)
_ALPHAS = (PI / 2, PI / 3, 2 * PI / 3)


def _exam_once(rng):
    a = rng.randint(1, 6)
    k = rng.randint(2, 9)
    alpha = rng.choice(_ALPHAS)
    surf = S.revolution(
        a * sympy.sin(u),
        a * sympy.cos(u),
        (0, alpha),
        (0, 2 * PI),
        "sphere",
        0,
        speed=a * a * sympy.sin(u),
        a=a,
        alpha=alpha,
    )
    S.verify_surface(surf)
    c = a * a
    mass = sympy.expand(PI * k * a**3 * sympy.sin(alpha) ** 2)
    area = sympy.expand(2 * PI * a * a * (1 - sympy.cos(alpha)))
    mean = sympy.expand(k * a * (1 + sympy.cos(alpha)) / 2)
    g = k * z
    check(
        V.close(S.numeric_area(surf, integrand=g, n=8), V.float_of(mass), 1e-7),
        "mass disagrees with the numerical integral",
    )
    check(V.close(S.numeric_area(surf, n=8), V.float_of(area), 1e-7), "area disagrees with the numerical integral")
    check(V.close(V.float_of(mass) / V.float_of(area), V.float_of(mean), 1e-12), "mean density mismatch")

    sa, ca = sympy.sin(alpha), sympy.cos(alpha)
    a_out = V.expr_answer(c)
    a_dist = _usable(
        a_out["answer"],
        a_out["answer_text"],
        [
            V.dexpr(a, "vc2-ssi-scale-dropped"),
            V.dexpr(a**3, "vc2-ssi-radius-cubed"),
            V.dexpr(2 * a, "vc2-ssi-derivative-slip"),
            V.dexpr(a + a * a, "vc2-ssi-scale-added"),
            V.dexpr(sympy.Integer(1), "vc2-ssi-area-element-dropped"),
        ],
    )
    b_out = V.expr_answer(mass)
    b_dist = _usable(
        b_out["answer"],
        b_out["answer_text"],
        [
            V.dexpr(PI * k * a**2 * sa**2, "vc2-ssi-scale-dropped"),
            V.dexpr(2 * PI * k * a**3 * sa**2, "vc2-ssi-trig-average-slip"),
            V.dexpr(2 * PI * k * a**3 * (1 - ca), "vc2-ssi-height-taken-constant"),
            V.dexpr(PI * k * a**3 * ca**2, "vc2-ssi-substitution-limits-slip"),
            V.dexpr(PI * k * a**3 * sa, "vc2-ssi-trig-power-slip"),
        ],
    )
    c_out = V.expr_answer(area)
    c_dist = _usable(
        c_out["answer"],
        c_out["answer_text"],
        [
            V.dexpr(2 * PI * a * (1 - ca), "vc2-ssi-scale-dropped"),
            V.dexpr(2 * PI * a * a * ca, "vc2-ssi-trig-average-slip"),
            V.dexpr(2 * PI * a * a * (1 + ca), "vc2-ssi-antiderivative-sign"),
            V.dexpr(PI * a * a * (1 - ca), "vc2-ssi-angle-halved"),
            V.dexpr(2 * PI * a * a * alpha, "vc2-ssi-sine-dropped"),
        ],
    )
    d_out = V.expr_answer(mean)
    d_dist = _usable(
        d_out["answer"],
        d_out["answer_text"],
        [
            V.dexpr(mass, "vc2-ssi-average-not-divided"),
            V.dexpr(area / mass, "vc2-ssi-average-inverted"),
            V.dexpr(k * a * sa**2 / (1 - ca), "vc2-ssi-trig-average-slip"),
            V.dexpr(k * a * (1 - ca) / 2, "vc2-ssi-antiderivative-sign"),
            V.dexpr(sympy.Integer(k * a), "vc2-ssi-average-height-wrong"),
        ],
    )
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    stem_text, stem_latex = V.fill(
        rng.choice(_EXAM_STEMS), r=S.rvec(surf), region=S.region_phrase(surf), k=V.m(sympy.Integer(k))
    )

    def part(template, out, dist, marks, working, follow=None):
        text, latex = V.fill(template)
        d_ = dict(prompt=text, prompt_latex=latex, marks=marks, working=working, distractors=dist, **out)
        if follow:
            d_["follow_through"] = follow
        return d_

    parts = [
        part(
            "The area element is dS = c sin(u) du dv for a constant c. Find c.",
            a_out,
            a_dist,
            2,
            ["Cross r_u and r_v: the length of the result is a² sin u, so c = a²."],
        ),
        part(
            "Find the exact total mass of the shell, in kilograms.",
            b_out,
            b_dist,
            3,
            [
                "Mass = the double integral of k z c sin u du dv with z = a cos u, which is k a c × 2π × (sin² of the upper limit)/2."
            ],
            follow=f"part_a*{a}*{k}*pi*sin({sympy.sstr(alpha)})**2",
        ),
        part(
            "Find the exact surface area of the shell, in square metres.",
            c_out,
            c_dist,
            2,
            ["Area = the double integral of c sin u du dv = c × 2π × (1 - cos of the upper limit)."],
            follow=f"part_a*2*pi*(1 - cos({sympy.sstr(alpha)}))",
        ),
        part(
            "Find the average surface density of the shell, which is its mass divided by its area, in kilograms per square metre.",
            d_out,
            d_dist,
            2,
            ["Average density = mass ÷ area, using your answers to parts (b) and (c)."],
            follow="part_b/part_c",
        ),
    ]
    return dict(text=stem_text, latex=stem_latex, parts=parts)


@generator("exd_x_s2a2_scalar_surface", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a2_scalar_surface(rng: random.Random, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam scalar surface integral with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")
