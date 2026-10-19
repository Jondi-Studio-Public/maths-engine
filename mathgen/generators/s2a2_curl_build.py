"""Generates `s2a2_curl` questions (easy, medium, hard) and the exam chain `exd_x_s2a2_curl`.

``uc2-curl-of-field`` (``multivariable-and-vector-calculus``, answer ``expression``): the curl
∇ × F = ⟨∂R/∂y - ∂Q/∂z, ∂P/∂z - ∂R/∂x, ∂Q/∂x - ∂P/∂y⟩ of a three-dimensional vector field. A vector answer is a
bracketed triple of expressions, graded component by component in the order i, j, k.

* easy: a polynomial field of degree up to three; the curl at a lattice point, as a triple of integers.
* medium: a field whose components carry sin, cos or exponential factors of products of the other variables; the curl
  as a triple of expressions in x, y and z (chain rule in every non-zero component).
* hard, one of
  (a) two unknown constants a and b in the field, and the value of the curl at a point given; find (a, b);
  (b) a rotation field ω × r plus the gradient of a scalar function: the curl is 2ω whatever the gradient part is;
  (c) a cyclic field of terms c x e^(yz), evaluated exactly at a point where some exponents vanish.
* exam (``exd_x_s2a2_curl``): the vorticity ∇ × v of a velocity field: (a) its i-component, (b) its j-component,
  (c) its k-component, each as an expression, then (d) its magnitude at a point, which follows through from all three.

Every exact curl is confirmed against fourth-order central differences of the field at lattice points.
"""

from __future__ import annotations

import random

from functools import lru_cache

import sympy

from ._s2_options import spot_check
from ..core import attempts, check, generator
from . import _s2a2_vec as V
from ._s2a2_vec import x, y, z

_NAMES = ("F", "F", "v", "G")
XYZ = (x, y, z)


class _Redraw(Exception):
    """A draw that cannot give enough distinct wrong answers."""


# ---------------------------------------------------------------------------------------------------------------
# the curl and its wrong versions


@lru_cache(maxsize=512)
def _cached_partials(F: tuple):
    return tuple(tuple(sympy.diff(c, v) for v in XYZ) for c in F)


def _partials(F):
    """D[i][j] = ∂F_i/∂x_j (differentiated once per field, however often it is asked for)."""
    return [list(row) for row in _cached_partials(tuple(F))]


def _curl_from(D):
    return (D[2][1] - D[1][2], D[0][2] - D[2][0], D[1][0] - D[0][1])


def _wrong_curls(D):
    """(vector, misconception id): the curls a learner writes after a standard slip."""
    i, j, k = _curl_from(D)
    return [
        ((i, -j, k), "vc2-curl-j-sign"),
        ((-i, -j, -k), "vc2-curl-reversed-order"),
        ((k, i, j), "vc2-curl-components-shifted"),
        ((D[2][1], D[0][2], D[1][0]), "vc2-curl-partial-dropped"),
        ((D[2][1] + D[1][2], D[0][2] + D[2][0], D[1][0] + D[0][1]), "vc2-curl-partials-added"),
    ]


def _simplify_vec(vec):
    return tuple(sympy.expand(c) for c in vec)


def _verify_curl(F, exact):
    for p in ((1, 2, -1), (-2, 1, 3)):
        want = V.fd_curl(F, p)
        got = tuple(V.float_of(V.at(c, p)) for c in exact)
        check(V.close_vec(want, got, 1e-5), "curl disagrees with the numerical difference")


def _vec_distractors(D, extra=()):
    out = [V.dvec(_simplify_vec(vec), tag) for vec, tag in _wrong_curls(D)]
    return list(extra) + out


def _sentence(template, name, F, **slots):
    return V.fill(template, op=V.curlop(name), F=V.cat(V.fdef(name), V.mv(F)), **slots)


# ---------------------------------------------------------------------------------------------------------------
# easy


def _easy(rng):
    for _ in attempts(200, "an easy curl"):
        F = tuple(V.rand_poly(rng, terms=3, max_degree=3) for _ in range(3))
        pt = V.lattice_point(rng, -2, 3)
        c = tuple(V.at(comp, pt) for comp in _curl_from(_partials(F)))
        if sum(1 for e in c if e != 0) >= 2 and all(abs(e) < 300 for e in c):
            return F, pt, c
    raise AssertionError("unreachable")


_EASY_STEMS = (
    "Find the curl {op} of the vector field {F} at the point {P}. Write your answer as an ordered triple "
    "(i-component, j-component, k-component).",
    "The vector field {F} is defined on all of three-dimensional space. Calculate {op} at the point {P}, written as an "
    "ordered triple (i-component, j-component, k-component).",
)


# ---------------------------------------------------------------------------------------------------------------
# medium: a function of a product of two variables in each component


def _medium_field(rng):
    """Components P, Q, R, each a polynomial plus c w f(a b), where f is sin, cos or exp, a and b are the two variables
    other than the component's own, and w is the component's own variable or 1. Returns the field and the data needed to
    form the chain-rule slips."""
    comps, data = [], []
    for own, (a, b) in zip(XYZ, ((y, z), (z, x), (x, y))):
        f = rng.choice([sympy.sin, sympy.cos, sympy.exp])
        c = V.nz(rng, -3, 4)
        w = rng.choice([sympy.Integer(1), own])
        poly = V.rand_poly(rng, terms=rng.choice([1, 2]), max_degree=2)
        g = a * b
        comps.append(c * w * f(g) + poly)
        data.append((poly, c * w, f, g))
    return tuple(comps), data


def _slipped_partials(F, data):
    """The matrix of partials with every derivative of a function-of-a-product taken without the inner factor."""
    D = _partials(F)
    S = [row[:] for row in D]
    for i, (poly, cw, f, g) in enumerate(data):
        t = sympy.Dummy("t")
        outer = sympy.diff(f(t), t).subs(t, g)  # f'(g), the chain-rule slip
        for j, var in enumerate(XYZ):
            if var in g.free_symbols:
                S[i][j] = sympy.diff(poly, var) + cw * outer + (sympy.diff(cw, var) * f(g))
    return S


def _medium(rng):
    for _ in attempts(100, "a medium curl"):
        F, data = _medium_field(rng)
        D = _partials(F)
        curl = _simplify_vec(_curl_from(D))
        if any(c == 0 for c in curl):
            continue
        slip = _simplify_vec(_curl_from(_slipped_partials(F, data)))
        return F, D, curl, slip
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# hard


def _hard_unknowns(rng):
    """A polynomial field with constants a and b in it, and a point at which the curl determines both."""
    a, b = sympy.symbols("a b")
    for _ in attempts(300, "a field with two unknown constants"):
        base = tuple(V.rand_poly(rng, terms=2, max_degree=3) for _ in range(3))
        ma = V.rand_poly(rng, terms=1, max_degree=3, lo=1, hi=1, min_degree=2)
        mb = V.rand_poly(rng, terms=1, max_degree=3, lo=1, hi=1, min_degree=2)
        slot_a, slot_b = rng.sample(range(3), 2)
        F = list(base)
        F[slot_a] = F[slot_a] + a * ma
        F[slot_b] = F[slot_b] + b * mb
        F = tuple(sympy.expand(c) for c in F)
        pt = V.lattice_point(rng, -2, 3, nonzero=True)
        curl = tuple(sympy.expand(V.at(c, pt)) for c in _curl_from(_partials(F)))
        rows = [[sympy.diff(c, s) for s in (a, b)] for c in curl]
        if sympy.Matrix(rows).rank() < 2:
            continue
        av, bv = rng.choice([n for n in range(-4, 5) if n]), rng.choice([n for n in range(-4, 5) if n])
        if av == bv:
            continue
        target = tuple(c.subs({a: av, b: bv}) for c in curl)
        if any(abs(t) > 400 for t in target):
            continue
        sol = sympy.solve([c - t for c, t in zip(curl, target)], [a, b], dict=True)
        check(sol == [{a: av, b: bv}], "the constants are not determined by the curl")
        G = tuple(c.subs({a: av, b: bv}) for c in F)
        check(V.close_vec(V.fd_curl(G, pt), target, 1e-5), "the curl at the point disagrees with the difference")
        return F, pt, target, av, bv
    raise AssertionError("unreachable")


def _hard_rotation(rng):
    w = [V.nz(rng, -4, 4) for _ in range(3)]
    r = (x, y, z)
    rot = V.cross(w, r)
    f = V.rand_poly(rng, terms=3, max_degree=3, min_degree=2)
    if rng.random() < 0.5:
        a, b = rng.sample(list(XYZ), 2)
        f = f + rng.choice([sympy.exp(a * b), sympy.sin(a * b)])
    grad = V.grad(f)
    F = tuple(sympy.expand(r_ + g_) for r_, g_ in zip(rot, grad))
    answer = tuple(sympy.Integer(2 * c) for c in w)
    check(V.close_vec(V.fd_curl(F, (1, 2, -1)), answer, 1e-5), "rotation curl disagrees with the difference")
    return F, answer, w


def _hard_exp(rng):
    pairs = [(x, y * z), (y, z * x), (z, x * y)]
    rng.shuffle(pairs)
    terms = [(V.nz(rng, -2, 3), a, arg) for a, arg in pairs]
    F = tuple(c * a * sympy.exp(arg) for c, a, arg in terms)
    for _ in attempts(300, "a point for the exponential curl"):
        pt = tuple(rng.randint(-2, 3) for _ in range(3))
        vals = [V.at(arg, pt) for _, _, arg in terms]
        if any(v == 0 for v in vals) and any(v != 0 for v in vals) and all(abs(v) <= 6 for v in vals):
            break
    D = _partials(F)
    exact = tuple(sympy.simplify(V.at(c, pt)) for c in _curl_from(D))
    check(V.close_vec(V.fd_curl(F, pt), tuple(V.float_of(c) for c in exact), 1e-5), "exponential curl disagrees")
    return F, pt, exact, D


def _hard(rng, name):
    roll = rng.random()
    if roll < 0.35:
        F, pt, target, av, bv = _hard_unknowns(rng)
        text, latex = V.fill(
            "The vector field {F} contains constants a and b. At the point {P}, {op} = {T}. Find a and b, "
            "and give your answer as the ordered pair (a, b).",
            F=V.cat(V.fdef(name), V.mv(F)),
            op=V.curlop(name),
            P=V.mpt(pt),
            T=V.mv(target),
        )
        out = V.vec_answer((av, bv))
        cands = [
            V.dvec((bv, av), "vc2-curl-parameters-swapped"),
            V.dvec((-av, bv), "vc2-curl-solve-sign"),
            V.dvec((av, -bv), "vc2-curl-solve-sign"),
            V.dvec((-av, -bv), "vc2-curl-solve-sign"),
        ]
        working = ["Each component of the curl at the point is a linear expression in a and b; solve two of them."]
    elif roll < 0.7:
        F, answer, w = _hard_rotation(rng)
        text, latex = _sentence(
            "Find {op} for {F}. Write your answer as an ordered triple (i-component, j-component, k-component).",
            name,
            F,
        )
        out = V.vec_answer(answer)
        D = _partials(F)
        cands = [
            V.dvec(tuple(sympy.Integer(c) for c in w), "vc2-curl-factor-two-dropped"),
            V.dvec(tuple(-2 * sympy.Integer(c) for c in w), "vc2-curl-reversed-order"),
            V.dvec((2 * w[0], -2 * w[1], 2 * w[2]), "vc2-curl-j-sign"),
            V.dvec((2 * w[2], 2 * w[0], 2 * w[1]), "vc2-curl-components-shifted"),
        ]
        _ = D
        working = ["The gradient part has zero curl, and the curl of ω × r is 2ω, so the curl is constant."]
    else:
        F, pt, exact, D = _hard_exp(rng)
        text, latex = _sentence(
            "Find the exact value of {op} at the point {P} for the field {F}. Write your answer as an ordered triple "
            "(i-component, j-component, k-component).",
            name,
            F,
            P=V.mpt(pt),
        )
        out = V.vec_answer(exact)
        cands = [V.dvec(tuple(sympy.simplify(V.at(c, pt)) for c in vec), tag) for vec, tag in _wrong_curls(D)]
        working = ["Each component needs the product rule and the chain rule; evaluate at the point."]
    dist = V.usable(out["answer"], out["answer_text"], cands)
    if len(dist) < 3:
        raise _Redraw
    return dict(text=text, latex=latex, working=working, distractors=dist, **out)


# ---------------------------------------------------------------------------------------------------------------


def _single(rng, tier):
    name = rng.choice(_NAMES)
    if tier == "hard":
        return _hard(rng, name)
    if tier == "easy":
        F, pt, c = _easy(rng)
        text, latex = _sentence(rng.choice(_EASY_STEMS), name, F, P=V.mpt(pt))
        out = V.vec_answer(c)
        D = _partials(F)
        cands = [V.dvec(tuple(V.at(e, pt) for e in vec), tag) for vec, tag in _wrong_curls(D)]
        cands.append(V.dexpr(V.at(V.div(F), pt), "vc2-curl-gave-divergence"))
        working = [f"curl F = ⟨R_y - Q_z, P_z - R_x, Q_x - P_y⟩, evaluated at the point: {out['answer_text']}."]
    else:
        F, D, curl, slip = _medium(rng)
        _verify_curl(F, curl)
        text, latex = _sentence(
            "Find the curl {op} of the vector field {F}. Give your answer as an ordered triple of expressions in x, y "
            "and z, in the order of the i, j and k components.",
            name,
            F,
        )
        out = V.vec_answer(curl)
        cands = [V.dvec(slip, "vc2-chain-factor-dropped")] + _vec_distractors(D)
        working = [
            "Use the chain rule on each factor f(ab), and the determinant pattern for the components of the curl."
        ]
    dist = V.usable(out["answer"], out["answer_text"], cands)
    if len(dist) < 3:
        raise _Redraw
    return dict(text=text, latex=latex, working=working, distractors=dist, **out)


@generator(
    "s2a2_curl",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a2_curl(rng: random.Random, tier="easy"):
    for _ in attempts(30, "a curl question with enough distractors"):
        try:
            return _single(rng, tier)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_EXAM_STEMS = (
    "Water circulating in a small tidal pool is modelled by the velocity field {F} metres per second, where x, y and z "
    "are in metres. The vorticity of the flow is the curl {op}.",
    "A wind-tunnel model gives the velocity field {F} (metres per second, with x, y and z in metres) for the air around "
    "a rotor. The vorticity of the air is the curl {op}.",
)


def _exam_once(rng):
    for _ in attempts(100, "an exam field"):
        F = tuple(V.rand_poly(rng, terms=3, max_degree=3) for _ in range(3))
        D = _partials(F)
        curl = _simplify_vec(_curl_from(D))
        pt = V.lattice_point(rng, -2, 3, nonzero=True)
        vals = [V.at(c, pt) for c in curl]
        n = sum(v * v for v in vals)
        if all(v != 0 for v in vals) and n < 400 and not sympy.sqrt(n).is_Integer:
            break
    mag = sympy.sqrt(n)
    if spot_check(F, pt):  # the finite-difference route is slow: one question in four
        _verify_curl(F, curl)
        check(
            V.close(V.float_of(mag), sum(float(v) ** 2 for v in V.fd_curl(F, pt)) ** 0.5, 1e-5), "magnitude disagrees"
        )
    wrongs = [
        # (i, j, k) wrong versions of each component: reversed, added, wrong variables, one partial only
        [
            (D[1][2] - D[2][1], "vc2-curl-reversed-order"),
            (D[2][1] + D[1][2], "vc2-curl-partials-added"),
            (D[2][2] - D[1][1], "vc2-curl-wrong-variables"),
            (D[2][1], "vc2-curl-partial-dropped"),
        ],
        [
            (D[2][0] - D[0][2], "vc2-curl-reversed-order"),
            (D[0][2] + D[2][0], "vc2-curl-partials-added"),
            (D[0][0] - D[2][2], "vc2-curl-wrong-variables"),
            (D[0][2], "vc2-curl-partial-dropped"),
        ],
        [
            (D[0][1] - D[1][0], "vc2-curl-reversed-order"),
            (D[1][0] + D[0][1], "vc2-curl-partials-added"),
            (D[1][1] - D[0][0], "vc2-curl-wrong-variables"),
            (D[1][0], "vc2-curl-partial-dropped"),
        ],
    ]
    comp_parts = []
    for comp, options in zip(curl, wrongs):
        out = V.expr_answer(comp)
        dist = V.usable(out["answer"], out["answer_text"], [V.dexpr(sympy.expand(e), t) for e, t in options])
        if len(dist) < 2:
            raise _Redraw
        comp_parts.append((out, dist))
    a_, b_, c_ = vals
    mag_out = V.expr_answer(mag)
    mag_cands = [
        V.dexpr(n, "vc2-curl-magnitude-not-rooted"),
        V.dexpr(abs(a_) + abs(b_) + abs(c_), "vc2-curl-magnitude-sum"),
        V.dexpr(sympy.sqrt(a_**2 + b_**2), "vc2-curl-magnitude-component-dropped"),
        V.dexpr(sympy.sqrt(a_**2 + c_**2), "vc2-curl-magnitude-component-dropped"),
        V.dexpr(sympy.sqrt(sum(vals)) if sum(vals) > 0 else sympy.sqrt(abs(sum(vals))), "vc2-curl-magnitude-sum"),
    ]
    mag_dist = V.usable(mag_out["answer"], mag_out["answer_text"], mag_cands)
    if len(mag_dist) < 2:
        raise _Redraw
    ax, ay, az = pt
    stem_text, stem_latex = V.fill(rng.choice(_EXAM_STEMS), F=V.cat(V.fdef("v"), V.mv(F)), op=V.curlop("v"))

    def part(prompt_template, out, dist, marks, working, follow=None, **slots):
        text, latex = V.fill(prompt_template, **slots)
        d_ = dict(prompt=text, prompt_latex=latex, marks=marks, working=working, distractors=dist, **out)
        if follow:
            d_["follow_through"] = follow
        return d_

    subs = f"({ax}, {ay}, {az})"
    follow = (
        f"sqrt(Subs(part_a, (x, y, z), {subs})**2 + Subs(part_b, (x, y, z), {subs})**2 "
        f"+ Subs(part_c, (x, y, z), {subs})**2)"
    )
    parts = [
        part(
            "Find the i-component of the vorticity, as an expression in x, y and z.",
            *comp_parts[0],
            2,
            ["The i-component is ∂R/∂y - ∂Q/∂z."],
        ),
        part(
            "Find the j-component of the vorticity, as an expression in x, y and z.",
            *comp_parts[1],
            2,
            ["The j-component is ∂P/∂z - ∂R/∂x."],
        ),
        part(
            "Find the k-component of the vorticity, as an expression in x, y and z.",
            *comp_parts[2],
            2,
            ["The k-component is ∂Q/∂x - ∂P/∂y."],
        ),
        part(
            "Hence find the magnitude of the vorticity at the point P{P}, in exact form.",
            mag_out,
            mag_dist,
            3,
            ["Substitute the point into the three components and take the square root of the sum of their squares."],
            follow=follow,
            P=V.mpt(pt),
        ),
    ]
    return dict(text=stem_text, latex=stem_latex, parts=parts)


@generator("exd_x_s2a2_curl", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a2_curl(rng: random.Random, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam curl with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")
