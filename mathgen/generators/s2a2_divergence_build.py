"""Generates `s2a2_divergence` questions (easy, medium, hard) and the exam chain `exd_x_s2a2_divergence`.

``uc2-divergence-of-field`` (``multivariable-and-vector-calculus``, answer ``expression``): the divergence
∇ · F = ∂P/∂x + ∂Q/∂y + ∂R/∂z of a vector field in three dimensions.

* easy: a polynomial field of degree up to three with mixed terms; the divergence as an expression in x, y and z.
* medium: a field whose components carry exponential and trigonometric factors, so the chain rule and the product
  rule are needed on each component's own variable; the divergence as an expression.
* hard, one of
  (a) a field that is the sum of a divergence-free part (the curl of a polynomial field) and one or two terms with
  unknown constants; find the constant(s) that make the field solenoidal (the divergence is zero everywhere);
  (b) the radial field (x² + y² + z²)^n ⟨x, y, z⟩ evaluated at a lattice point on a sphere of integer radius;
  (c) a cyclic field of products x e^(xy), y e^(yz), z e^(zx), evaluated exactly at a point where some exponents vanish.
* exam (``exd_x_s2a2_divergence``): (a) ∇ · v as an expression, (b) its value at a point (follows through from
  (a)), (c) the constant k that cancels it at the point when a term k z is added (follows through from (b)),
  (d) ∇ · (g v) at the point for a scalar density g, by the product rule (follows through from (b)).

Every exact divergence is confirmed against a fourth-order central difference of the field at lattice points; the
constants of the hard solenoidal questions are confirmed by substituting them back and differencing.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from . import _s2a2_vec as V
from ._s2a2_vec import x, y, z

_NAMES = ("F", "F", "v", "G")


class _Redraw(Exception):
    """A draw that cannot give enough distinct wrong answers."""


def _verify_div(F, exact):
    """The exact divergence (an expression) against a numerical difference of the field at lattice points."""
    for p in ((1, 2, -1), (-2, 1, 3)):
        check(V.close(V.fd_div(F, p), V.float_of(V.at(exact, p))), "divergence disagrees with the numerical difference")


def _own_poly(rng, var, terms=(2, 3), max_degree=3):
    for _ in attempts(200, "a component containing its own variable"):
        p = V.rand_poly(rng, terms=rng.choice(terms), max_degree=max_degree)
        if p.has(var):
            return p
    raise AssertionError("unreachable")


def _div_candidates(F):
    """The common wrong divergences of a field: the wrong variables, a dropped component, the curl components added,
    and the three partials left as a vector."""
    P, Q, R = F
    dPx, dQy, dRz = sympy.diff(P, x), sympy.diff(Q, y), sympy.diff(R, z)
    cyclic = sympy.diff(P, y) + sympy.diff(Q, z) + sympy.diff(R, x)
    return [
        V.dexpr(sympy.expand(cyclic), "vc2-div-wrong-variable"),
        V.dexpr(sympy.expand(dPx + dQy), "vc2-div-component-dropped"),
        V.dexpr(sympy.expand(sum(V.curl(F))), "vc2-div-confused-with-curl"),
        V.dvec((dPx, dQy, dRz), "vc2-div-left-as-vector"),
    ]


def _sentence(template, name, F):
    return V.fill(template, op=V.divop(name), F=V.cat(V.fdef(name), V.mv(F)))


_STEMS = (
    "Find the divergence {op} of the vector field {F}. Give your answer as an expression in x, y and z.",
    "The vector field {F} is defined on all of three-dimensional space. Calculate {op} as an expression in x, y and z.",
    "Calculate {op} for {F}. Write the result as an expression in x, y and z.",
)


# ---------------------------------------------------------------------------------------------------------------
# easy and medium


def _easy(rng):
    for _ in attempts(100, "an easy divergence"):
        F = (_own_poly(rng, x), _own_poly(rng, y), _own_poly(rng, z))
        d = sympy.expand(V.div(F))
        if len(sympy.Add.make_args(d)) >= 3:
            return F, d, []
    raise AssertionError("unreachable")


def _blocks(t, a):
    """(factor in the component's own variable t, its derivative with a rule dropped, misconception id), where a is
    another variable. The slip is what a learner writes after dropping the chain-rule factor or a product-rule term."""
    chain, product = "vc2-chain-factor-dropped", "vc2-product-rule-term-dropped"
    return [
        (sympy.sin(t * a), sympy.cos(t * a), chain),
        (sympy.exp(t * a), sympy.exp(t * a), chain),
        (sympy.cos(a * t**2), -sympy.sin(a * t**2), chain),
        (sympy.exp(a * t**2), sympy.exp(a * t**2), chain),
        (t * sympy.exp(t * a), t * a * sympy.exp(t * a), product),
        (t * sympy.cos(t * a), -t * a * sympy.sin(t * a), product),
    ]


def _medium(rng):
    for _ in attempts(100, "a medium divergence"):
        comps, slips = [], []
        for var in (x, y, z):
            a = rng.choice([w for w in (x, y, z) if w != var])
            expr, slip, tag = rng.choice(_blocks(var, a))
            c = V.nz(rng, -3, 4)
            comps.append(c * expr + V.rand_poly(rng, terms=rng.choice([1, 2]), max_degree=2))
            slips.append((c * sympy.diff(expr, var), c * slip, tag))
        F = tuple(comps)
        d = sympy.expand(V.div(F))
        if d == 0:
            continue
        extra, seen = [], set()
        for true_d, slip, tag in slips:
            if tag not in seen and sympy.simplify(true_d - slip) != 0:
                seen.add(tag)
                extra.append(V.dexpr(sympy.expand(d - true_d + slip), tag))
        return F, d, extra
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# hard


def _solenoidal_part(rng):
    """The curl of a random polynomial field: a field with zero divergence, every component non-zero."""
    for _ in attempts(200, "a divergence-free polynomial field"):
        A = tuple(V.rand_poly(rng, terms=rng.choice([1, 2]), max_degree=3, min_degree=2) for _ in range(3))
        S = tuple(sympy.expand(c) for c in V.curl(A))
        if all(c != 0 for c in S):
            check(sympy.expand(V.div(S)) == 0, "the curl of a field has non-zero divergence")
            return S
    raise AssertionError("unreachable")


def _hard_solenoidal(rng, two: bool):
    k, m = sympy.symbols("k m")
    S = _solenoidal_part(rng)
    p = rng.choice([0, 1])
    vz = rng.choice([z, z**2])
    c = V.nz(rng, -4, 4)
    # k x y^p vz in P has divergence k y^p vz; c y^(p+1) vz in Q has divergence (p+1) c y^p vz
    P, Q, R = S[0] + k * x * y**p * vz, S[1] + c * y ** (p + 1) * vz, S[2]
    names, answer = [k], [-(p + 1) * c]
    if two:
        s_, t_ = rng.choice([(0, 1), (1, 0), (1, 1), (0, 2)])
        e = V.nz(rng, -3, 3)
        # m z x^s y^t in R has divergence m x^s y^t; e x^(s+1) y^t in P has divergence (s+1) e x^s y^t
        R, P = R + m * z * x**s_ * y**t_, P + e * x ** (s_ + 1) * y**t_
        names.append(m)
        answer.append(-(s_ + 1) * e)
    F = (sympy.expand(P), sympy.expand(Q), sympy.expand(R))
    solution = dict(zip(names, answer))
    check(sympy.expand(V.div(F).subs(solution)) == 0, "the divergence does not vanish at the stated constants")
    G = tuple(comp.subs(solution) for comp in F)
    check(V.close(V.fd_div(G, (1, 2, -1)), 0), "numerical divergence is not zero")
    check(sympy.expand(V.div(F).subs({names[0]: answer[0] + 1, **dict(zip(names[1:], answer[1:]))})) != 0, "not unique")
    return F, names, answer, c, p


def _hard_radial(rng):
    n = rng.choice([-2, -1, 1, 2])
    pt = rng.choice([(1, 2, 2), (2, 1, 2), (2, 2, 1), (-1, 2, 2), (1, -2, 2), (2, 3, 6), (3, 2, 6), (1, 4, 8)])
    r2 = sum(c * c for c in pt)

    def val(coef, power):
        return Fraction(coef) * Fraction(r2) ** power

    value = val(3 + 2 * n, n)
    F = tuple(c * (x**2 + y**2 + z**2) ** n for c in (x, y, z))
    check(V.close(V.fd_div(F, pt), float(value)), "radial divergence disagrees with the numerical difference")
    cands = [
        (val(3, n), "vc2-div-power-factor-not-differentiated"),
        (val(3 + n, n), "vc2-div-chain-factor-dropped"),
        (val(3 + 2 * n, n - 1), "vc2-div-power-rule-slip"),
        (val(1 + 2 * n, n), "vc2-div-own-variable-undercounted"),
    ]
    return n, pt, value, cands


def _hard_exp(rng):
    partners = rng.choice([(y, z, x), (z, x, y)])
    cs = [V.nz(rng, -2, 3) for _ in range(3)]
    F = tuple(c * var * sympy.exp(var * w) for c, var, w in zip(cs, (x, y, z), partners))
    for _ in attempts(200, "a point for the exponential field"):
        pt = tuple(rng.randint(-2, 3) for _ in range(3))
        exps = [var * w for var, w in zip((x, y, z), partners)]
        vals = [V.at(e, pt) for e in exps]
        if any(e == 0 for e in vals) and any(e != 0 for e in vals) and all(abs(e) <= 6 for e in vals):
            break
    exact = sympy.simplify(V.at(V.div(F), pt))
    check(V.close(V.fd_div(F, pt), V.float_of(exact), 1e-5), "exponential divergence disagrees with the difference")
    product_dropped = chain_dropped = 0
    for c, var, w in zip(cs, (x, y, z), partners):
        e = sympy.exp(var * w)
        product_dropped += c * var * sympy.diff(e, var)
        chain_dropped += c * (e + var * e)
    slips = [
        (V.at(product_dropped, pt), "vc2-product-rule-term-dropped"),
        (V.at(chain_dropped, pt), "vc2-chain-factor-dropped"),
    ]
    # e^0 read as 0 in the terms whose exponent vanishes at the point
    misread = 0
    for c, var, w in zip(cs, (x, y, z), partners):
        exponent = V.at(var * w, pt)
        misread += 0 if exponent == 0 else V.at(c * sympy.exp(var * w) * (1 + var * w), pt)
    return F, pt, exact, slips, misread


def _hard(rng, name):
    roll = rng.random()
    if roll < 0.4:
        two = roll < 0.15
        F, names, answer, c, p = _hard_solenoidal(rng, two)
        text, latex = V.fill(
            "The vector field {F} contains "
            + ("the constants k and m" if two else "a constant k")
            + ". Find "
            + ("the values of k and m" if two else "the value of k")
            + " for which the field is solenoidal, that is, for which {op} = 0 at every point."
            + (" Give your answer as the ordered pair (k, m)." if two else ""),
            F=V.cat(V.fdef(name), V.mv(F)),
            op=V.divop(name),
        )
        if two:
            out = V.vec_answer(answer)
            swapped = [answer[1], answer[0]]
            cands = [
                V.dvec([-answer[0], answer[1]], "vc2-div-solve-sign"),
                V.dvec([answer[0], -answer[1]], "vc2-div-solve-sign"),
                V.dvec(swapped, "vc2-div-parameters-swapped"),
                V.dvec([-answer[0], -answer[1]], "vc2-div-solve-sign"),
            ]
        else:
            out = V.expr_answer(answer[0])
            true = answer[0]
            cands = [
                V.dexpr(-true, "vc2-div-solve-sign"),
                V.dexpr(-c, "vc2-div-param-factor-dropped"),
                V.dexpr(c, "vc2-div-solve-sign"),
                V.dexpr(true * (p + 1), "vc2-div-param-factor-dropped"),
                V.dexpr(true + 1, "vc2-div-param-off-by-one"),
            ]
        working = [
            "The curl of any field has zero divergence, so only the terms with the constants matter: "
            "set the sum of their partial derivatives to zero."
        ]
        kind = "expression"
    elif roll < 0.7:
        n, pt, value, cands_raw = _hard_radial(rng)
        text, latex = V.fill(
            "Let {F}. Find {op} at the point {P}.",
            F=V.cat(V.fdef(name), V.raw(_radial_plain(n) + "⟨x, y, z⟩", _radial_tex(n) + r"\langle x, y, z \rangle")),
            op=V.divop(name),
            P=V.mpt(pt),
        )
        out = V.expr_answer(sympy.Rational(value.numerator, value.denominator))
        cands = [V.dexpr(sympy.Rational(v.numerator, v.denominator), t) for v, t in cands_raw]
        working = [
            f"With r² = x² + y² + z², the divergence is (3 + 2n)(r²)^n = {V.nice(out['answer_text'])} at the point."
        ]
        kind = "expression"
    else:
        F, pt, exact, slips, zero_misread = _hard_exp(rng)
        text, latex = V.fill(
            "Find the exact value of {op} at the point {P} for the field {F}.",
            F=V.cat(V.fdef(name), V.mv(F)),
            op=V.divop(name),
            P=V.mpt(pt),
        )
        out = V.expr_answer(exact)
        cands = [V.dexpr(sympy.simplify(s), t) for s, t in slips]
        cands.append(V.dexpr(sympy.simplify(zero_misread), "vc2-div-exp-zero-as-zero"))
        working = [
            f"Each component needs the product rule and the chain rule; at the point ∇ · F = {V.nice(out['answer_text'])}."
        ]
        kind = "expression"
    dist = V.usable(out["answer"], out["answer_text"], cands, kind=kind)
    if len(dist) < 3:
        raise _Redraw
    return dict(text=text, latex=latex, working=working, distractors=dist, **out)


def _radial_plain(n: int) -> str:
    return "(x² + y² + z²)" + ({1: "", 2: "²"}.get(n) if n > 0 else f"^({n})")


def _radial_tex(n: int) -> str:
    return r"(x^{2} + y^{2} + z^{2})" + ("" if n == 1 else f"^{{{n}}}")


def _single(rng, tier):
    name = rng.choice(_NAMES)
    if tier == "hard":
        return _hard(rng, name)
    F, d, extra = _easy(rng) if tier == "easy" else _medium(rng)
    _verify_div(F, d)
    text, latex = _sentence(rng.choice(_STEMS), name, F)
    out = V.expr_answer(d)
    dist = V.usable(out["answer"], out["answer_text"], extra + _div_candidates(F))
    if len(dist) < 3:
        raise _Redraw
    if tier == "easy":
        working = [
            f"Add ∂P/∂x + ∂Q/∂y + ∂R/∂z, differentiating each component by its own variable: {V.nice(out['answer_text'])}."
        ]
    else:
        working = [
            "Differentiate each component with respect to its own variable, using the chain rule and the product rule "
            f"where they apply, and add: {V.nice(out['answer_text'])}."
        ]
    return dict(text=text, latex=latex, working=working, distractors=dist, **out)


@generator(
    "s2a2_divergence",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a2_divergence(rng: random.Random, tier="easy"):
    for _ in attempts(30, "a divergence question with enough distractors"):
        try:
            return _single(rng, tier)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_EXAM_STEMS = (
    "Air moves through a ventilation duct with velocity {F} metres per second, where x, y and z are in metres.",
    "A computer model of the airflow in a ventilation duct gives the velocity field {F}, in metres per second, with x, y and z in metres.",
)


def _exam_once(rng):
    for _ in attempts(100, "an exam field"):
        F = (_own_poly(rng, x), _own_poly(rng, y), _own_poly(rng, z))
        d = sympy.expand(V.div(F))
        pt = V.lattice_point(rng, -2, 3, nonzero=True)
        b = V.at(d, pt)
        if len(sympy.Add.make_args(d)) >= 3 and b != 0 and abs(b) < 400 and pt[2] not in (1, -1):
            break
    _verify_div(F, d)
    g = x**2 + y**2 + z**2
    g0 = V.at(g, pt)
    c0 = V.at(2 * (x * F[0] + y * F[1] + z * F[2]), pt)  # v . grad g
    full = g0 * b + c0
    # second route for (d): numerical divergence of g v
    gv = tuple(g * c for c in F)
    check(V.close(V.fd_div(gv, pt), float(full)), "div(g v) disagrees with the numerical difference")
    k = -b
    check(V.close(V.fd_div((F[0], F[1], F[2] + k * z), pt), 0), "the cancelling constant does not cancel")
    ax, ay, az = pt

    a_out = V.expr_answer(d)
    a_dist = V.usable(a_out["answer"], a_out["answer_text"], _div_candidates(F), n=4)
    b_out = V.frac_answer(b)
    b_dist = V.usable(
        b_out["answer"],
        b_out["answer_text"],
        [
            V.dfrac(
                V.at(sympy.diff(F[0], y) + sympy.diff(F[1], z) + sympy.diff(F[2], x), pt), "vc2-div-wrong-variable"
            ),
            V.dfrac(V.at(sympy.diff(F[0], x) + sympy.diff(F[1], y), pt), "vc2-div-component-dropped"),
            V.dfrac(sum(V.at(c, pt) for c in F), "vc2-div-field-value-not-divergence"),
            V.dfrac(V.at(sum(V.curl(F)), pt), "vc2-div-confused-with-curl"),
        ],
        kind="fraction",
    )
    c_out = V.frac_answer(k)
    c_dist = V.usable(
        c_out["answer"],
        c_out["answer_text"],
        [
            V.dfrac(b, "vc2-div-solve-sign"),
            V.dfrac(Fraction(int(-b), int(az)), "vc2-div-added-term-not-differentiated"),
            V.dfrac(-b * az, "vc2-div-added-term-not-differentiated"),
            V.dfrac(b + 1, "vc2-div-param-off-by-one"),
        ],
        kind="fraction",
    )
    d_out = V.frac_answer(full)
    d_dist = V.usable(
        d_out["answer"],
        d_out["answer_text"],
        [
            V.dfrac(g0 * b, "vc2-product-rule-term-dropped"),
            V.dfrac(c0, "vc2-product-rule-term-dropped"),
            V.dfrac(g0 * b - c0, "vc2-div-product-rule-sign"),
            V.dfrac(b + c0, "vc2-div-density-factor-dropped"),
        ],
        kind="fraction",
    )
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw
    p_txt = f"({ax}, {ay}, {az})"
    stem_text, stem_latex = V.fill(rng.choice(_EXAM_STEMS), F=V.cat(V.fdef("v"), V.mv(F)))
    ps = V.mpt(pt)

    def part(prompt_template, out, dist, marks, working, follow=None, **slots):
        text, latex = V.fill(prompt_template, **slots)
        d_ = dict(prompt=text, prompt_latex=latex, marks=marks, working=working, distractors=dist, **out)
        if follow:
            d_["follow_through"] = follow
        return d_

    parts = [
        part(
            "Find the divergence {op} as an expression in x, y and z.",
            a_out,
            a_dist,
            3,
            ["Differentiate each component with respect to its own variable and add."],
            op=V.divop("v"),
        ),
        part(
            "Hence find the divergence of v at the point P" + p_txt + ".",
            b_out,
            b_dist,
            1,
            [f"Substitute the coordinates of P into the divergence: {V.nice(b_out['answer_text'])}."],
            follow=f"Subs(part_a, (x, y, z), ({ax}, {ay}, {az}))",
        ),
        part(
            "A fan adds a term kz to the z-component of v, where k is a constant. Find the value of k for which the new "
            "field has zero divergence at P.",
            c_out,
            c_dist,
            2,
            ["The added term contributes k to the divergence everywhere, so k cancels the value found in part (b)."],
            follow="-part_b",
        ),
        part(
            "The air has density g(x, y, z) = x² + y² + z² kilograms per cubic metre. For the original field v, find the "
            "divergence of g v at P.",
            d_out,
            d_dist,
            3,
            ["Product rule: ∇ · (g v) = g ∇ · v + v · ∇g, with g = " + str(g0) + f" and v · ∇g = {c0} at P."],
            follow=f"{g0}*part_b + {c0}",
        ),
    ]
    _ = ps
    return dict(text=stem_text, latex=stem_latex, parts=parts)


@generator("exd_x_s2a2_divergence", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a2_divergence(rng: random.Random, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam divergence with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")
