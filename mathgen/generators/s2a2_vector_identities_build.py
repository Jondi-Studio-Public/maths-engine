"""Generates `s2a2_vector_identities` questions (easy, medium, hard) and the exam chain `exd_x_s2a2_vector_identities`.

``uc2-vector-identities`` (``multivariable-and-vector-calculus``, answer ``expression``): the product rules and the
second-derivative identities of vector calculus, applied to concrete fields and evaluated at a point. Each answer is a
number, or a bracketed triple of numbers for a vector.

* easy: ∇ × (f F) = f ∇ × F + ∇f × F, or ∇ · (f F) = f ∇ · F + F · ∇f, at a point.
* medium, one of: ∇ · (F × G) = G · ∇ × F - F · ∇ × G; ∇²(f g) = f ∇²g + g ∇²f + 2 ∇f · ∇g;
  ∇ · (∇f × G) = -∇f · ∇ × G (because ∇ × ∇f = 0).
* hard, one of: ∇²(x² + y² + z²)^(n/2) = n(n + 1) r^(n-2) at a point; Green's difference
  ∇ · (f ∇g - g ∇f) = f ∇²g - g ∇²f; the double curl ∇ × (∇ × F) = ∇(∇ · F) - ∇²F at a point.
* exam (``exd_x_s2a2_vector_identities``): (a) ∇ · F at a point, (b) F · ∇f at the point, (c) ∇ · (f F) at the point
  (follows through from (a) and (b)), (d) the identity that holds for every smooth f and F, chosen from five
  (multiple choice, the only classification in the chain).

Each value is worked out three ways: by expanding the field and differentiating it directly, by the identity, and by
a numerical difference of the exact curl or divergence.
"""

from __future__ import annotations

import random

import sympy

from ..core import attempts, check, generator
from . import _s2a2_vec as V
from ._s2a2_vec import x, y, z

XYZ = (x, y, z)
_VEC = r"\mathbf{%s}"


class _Redraw(Exception):
    """A draw that cannot give enough distinct wrong answers."""


# ---------------------------------------------------------------------------------------------------------------
# random data


def _scalar(rng, degree=3, terms=(2, 3)):
    return V.rand_poly(rng, terms=rng.choice(terms), max_degree=degree)


def _vector(rng, degree=2, terms=2):
    return tuple(V.rand_poly(rng, terms=terms, max_degree=degree) for _ in range(3))


def _point(rng):
    return V.lattice_point(rng, -2, 3)


def laplacian(e):
    return sum(sympy.diff(e, v, 2) for v in XYZ)


def vec_laplacian(F):
    return tuple(laplacian(c) for c in F)


def scale(a, F):
    return tuple(a * c for c in F)


def add(F, G):
    return tuple(p + q for p, q in zip(F, G))


def sub(F, G):
    return tuple(p - q for p, q in zip(F, G))


def evaluate(F, pt):
    return tuple(V.at(c, pt) for c in F)


def _vec_cands(items, pt):
    return [V.dvec(evaluate(vec, pt), tag) for vec, tag in items]


# ---------------------------------------------------------------------------------------------------------------
# the identities, each with the direct route (expand and differentiate) and the identity route


def curl_fF(f, F):
    direct = V.curl(scale(f, F))
    identity = add(scale(f, V.curl(F)), V.cross(V.grad(f), F))
    return direct, identity


def div_fF(f, F):
    return V.div(scale(f, F)), f * V.div(F) + V.dot(F, V.grad(f))


def div_FxG(F, G):
    return V.div(V.cross(F, G)), V.dot(G, V.curl(F)) - V.dot(F, V.curl(G))


def lap_fg(f, g):
    return laplacian(f * g), f * laplacian(g) + g * laplacian(f) + 2 * V.dot(V.grad(f), V.grad(g))


def div_gradf_x_G(f, G):
    return V.div(V.cross(V.grad(f), G)), -V.dot(V.grad(f), V.curl(G))


def green_difference(f, g):
    gf, gg = V.grad(f), V.grad(g)
    direct = V.div(sub(scale(f, gg), scale(g, gf)))
    return direct, f * laplacian(g) - g * laplacian(f)


def curl_curl(F):
    direct = V.curl(V.curl(F))
    grad_div = V.grad(V.div(F))
    return direct, sub(grad_div, vec_laplacian(F))


def _agree_scalar(direct, identity, pt, numeric=None):
    a, b = V.at(direct, pt), V.at(identity, pt)
    check(sympy.simplify(a - b) == 0, "the identity and the direct route disagree")
    if numeric is not None:
        check(V.close(numeric, V.float_of(a), 1e-5), "the numerical difference disagrees")
    return sympy.nsimplify(a)


def _agree_vector(direct, identity, pt, numeric=None):
    a, b = evaluate(direct, pt), evaluate(identity, pt)
    check(all(sympy.simplify(p - q) == 0 for p, q in zip(a, b)), "the identity and the direct route disagree")
    if numeric is not None:
        check(V.close_vec(numeric, tuple(V.float_of(c) for c in a), 1e-5), "the numerical difference disagrees")
    return a


# ---------------------------------------------------------------------------------------------------------------
# questions


def _prompt(template, **slots):
    return V.fill(template, **slots)


def _easy(rng):
    f, F, pt = _scalar(rng), _vector(rng), _point(rng)
    fd = lambda: V.sdef("f", f)  # noqa: E731
    Fd = V.cat(V.fdef("F"), V.mv(F))
    if rng.random() < 0.5:
        direct, identity = curl_fF(f, F)
        value = _agree_vector(direct, identity, pt, V.fd_curl(scale(f, F), pt))
        if sum(1 for c in value if c != 0) < 2:
            raise _Redraw
        text, latex = _prompt(
            "Let {f} and {F}. Find {op} at the point {P}. Write your answer as an ordered triple "
            "(i-component, j-component, k-component).",
            op=V.raw("∇ × (f F)", r"\nabla \times (f\mathbf{F})"),
            f=fd(),
            F=Fd,
            P=V.mpt(pt),
        )
        cf, cc, gf = f.subs(dict(zip(XYZ, pt))), evaluate(V.curl(F), pt), evaluate(V.grad(f), pt)
        fv, cr = evaluate(F, pt), evaluate(V.cross(V.grad(f), F), pt)
        cands = [
            V.dvec(sub(scale(cf, cc), cr), "vc2-id-cross-term-sign"),
            V.dvec(scale(cf, cc), "vc2-id-product-term-dropped"),
            V.dvec(cr, "vc2-id-product-term-dropped"),
            V.dvec(scale(cf, add(cc, cr)), "vc2-id-scalar-multiplied-twice"),
            V.dvec(V.cross(fv, gf), "vc2-id-cross-term-sign"),
        ]
        out = V.vec_answer(value)
        working = ["Product rule for the curl: ∇ × (f F) = f ∇ × F + ∇f × F, evaluated at the point."]
    else:
        direct, identity = div_fF(f, F)
        value = _agree_scalar(direct, identity, pt, V.fd_div(scale(f, F), pt))
        text, latex = _prompt(
            "Let {f} and {F}. Find {op} at the point {P}.",
            op=V.raw("∇ · (f F)", r"\nabla \cdot (f\mathbf{F})"),
            f=fd(),
            F=Fd,
            P=V.mpt(pt),
        )
        cf, d, dotv = f.subs(dict(zip(XYZ, pt))), V.at(V.div(F), pt), V.at(V.dot(F, V.grad(f)), pt)
        cands = [
            V.dexpr(cf * d - dotv, "vc2-id-product-term-sign"),
            V.dexpr(cf * d, "vc2-id-product-term-dropped"),
            V.dexpr(dotv, "vc2-id-product-term-dropped"),
            V.dexpr(d + dotv, "vc2-id-scalar-factor-dropped"),
        ]
        out = V.expr_answer(value)
        working = ["Product rule for the divergence: ∇ · (f F) = f ∇ · F + F · ∇f, evaluated at the point."]
    return out, cands, text, latex, working


def _medium(rng):
    which = rng.choice(["triple", "laplacian", "cross-grad"])
    pt = _point(rng)
    if which == "triple":
        F, G = _vector(rng), _vector(rng)
        direct, identity = div_FxG(F, G)
        value = _agree_scalar(direct, identity, pt, V.fd_div(V.cross(F, G), pt))
        text, latex = _prompt(
            "Let {F} and {G}. Find {op} at the point {P}.",
            op=V.raw("∇ · (F × G)", r"\nabla \cdot (\mathbf{F} \times \mathbf{G})"),
            F=V.cat(V.fdef("F"), V.mv(F)),
            G=V.cat(V.fdef("G"), V.mv(G)),
            P=V.mpt(pt),
        )
        a = V.at(V.dot(G, V.curl(F)), pt)
        b = V.at(V.dot(F, V.curl(G)), pt)
        cands = [
            V.dexpr(b - a, "vc2-id-triple-product-sign"),
            V.dexpr(a + b, "vc2-id-triple-product-plus"),
            V.dexpr(a, "vc2-id-product-term-dropped"),
            V.dexpr(V.at(V.div(F) * V.div(G), pt), "vc2-id-divergences-multiplied"),
        ]
        working = ["∇ · (F × G) = G · (∇ × F) - F · (∇ × G), evaluated at the point."]
    elif which == "laplacian":
        f, g = _scalar(rng, 2), _scalar(rng, 2)
        direct, identity = lap_fg(f, g)
        value = _agree_scalar(direct, identity, pt, _num_lap(f * g, pt))
        text, latex = _prompt(
            "Let {f} and {g}. Find {op} at the point {P}.",
            op=V.raw("∇²(f g)", r"\nabla^{2}(f g)"),
            f=V.sdef("f", f),
            g=V.sdef("g", g),
            P=V.mpt(pt),
        )
        fv, gv = f.subs(dict(zip(XYZ, pt))), g.subs(dict(zip(XYZ, pt)))
        lf, lg = V.at(laplacian(f), pt), V.at(laplacian(g), pt)
        cross = V.at(V.dot(V.grad(f), V.grad(g)), pt)
        cands = [
            V.dexpr(fv * lg + gv * lf, "vc2-id-cross-term-dropped"),
            V.dexpr(fv * lg + gv * lf + cross, "vc2-id-cross-term-factor-two-dropped"),
            V.dexpr(fv * lg + gv * lf - 2 * cross, "vc2-id-product-term-sign"),
            V.dexpr(lf * lg, "vc2-id-laplacians-multiplied"),
        ]
        working = ["∇²(f g) = f ∇²g + g ∇²f + 2 ∇f · ∇g, evaluated at the point."]
    else:
        f, G = _scalar(rng), _vector(rng)
        direct, identity = div_gradf_x_G(f, G)
        value = _agree_scalar(direct, identity, pt, V.fd_div(V.cross(V.grad(f), G), pt))
        text, latex = _prompt(
            "Let {f} and {G}. Find {op} at the point {P}.",
            op=V.raw("∇ · (∇f × G)", r"\nabla \cdot (\nabla f \times \mathbf{G})"),
            f=V.sdef("f", f),
            G=V.cat(V.fdef("G"), V.mv(G)),
            P=V.mpt(pt),
        )
        t = V.at(V.dot(V.grad(f), V.curl(G)), pt)
        cands = [
            V.dexpr(t, "vc2-id-triple-product-sign"),
            V.dexpr(0, "vc2-id-second-term-dropped"),
            V.dexpr(-V.at(V.dot(V.grad(f), G), pt), "vc2-id-dot-for-curl"),
            V.dexpr(V.at(V.dot(V.grad(f), V.curl(G)) * 2, pt), "vc2-id-triple-product-plus"),
        ]
        working = ["Because ∇ × ∇f = 0, only one term survives: ∇ · (∇f × G) = -∇f · (∇ × G)."]
    return V.expr_answer(value), cands, text, latex, working


def _num_lap(e, pt, h=1e-3):
    """A numerical Laplacian by a second central difference."""
    fn = sympy.lambdify(XYZ, e, "math")
    base = [float(c) for c in pt]
    total = 0.0
    for i in range(3):
        up, dn = list(base), list(base)
        up[i] += h
        dn[i] -= h
        total += (fn(*up) - 2 * fn(*base) + fn(*dn)) / h**2
    return total


def _hard(rng):
    which = rng.choice(["radial", "green", "doublecurl"])
    pt = _point(rng)
    if which == "radial":
        n = rng.choice([2, 3, 4, -2, -3, 5])
        pt = rng.choice([(1, 2, 2), (2, 1, 2), (2, 2, 1), (-1, 2, 2), (1, -2, 2), (2, 3, 6), (1, 4, 8)])
        r2 = sum(c * c for c in pt)
        f = (x**2 + y**2 + z**2) ** sympy.Rational(n, 2)
        value = sympy.Rational(n * (n + 1)) * sympy.Rational(r2) ** sympy.Rational(n - 2, 2)
        if not value.is_Rational:
            # odd n at a point with r2 = 9, 49 or 81 is rational; others are not
            raise _Redraw
        check(V.close(_num_lap(f, pt), float(value), 1e-4), "radial Laplacian disagrees with the second difference")
        check(sympy.simplify(V.at(laplacian(f), pt) - value) == 0, "radial Laplacian disagrees with sympy")
        base = "(x² + y² + z²)"
        if n % 2:
            n_text = f"{base}^({n}/2)"
        else:
            half = n // 2
            n_text = base + {1: "", 2: "²", 3: "³"}[half] if half in (1, 2, 3) else f"{base}^({half})"
        n_tex = r"(x^{2} + y^{2} + z^{2})^{" + (f"{n}/2" if n % 2 else str(n // 2)) + "}"
        text, latex = _prompt(
            "Let {f}. Find {op} at the point {P}.",
            op=V.raw("∇²f", r"\nabla^{2} f"),
            f=V.M("f(x, y, z) = " + n_text, "f(x, y, z) = " + n_tex),
            P=V.mpt(pt),
        )
        r_n = sympy.Rational(r2) ** sympy.Rational(n - 2, 2)
        cands = [
            V.dexpr(sympy.Rational(n * (n - 1)) * r_n, "vc2-id-radial-one-dimensional"),
            V.dexpr(
                sympy.Rational(n * (n + 1)) * sympy.Rational(r2) ** sympy.Rational(n, 2),
                "vc2-id-radial-power-not-lowered",
            ),
            V.dexpr(sympy.Rational(n * (n + 2)) * r_n, "vc2-id-radial-wrong-factor"),
            V.dexpr(sympy.Rational(n * n) * r_n, "vc2-id-radial-wrong-factor"),
        ]
        working = ["For f = r^n the Laplacian is n(n + 1) r^(n - 2), with r² = x² + y² + z²."]
        out = V.expr_answer(value)
    elif which == "green":
        f, g = _scalar(rng, 3), _scalar(rng, 3)
        direct, identity = green_difference(f, g)
        value = _agree_scalar(direct, identity, pt, V.fd_div(sub(scale(f, V.grad(g)), scale(g, V.grad(f))), pt))
        text, latex = _prompt(
            "Let {f} and {g}. Find {op} at the point {P}.",
            op=V.raw("∇ · (f ∇g - g ∇f)", r"\nabla \cdot (f \nabla g - g \nabla f)"),
            f=V.sdef("f", f),
            g=V.sdef("g", g),
            P=V.mpt(pt),
        )
        fv, gv = f.subs(dict(zip(XYZ, pt))), g.subs(dict(zip(XYZ, pt)))
        lf, lg = V.at(laplacian(f), pt), V.at(laplacian(g), pt)
        cross = V.at(V.dot(V.grad(f), V.grad(g)), pt)
        cands = [
            V.dexpr(fv * lg + gv * lf, "vc2-id-difference-as-sum"),
            V.dexpr(fv * lg - gv * lf + cross, "vc2-id-cross-term-kept"),
            V.dexpr(gv * lf - fv * lg, "vc2-id-triple-product-sign"),
            V.dexpr(fv * lg - gv * lf + 2 * cross, "vc2-id-cross-term-kept"),
        ]
        working = ["The terms ∇f · ∇g from the two products cancel, leaving f ∇²g - g ∇²f."]
        out = V.expr_answer(value)
    else:
        F = tuple(V.rand_poly(rng, terms=2, max_degree=3, min_degree=2) for _ in range(3))
        direct, identity = curl_curl(F)
        value = _agree_vector(direct, identity, pt, V.fd_curl(V.curl(F), pt))
        if sum(1 for c in value if c != 0) < 2:
            raise _Redraw
        text, latex = _prompt(
            "Let {F}. Find {op} at the point {P}. Write your answer as an ordered triple "
            "(i-component, j-component, k-component).",
            op=V.raw("∇ × (∇ × F)", r"\nabla \times (\nabla \times \mathbf{F})"),
            F=V.cat(V.fdef("F"), V.mv(F)),
            P=V.mpt(pt),
        )
        gd = evaluate(V.grad(V.div(F)), pt)
        lp = evaluate(vec_laplacian(F), pt)
        cands = [
            V.dvec(sub(lp, gd), "vc2-id-double-curl-sign"),
            V.dvec(gd, "vc2-id-laplacian-term-dropped"),
            V.dvec(scale(-1, lp), "vc2-id-gradient-term-dropped"),
            V.dvec(add(gd, lp), "vc2-id-double-curl-plus"),
        ]
        working = ["∇ × (∇ × F) = ∇(∇ · F) - ∇²F, with the Laplacian taken component by component."]
        out = V.vec_answer(value)
    return out, cands, text, latex, working


def _single(rng, tier):
    out, cands, text, latex, working = {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)
    dist = V.usable(out["answer"], out["answer_text"], cands)
    if len(dist) < 3:
        raise _Redraw
    return dict(text=text, latex=latex, working=working, distractors=dist, **out)


@generator(
    "s2a2_vector_identities",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a2_vector_identities(rng: random.Random, tier="easy"):
    for _ in attempts(60, "an identity question with enough distractors"):
        try:
            return _single(rng, tier)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# exam chain

# Four families of statements; in each, the first is true for every smooth f, v and w and the others are not.
_IDENTITY_FAMILIES = (
    (
        "∇ · (f v) = f (∇ · v) + v · ∇f",
        [
            ("vc2-id-product-term-sign", "∇ · (f v) = f (∇ · v) - v · ∇f"),
            ("vc2-id-product-term-dropped", "∇ · (f v) = f (∇ · v)"),
            ("vc2-id-gradient-times-divergence", "∇ · (f v) = (∇f)(∇ · v)"),
            ("vc2-id-scalar-factor-dropped", "∇ · (f v) = ∇ · v + v · ∇f"),
        ],
    ),
    (
        "∇ × (f v) = f (∇ × v) + ∇f × v",
        [
            ("vc2-id-cross-term-sign", "∇ × (f v) = f (∇ × v) - ∇f × v"),
            ("vc2-id-product-term-dropped", "∇ × (f v) = f (∇ × v)"),
            ("vc2-id-dot-for-cross", "∇ × (f v) = f (∇ × v) + ∇f · v"),
            ("vc2-id-scalar-factor-dropped", "∇ × (f v) = ∇ × v + ∇f × v"),
        ],
    ),
    (
        "∇ · (v × w) = w · (∇ × v) - v · (∇ × w)",
        [
            ("vc2-id-triple-product-sign", "∇ · (v × w) = v · (∇ × w) - w · (∇ × v)"),
            ("vc2-id-triple-product-plus", "∇ · (v × w) = w · (∇ × v) + v · (∇ × w)"),
            ("vc2-id-divergences-multiplied", "∇ · (v × w) = (∇ · v)(∇ · w)"),
            ("vc2-id-cross-of-divergences", "∇ · (v × w) = (∇ · v) × (∇ · w)"),
        ],
    ),
    (
        "∇ · (∇ × v) = 0",
        [
            ("vc2-id-div-curl-not-zero", "∇ · (∇ × v) = ∇²v"),
            ("vc2-id-laplacian-zero", "∇ · (∇f) = 0"),
            ("vc2-id-double-curl-zero", "∇ × (∇ × v) = 0"),
            ("vc2-id-grad-div-zero", "∇(∇ · v) = 0"),
        ],
    ),
)

_EXAM_STEMS = (
    "A temperature field f(x, y, z) = {f} (degrees Celsius) and an air-flow velocity field {F} (metres per second) are "
    "defined in a greenhouse, with x, y and z in metres. The point P is {P}.",
    "In a model of a heated room, the temperature is f(x, y, z) = {f} degrees Celsius and the air velocity is {F} "
    "metres per second, with x, y and z in metres. The point P is {P}.",
)


def _exam_once(rng):
    for _ in attempts(100, "an exam field"):
        f, F, pt = _scalar(rng, 3), _vector(rng, 2, 3), V.lattice_point(rng, -2, 3)
        direct, identity = div_fF(f, F)
        a = V.at(V.div(F), pt)
        b = V.at(V.dot(F, V.grad(f)), pt)
        f0 = V.at(f, pt)
        c = f0 * a + b
        if a != 0 and b != 0 and f0 not in (0, 1) and abs(c) < 600:
            break
    value = _agree_scalar(direct, identity, pt, V.fd_div(scale(f, F), pt))
    check(value == c, "the product rule value differs from the direct divergence")

    a_out = V.frac_answer(a)
    a_dist = V.usable(
        a_out["answer"],
        a_out["answer_text"],
        [
            V.dfrac(
                V.at(sympy.diff(F[0], y) + sympy.diff(F[1], z) + sympy.diff(F[2], x), pt), "vc2-div-wrong-variable"
            ),
            V.dfrac(V.at(sympy.diff(F[0], x) + sympy.diff(F[1], y), pt), "vc2-div-component-dropped"),
            V.dfrac(sum(V.at(q, pt) for q in F), "vc2-div-field-value-not-divergence"),
            V.dfrac(sum(V.at(q, pt) for q in V.curl(F)), "vc2-div-confused-with-curl"),
        ],
        kind="fraction",
    )
    gf = evaluate(V.grad(f), pt)
    fv = evaluate(F, pt)
    b_out = V.frac_answer(b)
    b_dist = V.usable(
        b_out["answer"],
        b_out["answer_text"],
        [
            V.dfrac(sum(p * q for p, q in zip(fv, (1, 1, 1))), "vc2-id-gradient-replaced-by-ones"),
            V.dfrac(sum(V.at(q, pt) for q in V.grad(f)), "vc2-id-dot-product-missing-field"),
            V.dfrac(sum(V.at(q, pt) for q in V.cross(F, V.grad(f))), "vc2-id-cross-for-dot"),
            V.dfrac(sum(p * q for p, q in zip(fv, gf)) + f0, "vc2-id-scalar-added"),
        ],
        kind="fraction",
    )
    c_out = V.frac_answer(c)
    c_dist = V.usable(
        c_out["answer"],
        c_out["answer_text"],
        [
            V.dfrac(f0 * a - b, "vc2-id-product-term-sign"),
            V.dfrac(f0 * a, "vc2-id-product-term-dropped"),
            V.dfrac(a + b, "vc2-id-scalar-factor-dropped"),
            V.dfrac(f0 * (a + b), "vc2-id-scalar-multiplied-twice"),
        ],
        kind="fraction",
    )
    right, wrongs = rng.choice(_IDENTITY_FAMILIES)
    mc_out = dict(answer=right, answer_text=right, answer_kind="mcq")
    mc_dist = [V.Distractor(text, tag, None) for tag, text in wrongs]
    if min(len(a_dist), len(b_dist), len(c_dist)) < 2:
        raise _Redraw

    stem_text, stem_latex = V.fill(
        rng.choice(_EXAM_STEMS), f=V.M(V.plain(f), V.tex(f)), F=V.cat(V.fdef("v"), V.mv(F)), P=V.mpt(pt)
    )
    stem_latex = stem_latex.replace("f(x, y, z) = ", "f(x, y, z) = ")
    ax, ay, az = pt

    def part(prompt_text, prompt_latex, out, dist, marks, working, follow=None):
        d_ = dict(prompt=prompt_text, prompt_latex=prompt_latex, marks=marks, working=working, distractors=dist, **out)
        if follow:
            d_["follow_through"] = follow
        return d_

    t_a, l_a = V.say("Find the divergence ", V.raw("∇ · v", r"\nabla \cdot \mathbf{v}"), " at P.")
    t_b, l_b = V.say(
        "Find ",
        V.raw("v · ∇f", r"\mathbf{v} \cdot \nabla f"),
        " at P, the rate at which the air carries temperature changes past P.",
    )
    t_c, l_c = V.say(
        "Hence find ",
        V.raw("∇ · (f v)", r"\nabla \cdot (f\mathbf{v})"),
        " at P, using the product rule and your answers to parts (a) and (b).",
    )
    t_d, l_d = V.say(
        "Which one of the following statements holds for all smooth scalar fields f and all smooth vector fields v and w?"
    )
    parts = [
        part(
            t_a,
            l_a,
            a_out,
            a_dist,
            1,
            ["Differentiate each component with respect to its own variable, add, and substitute P."],
        ),
        part(t_b, l_b, b_out, b_dist, 1, ["Find ∇f at P and take its dot product with v at P."]),
        part(
            t_c,
            l_c,
            c_out,
            c_dist,
            2,
            ["∇ · (f v) = f ∇ · v + v · ∇f, with f(P) = " + str(f0) + "."],
            follow=f"{f0}*part_a + part_b",
        ),
        part(
            t_d,
            l_d,
            mc_out,
            mc_dist,
            2,
            [
                "Check each statement against the product rules and the facts that the curl of a gradient and the divergence of a curl are zero."
            ],
        ),
    ]
    _ = (ax, ay, az)
    return dict(text=stem_text, latex=stem_latex, parts=parts)


@generator("exd_x_s2a2_vector_identities", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a2_vector_identities(rng: random.Random, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam identity chain with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")
