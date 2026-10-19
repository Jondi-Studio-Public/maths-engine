"""Generates `s2b3_gram_schmidt` questions (easy, medium, hard) and the exam chain `exd_x_s2b3_gram_schmidt`.

``uc2-gram-schmidt`` (``linear-algebra``, answer ``matrix``): the Gram-Schmidt process on two or three vectors, taken
in the order given and NOT normalised, so the orthogonal vectors are unique and are graded entry by entry.

* easy: the second orthogonal vector of two vectors in R^3 or R^4: one projection to find and subtract.
* medium: the third orthogonal vector of three vectors in R^3 or R^4: two projections, each onto an orthogonal vector
  already built (not onto an original vector).
* hard: the same process for polynomials of degree at most 2 under the integral inner product on [-1, 1] or [0, 1];
  the answer is the coefficient vector (a, b, c) of a + bx + cx^2. Every inner product is an integral.
* exam (``exd_x_s2b3_gram_schmidt``): three vectors in R^4. (a) the projection coefficient, (b) u2, (c) the squared
  length of u2 (follows through from (a)), (d) u3.

Every answer is worked out in exact fractions, and re-derived with SymPy's ``GramSchmidt`` (vectors) or by integrating
the products of the polynomials with ``sympy.integrate`` (polynomials).
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms, phrase
from ._s2b3_kit import (
    dot,
    fraction_answer,
    fraction_distractors,
    gram_schmidt,
    matrix_distractors,
    need,
    sizes_ok,
    sym_vector,
    to_sym,
    vadd,
    vec_answer,
    vec_text,
    vscale,
    vsub,
)
from ._ucv1_linalg import rank

_SUPER = {3: "ℝ³", 4: "ℝ⁴"}


# -- drawing --------------------------------------------------------------------------------------------------


def _draw_vectors(rng, count, dim, den=12):
    """`count` independent integer vectors of length `dim` whose Gram-Schmidt vectors are readable fractions and in
    which every projection is non-zero (so no step of the process can be skipped)."""
    for _ in attempts(4000, f"{count} vectors in R^{dim} with readable Gram-Schmidt vectors"):
        vs = [[rng.randint(-3, 3) for _ in range(dim)] for _ in range(count)]
        if any(sum(1 for x in v if x != 0) < 2 for v in vs):
            continue
        if rank([[Fraction(x) for x in v] for v in vs]) != count:
            continue
        us = gram_schmidt(vs)
        if any(dot(vs[k], us[j]) == 0 for k in range(1, count) for j in range(k)):
            continue
        if not sizes_ok(us[1:], 30, den) or any(all(x == 0 for x in u) for u in us):
            continue
        return vs, us
    raise AssertionError  # pragma: no cover


def _sympy_gs(vs):
    """SymPy's Gram-Schmidt vectors (not normalised) as lists of Fractions."""
    out = sympy.GramSchmidt([to_sym(list(v)) for v in vs], orthonormal=False)
    return [sym_vector(u) for u in out]


# -- errors that give a wrong vector ---------------------------------------------------------------------------


def _two_vector_errors(v1, v2):
    c = dot(v2, v1) / dot(v1, v1)
    return [
        (vscale(c, v1), "gs-returns-projection"),
        (vadd(v2, vscale(c, v1)), "gs-adds-projection"),
        (vsub(v2, vscale(dot(v2, v1), v1)), "gs-forgot-divide-by-norm-squared"),
        (vsub(v2, vscale(dot(v2, v1) / dot(v2, v2), v1)), "gs-divides-by-wrong-vector"),
    ]


def _three_vector_errors(vs, us):
    v1, v2, v3 = vs
    u1, u2, _u3 = us
    p1 = vscale(dot(v3, u1) / dot(u1, u1), u1)
    p2 = vscale(dot(v3, u2) / dot(u2, u2), u2)
    p2_orig = vscale(dot(v3, v2) / dot(v2, v2), v2)
    return [
        (vsub(vsub(v3, p1), p2_orig), "gs-projects-onto-original-vector"),
        (vsub(v3, p2), "gs-drops-first-projection"),
        (vsub(v3, p1), "gs-drops-second-projection"),
        (vadd(vadd(v3, p1), p2), "gs-adds-projection"),
    ]


_EASY_STEMS = (
    "Apply the Gram–Schmidt process to v1 = {v1} and v2 = {v2} in {space}, taking v1 first. Find the second "
    "orthogonal vector u2. Do not normalise it.",
    "In {space}, the vectors v1 = {v1} and v2 = {v2} are orthogonalised by the Gram–Schmidt process in that order, "
    "with no normalising. Find u2.",
)
_MEDIUM_STEMS = (
    "Apply the Gram–Schmidt process, in the order given and without normalising, to v1 = {v1}, v2 = {v2} and "
    "v3 = {v3} in {space}. Find the third vector u3.",
    "The vectors v1 = {v1}, v2 = {v2} and v3 = {v3} in {space} are orthogonalised in that order by the Gram–Schmidt "
    "process (no normalising). Find u3.",
)


def _easy(rng):
    dim = rng.choice((3, 3, 4))
    vs, us = _draw_vectors(rng, 2, dim, den=10)
    check(us == _sympy_gs(vs), "SymPy's Gram-Schmidt vector differs")
    check(dot(us[0], us[1]) == 0, "u1 and u2 are not orthogonal")
    cands = matrix_distractors(_two_vector_errors(*vs), us[1], vector=True)
    return dict(
        text=phrase(rng, _EASY_STEMS, v1=vec_text(vs[0]), v2=vec_text(vs[1]), space=_SUPER[dim]),
        distractors=need(cands),
        working=[
            f"The projection coefficient of v2 on u1 = v1 is v2·u1 / u1·u1 = {dot(vs[1], vs[0])}/{dot(vs[0], vs[0])}.",
            f"So u2 = v2 - ({dot(vs[1], vs[0])}/{dot(vs[0], vs[0])}) u1 = {vec_text(us[1])}.",
        ],
        **vec_answer(us[1]),
    )


def _medium(rng):
    dim = rng.choice((3, 3, 4))
    vs, us = _draw_vectors(rng, 3, dim, den=30)
    check(us == _sympy_gs(vs), "SymPy's Gram-Schmidt vectors differ")
    check(all(dot(us[i], us[j]) == 0 for i in range(3) for j in range(i)), "the vectors are not orthogonal")
    cands = matrix_distractors(_three_vector_errors(vs, us), us[2], vector=True)
    return dict(
        text=phrase(rng, _MEDIUM_STEMS, v1=vec_text(vs[0]), v2=vec_text(vs[1]), v3=vec_text(vs[2]), space=_SUPER[dim]),
        distractors=need(cands),
        working=[
            f"u1 = v1 and u2 = {vec_text(us[1])}.",
            f"u3 = v3 - (v3·u1 / u1·u1) u1 - (v3·u2 / u2·u2) u2 = {vec_text(us[2])}.",
        ],
        **vec_answer(us[2]),
    )


# -- hard: polynomials under an integral inner product ----------------------------------------------------------

_INTERVALS = ((-1, 1), (0, 1))


def _gram(lo, hi):
    """The matrix of <x^i, x^j> (i, j = 0..2) for the integral inner product on [lo, hi]."""
    return [[Fraction(hi ** (i + j + 1) - lo ** (i + j + 1), i + j + 1) for j in range(3)] for i in range(3)]


def _ip(g, p, q):
    return sum((Fraction(p[i]) * g[i][j] * Fraction(q[j]) for i in range(3) for j in range(3)), Fraction(0))


def _poly_text(c):
    return join_terms([(c[0], ""), (c[1], "x"), (c[2], "x²")])


def _poly_gs(g, ps):
    out: list[list[Fraction]] = []
    for p in ps:
        u = [Fraction(a) for a in p]
        for w in out:
            u = vsub(u, vscale(_ip(g, p, w) / _ip(g, w, w), w))
        out.append(u)
    return out


def _sympy_poly_gs(lo, hi, ps):
    x = sympy.symbols("x")

    def integral(f):
        antiderivative = sympy.Poly(f, x).integrate()
        return antiderivative.eval(hi) - antiderivative.eval(lo)

    polys = [sum(sympy.Integer(c) * x**k for k, c in enumerate(p)) for p in ps]
    us: list = []
    for p in polys:
        u = p
        for w in us:
            u = u - integral(p * w) / integral(w * w) * w
        us.append(sympy.expand(u))
    return [[frac_of(sympy.Poly(u, x).coeff_monomial(x**k)) for k in range(3)] for u in us]


def frac_of(value) -> Fraction:
    return Fraction(int(value.p), int(value.q))


def _draw_polys(rng, count, g):
    for _ in attempts(4000, f"{count} polynomials with readable Gram-Schmidt polynomials"):
        ps = [[rng.randint(-2, 2) for _ in range(3)] for _ in range(count)]
        if count == 3:
            ps = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] if rng.random() < 0.5 else ps
        if any(all(a == 0 for a in p) for p in ps) or rank([[Fraction(a) for a in p] for p in ps]) != count:
            continue
        # keep the statement readable: no polynomial of degree 2 among the first two when there are only two
        us = _poly_gs(g, ps)
        if any(_ip(g, ps[k], us[j]) == 0 for k in range(1, count) for j in range(k)):
            continue
        if not sizes_ok(us, 60, 40) or any(all(a == 0 for a in u) for u in us):
            continue
        return ps, us
    raise AssertionError  # pragma: no cover


def _poly_errors(g, ps, us):
    last = len(ps) - 1
    p = ps[last]
    ident = [[Fraction(int(i == j)) for j in range(3)] for i in range(3)]

    # the same process with the coefficient dot product instead of the integral
    def coeff_ip(a, b):
        return _ip(ident, a, b)

    flat: list[list[Fraction]] = []
    for q in ps:
        u = [Fraction(a) for a in q]
        for w in flat:
            u = vsub(u, vscale(coeff_ip(q, w) / coeff_ip(w, w), w))
        flat.append(u)
    pieces = [vscale(_ip(g, p, us[j]) / _ip(g, us[j], us[j]), us[j]) for j in range(last)]
    no_divide = list(map(Fraction, p))
    for j in range(last):
        no_divide = vsub(no_divide, vscale(_ip(g, p, us[j]), us[j]))
    out = [(flat[last], "gs-integral-replaced-by-coefficient-dot")]
    if last == 1:
        out.append((vadd(p, pieces[0]), "gs-adds-projection"))
    else:
        out.append((vsub(p, pieces[1]), "gs-drops-first-projection"))
        out.append(
            (
                vsub(vsub(p, pieces[0]), vscale(_ip(g, p, ps[1]) / _ip(g, ps[1], ps[1]), ps[1])),
                "gs-projects-onto-original-vector",
            )
        )
    out.append((no_divide, "gs-forgot-divide-by-norm-squared"))
    out.append((pieces[0] if last == 1 else vadd(pieces[0], pieces[1]), "gs-returns-projection"))
    return out


_POLY_STEMS = (
    "On the polynomials of degree at most 2, the inner product is ⟨p, q⟩ = ∫ from {lo} to 1 of p(x) q(x) dx. Apply "
    "the Gram–Schmidt process, in the order given and without normalising, to {plist}. Write the polynomial u{k} = "
    "a + bx + cx² as the vector (a, b, c).",
    "Let ⟨p, q⟩ = ∫ from {lo} to 1 of p(x) q(x) dx on the polynomials of degree at most 2. The polynomials {plist} "
    "are orthogonalised in that order by the Gram–Schmidt process, with no normalising. Find u{k}, giving the "
    "coefficient vector (a, b, c) of u{k} = a + bx + cx².",
)


def _hard_poly(rng):
    lo, hi = rng.choice(_INTERVALS)
    g = _gram(lo, hi)
    count = rng.choice((2, 3))
    ps, us = _draw_polys(rng, count, g)
    # Second route: integrate the products with SymPy.
    check(_sympy_poly_gs(lo, hi, ps) == us, "SymPy's integrals give a different Gram-Schmidt polynomial")
    names = [f"p{i + 1}(x) = {_poly_text(p)}" for i, p in enumerate(ps)]
    plist = ", ".join(names[:-1]) + " and " + names[-1] if count == 3 else " and ".join(names)
    cands = matrix_distractors(_poly_errors(g, ps, us), us[-1], vector=True)
    return dict(
        text=phrase(rng, _POLY_STEMS, lo=lo, plist=plist, k=count),
        distractors=need(cands),
        working=[
            f"Each inner product is an integral of a product of polynomials, e.g. ⟨p1, p1⟩ = {_ip(g, ps[0], ps[0])}.",
            f"u{count} = {_poly_text(us[-1])}, so the vector is {vec_text(us[-1])}.",
        ],
        **vec_answer(us[-1]),
    )


_WEIGHTS = ((1, 2, 1), (2, 1, 3), (1, 3, 2), (1, 2, 3), (3, 1, 2), (2, 3, 1))


def _wdot(w, u, v) -> Fraction:
    return sum((Fraction(a) * b * c for a, b, c in zip(w, u, v)), Fraction(0))


def _weighted_gs(w, vs):
    out: list[list[Fraction]] = []
    for v in vs:
        u = [Fraction(a) for a in v]
        for q in out:
            u = vsub(u, vscale(_wdot(w, v, q) / _wdot(w, q, q), q))
        out.append(u)
    return out


def _weighted_errors(w, vs, us):
    """Wrong vectors for the last step: the ordinary dot product used throughout, then the usual slips."""
    last = len(vs) - 1
    plain = gram_schmidt(vs)[last]
    v = vs[last]
    pieces = [vscale(_wdot(w, v, us[j]) / _wdot(w, us[j], us[j]), us[j]) for j in range(last)]
    out = [(plain, "gs-ignores-the-weights")]
    out.append(
        (vadd(v, pieces[0]), "gs-adds-projection") if last == 1 else (vsub(v, pieces[1]), "gs-drops-first-projection")
    )
    out.append((vsub(v, vscale(_wdot(w, v, us[0]), us[0])), "gs-forgot-divide-by-norm-squared"))
    out.append(
        (
            pieces[0]
            if last == 1
            else vsub(vsub(v, pieces[0]), vscale(_wdot(w, v, vs[1]) / _wdot(w, vs[1], vs[1]), vs[1])),
            "gs-returns-projection" if last == 1 else "gs-projects-onto-original-vector",
        )
    )
    return out


_WEIGHTED_STEMS = (
    "On ℝ³ the inner product is ⟨x, y⟩ = {form}. Apply the Gram–Schmidt process for this inner product, in the order "
    "given and without normalising, to {vlist}. Find u{k}.",
    "Let ⟨x, y⟩ = {form} for x, y in ℝ³. The vectors {vlist} are orthogonalised in that order by the Gram–Schmidt "
    "process for this inner product (no normalising). Find u{k}.",
)


def _hard_weighted(rng):
    """Gram-Schmidt for a weighted inner product on R^3: every projection coefficient uses the weights."""
    for _ in attempts(4000, "vectors with a readable weighted Gram-Schmidt vector"):
        w = rng.choice(_WEIGHTS)
        count = rng.choice((2, 3))
        vs = [[rng.randint(-3, 3) for _ in range(3)] for _ in range(count)]
        if any(sum(1 for x in v if x != 0) < 2 for v in vs) or rank([[Fraction(x) for x in v] for v in vs]) != count:
            continue
        us = _weighted_gs(w, vs)
        if any(_wdot(w, vs[k], us[j]) == 0 for k in range(1, count) for j in range(k)):
            continue
        if not sizes_ok(us[1:], 40, 30) or any(all(a == 0 for a in u) for u in us):
            continue
        # the ordinary process must give a different vector, or the weights would not matter
        if gram_schmidt(vs)[-1] == us[-1]:
            continue
        break
    # Second route: SymPy with the weight matrix W: <x, y> = x^T W y.
    weight = sympy.diag(*w)
    sym_us: list[sympy.Matrix] = []
    for v in vs:
        sv = to_sym(list(v))
        u = sv
        for q in sym_us:
            u = u - ((sv.T * weight * q)[0] / (q.T * weight * q)[0]) * q
        sym_us.append(u)
    check([sym_vector(u) for u in sym_us] == us, "SymPy's weighted Gram-Schmidt vectors differ")
    check(all(_wdot(w, us[i], us[j]) == 0 for i in range(count) for j in range(i)), "not orthogonal for the weights")
    form = " + ".join(f"{c}x{i + 1}y{i + 1}" if c != 1 else f"x{i + 1}y{i + 1}" for i, c in enumerate(w))
    names = [f"v{i + 1} = {vec_text(v)}" for i, v in enumerate(vs)]
    vlist = ", ".join(names[:-1]) + " and " + names[-1] if count == 3 else " and ".join(names)
    cands = matrix_distractors(_weighted_errors(w, vs, us), us[-1], vector=True)
    return dict(
        text=phrase(rng, _WEIGHTED_STEMS, form=form, vlist=vlist, k=count),
        distractors=need(cands),
        working=[
            "Every dot product is replaced by the weighted inner product, including the squared lengths in the denominators.",
            f"u{count} = {vec_text(us[-1])}.",
        ],
        **vec_answer(us[-1]),
    )


@generator(
    "s2b3_gram_schmidt",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2b3_gram_schmidt(rng, tier="easy"):
    """Gram-Schmidt on vectors in R^3 or R^4 and on polynomials; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice((_hard_poly, _hard_poly, _hard_weighted))(rng)


_EXAM_STEMS = (
    "The vectors v1 = {v1}, v2 = {v2} and v3 = {v3} in ℝ⁴ are orthogonalised by the Gram–Schmidt process in that "
    "order, without normalising: u1 = v1.",
    "Apply the Gram–Schmidt process to v1 = {v1}, v2 = {v2} and v3 = {v3} in ℝ⁴, in that order and without "
    "normalising. The first orthogonal vector is u1 = v1.",
    "In ℝ⁴ let v1 = {v1}, v2 = {v2} and v3 = {v3}. The orthogonal vectors u1 = v1, u2 and u3 come from the "
    "Gram–Schmidt process taken in this order, with no normalising.",
)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_s2b3_gram_schmidt", level=18, presets={"exam": dict(tier="exam")})
def exd_x_s2b3_gram_schmidt(rng, tier="exam"):
    """Three vectors in R^4: the projection coefficient, u2, the squared length of u2, and u3."""
    check(tier == "exam", f"unknown tier {tier!r}")
    vs, us = _draw_vectors(rng, 3, 4, den=40)
    check(us == _sympy_gs(vs), "SymPy's Gram-Schmidt vectors differ")
    v1, v2, v3 = vs
    u1, u2, u3 = us
    coeff = dot(v2, v1) / dot(v1, v1)
    n1, n2 = dot(v1, v1), dot(v2, v2)
    length_sq = dot(u2, u2)
    check(length_sq == n2 - coeff**2 * n1, "the squared length identity fails")
    check(length_sq == frac_of((to_sym(list(u2)).T * to_sym(list(u2)))[0]), "SymPy's |u2|^2 differs")

    a_dist = fraction_distractors(
        coeff,
        [
            (dot(v2, v1), "gs-forgot-divide-by-norm-squared"),
            (dot(v2, v1) / dot(v2, v2), "gs-divides-by-wrong-vector"),
            (-coeff, "gs-adds-projection"),
            (dot(v1, v1) / dot(v2, v1) if dot(v2, v1) else None, "gs-inverted-coefficient"),
        ],
    )
    b_dist = matrix_distractors(_two_vector_errors(v1, v2), u2, vector=True, limit=3)
    c_dist = fraction_distractors(
        length_sq,
        [
            (n2 - coeff * n1, "gs-coefficient-not-squared"),
            (n2 + coeff**2 * n1, "gs-adds-projection"),
            (n2, "gs-drops-projection-from-length"),
            (coeff**2 * n1, "gs-length-of-projection"),
        ],
        positive=True,
    )
    d_dist = matrix_distractors(_three_vector_errors(vs, us), u3, vector=True, limit=3)
    stem = phrase(
        rng,
        _EXAM_STEMS,
        v1=vec_text(v1),
        v2=vec_text(v2),
        v3=vec_text(v3),
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the number c for which u2 = v2 - c u1 is orthogonal to u1.",
                marks=2,
                working=[f"c = v2·u1 / u1·u1 = {dot(v2, v1)}/{n1}."],
                distractors=need(a_dist),
                **fraction_answer(coeff),
            ),
            dict(
                prompt="Hence find u2.",
                marks=2,
                working=[f"u2 = v2 - c u1 = {vec_text(u2)}."],
                distractors=need(b_dist),
                **vec_answer(u2),
            ),
            dict(
                prompt="Find the squared length |u2|² of u2.",
                marks=2,
                follow_through=f"{n2} - part_a**2*{n1}",
                sensible_range=(0, None),
                working=[f"|u2|² = |v2|² - c²|u1|² = {n2} - c²({n1}) = {length_sq}."],
                distractors=need(c_dist),
                **fraction_answer(length_sq),
            ),
            dict(
                prompt="Find u3.",
                marks=3,
                working=[f"u3 = v3 - (v3·u1/u1·u1) u1 - (v3·u2/u2·u2) u2 = {vec_text(u3)}."],
                distractors=need(d_dist),
                **vec_answer(u3),
            ),
        ],
    )
