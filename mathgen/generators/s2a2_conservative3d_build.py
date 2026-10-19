"""Generates `s2a2_conservative3d` questions (easy, medium, hard) and the exam chain `exd_x_s2a2_conservative3d`.

``uc2-conservative-field-potential`` (``multivariable-and-vector-calculus``, answer ``expression``): a field in three
dimensions is conservative when its curl is zero, and then it is the gradient ∇φ of a scalar potential φ, so the work
along any path is φ(B) - φ(A). (The plane case is ``ucv1-line-integral``; this skill is the curl test in space.)

* easy: the field is ∇φ except that one coefficient in one component is the unknown k. Find k so that the curl vanishes.
* medium: find the potential φ, with φ(0, 0, 0) = 0, of a conservative field that mixes all three variables.
* hard, one of
  (a) two unknown constants k and m; make the field conservative, then find the work between two points;
  (b) the work along an awkward curve (a twisted cubic or a helix) in a field that is conservative but not said to be.
* exam (``exd_x_s2a2_conservative3d``): (a) the constant k that makes the field conservative, (b) the potential φ
  (follows through from (a)), (c) the work between two points (from (b)), (d) the work along a curve between two
  other points (from (b)).

Every potential is confirmed by differencing φ against the field at lattice points and by a symbolic zero curl; every
work value is confirmed by Gauss-Legendre quadrature of F · r' along the path, which shares nothing with φ(B) - φ(A).
"""

from __future__ import annotations

import random

import sympy

from ._s2_options import spot_check
from ..core import attempts, check, generator
from . import _s2a2_vec as V
from ._s2a2_vec import x, y, z

_NAMES = ("F", "F", "G")
k_sym, m_sym = sympy.symbols("k m")
XYZ = (x, y, z)

_MIXED = [x**2 * y * z, x * y**2 * z, x * y * z**2, x * y * z]
_OTHER = [x**3, y**3, z**3, x**2 * y, y**2 * z, x * z**2, x**2 * z, y * z, x * y, x * z, y**2, z**2, x**2, x * y**2]


class _Redraw(Exception):
    """A draw that cannot give enough distinct wrong answers."""


def _potential(rng, need_mixed: int = 1, spread: int = 9):
    """φ with `need_mixed` genuinely three-variable monomials and a couple of other terms, and its terms by monomial."""
    mixed = rng.sample(_MIXED, need_mixed)
    others = rng.sample(_OTHER, rng.choice([2, 3]))
    terms = {mono: V.nz(rng, -spread, spread) if mono in _MIXED else V.nz(rng, -3, 4) for mono in mixed + others}
    return sum((c * mono for mono, c in terms.items()), sympy.Integer(0)), terms


def _unknown_in(F_parts, comp: int, mono_term, symbol):
    """Replace the term `mono_term` (a monomial times its coefficient) of component `comp` by symbol times the monomial,
    when that monomial appears nowhere else in the component. Returns the new field and the true coefficient, or None."""
    base = sympy.expand(F_parts[comp])
    coeff, mono = mono_term.as_coeff_Mul()
    rest = sympy.expand(base - mono_term)
    if rest.has(mono) and sympy.expand(rest).coeff(mono) != 0:
        return None
    new = list(F_parts)
    new[comp] = rest + symbol * mono
    return tuple(new), coeff, mono


def _check_potential(phi, F):
    """The field is the gradient of phi (symbolically) and is curl-free (by differences at two points)."""
    check(all(sympy.expand(g - f) == 0 for g, f in zip(V.grad(phi), F)), "the field is not the gradient of phi")
    for p in ((1, 2, -1), (-2, 1, 3)):
        check(V.close_vec(V.fd_grad(phi, p), tuple(V.float_of(V.at(c, p)) for c in F), 1e-5), "gradient differs")
        check(V.close_vec(V.fd_curl(F, p), (0, 0, 0), 1e-5), "the field has non-zero curl")


def _work_numeric(F, r, a, b):
    """∫ F(r(t)) · r'(t) dt from t = a to b, by Gauss-Legendre."""
    return V.line_integral_numeric(F, r, sympy.Symbol("t"), a, b)


# ---------------------------------------------------------------------------------------------------------------
# easy: the unknown constant


def _easy(rng):
    for _ in attempts(200, "a field with one unknown"):
        phi, terms = _potential(rng, spread=20)
        F = V.grad(phi)
        mixed = [mono for mono in terms if mono in _MIXED]
        mono = mixed[0]
        comp = rng.randrange(3)
        term = sympy.expand(sympy.diff(terms[mono] * mono, XYZ[comp]))
        got = _unknown_in(F, comp, term, k_sym)
        if got is None or abs(got[1]) < 2:
            continue
        G, true, _mono = got
        printed = {abs(int(c)) for comp_ in G for c in sympy.Poly(sympy.expand(comp_.subs(k_sym, 0)), x, y, z).coeffs()}
        if abs(int(true)) in printed:
            continue  # the answer must not be readable straight off the field
        # confirm: with the true value the field is a gradient; with any other it is not
        H = tuple(c.subs(k_sym, true) for c in G)
        check(all(sympy.expand(c) == 0 for c in V.curl(H)), "the curl does not vanish at the stated k")
        check(any(sympy.expand(c.subs(k_sym, true + 1)) != 0 for c in V.curl(G)), "k is not unique")
        check(V.close_vec(V.fd_curl(H, (1, 2, -1)), (0, 0, 0), 1e-5), "numerical curl is not zero")
        sol = sympy.solve([sympy.Poly(c, x, y, z).coeffs()[0] for c in V.curl(G) if c != 0][0], k_sym)
        check(sol == [true], "solving the curl condition gives a different k")
        return G, true, phi, comp, mono, terms
    raise AssertionError("unreachable")


_EASY_STEMS = (
    "The vector field {F} contains a constant k. Find the value of k for which F is conservative on all of space.",
    "For what value of the constant k is the field {F} conservative?",
)


def _easy_question(rng, name):
    G, true, phi, comp, mono, terms = _easy(rng)
    text, latex = V.fill(rng.choice(_EASY_STEMS).replace("F is", f"{name} is"), F=V.cat(V.fdef(name), V.mv(G)))
    # the coefficient as it would be read straight off the unknown's monomial, and common slips
    c_mono = terms[mono]
    exponent = sympy.degree(mono, XYZ[comp])
    cands = [
        V.dexpr(c_mono, "vc2-cons-coefficient-not-scaled"),
        V.dexpr(-true, "vc2-cons-solve-sign"),
        V.dexpr(c_mono * (exponent + 1), "vc2-cons-power-factor-extra"),
        V.dexpr(true + 1, "vc2-cons-off-by-one"),
        V.dexpr(sympy.Rational(true, 2), "vc2-cons-half"),
    ]
    out = V.expr_answer(true)
    working = [f"Setting the components of the curl to zero gives k = {true}."]
    return out, cands, text, latex, working


# ---------------------------------------------------------------------------------------------------------------
# medium: the potential


def _potential_wrongs(phi, F):
    P, Q, R = F
    only_p = V.pint(P, x)
    p_plus_q = only_p + V.pint(Q, y)
    missing_z = only_p + V.pint(sympy.expand(Q - sympy.diff(only_p, y)), y)
    both = p_plus_q + V.pint(R, z)
    return [
        (only_p, "vc2-cons-potential-one-variable-only"),
        (missing_z, "vc2-cons-potential-third-component-ignored"),
        (both, "vc2-cons-potential-double-counted"),
        (-phi, "vc2-cons-potential-sign"),
        (p_plus_q, "vc2-cons-potential-double-counted"),
    ]


def _medium(rng, name):
    for _ in attempts(100, "a medium potential"):
        phi, terms = _potential(rng, need_mixed=rng.choice([1, 2]))
        F = V.grad(phi)
        if all(c.has(v) for c, v in zip(F, XYZ)):
            break
    _check_potential(phi, F)
    out = V.expr_answer(phi)
    text, latex = V.fill(
        "The vector field {F} is conservative. Find the scalar potential φ(x, y, z) with {N} = ∇φ and φ(0, 0, 0) = 0.",
        F=V.cat(V.fdef(name), V.mv(F)),
        N=V.bold(name),
    )
    cands = [V.dexpr(sympy.expand(e), t) for e, t in _potential_wrongs(phi, F)]
    working = ["Integrate P with respect to x, then add the terms in y and z needed to match Q and R."]
    return out, cands, text, latex, working


# ---------------------------------------------------------------------------------------------------------------
# hard


def _deg(c, t) -> int:
    return 0 if c == 0 else int(sympy.Poly(c, t).degree())


def _curve(rng):
    """A path r(t): a cubic on [0, 1] with lattice endpoints that is not a straight line, or a helix on [0, π/2]
    or [0, π]. Returns (r, lo, hi, A, B)."""
    t = sympy.Symbol("t")
    if rng.random() < 0.5:
        for _ in attempts(200, "an awkward path"):
            r = tuple(
                sympy.sympify(rng.randint(-1, 2))
                + rng.randint(-2, 2) * t
                + rng.randint(-2, 2) * t**2
                + rng.choice([0, 0, 1, -1]) * t**3
                for _ in range(3)
            )
            if sum(_deg(c, t) >= 2 for c in r) < 2:
                continue
            A = tuple(sympy.sympify(c).subs(t, 0) for c in r)
            B = tuple(sympy.sympify(c).subs(t, 1) for c in r)
            if A != B:
                return r, 0, 1, A, B
        raise AssertionError("unreachable")
    ra, rc = rng.choice([1, 2]), rng.choice([1, 2, 3])
    top = rng.choice([sympy.pi / 2, sympy.pi])
    r = (ra * sympy.cos(t), ra * sympy.sin(t), rc * t)
    A = tuple(sympy.simplify(c.subs(t, 0)) for c in r)
    B = tuple(sympy.simplify(c.subs(t, top)) for c in r)
    return r, 0, top, A, B


def M_interval(hi):
    if hi == 1:
        return V.raw("0 ≤ t ≤ 1", r"0 \le t \le 1")
    if hi == sympy.pi / 2:
        return V.raw("0 ≤ t ≤ π/2", r"0 \le t \le \frac{\pi}{2}")
    return V.raw("0 ≤ t ≤ π", r"0 \le t \le \pi")


def _hard(rng, name):
    t = sympy.Symbol("t")
    if rng.random() < 0.5:
        for _ in attempts(100, "two unknowns"):
            phi, terms = _potential(rng, need_mixed=2)
            F = V.grad(phi)
            mixed = [mono for mono in terms if mono in _MIXED]
            comps = rng.sample(range(3), 2)
            G, trues, ok, monos = F, [], True, []
            for symbol, comp, mono in zip((k_sym, m_sym), comps, mixed):
                term = sympy.expand(sympy.diff(terms[mono] * mono, XYZ[comp]))
                got = _unknown_in(G, comp, term, symbol)
                if got is None or abs(got[1]) < 2:
                    ok = False
                    break
                G, true, base_mono = got
                trues.append(true)
                monos.append((comp, base_mono))
            if ok:
                break
        sol = {k_sym: trues[0], m_sym: trues[1]}
        H = tuple(c.subs(sol) for c in G)
        check(all(sympy.expand(c) == 0 for c in V.curl(H)), "the curl does not vanish at the stated constants")
        A, B = V.lattice_point(rng, -2, 2), V.lattice_point(rng, -2, 2)
        for _ in attempts(100, "distinct endpoints"):
            if A != B and V.at(phi, B) != V.at(phi, A):
                break
            A, B = V.lattice_point(rng, -2, 2), V.lattice_point(rng, -2, 2)
        W = V.at(phi, B) - V.at(phi, A)
        straight = tuple(a_ + (b_ - a_) * t for a_, b_ in zip(A, B))
        check(V.close(_work_numeric(H, straight, 0, 1), float(W), 1e-6), "work disagrees with quadrature")
        text, latex = V.fill(
            "The force field {F} (in newtons, with distances in metres) is conservative and contains constants k and m. "
            "Find the work done by the force in moving a particle from {A} to {B}, in joules.",
            F=V.cat(V.fdef(name), V.mv(G)),
            A=V.mpt(A),
            B=V.mpt(B),
        )
        pa, pb = V.at(phi, A), V.at(phi, B)
        # the potential a learner builds after taking k with the wrong sign: φ plus (-2k) times the integral of k's monomial
        comp0, mono0 = monos[0]
        piece = V.pint(mono0, XYZ[comp0])
        wrong_k = V.at(phi - 2 * trues[0] * piece, B) - V.at(phi - 2 * trues[0] * piece, A)
        cands = [
            V.dexpr(pa - pb, "vc2-cons-direction-sign"),
            V.dexpr(pa + pb, "vc2-cons-endpoints-added"),
            V.dexpr(pb, "vc2-cons-forgot-initial-potential"),
            V.dexpr(wrong_k, "vc2-cons-solve-sign"),
        ]
        working = ["Find k and m from the zero-curl conditions, then the work is φ(B) - φ(A)."]
        return V.expr_answer(W), cands, text, latex, working
    phi, terms = _potential(rng, need_mixed=rng.choice([1, 2]))
    F = V.grad(phi)
    r, lo, hi, A, B = _curve(rng)
    _check_potential(phi, F)
    W = sympy.simplify(V.at(phi, B) - V.at(phi, A))
    check(V.close(_work_numeric(F, r, lo, hi), V.float_of(W), 1e-6), "work disagrees with quadrature")
    pa, pb = V.at(phi, A), V.at(phi, B)
    interval = M_interval(hi)
    text, latex = V.fill(
        "Under the force {F} (newtons, metres), a particle moves along the curve C given by {C}. Find the work done by the "
        "force, in joules.",
        C=V.cat(V.raw("r(t) = ", r"\mathbf{r}(t) = "), V.mv(r), ", ", interval),
        F=V.cat(V.fdef(name), V.mv(F)),
    )
    cands = [
        V.dexpr(pa - pb, "vc2-cons-direction-sign"),
        V.dexpr(pa + pb, "vc2-cons-endpoints-added"),
        V.dexpr(pb, "vc2-cons-forgot-initial-potential"),
        V.dexpr(sympy.simplify(V.at(sum(f * (b_ - a_) for f, a_, b_ in zip(F, A, B)), B)), "vc2-cons-endpoint-field"),
    ]
    working = ["The curl of F is zero, so F = ∇φ and the work is φ(end) - φ(start), whatever the curve."]
    return V.expr_answer(W), cands, text, latex, working


def _single(rng, tier):
    name = rng.choice(_NAMES)
    if tier == "easy":
        out, cands, text, latex, working = _easy_question(rng, name)
    elif tier == "medium":
        out, cands, text, latex, working = _medium(rng, name)
    else:
        out, cands, text, latex, working = _hard(rng, name)
    dist = V.usable(out["answer"], out["answer_text"], cands)
    if len(dist) < 3:
        raise _Redraw
    return dict(text=text, latex=latex, working=working, distractors=dist, **out)


@generator(
    "s2a2_conservative3d",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2a2_conservative3d(rng: random.Random, tier="easy"):
    for _ in attempts(30, "a conservative-field question with enough distractors"):
        try:
            return _single(rng, tier)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------------------------
# exam chain

_EXAM_STEMS = (
    "A magnetic gripper exerts the force field {F} on a small steel object, in newtons, with x, y and z in metres. "
    "The constant k is chosen so that the field is conservative.",
    "A particle moves in the force field {F} (newtons, metres). The constant k is such that the field is conservative.",
)


def _exam_once(rng):
    for _ in attempts(100, "an exam field"):
        phi, terms = _potential(rng, need_mixed=rng.choice([1, 2]))
        F = V.grad(phi)
        mixed = [mono for mono in terms if mono in _MIXED and sympy.degree(mono, x) >= 1]
        if not mixed:
            continue
        mono = mixed[0]
        term = sympy.expand(sympy.diff(terms[mono] * mono, x))
        got = _unknown_in(F, 0, term, k_sym)
        if got is None or abs(got[1]) < 2:
            continue
        G, true, base_mono = got
        break
    H = tuple(c.subs(k_sym, true) for c in G)
    _check_potential(phi, H)
    # the potential built from P = ... + k * base_mono, with k left in: phi_k
    piece = V.pint(base_mono, x)  # the integral of the monomial that k multiplies
    phi_k = sympy.expand(phi - true * piece + k_sym * piece)
    check(sympy.expand(phi_k.subs(k_sym, true) - phi) == 0, "the potential with k is wrong")
    A = V.lattice_point(rng, -1, 2)
    B = V.lattice_point(rng, -1, 3)
    r, lo, hi, C0, C1 = _curve_lattice(rng, {A, B})
    for _ in attempts(100, "distinct endpoints"):
        if A != B and V.at(phi, B) != V.at(phi, A):
            break
        B = V.lattice_point(rng, -1, 3)
    W_ab = V.at(phi, B) - V.at(phi, A)
    W_c = V.at(phi, C1) - V.at(phi, C0)
    if W_ab == 0 or W_c == 0 or abs(W_ab) > 600 or abs(W_c) > 600:
        raise _Redraw
    t = sympy.Symbol("t")
    straight = tuple(a_ + (b_ - a_) * t for a_, b_ in zip(A, B))
    if spot_check(A, B, C0, C1):  # the quadrature is the slow second route: one question in four
        check(V.close(_work_numeric(H, straight, 0, 1), float(W_ab), 1e-6), "work (b) disagrees with quadrature")
        check(V.close(_work_numeric(H, r, lo, hi), float(W_c), 1e-6), "work (c) disagrees with quadrature")

    a_out = V.frac_answer(true)
    a_dist = V.usable(
        a_out["answer"],
        a_out["answer_text"],
        [
            V.dfrac(terms[mono], "vc2-cons-coefficient-not-scaled"),
            V.dfrac(-true, "vc2-cons-solve-sign"),
            V.dfrac(true + 1, "vc2-cons-off-by-one"),
            V.dfrac(sympy.Rational(true, 2), "vc2-cons-half"),
        ],
        kind="fraction",
    )
    b_out = V.expr_answer(phi)
    b_dist = V.usable(
        b_out["answer"],
        b_out["answer_text"],
        [V.dexpr(sympy.expand(e), t_) for e, t_ in _potential_wrongs(phi, H)],
    )

    def work_dists(P0, P1):
        pa, pb = V.at(phi, P0), V.at(phi, P1)
        return [
            V.dfrac(pa - pb, "vc2-cons-direction-sign"),
            V.dfrac(pa + pb, "vc2-cons-endpoints-added"),
            V.dfrac(pb, "vc2-cons-forgot-initial-potential"),
            V.dfrac(V.at(dot_field(H, [q - p for p, q in zip(P0, P1)]), P1), "vc2-cons-endpoint-field"),
        ]

    c_out = V.frac_answer(W_ab)
    c_dist = V.usable(c_out["answer"], c_out["answer_text"], work_dists(A, B), kind="fraction")
    d_out = V.frac_answer(W_c)
    d_dist = V.usable(d_out["answer"], d_out["answer_text"], work_dists(C0, C1), kind="fraction")
    if min(len(a_dist), len(b_dist), len(c_dist), len(d_dist)) < 2:
        raise _Redraw

    stem_text, stem_latex = V.fill(rng.choice(_EXAM_STEMS), F=V.cat(V.fdef("F"), V.mv(G)))

    def part(prompt_template, out, dist, marks, working, follow=None, **slots):
        text, latex = V.fill(prompt_template, **slots)
        d_ = dict(prompt=text, prompt_latex=latex, marks=marks, working=working, distractors=dist, **out)
        if follow:
            d_["follow_through"] = follow
        return d_

    def sub(label, p):
        return f"Subs(part_{label}, (x, y, z), ({p[0]}, {p[1]}, {p[2]}))"

    phi_ft = str(phi_k).replace("k", "part_a")
    interval = M_interval(hi)
    parts = [
        part(
            "Find the value of k.",
            a_out,
            a_dist,
            2,
            ["The curl of F must be zero; the components of the curl give k."],
        ),
        part(
            "Find the potential function φ(x, y, z) with F = ∇φ and φ(0, 0, 0) = 0.",
            b_out,
            b_dist,
            3,
            ["Integrate P with respect to x, then add the terms in y and z needed to match Q and R."],
            follow=phi_ft,
        ),
        part(
            "Find the work done by the force, in joules, in moving the object from {A} to {B} along any path.",
            c_out,
            c_dist,
            2,
            ["The work is φ(B) - φ(A)."],
            follow=f"{sub('b', B)} - {sub('b', A)}",
            A=V.mpt(A),
            B=V.mpt(B),
        ),
        part(
            "The object is moved instead along the curve C given by {C}. Find the work done by the force, in joules.",
            d_out,
            d_dist,
            2,
            [
                "The field is conservative, so the work is φ(end) - φ(start), where the curve starts at "
                f"{V.pt_plain(C0)} and ends at {V.pt_plain(C1)}."
            ],
            follow=f"{sub('b', C1)} - {sub('b', C0)}",
            C=V.cat(V.raw("r(t) = ", r"\mathbf{r}(t) = "), V.mv(r), ", ", interval),
        ),
    ]
    return dict(text=stem_text, latex=stem_latex, parts=parts)


def dot_field(H, d):
    """F · d for a displacement d, as an expression in x, y, z."""
    return sum(c * di for c, di in zip(H, d))


def _curve_lattice(rng, avoid):
    for _ in attempts(100, "a curve avoiding the endpoints"):
        r, lo, hi, A, B = _curve(rng)
        if all(isinstance(c, sympy.Integer) for c in A + B) and A not in avoid and B not in avoid:
            return r, lo, hi, A, B
    raise _Redraw


@generator("exd_x_s2a2_conservative3d", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a2_conservative3d(rng: random.Random, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam potential with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")
