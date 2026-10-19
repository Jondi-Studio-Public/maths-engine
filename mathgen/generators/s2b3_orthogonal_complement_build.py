"""Generates `s2b3_orthogonal_complement` questions (easy, medium, hard) and the exam chain
`exd_x_s2b3_orthogonal_complement`.

``uc2-orthogonal-complement`` (``linear-algebra``, answer ``matrix``): the orthogonal complement W^perp of a subspace W
of R^n, written as the unique basis whose vectors are the non-zero rows of a reduced row echelon form, so the answer is
graded entry by entry.

* easy: W spanned by one or two vectors of R^3; W^perp is a plane or a line.
* medium: W spanned by one, two or three independent vectors of R^4.
* hard: W given with redundancy, either as the span of three or four vectors one of which is a combination of the
  others, or as the solution set of a system with a dependent equation (so W^perp is the row space of the system).
* exam (``exd_x_s2b3_orthogonal_complement``): three vectors in R^5 of which one is redundant. (a) dim W, (b) dim W^perp
  (follows through from (a)), (c) the RREF basis of W^perp, (d) the value of k that puts a vector with one unknown
  entry in W^perp.

The basis is found by Gauss-Jordan elimination in exact fractions and re-derived with SymPy's ``nullspace`` and
``rref``; the exam value of k is checked by SymPy at k and at k + 1.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, join_terms, phrase, pick_distractors
from ._s2b3_kit import (
    dot,
    fraction_distractors,
    mat_answer,
    mat_text,
    matrix_distractors,
    need,
    number_answer,
    sizes_ok,
    to_sym,
    vec_text,
)
from ._ucv1_linalg import nullspace, rank, rref

_SPACE = {3: "ℝ³", 4: "ℝ⁴", 5: "ℝ⁵"}


def _fm(rows):
    return [[Fraction(v) for v in row] for row in rows]


def _rref_rows(rows):
    """The non-zero rows of the reduced row echelon form of `rows`."""
    reduced, pivots = rref(_fm(rows))
    return reduced[: len(pivots)]


def _complement(spanning):
    """RREF basis of the orthogonal complement of the span of `spanning` (rows)."""
    return _rref_rows(nullspace(_fm(spanning)))


def _sympy_complement(spanning):
    """SymPy: the null space of the matrix whose rows span W, row-reduced into its normal form."""
    basis = to_sym(_fm(spanning)).nullspace()
    reduced, pivots = sympy.Matrix.hstack(*basis).T.rref()
    return [[Fraction(int(v.p), int(v.q)) for v in reduced.row(i)] for i in range(len(pivots))]


def _draw_span(rng, count, dim, *, rank_wanted=None, bound_num=24, bound_den=12):
    """`count` integer vectors in R^dim, independent unless `rank_wanted` is lower (then the later ones are made as
    combinations of the first), with a readable complement basis. Returns (vectors, complement rows)."""
    rank_wanted = rank_wanted or count
    for _ in attempts(4000, f"{count} vectors in R^{dim} with a readable complement"):
        base = [[rng.randint(-3, 3) for _ in range(dim)] for _ in range(rank_wanted)]
        vs = [list(v) for v in base]
        for _extra in range(count - rank_wanted):
            p, q = rng.choice((-2, -1, 1, 2)), rng.choice((-2, -1, 1, 2))
            vs.append([p * a + q * b for a, b in zip(base[0], base[1])])
        rng.shuffle(vs)
        if any(sum(1 for x in v if x != 0) < 2 for v in vs) or max(abs(x) for v in vs for x in v) > 6:
            continue
        if rank(_fm(vs)) != rank_wanted:
            continue
        comp = _complement(vs)
        if len(comp) != dim - rank_wanted or not sizes_ok(comp, bound_num, bound_den):
            continue
        if len(comp) > 1 and len({tuple(r) for r in comp}) != len(comp):
            continue
        return vs, comp
    raise AssertionError  # pragma: no cover


def _verify(vs, comp):
    """The rows of `comp` are orthogonal to every spanning vector, are in RREF, and agree with SymPy's null space."""
    check(all(dot(r, v) == 0 for r in comp for v in vs), "a complement vector is not orthogonal to W")
    check(_sympy_complement(vs) == comp, "SymPy's complement basis differs")
    check(rank(_fm(vs)) + len(comp) == len(vs[0]), "dim W + dim W-perp is not n")


def _lcm_clear(row):
    den = 1
    for v in row:
        den = den * v.denominator // gcd(den, v.denominator)
    return [v * den for v in row]


def _errors(vs, comp):
    """Wrong matrices a learner reaches: W itself, a sign slip in the free-variable coefficients, and so on."""
    reduced, pivots = rref(_fm(vs))
    n = len(vs[0])
    free = [c for c in range(n) if c not in pivots]
    flipped = []
    for f in free:
        vec = [Fraction(0)] * n
        vec[f] = Fraction(1)
        for row, pc in zip(reduced, pivots):
            vec[pc] = row[f]  # the sign of -row[f] forgotten
        flipped.append(vec)
    out = [
        (_rref_rows(vs), "oc-gives-w-itself"),
        (_rref_rows(flipped) if flipped else None, "oc-sign-error-in-null-vectors"),
    ]
    if any(v.denominator != 1 for r in comp for v in r):
        out.append(([_lcm_clear(r) for r in comp], "oc-not-in-reduced-form"))
    if len(vs) > 1:
        only_first = nullspace(_fm(vs[:1]))
        out.append((_rref_rows(only_first), "oc-only-first-vector"))
    if len(comp) > 1:
        out.append((comp[:1], "oc-too-few-vectors"))
    return out


_STEMS = (
    "Let W be the subspace of {space} spanned by {vectors}. Find a basis of the orthogonal complement W⊥ and give it as the rows of a "
    "matrix in reduced row echelon form (the non-zero rows of the RREF, so the answer is unique).",
    "In {space}, W is the span of {vectors}. Write the orthogonal complement W⊥ in the form of the non-zero rows of a "
    "reduced row echelon form: enter the matrix whose rows are these basis vectors.",
)


def _spans_question(rng, vs, comp, dim):
    _verify(vs, comp)
    cands = matrix_distractors(_errors(vs, comp), comp)
    return dict(
        text=phrase(rng, _STEMS, vectors=", ".join(vec_text(v) for v in vs), space=_SPACE[dim]),
        distractors=need(cands),
        working=[
            "W⊥ is the null space of the matrix whose rows are the spanning vectors.",
            f"Solving and reducing gives the basis {mat_text(comp)} in reduced row echelon form.",
        ],
        **mat_answer(comp),
    )


def _easy(rng):
    count = rng.choice((1, 2, 2))
    vs, comp = _draw_span(rng, count, 3, bound_den=12)
    return _spans_question(rng, vs, comp, 3)


def _medium(rng):
    count = rng.choice((2, 2, 3))
    vs, comp = _draw_span(rng, count, 4)
    return _spans_question(rng, vs, comp, 4)


# -- hard -------------------------------------------------------------------------------------------------------


def _hard_redundant(rng):
    dim, count = rng.choice(((4, 3), (4, 3), (5, 4)))
    vs, comp = _draw_span(rng, count, dim, rank_wanted=2, bound_num=30)
    return _spans_question(rng, vs, comp, dim)


def _equation(row, names="xyzwv"):
    return join_terms([(int(c), name) for c, name in zip(row, names)]) + " = 0"


_SYSTEM_STEMS = (
    "The subspace W of {space} is the solution set of the system {equations}. Find a basis of W⊥ and give it as the "
    "rows of a matrix in reduced row echelon form (the non-zero rows of the RREF, so the answer is unique).",
    "Let W be the set of all (x, y, z{extra}) in {space} that satisfy {equations}. Write a basis of the orthogonal complement W⊥ as the "
    "non-zero rows of a reduced row echelon form, entered as a matrix.",
)


def _hard_system(rng):
    """W = solution set of a system with a redundant equation, so W^perp is the row space of the system."""
    for _ in attempts(4000, "a system with a redundant equation and a readable row space"):
        dim = rng.choice((4, 4, 5))
        base = [[rng.randint(-3, 3) for _ in range(dim)] for _ in range(2)]
        p, q = rng.choice((-2, -1, 1, 2)), rng.choice((-2, -1, 1, 2))
        extra = [p * a + q * b for a, b in zip(*base)]
        rows = base + [extra]
        rng.shuffle(rows)
        if (
            rank(_fm(rows)) != 2
            or any(sum(1 for x in r if x != 0) < 2 for r in rows)
            or max(abs(x) for r in rows for x in r) > 7
        ):
            continue
        comp = _rref_rows(rows)
        if not sizes_ok(comp, 24, 12):
            continue
        break
    # Second route: W = null(E), W-perp = null(W-basis rows): SymPy's complement of the solution set recovers the row space.
    w_basis = [[Fraction(int(v.p), int(v.q)) for v in vec] for vec in to_sym(_fm(rows)).nullspace()]
    check(_sympy_complement(w_basis) == comp, "the complement of the solution set is not the row space")
    check(
        all(dot(r, w) == 0 for r in comp for w in w_basis), "a row-space vector is not orthogonal to the solution set"
    )
    names = "xyzwv"[:dim]
    equations = ", ".join(_equation(r, names) for r in rows)
    solutions = _rref_rows(w_basis)
    cands = matrix_distractors(
        [
            (solutions, "oc-gives-w-itself"),
            ([r for r in rows], "oc-keeps-the-dependent-equation"),
            (rows[:2], "oc-not-in-reduced-form"),
        ],
        comp,
    )
    return dict(
        text=phrase(
            rng, _SYSTEM_STEMS, space=_SPACE[dim], equations=equations, extra="".join(", " + n for n in names[3:])
        ),
        distractors=need(cands),
        working=[
            "The coefficient rows of the equations are orthogonal to every solution, so W⊥ is their row space.",
            f"One equation is a combination of the others. Row-reducing leaves {mat_text(comp)}.",
        ],
        **mat_answer(comp),
    )


@generator(
    "s2b3_orthogonal_complement",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2b3_orthogonal_complement(rng, tier="easy"):
    """The RREF basis of the orthogonal complement of a subspace; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice((_hard_redundant, _hard_system))(rng)


# -- exam -------------------------------------------------------------------------------------------------------


def _k_distractors(vs, z, t):
    """Wrong values of k from the first spanning vector with a non-zero entry at the unknown position."""
    k = z[t]
    eq_vec = next(v for v in vs if v[t] != 0)
    others = [i for i in range(len(z)) if i != t]
    total = sum((eq_vec[i] * z[i] for i in others), Fraction(0))
    cands = [(-k, "sign-error-moving-term")]
    if abs(eq_vec[t]) != 1:
        cands.append((-total, "oc-forgot-coefficient-of-k"))
    for i in others:
        if eq_vec[i] * z[i] != 0:
            cands.append((-(total - 2 * eq_vec[i] * z[i]) / eq_vec[t], "sign-error-moving-term"))
            cands.append((-(total - eq_vec[i] * z[i]) / eq_vec[t], "oc-dropped-a-term"))
    return fraction_distractors(k, cands)


def _primitive(row):
    cleared = _lcm_clear(row)
    g = 0
    for v in cleared:
        g = gcd(g, int(abs(v)))
    return [v / g for v in cleared]


@generator("exd_x_s2b3_orthogonal_complement", level=18, presets={"exam": dict(tier="exam")})
def exd_x_s2b3_orthogonal_complement(rng, tier="exam"):
    """Three vectors in R^5 with one redundant: dim W, dim W-perp, the RREF basis of W-perp, and a missing entry k."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "an exam subspace with a readable complement and an unknown entry"):
        vs, comp = _draw_span(rng, 3, 5, rank_wanted=2, bound_num=24, bound_den=12)
        ints = [_primitive(r) for r in comp]
        coeffs = [rng.randint(-2, 2) for _ in ints]
        if all(c == 0 for c in coeffs) or sum(1 for c in coeffs if c != 0) < 2:
            continue
        z = [sum((c * r[i] for c, r in zip(coeffs, ints)), Fraction(0)) for i in range(5)]
        spots = [t for t in range(5) if any(v[t] != 0 for v in vs) and z[t] != 0]
        if z != [int(v) for v in z] or not spots or max(abs(v) for v in z) > 12:
            continue
        t = rng.choice(spots)
        d_dist = _k_distractors(vs, z, t)
        if len(d_dist) >= 2:
            break
    _verify(vs, comp)
    k = int(z[t])
    # SymPy: z is in W-perp at k, and not at k + 1.
    good, bad = [int(v) for v in z], [int(v) for v in z]
    bad[t] = k + 1
    sym_w = to_sym(_fm(vs))
    check(sym_w * to_sym(good) == sympy.zeros(sym_w.rows, 1), "z is not in W-perp at k")
    check(sym_w * to_sym(bad) != sympy.zeros(sym_w.rows, 1), "z is still in W-perp at k + 1")

    z_text = "(" + ", ".join("k" if i == t else str(int(v)) for i, v in enumerate(z)) + ")"
    a_dist = [
        Distractor(str(v), tag, exact_value(v))
        for v, tag in ((3, "rank-nullity-confused"), (1, "rank-off-by-one"), (5, "nullity-counts-columns"))
    ]
    b_dist = [
        Distractor(str(v), tag, exact_value(v))
        for v, tag in ((2, "rank-nullity-confused"), (5, "nullity-counts-columns"), (4, "rank-off-by-one"))
    ]
    c_dist = matrix_distractors(_errors(vs, comp), comp, limit=3)
    check(len(pick_distractors("3", [str(d) for d in b_dist])) >= 2, "too few dimension distractors")
    stem = f"Let W be the subspace of ℝ⁵ spanned by {', '.join(vec_text(v) for v in vs)}."
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the dimension of W.",
                marks=2,
                sensible_range=(0, 6),
                working=["Row-reducing the spanning vectors leaves two non-zero rows, so dim W = 2."],
                distractors=need(a_dist),
                **number_answer(2),
            ),
            dict(
                prompt="Hence write down the dimension of W⊥.",
                marks=1,
                follow_through="5 - part_a",
                sensible_range=(0, 6),
                working=["dim W + dim W⊥ = 5, so dim W⊥ = 3."],
                distractors=need(b_dist),
                **number_answer(3),
            ),
            dict(
                prompt="Find a basis of W⊥ as the rows of a matrix in reduced row echelon form.",
                marks=3,
                working=[f"W⊥ is the null space of the matrix whose rows span W: {mat_text(comp)}."],
                distractors=need(c_dist),
                **mat_answer(comp),
            ),
            dict(
                prompt=f"The vector z = {z_text} lies in W⊥. Find the value of k.",
                marks=2,
                working=[
                    "z is orthogonal to each spanning vector of W; one such equation is linear in k and gives its value."
                ],
                distractors=need(d_dist),
                **number_answer(k),
            ),
        ],
    )
