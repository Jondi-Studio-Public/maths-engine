"""Generates `s2b1_basis_from_equations` questions (easy, medium, hard) and the exam chain `exd_x_s2b1_basis_equations`.

``uc2-basis-from-equations`` (``linear-algebra``, answer ``matrix``): a basis, and so the dimension, of a subspace
defined by homogeneous linear conditions, or of an intersection. A subspace has many bases, so every question names the
one normal form it wants and the answer is compared exactly:

* the null-space basis of the reduced row echelon form: one vector per free variable, in the order of the free
  variables, with that free variable 1 and the others 0 (easy, medium and the exam chain); or
* the reduced row echelon form of the matrix whose rows are any basis (hard), which is unique too.

The basis vectors are the rows of the answer matrix.

* easy: two independent equations in R^4 (the equations are random integer combinations of a reduced system, so the
  free variables are not simply the last two).
* medium: conditions that must first be turned into equations: p(1) = 0, p'(0) = p(-1) and the like on the coefficients
  of a cubic; trace and symmetry conditions or AB = BA on 2x2 matrices; three equations in R^5 of which one is a
  combination of the others; the vectors of R^5 orthogonal to two or three given vectors.
* hard: an intersection. A subspace given by equations meets a subspace given by spanning vectors, or two subspaces given
  by spanning vectors meet (the generators are shuffled so that the common vector is not visible).
* exam (``exd_x_s2b1_basis_equations``): three equations in R^4 with a parameter k in one coefficient. (a) the k at
  which the equations become dependent, (b) the dimension of the solution space there, (c) its basis in the null-space
  normal form, (d) the dimension for every other k (follows through from (b): rank rises by exactly one).

Every system is built from a reduced row echelon form chosen first, so its solution space is known by construction; the
basis and dimension are re-derived by Gauss-Jordan elimination and again with SymPy's `nullspace()` and `rref()`.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms, phrase, pick_distractors
from ._s2b1_linalg import (
    plain_stem,
    X,
    det_root_distractors,
    int_distractors,
    mat_answer,
    mat_distractors,
    sym,
)
from ._ucv1_linalg import fm, mmul, nullspace, primitive, rank, rref, transpose, unimodular

LEVEL = 19
_SUB = "₀₁₂₃₄₅₆₇₈₉"


def _names(n: int) -> list[str]:
    return [f"x{_SUB[i + 1]}" for i in range(n)]


def _equation(coeffs, names) -> str:
    return join_terms([(int(c), v) for c, v in zip(coeffs, names)]) + " = 0"


def _equations_text(rows, names) -> str:
    return ", ".join(_equation(r, names) for r in rows)


def _reduced_system(rng, n, r):
    """(R, pivots): an r×n reduced row echelon matrix with integer entries and random pivot columns."""
    for _ in attempts(200, "a reduced system with non-trivial free entries"):
        pivots = sorted(rng.sample(range(n), r))
        free = [c for c in range(n) if c not in pivots]
        rows = []
        for i, pc in enumerate(pivots):
            row = [Fraction(0)] * n
            row[pc] = Fraction(1)
            for f in free:
                if f > pc:
                    row[f] = Fraction(rng.choice([-3, -2, -1, 0, 1, 2, 3]))
            rows.append(row)
        # entries in free columns to the left of a pivot are not part of RREF; they stay zero.
        if all(any(row[f] != 0 for f in free if f > pivots[i]) for i, row in enumerate(rows)):
            return rows, pivots
    raise AssertionError  # pragma: no cover


def _mix(rng, rows, bound=2, cap=14):
    """Integer combinations of `rows` (invertible mixing), so the equations are not already reduced."""
    r = len(rows)
    for _ in attempts(500, "a mixing of the reduced equations with small entries"):
        u = unimodular(rng, r, bound) if r > 1 else [[Fraction(rng.choice([-1, 1]))]]
        mixed = mmul(u, rows)
        if all(abs(v) <= cap for row in mixed for v in row) and mixed != rows:
            return mixed
    raise AssertionError  # pragma: no cover


def _basis_candidates(equations, basis, pivots_free):
    """Wrong bases (lists of vectors) with misconception tags."""
    n = len(basis[0])
    cands = []
    cands.append((transpose(basis), "basis-vectors-as-columns"))
    cands.append(([[-v for v in b] for b in basis], "nullvec-component-sign-slip"))
    cands.append(([row[:] for row in equations[: len(basis)]], "nullvec-row-as-vector"))
    flipped = [b[:] for b in basis]
    flipped[0] = [-flipped[0][0], *flipped[0][1:]] if flipped[0][0] != 0 else flipped[0]
    cands.append((flipped, "nullvec-component-sign-slip"))
    cands.append((list(reversed(basis)), "basis-order-ignored"))
    # Wrong free variables: pivots and free variables exchanged for the first free column.
    cands.append(([[b[i] for i in range(n)][::-1] for b in basis], "basis-free-variables-misidentified"))
    return cands


def _basis_dict(text, basis, equations, working, n_dist=4):
    answer = mat_answer(basis)
    cands = _basis_candidates(equations, basis, None)
    distractors = pick_distractors(answer["answer_text"], mat_distractors(cands, basis), n_dist)
    if len(distractors) < 3:
        return None
    return dict(text=text, working=working, distractors=distractors, **answer)


def _verify_basis(equations, basis):
    """Independent checks: every basis vector solves the system, the count is n - rank, and SymPy agrees."""
    e = sym(equations)
    b = sym(basis)
    check((e * b.T).is_zero_matrix, "a basis vector does not solve the system")
    check(len(basis) == len(equations[0]) - e.rank(), "the basis size differs from n - rank")
    sym_basis = e.nullspace()
    check(len(sym_basis) == len(basis), "SymPy's nullity differs")
    both = sympy.Matrix.vstack(b, sympy.Matrix.hstack(*sym_basis).T)
    check(both.rank() == len(basis), "the basis and SymPy's null space span different spaces")


_FREE_RULE = (
    "Give your basis as the rows of a matrix: take the free variables (those without a pivot in the reduced row echelon "
    "form of the system) in order, and let the k-th row have the k-th free variable equal to 1 and the other free "
    "variables equal to 0."
)


# -- easy ----------------------------------------------------------------------------------------------------------------


def _easy(rng):
    n = 4
    reduced, pivots = _reduced_system(rng, n, 2)
    equations = _mix(rng, reduced)
    basis = nullspace(fm(equations))
    check(nullspace(fm(reduced)) == basis, "the mixed system has a different solution space")
    _verify_basis(equations, basis)
    names = _names(n)
    stem = phrase(
        rng,
        (
            "Let U be the subspace of ℝ⁴ of all vectors (x₁, x₂, x₃, x₄) satisfying "
            + "{eqs}. Find the basis of U in the form described below. "
            + _FREE_RULE,
            "The subspace U of ℝ⁴ consists of the vectors (x₁, x₂, x₃, x₄) with {eqs}. Find a basis of U. "
            + _FREE_RULE,
        ),
        eqs=" and ".join(_equation(r, names) for r in equations),
    )
    reduced_rows, pivot_cols = rref(fm(equations))
    working = [
        f"Row reduce the system to {', '.join(_equation(r, names) for r in reduced_rows[:2])}; the pivot columns are {[p + 1 for p in pivot_cols]}.",
        f"Each free variable gives one basis vector: the rows of {mat_answer(basis)['answer_text']}, so dim U = {len(basis)}.",
    ]
    return _basis_dict(stem, basis, equations, working)


# -- medium ----------------------------------------------------------------------------------------------------------------


def _poly_conditions(rng):
    a, b = rng.sample([-2, -1, 1, 2], 2)
    pool = [
        (f"p({a}) = 0", lambda p: p.subs(X, a)),
        (f"p'({a}) = 0", lambda p: sympy.diff(p, X).subs(X, a)),
        (f"p({a}) = p({b})", lambda p: p.subs(X, a) - p.subs(X, b)),
        (f"p'({a}) = p({b})", lambda p: sympy.diff(p, X).subs(X, a) - p.subs(X, b)),
        ("p(0) + p(1) = 0", lambda p: p.subs(X, 0) + p.subs(X, 1)),
        ("p''(0) = p'(1)", lambda p: sympy.diff(p, X, 2).subs(X, 0) - sympy.diff(p, X).subs(X, 1)),
        ("p(0) = p'(0)", lambda p: p.subs(X, 0) - sympy.diff(p, X).subs(X, 0)),
    ]
    return pool


def _medium_poly(rng):
    for _ in attempts(200, "polynomial conditions of the right rank"):
        chosen = rng.sample(_poly_conditions(rng), rng.choice([2, 2, 3]))
        rows = [[Fraction(int(f(X**j))) for j in range(4)] for _, f in chosen]
        if rank(rows) == len(rows) and len(set(map(str, rows))) == len(rows):
            break
    basis = nullspace(rows)
    _verify_basis(rows, basis)
    conds = " and ".join(t for t, _ in chosen)
    stem = phrase(
        rng,
        (
            "Let U be the subspace of the cubic polynomials p(x) = a₀ + a₁x + a₂x² + a₃x³ satisfying {conds}. Work with the coordinates "
            "(a₀, a₁, a₂, a₃) of p. Find the basis of U in the form described below. " + _FREE_RULE,
            "In the space P₃ of polynomials of degree at most 3, let U = {{p : {conds}}}. Write each p as (a₀, a₁, a₂, a₃) for "
            "p(x) = a₀ + a₁x + a₂x² + a₃x³. Find the basis of U in the form described below. " + _FREE_RULE,
        ),
        conds=conds,
    )
    names = [f"a{_SUB[i]}" for i in range(4)]
    working = [
        f"Each condition is a linear equation in the coefficients: {'; '.join(_equation(r, names) for r in rows)}.",
        f"Solving gives the basis {mat_answer(basis)['answer_text']} (one row per free variable), so dim U = {len(basis)}.",
    ]
    return _basis_dict(stem, basis, rows, working)


def _matrix_conditions(rng):
    bm = sympy.Matrix(2, 2, lambda i, j: rng.randint(-2, 2))
    while bm[0, 1] == 0 and bm[1, 0] == 0 and bm[0, 0] == bm[1, 1]:  # pragma: no cover - bounded in practice
        bm = sympy.Matrix(2, 2, lambda i, j: rng.randint(-2, 2))
    bt = f"[[{bm[0, 0]}, {bm[0, 1]}], [{bm[1, 0]}, {bm[1, 1]}]]"

    def commutes(m):
        d = m * bm - bm * m
        return [d[0, 0], d[0, 1], d[1, 0], d[1, 1]]

    return [
        ("the trace of A is 0", lambda m: [m.trace()]),
        ("A is symmetric", lambda m: [m[0, 1] - m[1, 0]]),
        ("the first row of A sums to 0", lambda m: [m[0, 0] + m[0, 1]]),
        ("the second row of A sums to 0", lambda m: [m[1, 0] + m[1, 1]]),
        ("the diagonal entries of A are equal", lambda m: [m[0, 0] - m[1, 1]]),
        (f"AB = BA, where B = {bt}", commutes),
    ]


def _medium_matrix(rng):
    for _ in attempts(300, "2×2 matrix conditions of the right rank"):
        pool = _matrix_conditions(rng)
        chosen = rng.sample(pool, rng.choice([2, 2, 3]))
        rows = []
        for _, f in chosen:
            cols = []
            for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                e = sympy.zeros(2, 2)
                e[i, j] = 1
                cols.append(f(e))
            for k in range(len(cols[0])):
                rows.append([Fraction(int(cols[c][k])) for c in range(4)])
        rows = [r for r in rows if any(v != 0 for v in r)]
        if rows and rank(rows) <= 3 and rank(rows) >= 2 and rank(rows) < 4:
            # keep a minimal independent set of equations as the system to display
            break
    basis = nullspace(rows)
    _verify_basis(rows, basis)
    conds = " and ".join(t for t, _ in chosen)
    stem = phrase(
        rng,
        (
            "Let U be the subspace of the 2×2 real matrices A = [[a, b], [c, d]] for which {conds}. Work with the coordinates (a, b, c, d) "
            "of A. Find the basis of U in the form described below. " + _FREE_RULE,
            "In the space of 2×2 real matrices A = [[a, b], [c, d]], U is the set of A for which {conds}. Use the coordinates "
            "(a, b, c, d) for A. Find the basis of U in the form described below. " + _FREE_RULE,
        ),
        conds=conds,
    )
    names = ["a", "b", "c", "d"]
    working = [
        f"The conditions become the linear system {'; '.join(_equation(r, names) for r in rows)}.",
        f"The solution space has the basis {mat_answer(basis)['answer_text']} (one row per free variable), so dim U = {len(basis)}.",
    ]
    return _basis_dict(stem, basis, rows, working)


def _medium_dependent(rng):
    n = rng.choice([4, 5])
    r = 2
    reduced, pivots = _reduced_system(rng, n, r)
    mixed = _mix(rng, reduced)
    combo = [Fraction(rng.choice([-2, -1, 1, 2])) for _ in range(r)]
    third = [sum(combo[i] * mixed[i][c] for i in range(r)) for c in range(n)]
    if max(abs(v) for v in third) > 14:
        return None
    rows = [list(mixed[0]), list(mixed[1])]
    rows.insert(rng.randrange(3), third)
    if rank(rows) != 2:
        return None
    basis = nullspace(rows)
    _verify_basis(rows, basis)
    names = _names(n)
    stem = phrase(
        rng,
        (
            "Let U be the set of vectors in ℝ{sup} that satisfy all of {eqs}. Find the basis of U in the form described below. "
            + _FREE_RULE,
            "The subspace U of ℝ{sup} is defined by the three equations {eqs}. Find the basis of U in the form described below. "
            + _FREE_RULE,
        ),
        sup="⁴" if n == 4 else "⁵",
        eqs=", ".join(_equation(r_, names) for r_ in rows),
    )
    working = [
        "Row reduction shows one equation is a combination of the other two, so the rank is 2, not 3.",
        f"With {n - 2} free variables the basis is {mat_answer(basis)['answer_text']} (one row per free variable), so dim U = {n - 2}.",
    ]
    return _basis_dict(stem, basis, rows, working)


def _medium_orthogonal(rng):
    n = 5
    k = rng.choice([2, 3])
    for _ in attempts(300, "independent vectors with a clean complement"):
        vecs = [[Fraction(rng.randint(-3, 3)) for _ in range(n)] for _ in range(k)]
        if rank(vecs) == k and all(any(v != 0 for v in row) for row in vecs):
            if max(v.denominator for b in nullspace(vecs) for v in b) <= 3:
                break
    basis = nullspace(vecs)
    _verify_basis(vecs, basis)
    shown = ", ".join("(" + ", ".join(str(int(v)) for v in row) + ")" for row in vecs)
    stem = phrase(
        rng,
        (
            "Let U be the subspace of ℝ⁵ of all vectors orthogonal to each of {shown}. Find the basis of U in the form described below. "
            + _FREE_RULE,
            "Find, in the form described below, the basis of the subspace of ℝ⁵ consisting of the vectors x with x · v = 0 for each v in {{{shown}}}. "
            + _FREE_RULE,
        ),
        shown=shown,
    )
    names = _names(n)
    working = [
        f"x · v = 0 for each v gives the system {'; '.join(_equation(r, names) for r in vecs)}.",
        f"The solution space has the basis {mat_answer(basis)['answer_text']} (one row per free variable), so dim U = {len(basis)}.",
    ]
    return _basis_dict(stem, basis, vecs, working)


# -- hard ------------------------------------------------------------------------------------------------------------------

_RREF_RULE = (
    "Give your basis as the non-zero rows of the reduced row echelon form of the matrix whose rows are a basis (this form is "
    "the same for every basis of the subspace)."
)


def _rref_rows(vectors):
    reduced, pivots = rref(fm(vectors))
    return [row for row in reduced[: len(pivots)]]


def _span_text(vectors):
    return ", ".join("(" + ", ".join(str(int(v)) for v in row) + ")" for row in vectors)


def _intersection_basis_via_sympy(gens_u, gens_w):
    """A basis of span(U) ∩ span(W) by solving u-combination = w-combination in SymPy."""
    u = sympy.Matrix([[sympy.Rational(v.numerator, v.denominator) for v in row] for row in gens_u]).T
    w = sympy.Matrix([[sympy.Rational(v.numerator, v.denominator) for v in row] for row in gens_w]).T
    joint = u.row_join(-w)
    vectors = []
    for sol in joint.nullspace():
        v = u * sol[: u.cols, :]
        vectors.append([Fraction(int(c.p), int(c.q)) for c in v])
    return vectors


def _shuffled_generators(rng, vectors, cap=9):
    """`vectors` replaced by unimodular combinations of themselves, so the original vectors are not visible."""
    k = len(vectors)
    for _ in attempts(300, "shuffled generators with small entries"):
        u = unimodular(rng, k, 2)
        mixed = mmul(u, vectors)
        if (
            all(abs(v) <= cap for row in mixed for v in row)
            and all(any(v != 0 for v in row) for row in mixed)
            and mixed != vectors
        ):
            return mixed
    raise AssertionError  # pragma: no cover


def _hard_two_spans(rng):
    n = 4
    for _ in attempts(500, "two planes of R^4 meeting in a line"):
        w, p, q = [[Fraction(rng.randint(-2, 3)) for _ in range(n)] for _ in range(3)]
        if rank([w, p, q]) != 3:
            continue
        gens_u = _shuffled_generators(rng, [w, p])
        gens_w = _shuffled_generators(rng, [w, q])
        if any(rank([gu, gw]) < 2 for gu in gens_u for gw in gens_w):
            continue
        inter = _intersection_basis_via_sympy(gens_u, gens_w)
        expected = _rref_rows([w])
        if len(inter) == 1 and _rref_rows(inter) == expected and rank(gens_u) == 2 and rank(gens_w) == 2:
            break
    basis = expected
    check(rank([*gens_u, *gens_w]) == 3, "the sum of the planes is not 3-dimensional")
    stem = phrase(
        rng,
        (
            "In ℝ⁴ let U = span{{{u}}} and W = span{{{w}}}. Find a basis of the intersection U ∩ W. " + _RREF_RULE,
            "The subspaces U = span{{{u}}} and W = span{{{w}}} of ℝ⁴ meet in a line. Find a basis of U ∩ W. "
            + _RREF_RULE,
        ),
        u=_span_text(gens_u),
        w=_span_text(gens_w),
    )
    answer = mat_answer(basis)
    wrong_u = _rref_rows(gens_u)
    wrong_w = _rref_rows(gens_w)
    primitive_row = [[Fraction(v) for v in primitive(basis[0])]]
    cands = [
        (wrong_u[:1], "intersection-used-one-subspace"),
        (wrong_w[:1], "intersection-used-one-subspace"),
        (primitive_row, "basis-not-normalised"),
        ([[-v for v in basis[0]]], "nullvec-component-sign-slip"),
        ([[sum(g[i] for g in [gens_u[0], gens_w[0]]) for i in range(n)]], "intersection-sum-for-intersection"),
        ([[basis[0][i] for i in reversed(range(n))]], "basis-free-variables-misidentified"),
    ]
    distractors = pick_distractors(answer["answer_text"], mat_distractors(cands, basis), 4)
    if len(distractors) < 3:
        return None
    working = [
        "A vector in both subspaces is a combination of the generators of U that equals a combination of the generators of W: solve a₁u₁ + a₂u₂ = b₁w₁ + b₂w₂.",
        f"The solution space is a line, spanned by a vector that reduces to {answer['answer_text']}.",
    ]
    return dict(text=stem, working=working, distractors=distractors, **answer)


def _hard_equation_and_span(rng):
    n = 4
    for _ in attempts(500, "an equation-defined and a span-defined subspace of R^4"):
        reduced, _pivots = _reduced_system(rng, n, 1)
        equations = _mix(rng, reduced, 1, 9)
        gens = [[Fraction(rng.randint(-3, 3)) for _ in range(n)] for _ in range(2)]
        if rank(gens) != 2:
            continue
        # u = a g1 + b g2 satisfies e·u = 0: one linear condition on (a, b).
        e = equations[0]
        c1 = sum(e[i] * gens[0][i] for i in range(n))
        c2 = sum(e[i] * gens[1][i] for i in range(n))
        if c1 == 0 or c2 == 0:
            continue
        direction = [c2 * gens[0][i] - c1 * gens[1][i] for i in range(n)]
        if all(v == 0 for v in direction):
            continue
        break
    basis = _rref_rows([direction])
    # Independent route: SymPy intersects the null space of the equation with the span of the generators.
    inter = _intersection_basis_via_sympy(gens, nullspace(fm(equations)))
    check(len(inter) == 1 and _rref_rows(inter) == basis, "SymPy's intersection differs")
    names = _names(n)
    stem = phrase(
        rng,
        (
            "In ℝ⁴ let U = {{x : {eq}}} and W = span{{{w}}}. Find a basis of U ∩ W. " + _RREF_RULE,
            "Let U be the subspace of ℝ⁴ defined by {eq}, and let W = span{{{w}}}. Find a basis of the intersection U ∩ W. "
            + _RREF_RULE,
        ),
        eq=_equation(equations[0], names),
        w=_span_text(gens),
    )
    answer = mat_answer(basis)
    wrong_w = _rref_rows(gens)
    cands = [
        (wrong_w[:1], "intersection-used-one-subspace"),
        ([[Fraction(v) for v in primitive(direction)]], "basis-not-normalised"),
        ([[-v for v in basis[0]]], "nullvec-component-sign-slip"),
        ([[c1 * gens[0][i] - c2 * gens[1][i] for i in range(n)]], "intersection-substitution-sign-slip"),
        ([[c1 * gens[0][i] + c2 * gens[1][i] for i in range(n)]], "intersection-substitution-sign-slip"),
        ([[v for v in reversed(basis[0])]], "basis-free-variables-misidentified"),
    ]
    cands = [
        (_rref_rows(c)[:1] if any(v != 0 for v in c[0]) and tag != "basis-not-normalised" else c, tag)
        for c, tag in cands
    ]
    distractors = pick_distractors(answer["answer_text"], mat_distractors(cands, basis), 4)
    if len(distractors) < 3:
        return None
    working = [
        "Write a vector of W as a·w₁ + b·w₂ and substitute into the equation of U: one equation in a and b, so one free parameter.",
        f"The solutions are multiples of {answer['answer_text']} once reduced, so U ∩ W is a line.",
    ]
    return dict(text=stem, working=working, distractors=distractors, **answer)


def _retry(rng, build):
    for _ in attempts(200, "a question with enough distractors"):
        result = build(rng)
        if result is not None:
            return result
    raise AssertionError  # pragma: no cover


@generator(
    "s2b1_basis_from_equations",
    level=LEVEL,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@plain_stem
def s2b1_basis_from_equations(rng, tier="easy"):
    """Basis of a subspace defined by equations or an intersection; the tiers are described in the module docstring."""
    if tier == "easy":
        return _retry(rng, _easy)
    if tier == "medium":
        return _retry(rng, rng.choice([_medium_poly, _medium_matrix, _medium_dependent, _medium_orthogonal]))
    check(tier == "hard", f"unknown tier {tier!r}")
    return _retry(rng, rng.choice([_hard_two_spans, _hard_equation_and_span]))


# -- exam ----------------------------------------------------------------------------------------------------------------------


def _exam_try(rng):
    n = 4
    k = sympy.symbols("k")
    reduced, _pivots = _reduced_system(rng, n, 2)
    mixed = _mix(rng, reduced)
    r1, r2 = mixed
    combo = [rng.choice([-2, -1, 1, 2]), rng.choice([-2, -1, 1, 2])]
    base = [combo[0] * r1[c] + combo[1] * r2[c] for c in range(n)]
    j = rng.randrange(n)
    k0 = rng.choice([v for v in range(-4, 5) if v != 0])
    # Third equation: base + (k - k0) in position j, so it equals combo1 r1 + combo2 r2 exactly when k = k0.
    third = [sympy.Integer(int(v)) for v in base]
    third[j] = third[j] + k - k0
    if max(abs(v) for v in base) > 12 or abs(int(base[j]) - k0) > 12:
        return None
    rows_k = [[sympy.Integer(int(v)) for v in r1], [sympy.Integer(int(v)) for v in r2], third]
    matrix_k = sympy.Matrix(rows_k)
    check(matrix_k.subs(k, k0).rank() == 2 and matrix_k.subs(k, k0 + 1).rank() == 3, "rank pattern differs")
    # The 3×3 minor on columns excluding one chosen column is non-zero except at k0; use SymPy to confirm k0 is the only root
    # of every minor that involves the parameter.
    roots = set()
    for cols in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]):
        minor = matrix_k[:, cols].det()
        if minor.has(k):
            roots |= set(sympy.solve(minor, k))
    check(roots <= {k0}, "another value of k makes a minor vanish")
    solution_at_k0 = [[Fraction(int(v)) for v in row] for row in matrix_k.subs(k, k0).tolist()]
    basis = nullspace(solution_at_k0)
    _verify_basis(solution_at_k0, basis)
    check(len(basis) == 2, "the solution space at k0 is not 2-dimensional")

    names = _names(n)

    def eq_k(row):
        terms = []
        for c, v in enumerate(names):
            if row[c].has(k):
                const = int(row[c].subs(k, 0))
                inner = f"(k + {const})" if const > 0 else f"(k - {-const})" if const < 0 else "k"
                terms.append((1, f"{inner}{v}"))
            else:
                terms.append((int(row[c]), v))
        return join_terms(terms) + " = 0"

    eqs = ", ".join(eq_k(row) for row in rows_k)
    a_dist = pick_distractors(
        str(k0),
        int_distractors(
            [
                (-k0, "sign-error-moving-term"),
                (k0 + 1, "rank-off-by-one"),
                (int(base[j]), "colspace-forgot-multiplier"),
                *[(Fraction(r), t) for r, t in det_root_distractors_rows(rows_k, k, k0)],
            ],
            k0,
        ),
        3,
    )
    b_dist = pick_distractors(
        "2",
        int_distractors(
            [(1, "dimension-counts-equations"), (4, "nullity-counts-columns"), (0, "kernel-assumed-trivial")], 2
        ),
        3,
    )
    c_answer = mat_answer(basis)
    c_dist = pick_distractors(
        c_answer["answer_text"], mat_distractors(_basis_candidates(solution_at_k0, basis, None), basis), 3
    )
    d_dist = pick_distractors(
        "1", int_distractors([(2, "dimension-ignores-parameter"), (0, "kernel-assumed-trivial")], 1), 3
    )
    if not all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)):
        return None
    stem = f"For a real number k let U_k be the subspace of ℝ⁴ of all vectors (x₁, x₂, x₃, x₄) satisfying the three equations {eqs}."
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the value of k for which the three equations are not independent (the rank of the system drops).",
                answer_kind="number",
                answer=k0,
                answer_text=str(k0),
                marks=3,
                working=[
                    f"The third equation equals {join_terms([(combo[0], '(first)'), (combo[1], '(second)')])} exactly when k = {k0}."
                ],
                distractors=a_dist,
            ),
            dict(
                prompt="Find the dimension of U_k for this value of k.",
                answer_kind="number",
                answer=2,
                answer_text="2",
                marks=2,
                working=["At this k the rank is 2, so the dimension is 4 - 2 = 2."],
                distractors=b_dist,
            ),
            dict(
                prompt="For this value of k, find the basis of U_k in the stated form, as the rows of a matrix: take the free variables (those without a pivot in the reduced row echelon form) in order, and let the i-th row have the i-th free variable equal to 1 and the other free variable 0.",
                marks=3,
                working=[f"Solving the rank-2 system gives the rows {c_answer['answer_text']}."],
                distractors=c_dist,
                **c_answer,
            ),
            dict(
                prompt="Find the dimension of U_k for every other value of k.",
                answer_kind="number",
                answer=1,
                answer_text="1",
                marks=1,
                follow_through="part_b - 1",
                working=["For any other k the rank is 3, one more than at k = " + str(k0) + ", so the dimension is 1."],
                distractors=d_dist,
            ),
        ],
    )


def det_root_distractors_rows(rows, k, k0):
    """Wrong roots from sign slips in a 3×3 minor linear in k: use the first three columns that contain k."""
    matrix = sympy.Matrix(rows)
    for cols in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]):
        sub = matrix[:, cols]
        if sub.det().has(k):
            return det_root_distractors([[sub[i, j] for j in range(3)] for i in range(3)], k, k0)
    return []


@generator("exd_x_s2b1_basis_equations", level=LEVEL, presets={"exam": dict(tier="exam")})
@plain_stem
def exd_x_s2b1_basis_equations(rng, tier="exam"):
    """Exam chain: the k at which three equations become dependent, then the dimension, a basis and the generic dimension."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _retry(rng, _exam_try)


__all__ = ["s2b1_basis_from_equations", "exd_x_s2b1_basis_equations", "transpose"]
