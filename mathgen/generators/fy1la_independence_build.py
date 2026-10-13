"""Generates `fy1la_independence` questions (easy, medium, hard) and the exam chain `exd_x_fy1la_independence`.

``ucv1-linear-independence-r3`` (``linear-algebra``, answer ``number``): three vectors u, v, w in ℝ³ are linearly
dependent exactly when det[u v w] = 0, and then some non-trivial relation c1·u + c2·v + c3·w = 0 holds.

* easy: one entry of one vector is an unknown k; the value of k for which the three vectors are linearly dependent
  (det is linear in k).
* medium: the vectors are dependent already, with u and v independent; the coefficient c1 in the dependence relation
  c1·u + c2·v + w = 0 (a linear system for c1 and c2, then a sign change from w = -c1·u - c2·v).
* hard: k appears in TWO entries, so det is quadratic in k with one positive and one negative root, and the question
  states k > 0; the learner solves the quadratic and uses the condition to choose the root.
* exam (``exd_x_fy1la_independence``): k in one entry of u. (a) det[u v w] as an expression in k, (b) the k for which
  the vectors are dependent, (c) the coefficient c1 and (d) the coefficient c2 in the relation c1·u + c2·v + w = 0 at that k.

The determinant is computed by exact elimination and re-derived by SymPy; each dependence relation is checked by
substitution and by SymPy's null space.
"""

from __future__ import annotations

import sympy

from ..core import VerificationError, attempts, check, generator, join_terms, phrase
from ._fy1la_kit import (
    det,
    expression_distractors,
    expression_part,
    fm,
    int_roots,
    nullspace,
    num_distractors,
    number_part,
    quadratic_in_k,
    rank,
    sym,
    vector_with_k,
    with_columns,
)


def _vecs_text(shape) -> str:
    u, v, w = (vector_with_k(vec) for vec in shape)
    return f"u = {u}, v = {v} and w = {w}"


def _build(shape, kval):
    return [[kval if x == "k" else x for x in vec] for vec in shape]


def _matrix(shape, kval):
    return with_columns(_build(shape, kval))


def _relation(u, v, w):
    """(c1, c2) with c1 u + c2 v + w = 0 for independent u, v, or None; the null vector scaled so the w-coefficient is 1."""
    null = nullspace(with_columns([u, v, w]))
    if len(null) != 1 or null[0][2] == 0:
        return None
    scale = null[0][2]
    return null[0][0] / scale, null[0][1] / scale


def _verify_det(shape, C, D, k0):
    k = sympy.Symbol("k")
    m = sympy.Matrix(3, 3, lambda r, c: k if shape[c][r] == "k" else shape[c][r])
    check(sympy.expand(m.det() - (C * k + D)) == 0, "SymPy's determinant differs")
    check(sympy.solve(m.det(), k) == [k0], "SymPy's root differs")


def _draw_linear(rng):
    """(shape, relation, (C, D), k0): three vectors with one "k" entry, dependent at the integer k0 through the
    relation c1 u + c2 v + w = 0 (so u and v are independent at k0)."""
    nz = [x for x in range(-3, 4) if x]
    for _ in attempts(3000, "a dependent triple with one parameter"):
        u0 = [rng.randint(-3, 4) for _ in range(3)]
        v = [rng.randint(-3, 4) for _ in range(3)]
        c1, c2 = rng.choice(nz), rng.choice(nz)
        w = [-c1 * a - c2 * b for a, b in zip(u0, v)]
        if max(abs(x) for x in w) > 12 or rank(fm([u0, v])) != 2:
            continue
        m = rng.randrange(3)
        t = rng.randrange(3)
        vecs = [list(u0), list(v), list(w)]
        k0 = vecs[m][t]
        if k0 == 0 or abs(k0) > 9:
            continue
        shape = [list(vec) for vec in vecs]
        shape[m][t] = "k"
        d0, d1 = det(_matrix(shape, 0)), det(_matrix(shape, 1))
        C, D = int(d1 - d0), int(d0)
        if C == 0 or D != -C * k0:
            continue
        if sum(1 for vec in shape for x in vec if x == 0) > 2:
            continue
        # k in w would let k0 change the relation's meaning only through w, which is fine; k in u or v needs the
        # relation recomputed at k0, which `_relation` does below.
        return shape, (C, D), k0
    raise VerificationError("unreachable")  # pragma: no cover - attempts() raises first


def _linear_distractors(k0, C, D):
    out = [
        (-k0, "sign-error-moving-term"),
        (-D, "forgot-to-divide-by-coefficient"),
        (D, "forgot-to-divide-by-coefficient"),
    ]
    if D and C % D == 0:
        out.append((C // D, "linear-solution-inverted"))
    return out


def _det_text(C, D) -> str:
    return join_terms([(C, "k"), (D, "")])


def _easy(rng):
    for _ in attempts(300, "an easy independence question"):
        shape, (C, D), k0 = _draw_linear(rng)
        _verify_det(shape, C, D, k0)
        u, v, w = _build(shape, k0)
        check(rank(with_columns([u, v, w])) == 2, "not dependent at k0")
        dist = num_distractors(k0, _linear_distractors(k0, C, D))
        if dist is None:
            continue
        stems = (
            "Find the value of k for which the vectors {vs} are linearly dependent.",
            "The vectors {vs} in ℝ³ depend on a constant k. For what value of k are they linearly dependent?",
        )
        return dict(
            text=phrase(rng, stems, vs=_vecs_text(shape)),
            answer_kind="number",
            answer=k0,
            answer_text=str(k0),
            answer_latex=str(k0),
            working=[
                f"Three vectors in ℝ³ are dependent exactly when det[u v w] = 0; here det = {_det_text(C, D)}.",
                f"This is zero for k = {k0}.",
            ],
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


def _medium(rng):
    nz = [x for x in range(-3, 4) if x]
    for _ in attempts(1000, "a dependence-relation question"):
        u = [rng.randint(-3, 4) for _ in range(3)]
        v = [rng.randint(-3, 4) for _ in range(3)]
        c1, c2 = rng.choice(nz), rng.choice(nz)
        w = [-c1 * a - c2 * b for a, b in zip(u, v)]
        if (
            sum(1 for x in u + v if x == 0) > 1
            or max(abs(x) for x in w) > 12
            or any(all(x == 0 for x in vec) for vec in (u, v, w))
        ):
            continue
        rel = _relation(u, v, w)
        if rel is None or rel != (c1, c2):
            continue
        check(det(with_columns([u, v, w])) == 0, "not dependent")
        null = sym(with_columns([u, v, w])).nullspace()
        check(len(null) == 1 and null[0][2] != 0 and null[0][0] / null[0][2] == c1, "SymPy's relation differs")
        dist = num_distractors(
            c1,
            [
                (-c1, "dependence-relation-sign-slip"),
                (c2, "dependence-coefficients-swapped"),
                (-c2, "dependence-coefficients-swapped"),
                (1, "dependence-coefficient-of-w-reported"),
            ],
        )
        if dist is None:
            continue
        vs = f"u = {vector_with_k(u)}, v = {vector_with_k(v)} and w = {vector_with_k(w)}"
        stems = (
            "The vectors {vs} are linearly dependent, with c1·u + c2·v + w = 0 for some constants c1 and c2. Find c1.",
            "For the vectors {vs}, there are constants c1 and c2 such that c1·u + c2·v + w = 0. Find c1.",
            "Given the vectors {vs}, the equation c1·u + c2·v + w = 0 holds for some constants c1 and c2. What is c1?",
        )
        return dict(
            text=phrase(rng, stems, vs=vs),
            answer_kind="number",
            answer=c1,
            answer_text=str(c1),
            answer_latex=str(c1),
            working=[
                "Write c1·u + c2·v = -w coordinate by coordinate and solve two of the equations for c1 and c2.",
                f"This gives c1 = {c1} and c2 = {c2}, and the third coordinate checks.",
            ],
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


def _hard(rng):
    for _ in attempts(8000, "a quadratic independence question"):
        shape = [[rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(3)] for _ in range(3)]
        for i, j in rng.sample([(a, b) for a in range(3) for b in range(3)], 2):
            shape[i][j] = "k"
        c2, c1, c0 = quadratic_in_k(lambda kv: _matrix(shape, kv))
        roots = int_roots(c2, c1, c0)
        if roots is None or roots[0] >= 0 or roots[1] <= 0 or max(abs(r) for r in roots) > 8:
            continue
        good, bad = roots[1], roots[0]
        if rank(_matrix(shape, good)) != 2:
            continue
        k = sympy.Symbol("k")
        m = sympy.Matrix(3, 3, lambda r, c: k if shape[c][r] == "k" else shape[c][r])
        check(set(sympy.solve(m.det(), k)) == {good, bad}, "SymPy's roots differ")
        dist = num_distractors(
            good,
            [
                (bad, "root-chosen-ignoring-condition"),
                (-good, "sign-error-moving-term"),
                (-c1 / c2, "sum-of-roots-as-root"),
                (c0 / c2, "product-of-roots-as-root"),
            ],
        )
        if dist is None:
            continue
        stems = (
            "The vectors {vs} depend on a positive constant k. Find the value of k for which they are linearly dependent.",
            "Given that k > 0, find the value of k for which the vectors {vs} are linearly dependent.",
        )
        return dict(
            text=phrase(rng, stems, vs=_vecs_text(shape)),
            answer_kind="number",
            answer=good,
            answer_text=str(good),
            answer_latex=str(good),
            working=[
                f"det[u v w] = {join_terms([(int(c2), 'k²'), (int(c1), 'k'), (int(c0), '')])} = 0 gives k = {bad} or k = {good}.",
                f"Since k > 0, k = {good}.",
            ],
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


@generator(
    "fy1la_independence",
    level=16,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1la_independence(rng, tier="easy"):
    """Linear dependence of three vectors in ℝ³; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_fy1la_independence", level=16, presets={"exam": dict(tier="exam")})
def exd_x_fy1la_independence(rng, tier="exam"):
    """Exam chain: det in k, the dependent k, and the two coefficients of the dependence relation at that k."""
    check(tier == "exam", f"unknown tier {tier!r}")
    k = sympy.Symbol("k")
    for _ in attempts(1000, "an exam independence chain"):
        shape, (C, D), k0 = _draw_linear(rng)
        _verify_det(shape, C, D, k0)
        u, v, w = _build(shape, k0)
        rel = _relation(u, v, w)
        if rel is None or rel[0].denominator != 1 or rel[1].denominator != 1 or rank(with_columns([u, v])) != 2:
            continue
        c1, c2 = int(rel[0]), int(rel[1])
        check(all(c1 * a + c2 * b + c == 0 for a, b, c in zip(u, v, w)), "the relation fails")
        dist_a = expression_distractors(
            str(C * k + D),
            [
                (join_terms([(C, "k"), (-D, "")]), "det-expansion-sign-slip"),
                (join_terms([(-C, "k"), (D, "")]), "det-expansion-sign-slip"),
                (join_terms([(C, "k")]), "dropped-constant-term-det"),
                (join_terms([(-C, "k"), (-D, "")]), "det-expansion-sign-slip"),
            ],
            "Find det[u v w] in terms of k.",
        )
        dist_b = num_distractors(k0, _linear_distractors(k0, C, D))
        dist_c = num_distractors(
            c1,
            [
                (-c1, "dependence-relation-sign-slip"),
                (c2, "dependence-coefficients-swapped"),
                (-c2, "dependence-coefficients-swapped"),
                (1, "dependence-coefficient-of-w-reported"),
            ],
        )
        dist_d = num_distractors(
            c2,
            [
                (-c2, "dependence-relation-sign-slip"),
                (c1, "dependence-coefficients-swapped"),
                (-c1, "dependence-coefficients-swapped"),
                (1, "dependence-coefficient-of-w-reported"),
            ],
        )
        if None in (dist_a, dist_b, dist_c, dist_d):
            continue
        return dict(
            text=f"Let {_vecs_text(shape)} in ℝ³, where k is a constant, and let M be the 3×3 matrix with columns u, v and w.",
            parts=[
                expression_part(
                    "Find det(M) in terms of k.",
                    C * k + D,
                    _det_text(C, D),
                    dist_a,
                    marks=2,
                    working=["Expand along the row or column containing k."],
                ),
                number_part(
                    "Find the value of k for which u, v and w are linearly dependent.",
                    k0,
                    dist_b,
                    marks=1,
                    working=[f"The vectors are dependent when det(M) = 0: {_det_text(C, D)} = 0, so k = {k0}."],
                ),
                number_part(
                    "For this k, the vectors satisfy c1·u + c2·v + w = 0. Find c1.",
                    c1,
                    dist_c,
                    marks=2,
                    working=[
                        f"At k = {k0}, u = {vector_with_k(u)}, v = {vector_with_k(v)}, w = {vector_with_k(w)}; solving c1·u + c2·v = -w gives c1 = {c1}."
                    ],
                ),
                number_part(
                    "For the same relation, find c2.",
                    c2,
                    dist_d,
                    marks=2,
                    working=[f"The same solution gives c2 = {c2}; check: {c1}u + ({c2})v + w = 0."],
                ),
            ],
        )
    raise VerificationError("unreachable")  # pragma: no cover
