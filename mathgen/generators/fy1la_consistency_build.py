"""Generates `fy1la_consistency` questions (easy, medium, hard) and the exam chain `exd_x_fy1la_consistency`.

``ucv1-system-consistency-parameter`` (``linear-algebra``, answer ``number``): how many solutions a linear system has
when one coefficient is an unknown constant k, found from the determinant of the coefficient matrix and a check of
the singular case.

* easy: two equations in x and y with k in one coefficient. The value of k for which the system has NO solution (the
  lines are parallel and distinct).
* medium: three equations in x, y and z with k in one coefficient. The value of k for which the system does not have a
  unique solution (det A = 0, linear in k).
* hard: three equations with k in TWO coefficients, so det A is quadratic in k and has two roots. The right-hand side is
  chosen so the system is consistent at one root and inconsistent at the other; the question asks for the value of k
  that gives infinitely many solutions, so the singular case has to be checked, not just the determinant solved.
* exam (``exd_x_fy1la_consistency``): a 3×3 system with k in one coefficient. (a) det A as an expression in k, (b) the
  k for which there is no unique solution, (c) what happens at that k (no solution or infinitely many), (d) one
  unknown at a stated k for which the solution is unique.

Each determinant is computed by exact elimination and re-derived by SymPy, and each singular case is classified by
the ranks of A and [A | b] in two independent routes.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import VerificationError, attempts, check, generator, join_terms, phrase
from ._fy1la_kit import (
    consistent,
    det,
    expression_distractors,
    int_roots,
    expression_part,
    mcq_part,
    num_distractors,
    number_part,
    quadratic_in_k,
    rank,
    solve_unique,
    sym,
    system_text,
)

_VARS = ("x", "y", "z")
_OPTIONS = ("No solution", "Exactly one solution", "Infinitely many solutions")


def _system_text(shape, b, names) -> str:
    return system_text(shape, b, names)


def _build(shape, kval) -> list[list[Fraction]]:
    return [[Fraction(kval) if c == "k" else Fraction(c) for c in row] for row in shape]


def _det_line(shape):
    """(C, D) with det A(k) = C k + D, for a shape with k in one place."""
    d0, d1 = det(_build(shape, 0)), det(_build(shape, 1))
    return d1 - d0, d0


def _draw_linear(rng, n, want_inconsistent):
    """(shape, b, C, D, k0): an n×n coefficient shape with k in one place, a right-hand side, and the integer k0 where
    det A(k0) = 0. With `want_inconsistent` the system has no solution at k0."""
    for _ in attempts(3000, f"an {n}×{n} system with one parameter"):
        shape = [[rng.choice([-3, -2, -1, 1, 2, 3, 4]) for _ in range(n)] for _ in range(n)]
        i, j = rng.randrange(n), rng.randrange(n)
        shape[i][j] = "k"
        C, D = _det_line(shape)
        if C == 0 or D % C != 0:
            continue
        k0 = int(-D / C)
        if k0 == 0 or abs(k0) > 9:
            continue
        a0 = _build(shape, k0)
        if rank(a0) != n - 1:
            continue
        b = [rng.randint(-8, 9) for _ in range(n)]
        if not any(b) or consistent(a0, b) == want_inconsistent:
            continue
        return shape, b, int(C), int(D), k0
    raise VerificationError("unreachable")  # pragma: no cover - attempts() raises first


def _verify_linear(shape, C, D, k0):
    """SymPy: det A(k) = C k + D with the single root k0."""
    k = sympy.Symbol("k")
    m = sympy.Matrix(len(shape), len(shape), lambda r, c: k if shape[r][c] == "k" else shape[r][c])
    check(sympy.expand(m.det() - (C * k + D)) == 0, "SymPy's determinant differs")
    check(sympy.solve(m.det(), k) == [k0], "SymPy's root differs")


def _linear_distractors(k0, C, D):
    out = [
        (-k0, "sign-error-moving-term"),
        (-D, "forgot-to-divide-by-coefficient"),
        (D, "forgot-to-divide-by-coefficient"),
    ]
    if D != 0 and C % D == 0:
        out.append((Fraction(C, D), "linear-solution-inverted"))
    return out


def _det_text(C, D) -> str:
    return join_terms([(C, "k"), (D, "")])


def _easy(rng):
    for _ in attempts(300, "a 2×2 no-solution system"):
        shape, b, C, D, k0 = _draw_linear(rng, 2, True)
        a0 = _build(shape, k0)
        _verify_linear(shape, C, D, k0)
        check(sym(a0).rank() == 1 and sym([r + [b[i]] for i, r in enumerate(a0)]).rank() == 2, "not inconsistent at k0")
        for off in (-1, 1):
            check(det(_build(shape, k0 + off)) != 0, "another k is singular")
        dist = num_distractors(k0, _linear_distractors(k0, C, D))
        if dist is None:
            continue
        stems = (
            "Consider the system {s}, where k is a constant. Find the value of k for which the system has no solution.",
            "The simultaneous equations {s} contain a constant k. For what value of k do they have no solution?",
        )
        return dict(
            text=phrase(rng, stems, s=_system_text(shape, b, _VARS[:2])),
            answer_kind="number",
            answer=k0,
            answer_text=str(k0),
            answer_latex=str(k0),
            working=[
                f"det A = {_det_text(C, D)}, which is zero only for k = {k0}.",
                f"At k = {k0} the left sides are proportional but the right sides are not, so there is no solution.",
            ],
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


def _medium(rng):
    for _ in attempts(300, "a 3×3 parameter system"):
        shape, b, C, D, k0 = _draw_linear(rng, 3, rng.random() < 0.5)
        _verify_linear(shape, C, D, k0)
        dist = num_distractors(k0, _linear_distractors(k0, C, D))
        if dist is None:
            continue
        stems = (
            "Consider the system {s}, where k is a constant. Find the value of k for which the system does not have a unique solution.",
            "The system {s} contains a constant k. For what value of k does it fail to have exactly one solution?",
        )
        return dict(
            text=phrase(rng, stems, s=_system_text(shape, b, _VARS)),
            answer_kind="number",
            answer=k0,
            answer_text=str(k0),
            answer_latex=str(k0),
            working=[
                f"There is a unique solution exactly when det A is not zero; det A = {_det_text(C, D)}.",
                f"This is zero for k = {k0}.",
            ],
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


def _draw_hard(rng):
    for _ in attempts(6000, "a 3×3 system with a quadratic determinant"):
        shape = [[rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(3)] for _ in range(3)]
        cells = rng.sample([(i, j) for i in range(3) for j in range(3)], 2)
        for i, j in cells:
            shape[i][j] = "k"
        c2, c1, c0 = quadratic_in_k(lambda kv: _build(shape, kv))
        roots = int_roots(c2, c1, c0)
        if roots is None or 0 in roots or max(abs(r) for r in roots) > 8:
            continue
        good, bad = (roots[0], roots[1]) if rng.random() < 0.5 else (roots[1], roots[0])
        a_good, a_bad = _build(shape, good), _build(shape, bad)
        if rank(a_good) != 2 or rank(a_bad) != 2:
            continue
        x0 = [rng.randint(-3, 4) for _ in range(3)]
        b = [int(sum(a_good[r][c] * x0[c] for c in range(3))) for r in range(3)]
        if not any(b) or max(abs(v) for v in b) > 20 or consistent(a_bad, b):
            continue
        return shape, b, (int(c2), int(c1), int(c0)), good, bad
    raise VerificationError("unreachable")  # pragma: no cover


def _hard(rng):
    for _ in attempts(300, "a hard consistency question"):
        shape, b, (c2, c1, c0), good, bad = _draw_hard(rng)
        k = sympy.Symbol("k")
        m = sympy.Matrix(3, 3, lambda r, c: k if shape[r][c] == "k" else shape[r][c])
        check(set(sympy.solve(m.det(), k)) == {good, bad}, "SymPy's roots differ")
        aug = lambda kv: [r + [b[i]] for i, r in enumerate(_build(shape, kv))]  # noqa: E731
        check(sym(_build(shape, good)).rank() == sym(aug(good)).rank() == 2, "not consistent at the good root")
        check(sym(_build(shape, bad)).rank() == 2 and sym(aug(bad)).rank() == 3, "not inconsistent at the other root")
        sum_roots = Fraction(-c1, c2)
        dist = num_distractors(
            good,
            [
                (bad, "singular-means-infinitely-many"),
                (-good, "sign-error-moving-term"),
                (sum_roots, "sum-of-roots-as-root"),
                (Fraction(c0, c2), "product-of-roots-as-root"),
            ],
        )
        if dist is None:
            continue
        stems = (
            "Consider the system {s}, where k is a constant. Find the value of k for which the system has infinitely many solutions.",
            "The system {s} contains a constant k. For what value of k does it have infinitely many solutions?",
        )
        return dict(
            text=phrase(rng, stems, s=_system_text(shape, b, _VARS)),
            answer_kind="number",
            answer=good,
            answer_text=str(good),
            answer_latex=str(good),
            working=[
                f"det A = {join_terms([(c2, 'k²'), (c1, 'k'), (c0, '')])} = 0 gives k = {min(good, bad)} or k = {max(good, bad)}, "
                "so A is singular at both.",
                f"At k = {bad} the system is inconsistent (no solution). At k = {good} the rank of A and of [A | b] is 2, "
                f"so there are infinitely many solutions.",
            ],
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


@generator(
    "fy1la_consistency",
    level=16,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1la_consistency(rng, tier="easy"):
    """How many solutions a system with a parameter has; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_fy1la_consistency", level=16, presets={"exam": dict(tier="exam")})
def exd_x_fy1la_consistency(rng, tier="exam"):
    """Exam chain: det A in k, the singular k, what happens there, and one unknown at a stated k."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "an exam consistency chain"):
        inconsistent = rng.random() < 0.5
        shape, b, C, D, k0 = _draw_linear(rng, 3, inconsistent)
        _verify_linear(shape, C, D, k0)
        a0 = _build(shape, k0)
        check(consistent(a0, b) != inconsistent, "classification at k0 differs")
        # (d) a k with a unique solution and a component that is a whole number.
        pick = None
        for k1 in rng.sample([v for v in range(-6, 7) if v not in (0, k0)], 10):
            a1 = _build(shape, k1)
            d1 = det(a1)
            if abs(d1) < 2:
                continue
            sol = solve_unique(a1, b)
            for idx in rng.sample(range(3), 3):
                if sol[idx].denominator == 1 and sol[idx] != 0:
                    pick = (k1, idx, int(sol[idx]), a1, d1)
                    break
            if pick:
                break
        if pick is None:
            continue
        k1, idx, value, a1, d1 = pick
        check(sym(a1).LUsolve(sympy.Matrix(b))[idx] == value, "SymPy's solution differs")
        numerator = det([[Fraction(b[r]) if c == idx else a1[r][c] for c in range(3)] for r in range(3)])
        dist_d = num_distractors(
            value,
            [
                (-value, "solution-sign-slip"),
                (numerator, "cramer-numerator-not-divided"),
                (Fraction(d1) / numerator if numerator else 0, "cramer-ratio-inverted"),
            ],
        )
        k = sympy.Symbol("k")
        dist_a = expression_distractors(
            str(C * k + D),
            [
                (_det_text(C, -D), "det-expansion-sign-slip"),
                (_det_text(-C, D), "det-expansion-sign-slip"),
                (_det_text(C, 0), "dropped-constant-term-det"),
                (_det_text(-C, -D), "det-expansion-sign-slip"),
            ],
            "Find det(A) in terms of k.",
        )
        dist_b = num_distractors(k0, _linear_distractors(k0, C, D))
        answer_c = _OPTIONS[0] if inconsistent else _OPTIONS[2]
        other = _OPTIONS[2] if inconsistent else _OPTIONS[0]
        dist_c = [
            Distractor(other, "singular-means-infinitely-many" if inconsistent else "singular-means-no-solution"),
            Distractor(_OPTIONS[1], "singular-still-unique"),
        ]
        if None in (dist_a, dist_b, dist_d):
            continue
        names = _VARS
        return dict(
            text=f"Consider the system Ax = b given by {_system_text(shape, b, names)}, where k is a constant.",
            parts=[
                expression_part(
                    "Find det(A) in terms of k.",
                    C * k + D,
                    _det_text(C, D),
                    dist_a,
                    marks=2,
                    working=["Expand along a row, keeping the k term."],
                ),
                number_part(
                    "Find the value of k for which the system does not have a unique solution.",
                    k0,
                    dist_b,
                    marks=1,
                    working=[f"A is singular when {_det_text(C, D)} = 0, so k = {k0}."],
                ),
                mcq_part(
                    f"How many solutions does the system have when k = {k0}?",
                    answer_c,
                    dist_c,
                    marks=2,
                    working=[
                        f"At k = {k0}, rank A = 2 and rank [A | b] = {2 if not inconsistent else 3}, so the system has "
                        f"{'no solution' if inconsistent else 'infinitely many solutions'}."
                    ],
                ),
                number_part(
                    f"Find the value of {names[idx]} when k = {k1}.",
                    value,
                    dist_d,
                    marks=3,
                    working=[
                        f"At k = {k1}, det A = {d1} is not zero, so the solution is unique; solving gives {names[idx]} = {value}."
                    ],
                ),
            ],
        )
    raise VerificationError("unreachable")  # pragma: no cover
