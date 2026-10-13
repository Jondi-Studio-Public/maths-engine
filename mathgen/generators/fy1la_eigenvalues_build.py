"""Generates `fy1la_eigenvalues` questions (easy, medium, hard) and two exam chains.

``ucv1-eigenvalues-characteristic-polynomial`` (``linear-algebra``, answer ``list``): the eigenvalues of a matrix are
the roots of det(A - λI) = 0. The existing ``two-by-two-eigenvalues`` skill covers only an upper-triangular 2×2
matrix, where the eigenvalues can be read off the diagonal; this one needs the characteristic polynomial.

* easy: a 2×2 integer matrix with both off-diagonal entries non-zero and two distinct integer eigenvalues.
* medium: a 3×3 matrix with one eigenvalue isolated (a 1×1 block, in any position) and a full 2×2 block, so the
  characteristic polynomial factors at sight and only a quadratic has to be solved.
* hard: a dense 3×3 matrix with three distinct non-zero integer eigenvalues, so the cubic has to be solved by finding a
  rational root first.
* exam (``exd_x_fy1la_eigenvalues``): a dense 3×3 matrix. (a) the trace, (b) the determinant, (c) the product of the two
  eigenvalues other than a stated one (follows through from (b)), (d) all three eigenvalues.
* exam (``exd_x_fy1la_eigenvalues_2x2``, appended to ``two-by-two-eigenvalues``): a full 2×2 matrix. (a) the trace,
  (b) the determinant, (c) the discriminant t² - 4d (follows through from (a) and (b)), (d) the eigenvalues.

Each matrix is built as P D P⁻¹ with D diagonal and P unimodular, so its eigenvalues are known; they are re-derived
by SymPy's `eigenvals()` and the characteristic polynomial is checked to factor as (λ - λ1)(λ - λ2)(λ - λ3).
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import VerificationError, attempts, check, generator, join_terms, phrase
from ._fy1la_kit import (
    det,
    fm,
    inverse,
    list_distractors,
    mat_str,
    max_abs,
    mmul,
    num_distractors,
    number_part,
    sym,
    unimodular,
)


def _conjugate(p, eigenvalues):
    n = len(eigenvalues)
    d = [[eigenvalues[i] if i == j else 0 for j in range(n)] for i in range(n)]
    return mmul(mmul(p, fm(d)), inverse(p))


def _full_2x2(rng, lo, hi, nonzero=False, cap=20):
    """(A, eigenvalues): an integer 2×2 with both off-diagonal entries non-zero and distinct integer eigenvalues."""
    values = [v for v in range(lo, hi + 1) if v or not nonzero]
    for _ in attempts(1000, "a full 2×2 matrix with integer eigenvalues"):
        l1, l2 = rng.sample(values, 2)
        a = _conjugate(unimodular(rng, 2, 3), [l1, l2])
        if a[0][1] == 0 or a[1][0] == 0 or max_abs(a) > cap:
            continue
        return a, sorted([l1, l2])
    raise VerificationError("unreachable")  # pragma: no cover


def _dense_3x3(rng, nonzero=True, cap=20):
    values = [v for v in range(-5, 7) if v or not nonzero]
    for _ in attempts(3000, "a dense 3×3 matrix with integer eigenvalues"):
        eigenvalues = rng.sample(values, 3)
        a = _conjugate(unimodular(rng, 3, 2), eigenvalues)
        if max_abs(a) > cap or sum(1 for row in a for v in row if v == 0) > 1:
            continue
        return a, sorted(eigenvalues)
    raise VerificationError("unreachable")  # pragma: no cover


def _block_3x3(rng, cap=15):
    """A 3×3 matrix with one isolated eigenvalue and a full 2×2 block, the isolated entry at a random position."""
    values = list(range(-5, 7))
    for _ in attempts(3000, "a block 3×3 matrix"):
        l1, l2, l3 = rng.sample(values, 3)
        block = _conjugate(unimodular(rng, 2, 3), [l2, l3])
        if block[0][1] == 0 or block[1][0] == 0 or max_abs(block) > cap:
            continue
        pos = rng.randrange(3)
        order = [i for i in range(3) if i != pos]
        a = [[Fraction(0)] * 3 for _ in range(3)]
        a[pos][pos] = Fraction(l1)
        for bi, i in enumerate(order):
            for bj, j in enumerate(order):
                a[i][j] = block[bi][bj]
        return a, sorted([l1, l2, l3])
    raise VerificationError("unreachable")  # pragma: no cover


def _verify(a, eigenvalues):
    n = len(a)
    m = sym(a)
    found = m.eigenvals()
    check(found == {sympy.Integer(v): 1 for v in eigenvalues}, "SymPy's eigenvalues differ")
    lam = sympy.Symbol("lam")
    poly = sympy.expand(m.charpoly(lam).as_expr())
    check(poly == sympy.expand(sympy.prod([lam - v for v in eigenvalues])), "the characteristic polynomial differs")
    check(sum(eigenvalues) == sum(a[i][i] for i in range(n)), "trace differs from the sum of eigenvalues")
    prod = 1
    for v in eigenvalues:
        prod *= v
    check(det(a) == prod, "determinant differs from the product of eigenvalues")


def _eigen_candidates(a, eigenvalues):
    n = len(a)
    diag = [a[i][i] for i in range(n)]
    out = [(diag, "eigenvalue-diagonal-entries-reported"), ([-v for v in eigenvalues], "eigenvalue-sign-flipped")]
    if n == 3:
        out.append(([eigenvalues[0], eigenvalues[1], -eigenvalues[2]], "eigenvalue-third-from-trace-sign-slip"))
        out.append((eigenvalues[:2], "eigenvalue-missed-one-root"))
        out.append(([eigenvalues[0], eigenvalues[2], sum(diag)], "eigenvalue-trace-reported-as-root"))
    else:
        out.append(([sum(diag), det(a)], "eigenvalue-trace-and-determinant-reported"))
        out.append(([sum(diag)], "eigenvalue-trace-reported-as-root"))
    return out


def _single(rng, which):
    for _ in attempts(300, "an eigenvalue question"):
        if which == "easy":
            a, eigenvalues = _full_2x2(rng, -4, 6)
        elif which == "medium":
            a, eigenvalues = _block_3x3(rng)
        else:
            a, eigenvalues = _dense_3x3(rng)
        _verify(a, eigenvalues)
        dist = list_distractors(eigenvalues, _eigen_candidates(a, eigenvalues))
        if dist is None:
            continue
        text = ", ".join(str(v) for v in eigenvalues)
        n = len(a)
        stems = (
            "Find the eigenvalues of A = {m}.",
            "Determine the eigenvalues of A = {m}.",
            "Find all the eigenvalues of the {n}×{n} matrix A = {m}.",
        )
        lam = join_terms([(1, "λ²"), (-sum(a[i][i] for i in range(n)), "λ"), (det(a), "")]) if n == 2 else "a cubic"
        return dict(
            text=phrase(rng, stems, m=mat_str(a), n=n),
            answer_kind="list",
            answer=list(eigenvalues),
            answer_text=text,
            answer_latex=text,
            working=[
                f"det(A - λI) = {lam}; its roots are {text}."
                if n == 2
                else f"Expand det(A - λI) = 0 and find its three roots: {text}.",
                f"Check: the eigenvalues add to the trace {sum(a[i][i] for i in range(n))} and multiply to det A = {det(a)}.",
            ],
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


@generator(
    "fy1la_eigenvalues",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1la_eigenvalues(rng, tier="easy"):
    """Eigenvalues from the characteristic polynomial; the tiers are described in the module docstring."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return _single(rng, tier)


# -- exam -------------------------------------------------------------------------------------------------------


def _trace_candidates(a):
    n = len(a)
    tr = sum(a[i][i] for i in range(n))
    return [
        (sum(v for row in a for v in row), "sum-of-all-entries-as-trace"),
        (-tr, "trace-sign-slip"),
        (tr - a[n - 1][n - 1], "trace-misses-a-diagonal-entry"),
        (tr - a[0][0], "trace-misses-a-diagonal-entry"),
    ]


def _det_candidates(a, d):
    n = len(a)
    diag_product = 1
    for i in range(n):
        diag_product *= a[i][i]
    return [
        (-d, "det-expansion-sign-slip"),
        (sum(a[i][i] for i in range(n)), "trace-reported-as-determinant"),
        (diag_product, "determinant-as-product-of-diagonal"),
    ]


@generator("exd_x_fy1la_eigenvalues", level=17, presets={"exam": dict(tier="exam")})
def exd_x_fy1la_eigenvalues(rng, tier="exam"):
    """Exam chain on a dense 3×3: trace, determinant, product of the other two eigenvalues, all eigenvalues."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "an exam eigenvalue chain"):
        a, eigenvalues = _dense_3x3(rng)
        _verify(a, eigenvalues)
        tr, d = int(sum(a[i][i] for i in range(3))), int(det(a))
        e = rng.choice(eigenvalues)
        others = [v for v in eigenvalues if v != e]
        product = others[0] * others[1]
        check(d == e * product, "determinant is not e times the product of the others")
        dist_a = num_distractors(tr, _trace_candidates(a))
        dist_b = num_distractors(d, _det_candidates(a, d))
        dist_c = num_distractors(
            product,
            [
                (-product, "product-of-others-sign-slip"),
                (d * e, "product-multiplied-instead-of-divided"),
                (d, "whole-determinant-reported-as-product"),
                (tr - e, "sum-of-others-reported-as-product"),
            ],
        )
        dist_d = list_distractors(eigenvalues, _eigen_candidates(a, eigenvalues))
        if None in (dist_a, dist_b, dist_c, dist_d):
            continue
        text = ", ".join(str(v) for v in eigenvalues)
        return dict(
            text=f"Let A = {mat_str(a)}.",
            parts=[
                number_part(
                    "Find the trace of A.",
                    tr,
                    dist_a,
                    marks=1,
                    working=[f"The trace is the sum of the diagonal entries: {tr}."],
                ),
                number_part(
                    "Find det(A).",
                    d,
                    dist_b,
                    marks=2,
                    working=[f"Expanding along a row gives det(A) = {d}."],
                ),
                number_part(
                    f"Given that {e} is an eigenvalue of A, find the product of the other two eigenvalues.",
                    product,
                    dist_c,
                    marks=1,
                    follow_through=f"part_b/({e})",
                    working=[
                        f"The three eigenvalues multiply to det(A) = {d}, so the other two multiply to {d}/{e} = {product}."
                    ],
                ),
                dict(
                    prompt="Find all three eigenvalues of A.",
                    answer_kind="list",
                    answer=list(eigenvalues),
                    answer_text=text,
                    answer_latex=text,
                    marks=3,
                    working=[
                        f"The other two eigenvalues have sum {tr} - ({e}) = {tr - e} and product {product}, "
                        f"so they are the roots of {join_terms([(1, 'λ²'), (-(tr - e), 'λ'), (product, '')])}. The eigenvalues are {text}."
                    ],
                    distractors=dist_d,
                ),
            ],
        )
    raise VerificationError("unreachable")  # pragma: no cover


@generator("exd_x_fy1la_eigenvalues_2x2", level=14, presets={"exam": dict(tier="exam")})
def exd_x_fy1la_eigenvalues_2x2(rng, tier="exam"):
    """Exam chain on a full 2×2: trace, determinant, discriminant of the characteristic polynomial, eigenvalues."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "an exam 2×2 eigenvalue chain"):
        a, eigenvalues = _full_2x2(rng, -5, 7)
        _verify(a, eigenvalues)
        tr, d = int(a[0][0] + a[1][1]), int(det(a))
        disc = tr * tr - 4 * d
        check(disc == (eigenvalues[1] - eigenvalues[0]) ** 2, "discriminant is not the squared gap between the roots")
        dist_a = num_distractors(tr, _trace_candidates(a))
        dist_b = num_distractors(
            d,
            [
                (a[0][0] * a[1][1], "determinant-as-product-of-diagonal"),
                (a[0][0] * a[1][1] + a[0][1] * a[1][0], "det-formula-sign-slip"),
                (a[0][1] * a[1][0] - a[0][0] * a[1][1], "det-formula-sign-slip"),
                (-d, "det-expansion-sign-slip"),
            ],
        )
        dist_c = num_distractors(
            disc,
            [
                (tr * tr + 4 * d, "discriminant-sign-slip"),
                (tr - 4 * d, "discriminant-forgot-to-square"),
                (tr * tr - 2 * d, "discriminant-wrong-factor"),
                (tr * tr - d, "discriminant-wrong-factor"),
            ],
        )
        dist_d = list_distractors(eigenvalues, _eigen_candidates(a, eigenvalues))
        if None in (dist_a, dist_b, dist_c, dist_d):
            continue
        text = ", ".join(str(v) for v in eigenvalues)
        return dict(
            text=f"Let A = {mat_str(a)}, with characteristic polynomial λ² - tλ + d, where t = tr(A) and d = det(A).",
            parts=[
                number_part("Find t.", tr, dist_a, marks=1, working=[f"t is the sum of the diagonal entries: {tr}."]),
                number_part(
                    "Find d.",
                    d,
                    dist_b,
                    marks=2,
                    working=[f"d = ({a[0][0]})({a[1][1]}) - ({a[0][1]})({a[1][0]}) = {d}."],
                ),
                number_part(
                    "Find the discriminant t² - 4d of the characteristic polynomial.",
                    disc,
                    dist_c,
                    marks=1,
                    follow_through="part_a**2 - 4*part_b",
                    working=[f"t² - 4d = {tr}² - 4({d}) = {disc}."],
                ),
                dict(
                    prompt="Find the eigenvalues of A.",
                    answer_kind="list",
                    answer=list(eigenvalues),
                    answer_text=text,
                    answer_latex=text,
                    marks=2,
                    working=[f"λ = ({tr} ± √{disc})/2, so the eigenvalues are {text}."],
                    distractors=dist_d,
                ),
            ],
        )
    raise VerificationError("unreachable")  # pragma: no cover
