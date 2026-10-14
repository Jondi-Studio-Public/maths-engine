"""Induced matrix norms and the condition number: `r4h_matrix_norm_condition` (easy, medium, hard) and the exam chain
`exd_x_r4h_matrix_norm_condition`.

``uc2-matrix-norm-1-and-infinity`` (``numerical-and-computational-mathematics``). ``‖A‖₁`` is the largest absolute
column sum, ``‖A‖∞`` the largest absolute row sum and ``κ(A) = ‖A‖ ‖A⁻¹‖``.

* easy: ``‖A‖₁`` or ``‖A‖∞`` of a 3×3 matrix with negative entries (a whole number).
* medium: ``κ∞(A)`` of a 2×2 matrix (a fraction).
* hard: ``κ₁(A)`` of a 3×3 matrix with a small determinant (a fraction).
* exam: a 2×2 matrix: ``‖A‖₁``, ``‖A⁻¹‖₁``, ``κ₁(A)`` (follow-through from both) and the bound on the relative error
  of the solution of ``Ax = b`` when ``b`` is perturbed by a stated relative amount (follow-through from ``κ₁``).

Exact arithmetic with `Fraction`; the inverse is the adjugate over the determinant.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._r4h_kit import (
    Redraw,
    finish_parts,
    fraction_part,
    ft_symbol,
    integer_part,
    mat,
    shown,
    single_from,
    with_redraw,
)

MISC = {
    "num2t-norm-rows-for-columns": "Adding along rows when the 1-norm (largest column sum) was asked for.",
    "num2t-norm-columns-for-rows": "Adding down columns when the ∞-norm (largest row sum) was asked for.",
    "num2t-norm-signed-sums": "Adding the entries with their signs instead of their absolute values.",
    "num2t-norm-largest-entry": "Giving the entry of largest size instead of the largest absolute row or column sum.",
    "num2t-norm-all-entries": "Adding the absolute values of every entry in the matrix.",
    "num2t-norm-smallest-sum": "Choosing the smallest absolute sum instead of the largest.",
    "num2t-norm-sum-not-max": "Adding the row (or column) sums instead of taking the largest.",
    "num2t-cond-no-inverse": "Multiplying ‖A‖ by itself instead of by ‖A⁻¹‖.",
    "num2t-cond-adjugate-undivided": "Using the adjugate in place of A⁻¹, forgetting to divide by the determinant.",
    "num2t-cond-wrong-norm-on-inverse": "Using the 1-norm for one factor and the ∞-norm for the other.",
    "num2t-cond-quotient": "Dividing ‖A‖ by ‖A⁻¹‖ instead of multiplying.",
    "num2t-cond-determinant": "Giving the determinant or its reciprocal as the condition number.",
    "num2t-cond-other-norm": "Giving the condition number in the other norm (∞ for 1, or 1 for ∞).",
    "num2t-cond-inverse-sign": "Forming the inverse of a 2×2 matrix with the off-diagonal signs unchanged.",
    "num2t-cond-inverse-entries-unswapped": "Forming the inverse of a 2×2 matrix without swapping the diagonal entries.",
    "num2t-cond-bound-divided": "Dividing the relative perturbation by the condition number instead of multiplying.",
    "num2t-cond-bound-forgotten": "Giving the relative perturbation of b as the bound on the relative error of x.",
    "num2t-cond-bound-norm-only": "Multiplying the relative perturbation by ‖A⁻¹‖ alone.",
}
M_ROWS = "num2t-norm-rows-for-columns"
M_COLS = "num2t-norm-columns-for-rows"
M_SIGNED = "num2t-norm-signed-sums"
M_ENTRY = "num2t-norm-largest-entry"
M_ALL = "num2t-norm-all-entries"
M_SMALL = "num2t-norm-smallest-sum"
M_SUM = "num2t-norm-sum-not-max"
M_NOINV = "num2t-cond-no-inverse"
M_ADJ = "num2t-cond-adjugate-undivided"
M_MIXED = "num2t-cond-wrong-norm-on-inverse"
M_QUOT = "num2t-cond-quotient"
M_DET = "num2t-cond-determinant"
M_OTHER = "num2t-cond-other-norm"
M_INV_SIGN = "num2t-cond-inverse-sign"
M_INV_SWAP = "num2t-cond-inverse-entries-unswapped"


def norm1(m) -> Fraction:
    return max(sum(abs(Fraction(m[i][j])) for i in range(len(m))) for j in range(len(m[0])))


def norminf(m) -> Fraction:
    return max(sum(abs(Fraction(v)) for v in row) for row in m)


def det2(a) -> int:
    return a[0][0] * a[1][1] - a[0][1] * a[1][0]


def det3(a) -> int:
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def adj2(a):
    return [[a[1][1], -a[0][1]], [-a[1][0], a[0][0]]]


def adj3(a):
    """Adjugate (transpose of the cofactor matrix) of a 3x3."""

    def minor(i, j):
        r = [x for x in range(3) if x != i]
        c = [x for x in range(3) if x != j]
        return a[r[0]][c[0]] * a[r[1]][c[1]] - a[r[0]][c[1]] * a[r[1]][c[0]]

    return [[(-1) ** (i + j) * minor(j, i) for j in range(3)] for i in range(3)]


def inverse(a):
    n = len(a)
    d = det2(a) if n == 2 else det3(a)
    adj = adj2(a) if n == 2 else adj3(a)
    return [[Fraction(v, d) for v in row] for row in adj]


def _draw2(rng, lo=-6, hi=9):
    for _ in attempts(2000, "a 2x2 matrix"):
        a = [[rng.randint(lo, hi) for _ in range(2)] for _ in range(2)]
        d = det2(a)
        if d == 0 or abs(d) > 12 or abs(d) == 1 and rng.random() < 0.7:
            continue
        if any(v == 0 for row in a for v in row):
            continue
        inv = inverse(a)
        if norm1(a) == norminf(a) or norm1(inv) == norminf(inv):
            continue
        return a
    raise AssertionError("unreachable")


def _draw3(rng):
    for _ in attempts(5000, "a 3x3 matrix"):
        a = [[rng.randint(-4, 6) for _ in range(3)] for _ in range(3)]
        d = det3(a)
        if abs(d) not in range(2, 10):
            continue
        inv = inverse(a)
        if norm1(a) == norminf(a) or norm1(inv) == norminf(inv):
            continue
        if sum(1 for row in a for v in row if v == 0) > 1:
            continue
        return a
    raise AssertionError("unreachable")


@generator(
    "r4h_matrix_norm_condition",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4h_matrix_norm_condition(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return with_redraw(lambda: _single(rng, tier), "a norm question with enough distractors")


def _single(rng, tier):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        a = _draw2(rng)
        nrm = rng.choice([norminf, norm1])
        sym = "∞" if nrm is norminf else "1"
        other = norm1 if nrm is norminf else norminf
        inv = inverse(a)
        value = nrm(a) * nrm(inv)
        d = det2(a)
        cands = [
            (M_NOINV, nrm(a) * nrm(a)),
            (M_ADJ, nrm(a) * nrm(adj2(a))),
            (M_MIXED, nrm(a) * other(inv)),
            (M_QUOT, nrm(a) / nrm(inv)),
            (M_OTHER, other(a) * other(inv)),
            (
                M_INV_SWAP,
                nrm(a) * nrm([[Fraction(v, d) for v in row] for row in [[a[0][0], a[0][1]], [a[1][0], a[1][1]]]]),
            ),
        ]
        part = fraction_part("", value, cands, 1, min_options=3)
        am, k = shown(mat(a)), _sub(sym)
        stem = rng.choice(
            [
                f"For «A = {am}», find the condition number «κ{k}(A) = ‖A‖{k} ‖A⁻¹‖{k}» as a fraction in lowest terms.",
                f"Compute «κ{k}(A)», the condition number of «A = {am}» in the «{k}»-norm, giving the answer in lowest "
                "terms.",
                f"The matrix «A = {am}» is to be inverted. How ill-conditioned is it in the «{k}»-norm? Give "
                f"«κ{k}(A) = ‖A‖{k} ‖A⁻¹‖{k}» as a fraction in lowest terms.",
            ]
        )
        return single_from(
            part,
            stem,
            working=[
                f"det A = {d}, so A⁻¹ = (1/{d}){shown(mat(adj2(a)))}.",
                f"‖A‖{_sub(sym)} = {nrm(a)} and ‖A⁻¹‖{_sub(sym)} = {nrm(inv)}, so κ = {value}.",
            ],
        )
    a = _draw3(rng)
    inv = inverse(a)
    value = norm1(a) * norm1(inv)
    d = det3(a)
    cands = [
        (M_ADJ, norm1(a) * norm1(adj3(a))),
        (M_OTHER, norminf(a) * norminf(inv)),
        (M_MIXED, norm1(a) * norminf(inv)),
        (M_QUOT, norm1(a) / norm1(inv)),
        (M_NOINV, norm1(a) ** 2),
        (M_DET, Fraction(abs(d))),
    ]
    part = fraction_part("", value, cands, 1, min_options=3)
    am = shown(mat(a))
    stem = rng.choice(
        [
            f"For «A = {am}», find the condition number «κ₁(A) = ‖A‖₁ ‖A⁻¹‖₁» as a fraction in lowest terms.",
            f"Compute the condition number of «A = {am}» in the 1-norm, «κ₁(A)», giving the answer in lowest terms.",
            f"The matrix «A = {am}» has inverse «A⁻¹». Find «‖A‖₁ ‖A⁻¹‖₁» as a fraction in lowest terms.",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"det A = {d} and A⁻¹ = (1/{d}) adj A = {shown(inv)}.",
            f"‖A‖₁ = {norm1(a)} and ‖A⁻¹‖₁ = {norm1(inv)}, so κ₁ = {value}.",
        ],
    )


def _sub(sym: str) -> str:
    return "∞" if sym == "∞" else "₁"


def _easy(rng):
    for _ in attempts(2000, "an easy norm matrix"):
        a = [[rng.randint(-9, 9) for _ in range(3)] for _ in range(3)]
        if sum(1 for row in a for v in row if v < 0) < 3 or any(v == 0 for row in a for v in row):
            continue
        n1, ni = norm1(a), norminf(a)
        if n1 == ni:
            continue
        rows = [sum(abs(v) for v in r) for r in a]
        cols = [sum(abs(a[i][j]) for i in range(3)) for j in range(3)]
        if len(set(rows)) < 3 or len(set(cols)) < 3:
            continue
        break
    use_one = rng.random() < 0.5
    nrm, other = (norm1, norminf) if use_one else (norminf, norm1)
    sums = cols if use_one else rows
    others = rows if use_one else cols
    signed = [sum(a[i][j] for i in range(3)) for j in range(3)] if use_one else [sum(r) for r in a]
    cands = [
        (M_ROWS if use_one else M_COLS, other(a)),
        (M_SIGNED, max(abs(s) for s in signed) if max(abs(s) for s in signed) != nrm(a) else max(signed)),
        (M_ENTRY, Fraction(max(abs(v) for row in a for v in row))),
        (M_ALL, Fraction(sum(abs(v) for row in a for v in row))),
        (M_SMALL, Fraction(min(sums))),
        (M_SUM, Fraction(sum(sums))),
    ]
    part = integer_like(nrm(a), cands)
    sym = "1" if use_one else "∞"
    am, k = shown(mat(a)), _sub(sym)
    stem = rng.choice(
        [
            f"Find «‖A‖{k}» for «A = {am}».",
            f"Compute the induced «{k}»-norm of «A = {am}».",
            f"Evaluate the matrix norm «‖A‖{k}» for «A = {am}».",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"The absolute {'column' if use_one else 'row'} sums are {', '.join(str(s) for s in sums)}.",
            f"‖A‖{_sub(sym)} is the largest, {nrm(a)}.",
        ],
    )


def integer_like(value, cands):
    value = int(value)
    return integer_part(
        "",
        value,
        [(t, v) for v, t in [(v, t) for t, v in cands] if v is not None and Fraction(v).denominator == 1],
        1,
        n=4,
        min_options=3,
    )


@generator("exd_x_r4h_matrix_norm_condition", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4h_matrix_norm_condition(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam norm question")


def _exam(rng):
    a = _draw2(rng, lo=-5, hi=8)
    inv = inverse(a)
    d = det2(a)
    n_a, n_inv = norm1(a), norm1(inv)
    kappa = n_a * n_inv
    eps = rng.choice([Fraction(1, 100), Fraction(1, 200), Fraction(1, 50), Fraction(1, 1000), Fraction(3, 500)])
    bound = kappa * eps
    p_a = fraction_part(
        "Find «‖A‖₁», the largest absolute column sum.",
        n_a,
        [
            (M_ROWS, norminf(a)),
            (M_SIGNED, max(abs(a[0][j] + a[1][j]) for j in range(2))),
            (M_ENTRY, Fraction(max(abs(v) for r in a for v in r))),
            (M_ALL, Fraction(sum(abs(v) for r in a for v in r))),
        ],
        1,
        working=[f"‖A‖₁ = {n_a}."],
        value_range=(0, None),
    )
    p_b = fraction_part(
        "Find «‖A⁻¹‖₁».",
        n_inv,
        [
            (M_ADJ, norm1(adj2(a))),
            (M_OTHER, norminf(inv)),
            (M_INV_SIGN, norm1([[Fraction(v, d) for v in row] for row in [[a[1][1], a[0][1]], [a[1][0], a[0][0]]]])),
            (M_INV_SWAP, norm1([[Fraction(v, d) for v in row] for row in [[a[0][0], -a[0][1]], [-a[1][0], a[1][1]]]])),
        ],
        1,
        working=[f"det A = {d}, A⁻¹ = (1/{d}){shown(mat(adj2(a)))}, so ‖A⁻¹‖₁ = {n_inv}."],
        value_range=(0, None),
    )
    x, y = ft_symbol("part_a"), ft_symbol("part_b")
    p_c = fraction_part(
        "Use your two norms to find the condition number «κ₁(A)».",
        kappa,
        [(M_NOINV, n_a * n_a), (M_QUOT, n_a / n_inv), (M_OTHER, norminf(a) * norminf(inv)), (M_DET, Fraction(abs(d)))],
        1,
        ft=str(x * y),
        working=[f"κ₁ = {n_a}·{n_inv} = {kappa}."],
        value_range=(0, None),
    )
    z = ft_symbol("part_c")
    from ._r4h_kit import rat

    p_d = fraction_part(
        f"The system «Ax = b» is solved with «b» replaced by a perturbed «b + δb» where «‖δb‖₁/‖b‖₁ = {_ratio(eps)}». "
        "Use your condition number to bound the relative error «‖δx‖₁/‖x‖₁» of the solution.",
        bound,
        [
            ("num2t-cond-bound-divided", eps / kappa),
            ("num2t-cond-bound-forgotten", eps),
            ("num2t-cond-bound-norm-only", n_inv * eps),
            ("num2t-cond-bound-norm-only", n_a * eps),
        ],
        2,
        ft=str(z * rat(eps)),
        working=[f"‖δx‖/‖x‖ ≤ κ₁ ‖δb‖/‖b‖ = {kappa}·{_ratio(eps)} = {bound}."],
        value_range=(0, None),
    )
    am = shown(mat(a))
    stem = rng.choice(
        [
            f"Let «A = {am}».",
            f"The matrix «A = {am}» is the coefficient matrix of a linear system «Ax = b».",
            f"Consider «A = {am}» and the system «Ax = b».",
        ]
    )
    return finish_parts([p_a, p_b, p_c, p_d], stem)


def _ratio(eps: Fraction) -> str:
    return f"{eps.numerator}/{eps.denominator}"
