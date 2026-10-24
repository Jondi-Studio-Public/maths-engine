"""Gaussian elimination with partial pivoting: `r4h_gaussian_partial_pivoting` (easy, medium, hard) and the exam chain
`exd_x_r4h_gaussian_partial_pivoting`.

``uc2-gaussian-elimination-partial-pivoting`` (``numerical-and-computational-mathematics``). Every question states the
pivot rule: before eliminating below a pivot, swap into the pivot row the row (at or below it) whose entry in the
pivot column is largest in absolute value. The matrices are built so that the largest entry is unique at each stage.

* easy: a 3×3 matrix; give the matrix after the first stage (swap, then eliminate below the pivot).
* medium: a 3×3 matrix with a row interchange at both stages; give the upper triangular matrix ``U``.
* hard: a 3×3 system ``Ax = b`` with a whole-number solution; give ``x``.
* exam: a 3×3 system in four parts: the first pivot, the first multiplier (follow-through from the pivot), the
  updated entry in the second row and column (follow-through from the multiplier) and ``det A`` (follow-through from
  the pivot).

All arithmetic is exact (`Fraction`).
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from ._r4h_kit import (
    Redraw,
    answer_dict,
    finish_parts,
    fraction_part,
    ft_symbol,
    mat,
    rat,
    shown,
    single_from,
    tagged,
    with_redraw,
)

MISC = {
    "num2t-gauss-no-row-swap": "Eliminating with the entry in the pivot position, ignoring the rule to swap in the row with the largest entry.",
    "num2t-gauss-smallest-pivot": "Swapping in the row with the smallest non-zero entry instead of the largest.",
    "num2t-gauss-multiplier-sign": "Adding the multiple of the pivot row instead of subtracting it (the sign of the multiplier reversed).",
    "num2t-gauss-pivot-first-stage-only": "Pivoting at the first stage but not at the second.",
    "num2t-gauss-pivot-second-stage-only": "Pivoting at the second stage but not at the first.",
    "num2t-gauss-rows-restored": "Putting the rows back in their original order after the swap.",
    "num2t-gauss-multiplier-inverted": "Forming the multiplier as pivot divided by the entry, instead of entry divided by pivot.",
    "num2t-gauss-swap-b-forgotten": "Swapping rows of the matrix but not the matching entries of the right-hand side.",
    "num2t-gauss-rhs-not-updated": "Applying the row operations to the matrix but not to the right-hand side.",
    "num2t-gauss-back-substitution-undivided": "Back-substituting without dividing by the diagonal entry.",
    "num2t-gauss-solution-order-reversed": "Listing the solution components in reverse order.",
    "num2t-gauss-pivot-first-entry": "Taking the first entry of the first column as the pivot.",
    "num2t-gauss-pivot-absolute-value": "Giving the size of the pivot and dropping its sign.",
    "num2t-gauss-pivot-smallest": "Taking the entry of smallest size in the column as the pivot.",
    "num2t-gauss-pivot-largest-signed": "Choosing the largest signed entry rather than the largest in absolute value.",
    "num2t-gauss-multiplier-wrong-row": "Using the first-column entry of a different row to form the multiplier.",
    "num2t-gauss-multiplier-unswapped": "Forming the multiplier with the pivot from before the row swap.",
    "num2t-gauss-multiplier-sign-flipped": "Giving the negative of the multiplier.",
    "num2t-gauss-update-added": "Adding instead of subtracting the multiple of the pivot row when updating the entry.",
    "num2t-gauss-update-unswapped": "Updating the entry from the row order before the swap.",
    "num2t-gauss-update-not-reduced": "Reporting the entry before subtracting the multiple of the pivot row.",
    "num2t-gauss-det-swap-sign-ignored": "Ignoring the sign change from each row interchange when finding the determinant.",
    "num2t-gauss-det-swap-sign-extra": "Changing the sign of the determinant for each stage instead of for each row interchange.",
    "num2t-gauss-det-pivots-added": "Adding the pivots instead of multiplying them.",
    "num2t-gauss-det-original-diagonal": "Multiplying the diagonal entries of the original matrix.",
}
M_NOSWAP = "num2t-gauss-no-row-swap"
M_SMALL = "num2t-gauss-smallest-pivot"
M_SIGN = "num2t-gauss-multiplier-sign"
M_FIRST = "num2t-gauss-pivot-first-stage-only"
M_SECOND = "num2t-gauss-pivot-second-stage-only"
M_RESTORED = "num2t-gauss-rows-restored"
M_INVERTED = "num2t-gauss-multiplier-inverted"
M_BSWAP = "num2t-gauss-swap-b-forgotten"
M_BUPD = "num2t-gauss-rhs-not-updated"
M_UNDIV = "num2t-gauss-back-substitution-undivided"
M_REV = "num2t-gauss-solution-order-reversed"

RULES = (
    "Use partial pivoting: at each stage, first swap into the pivot position the row, at or below it, whose entry in "
    "the pivot column is largest in absolute value, then subtract the multiple (entry ÷ pivot) of the pivot row from "
    "each row below it.",
    "Follow the partial pivoting rule: before clearing a column, interchange rows so that the entry of largest absolute "
    "value (among the pivot position and the rows below it) becomes the pivot, then subtract (entry ÷ pivot) times "
    "the pivot row from every row beneath.",
    "Eliminate with partial pivoting. At each stage the pivot is the entry of greatest absolute value in the column, "
    "taken from the pivot position or below and moved up by a row swap; each row beneath loses its multiplier "
    "(entry ÷ pivot) times the pivot row.",
)


def gauss(a, b=None, rules=("max", "max"), msign=1, b_follow=True, b_update=True, stages=2, inverted=False):
    """Eliminate in exact arithmetic. Returns (matrix, swaps, pivots, multipliers) or None on a zero pivot.
    `rules[k]` is 'max', 'min' or 'none' for stage k."""
    n = len(a)
    m = [[Fraction(v) for v in row] + ([Fraction(b[i])] if b is not None else []) for i, row in enumerate(a)]
    swaps, pivots, mults = 0, [], []
    for k in range(stages):
        col = [(abs(m[i][k]), i) for i in range(k, n)]
        if rules[k] == "max":
            r = max(col, key=lambda t: (t[0], -t[1]))[1]
        elif rules[k] == "min":
            nz = [t for t in col if t[0] != 0]
            if not nz:
                return None
            r = min(nz, key=lambda t: (t[0], t[1]))[1]
        else:
            r = k
        if m[r][k] == 0:
            return None
        if r != k:
            swaps += 1
            for j in range(len(m[0])):
                if j == n and not b_follow:
                    continue
                m[k][j], m[r][j] = m[r][j], m[k][j]
        pivots.append(m[k][k])
        for i in range(k + 1, n):
            mult = m[i][k] / m[k][k] if not inverted else m[k][k] / m[i][k] if m[i][k] else None
            if mult is None:
                return None
            mults.append(mult)
            mult *= msign
            cols = len(m[0]) if b_update else n
            for j in range(cols):
                m[i][j] -= mult * m[k][j]
    return m, swaps, pivots, mults


def back_sub(u, c, divide=True):
    n = len(u)
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        s = c[i] - sum(u[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / u[i][i] if divide else s
    return x


def det3(a) -> int:
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def _unique_max(m, k):
    col = sorted((abs(m[i][k]) for i in range(k, len(m))), reverse=True)
    return col[0] != col[1]


def draw_matrix(rng, *, swap1=True, swap2=False, lim=9):
    for _ in attempts(5000, "a pivoting matrix"):
        a = [[rng.choice([v for v in range(-lim, lim + 1) if v]) for _ in range(3)] for _ in range(3)]
        if det3(a) == 0:
            continue
        col0 = sorted(abs(a[i][0]) for i in range(3))
        if col0[-1] == col0[-2] or col0[0] == col0[1]:
            continue
        r = max(range(3), key=lambda i: abs(a[i][0]))
        if swap1 != (r != 0):
            continue
        res = gauss(a, stages=1)
        if res is None:
            continue
        m1 = res[0]
        if not _unique_max(m1, 1) or m1[1][1] == 0 and m1[2][1] == 0:
            continue
        r2 = max(range(1, 3), key=lambda i: abs(m1[i][1]))
        if swap2 != (r2 != 1):
            continue
        full = gauss(a)
        if full is None or full[0][2][2] == 0:
            continue
        if any(v.denominator > 60 for row in full[0] for v in row):
            continue
        if any(abs(v) > 80 for row in full[0] for v in row):
            continue
        return a
    raise AssertionError("unreachable")


@generator(
    "r4h_gaussian_partial_pivoting",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4h_gaussian_partial_pivoting(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return with_redraw(lambda: _single(rng, tier), "a pivoting question with enough distractors")


def _single(rng, tier):
    if tier == "easy":
        a = draw_matrix(rng, swap1=True)
        m = gauss(a, stages=1)[0]
        ans, rows, text = answer_dict("M = ", m)
        cands = [
            (_g(a, rules=("none",), stages=1), M_NOSWAP),
            (_g(a, rules=("min",), stages=1), M_SMALL),
            (_g(a, rules=("max",), msign=-1, stages=1), M_SIGN),
            (_g(a, rules=("max",), inverted=True, stages=1), M_INVERTED),
            (_restored(a), M_RESTORED),
        ]
        dist = tagged("M = ", rows, text, cands)
        if len(dist) < 3:
            raise Redraw
        stem = (
            f"{rng.choice(RULES)} Apply the first stage of Gaussian elimination to «A = {shown(mat(a))}» and give the matrix «M» "
            "that results, with the rows in the order they then occupy."
        )
        return dict(
            text=stem.replace("«", "").replace("»", ""),
            answer_kind="matrix",
            distractors=dist,
            **ans,
            working=[
                "Swap the row with the largest first-column entry to the top, then clear the first column below it.",
                f"M = {shown(m)}",
            ],
        ) | _latex(stem)
    if tier == "medium":
        a = draw_matrix(rng, swap1=True, swap2=True)
        u = gauss(a)[0]
        ans, rows, text = answer_dict("U = ", u)
        cands = [
            (_g(a, rules=("none", "none")), M_NOSWAP),
            (_g(a, rules=("max", "none")), M_FIRST),
            (_g(a, rules=("none", "max")), M_SECOND),
            (_g(a, rules=("min", "min")), M_SMALL),
            (_g(a, msign=-1), M_SIGN),
        ]
        dist = tagged("U = ", rows, text, cands)
        if len(dist) < 3:
            raise Redraw
        stem = (
            f"{rng.choice(RULES)} Reduce «A = {shown(mat(a))}» to upper triangular form «U» by Gaussian elimination and give "
            "«U»."
        )
        return dict(
            text=stem.replace("«", "").replace("»", ""),
            answer_kind="matrix",
            distractors=dist,
            **ans,
            working=["Pivot and eliminate on column 1, then pivot and eliminate on column 2.", f"U = {shown(u)}"],
        ) | _latex(stem)
    a = draw_matrix(rng, swap1=True, swap2=rng.random() < 0.5)
    for _ in attempts(500, "a solution"):
        x = [rng.randint(-4, 4) for _ in range(3)]
        if sum(1 for v in x if v) >= 2:
            break
    b = [sum(a[i][j] * x[j] for j in range(3)) for i in range(3)]
    full, swaps, piv, _m = gauss(a, b)
    sol = back_sub([r[:3] for r in full], [r[3] for r in full])
    check(sol == [Fraction(v) for v in x], "elimination disagrees with the built solution")
    ans, rows, text = answer_dict("x = ", [[v] for v in sol])

    def col(vec):
        return None if vec is None else _plausible([[v] for v in vec])

    def solve(**kw):
        res = gauss(a, b, **kw)
        if res is None:
            return None
        mm = res[0]
        if kw.get("divide", True) and any(mm[i][i] == 0 for i in range(3)):
            return None  # this slip leaves a zero pivot, so it gives no solution column to offer
        return back_sub([r[:3] for r in mm], [r[3] for r in mm], divide=kw.get("divide", True))

    cands = [
        (col(solve(b_follow=False)), M_BSWAP),
        (col(solve(b_update=False)), M_BUPD),
        (col(back_sub([r[:3] for r in full], [r[3] for r in full], divide=False)), M_UNDIV),
        (col(list(reversed(sol))), M_REV),
        (col(solve(msign=-1)), M_SIGN),
    ]
    dist = tagged("x = ", rows, text, cands)
    if len(dist) < 3:
        raise Redraw
    stem = (
        f"{rng.choice(RULES)} Solve «Ax = b» for «A = {shown(mat(a))}» and «b = {shown([[v] for v in b])}» by Gaussian "
        "elimination followed by back substitution, and give the solution column «x»."
    )
    return dict(
        text=stem.replace("«", "").replace("»", ""),
        answer_kind="matrix",
        distractors=dist,
        **ans,
        working=[f"The reduced system is {shown(full)}.", f"Back substitution gives x = {shown([[v] for v in sol])}."],
    ) | _latex(stem)


def _latex(stem):
    from ._ucv1_texsay import say_marked

    text, latex = say_marked(stem)
    return {"text": text, "latex": latex}


def _plausible(m):
    """None when an entry is too unwieldy for a reply a student could plausibly reach."""
    if m is None:
        return None
    if any(v.denominator > 40 or abs(v) > 250 for row in m for v in row):
        return None
    return m


def _g(a, **kw):
    res = gauss(a, **kw)
    return None if res is None else _plausible(res[0])


def _restored(a):
    """Stage 1 with the swap undone afterwards: the eliminated rows put back in the original positions."""
    res = gauss(a, stages=1)
    r = max(range(3), key=lambda i: abs(a[i][0]))
    m = [row[:] for row in res[0]]
    m[0], m[r] = m[r], m[0]
    return m


# ---------------------------------------------------------------------------------------------------------------
# exam chain


@generator("exd_x_r4h_gaussian_partial_pivoting", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4h_gaussian_partial_pivoting(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam pivoting question")


def _exam(rng):
    a = draw_matrix(rng, swap1=True, swap2=rng.random() < 0.5)
    full, swaps, piv, mults = gauss(a)
    stage1, _s, _p, _m = gauss(a, stages=1)
    r = max(range(3), key=lambda i: abs(a[i][0]))
    swapped = [row[:] for row in a]
    swapped[0], swapped[r] = swapped[r], swapped[0]
    p1 = piv[0]
    m21 = mults[0]
    entry = stage1[1][1]
    det = Fraction(det3(a))
    check(det == (-1) ** swaps * piv[0] * piv[1] * full[2][2], "determinant by pivots disagrees with cofactors")
    col0 = [a[i][0] for i in range(3)]
    big_signed = max(col0)
    p_a = fraction_part(
        "Find the first pivot: the entry that ends up in the first row and column after the first interchange.",
        p1,
        [
            ("num2t-gauss-pivot-first-entry", a[0][0]),
            ("num2t-gauss-pivot-absolute-value", abs(p1)),
            ("num2t-gauss-pivot-smallest", min(col0, key=abs)),
            ("num2t-gauss-pivot-largest-signed", big_signed),
        ],
        1,
        working=[f"The first column is {col0}; the largest in absolute value is {p1}."],
    )
    a21 = swapped[1][0]
    other = swapped[2][0]
    pa = ft_symbol("part_a")
    p_b = fraction_part(
        "After the interchange, the second row begins with the entry below the pivot. Find the multiplier "
        "m₂₁, the entry below the pivot divided by the pivot, used to clear it, using your pivot.",
        m21,
        [
            ("num2t-gauss-multiplier-inverted", p1 / a21),
            ("num2t-gauss-multiplier-wrong-row", Fraction(other) / p1),
            ("num2t-gauss-multiplier-unswapped", Fraction(a21) / a[0][0] if a[0][0] else None),
            ("num2t-gauss-multiplier-sign-flipped", -m21),
        ],
        1,
        ft=str(rat(a21) / pa),
        working=[f"m₂₁ = {a21}/{p1} = {m21}."],
    )
    pb = ft_symbol("part_b")
    p_c = fraction_part(
        "Use your multiplier to find the updated entry in the second row and second column after the first stage.",
        entry,
        [
            ("num2t-gauss-update-added", Fraction(swapped[1][1]) + m21 * swapped[0][1]),
            ("num2t-gauss-update-unswapped", Fraction(a[1][1]) - m21 * a[0][1]),
            ("num2t-gauss-update-not-reduced", Fraction(swapped[1][1])),
            ("num2t-gauss-multiplier-sign-flipped", -entry),
        ],
        2,
        ft=str(rat(swapped[1][1]) - pb * rat(swapped[0][1])),
        working=[f"{swapped[1][1]} − ({m21})({swapped[0][1]}) = {entry}."],
    )
    k = rat(det / p1)
    p_d = fraction_part(
        "Complete the elimination with partial pivoting and use the pivots to find the determinant of «A», remembering the effect of "
        "any row interchanges.",
        det,
        [
            ("num2t-gauss-det-swap-sign-ignored", piv[0] * piv[1] * full[2][2]),
            ("num2t-gauss-det-swap-sign-extra", -piv[0] * piv[1] * full[2][2] if swaps % 2 == 0 else None),
            ("num2t-gauss-det-pivots-added", piv[0] + piv[1] + full[2][2]),
            ("num2t-gauss-det-original-diagonal", Fraction(a[0][0] * a[1][1] * a[2][2])),
        ],
        2,
        ft=str(k * pa),
        working=[f"The pivots are {p1}, {piv[1]}, {full[2][2]} with {swaps} interchange(s): det A = {det}."],
    )
    stem = f"{rng.choice(RULES)} Gaussian elimination with partial pivoting is applied to «A = {shown(mat(a))}»."
    return finish_parts([p_a, p_b, p_c, p_d], stem)
