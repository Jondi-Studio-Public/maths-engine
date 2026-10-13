"""Matrix inverses: 2×2 and 3×3, each with easy/medium/hard practice and an exam chain. Answers are graded as `matrix`.

Topics:

* ``mat_inv_2x2`` (``matrix-inverse-2x2``): easy, determinant ±1 so the inverse has whole entries; medium,
  determinant other than ±1 so the entries are fractions; hard, solve ``AX = B`` or ``XA = B`` with the inverse, or
  find the inverse of a matrix with a parameter at a stated value.
* ``mat_inv_3x3`` (``matrix-inverse-3x3``): easy, determinant ±1; medium, a small determinant other than ±1 (cofactors,
  transpose, divide); hard, solve ``Ax = b`` for a column ``x`` with the inverse, or invert a matrix with a parameter
  at a stated value.
* ``exd_x_mat_inv_2x2`` and ``exd_x_mat_inv_3x3``: four-part chains on a matrix with a parameter ``k``: (a) the
  determinant as an expression in ``k`` (expression), (b) the value of ``k`` that makes the matrix singular (number),
  (c) the inverse at a stated ``k`` (matrix), (d) the solution of ``AX = B`` at that ``k`` (matrix).

All arithmetic is exact (`Fraction`). Each inverse is built from cofactors, then checked three ways: ``A·A⁻¹ = I`` in
`Fraction`, SymPy's ``Matrix.inv()``, and (for solutions) back-substitution. Every distractor is a recognised slip and
is read by the real matrix grader, which must refuse it.
"""

from __future__ import annotations

import random
from fractions import Fraction as Fr
from types import SimpleNamespace

import sympy

from ..answers import Distractor, exact_value
from ..core import VerificationError, check, generator, join_terms, paren_negative
from ..grading import GRADERS
from ..grading.matrix import matrix_text

_MAX_DRAWS = 600

M_NO_SWAP = "inverse-diagonal-not-swapped"
M_NO_NEGATE = "inverse-offdiag-not-negated"
M_NO_DET = "inverse-not-divided-by-det"
M_RECIPROCAL = "inverse-entries-reciprocated"
M_NO_TRANSPOSE = "inverse-cofactors-not-transposed"
M_NO_SIGNS = "inverse-cofactor-signs-missed"
M_GLOBAL_SIGN = "inverse-global-sign"
M_WRONG_SIDE = "matrix-equation-wrong-side"
M_DET_SIGN = "det-formula-sign-slip"
M_SIGN = "sign-error-moving-term"
M_NO_DIVIDE = "forgot-to-divide-by-coefficient"


# -- exact matrix helpers (lists of lists of Fraction) ---------------------------------------------------------------


def _mat(rows) -> list[list[Fr]]:
    return [[Fr(x) for x in row] for row in rows]


def _minor(M, i, j):
    return [[M[r][c] for c in range(len(M)) if c != j] for r in range(len(M)) if r != i]


def _det(M) -> Fr:
    n = len(M)
    if n == 1:
        return M[0][0]
    return sum(((-1) ** j) * M[0][j] * _det(_minor(M, 0, j)) for j in range(n))


def _cofactors(M):
    n = len(M)
    return [[((-1) ** (i + j)) * _det(_minor(M, i, j)) for j in range(n)] for i in range(n)]


def _minors(M):
    n = len(M)
    return [[_det(_minor(M, i, j)) for j in range(n)] for i in range(n)]


def _t(M):
    return [list(r) for r in zip(*M)]


def _scale(M, s):
    return [[x * s for x in row] for row in M]


def _mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def _adj(M):
    if len(M) == 2:
        return [[M[1][1], -M[0][1]], [-M[1][0], M[0][0]]]
    return _t(_cofactors(M))


def _inverse(M):
    d = _det(M)
    check(d != 0, "singular matrix")
    return _scale(_adj(M), Fr(1) / d)


def _eye(n):
    return [[Fr(int(i == j)) for j in range(n)] for i in range(n)]


def _entry(x: Fr):
    return exact_value(x)


def _rows(M):
    return [[_entry(x) for x in row] for row in M]


def _shown(M) -> str:
    """The matrix as the question prints it: ``[[2, 1], [5, 3]]``."""
    return matrix_text(_rows(M))


def _check_inverse(M, Minv) -> None:
    n = len(M)
    check(_mul(M, Minv) == _eye(n) and _mul(Minv, M) == _eye(n), "A·A⁻¹ is not the identity")
    ref = sympy.Matrix(M).inv()
    check(
        all(
            sympy.Rational(Minv[i][j].numerator, Minv[i][j].denominator) == ref[i, j]
            for i in range(n)
            for j in range(n)
        ),
        "SymPy disagrees",
    )


def _accepts(answer_rows, answer_text: str, reply: str) -> bool:
    question = SimpleNamespace(answer=answer_rows, answer_text=answer_text, answer_form=None, text="", parts=None)
    return bool(GRADERS["matrix"](question, reply).correct)


def _tagged(label: str, answer_rows, answer_text: str, cands, *, limit: int = 3, rng=None):
    """Distractors from ``(matrix, misconception)`` candidates: not the answer, not repeated, none the grader accepts."""
    out, seen = [], {answer_text}
    cands = list(cands)
    if rng is not None:
        rng.shuffle(cands)  # which three slips are offered varies, so every catalogued slip turns up across seeds
    for M, mid in cands:
        rows = _rows(M)
        text = f"{label}{matrix_text(rows)}"
        if text in seen or _accepts(answer_rows, answer_text, text):
            continue
        seen.add(text)
        out.append(Distractor(text, mid, value=rows))
        if len(out) == limit:
            break
    return out if len(out) >= 2 else None


_FORMAT_NOTE = (
    "Give the entries as whole numbers or fractions; a scalar times a matrix, such as 1/7 [[3, -1], [2, 4]], is fine."
)
_WHOLE_NOTE = "Give the entries as whole numbers."


def _answer(label: str, M) -> dict:
    rows = _rows(M)
    text = f"{label}{matrix_text(rows)}"
    # answer_latex stays in the `plain:` form: the LaTeX matrix environment is a 7-letter run to the answer-latex sweep.
    return dict(answer=rows, answer_text=text, answer_latex="plain:" + text)


def _nz(rng, lo, hi, *avoid):
    return rng.choice([v for v in range(lo, hi + 1) if v != 0 and v not in avoid])


# -- 2×2 draws -------------------------------------------------------------------------------------------------------


def _draw_2x2(rng, dets, lim=6, *, need_fraction: bool = False):
    """A 2×2 integer matrix with b, c nonzero, a != d and a determinant in `dets`."""
    for _ in range(_MAX_DRAWS):
        a, b, c = _nz(rng, -lim, lim), _nz(rng, -lim, lim), _nz(rng, -lim, lim)
        d = rng.randint(-lim, lim)
        M = _mat([[a, b], [c, d]])
        det = _det(M)
        if det not in dets or a == d or d == 0:
            continue
        if need_fraction and all(x % det == 0 for x in (a, b, c, d)):
            continue
        return M
    raise VerificationError("no 2×2 matrix with the wanted determinant")


def _inverse_slips_2x2(M):
    a, b, c, d = M[0][0], M[0][1], M[1][0], M[1][1]
    det = _det(M)
    inv = _inverse(M)
    cands = [
        (_scale([[a, -b], [-c, d]], Fr(1) / det), M_NO_SWAP),
        (_scale([[d, b], [c, a]], Fr(1) / det), M_NO_NEGATE),
        (_adj(M), M_NO_DET),
        (_scale(inv, -1), M_GLOBAL_SIGN),
    ]
    if all(x != 0 for x in (a, b, c, d)):
        cands.append(([[Fr(1) / x for x in row] for row in M], M_RECIPROCAL))
    if a * d + b * c != 0:
        cands.append((_scale(_adj(M), Fr(1) / (a * d + b * c)), M_DET_SIGN))
    return cands


def _inv_2x2_body(rng, dets, note, *, need_fraction):
    M = _draw_2x2(rng, dets, need_fraction=need_fraction)
    inv = _inverse(M)
    _check_inverse(M, inv)
    ans = _answer("A⁻¹ = ", inv)
    dis = _tagged("A⁻¹ = ", ans["answer"], ans["answer_text"], _inverse_slips_2x2(M), rng=rng)
    if dis is None:
        return None
    det = _det(M)
    return dict(
        text=f"Find the inverse of A = {_shown(M)}. {note}",
        working=[
            f"det(A) = {M[0][0]}×{paren_negative(M[1][1])} - {paren_negative(M[0][1])}×{paren_negative(M[1][0])} = {det}.",
            "Swap the diagonal, negate the others, divide by det(A).",
        ],
        distractors=dis,
        **ans,
    )


def _draw_solve_2x2(rng):
    """AX = B or XA = B with a 2×2 A of small determinant."""
    for _ in range(_MAX_DRAWS):
        A = _draw_2x2(rng, {Fr(v) for v in (-5, -4, -3, -2, 1, 2, 3, 4, 5)}, lim=5)
        left = rng.random() < 0.5  # AX = B
        if rng.random() < 0.5:
            X = _mat([[rng.randint(-4, 5) for _ in range(2)] for _ in range(2)])
            B = _mul(A, X) if left else _mul(X, A)
        else:
            B = _mat([[rng.randint(-6, 6) for _ in range(2)] for _ in range(2)])
            X = _mul(_inverse(A), B) if left else _mul(B, _inverse(A))
        if len({x for row in X for x in row}) < 3 or X == B:
            continue
        inv = _inverse(A)
        wrong = _mul(B, inv) if left else _mul(inv, B)
        a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
        bad_inv = _scale([[a, -b], [-c, d]], Fr(1) / _det(A))
        bad_prod = _mul(bad_inv, B) if left else _mul(B, bad_inv)
        cands = [
            (wrong, M_WRONG_SIDE),
            (_mul(_adj(A), B) if left else _mul(B, _adj(A)), M_NO_DET),
            (bad_prod, M_NO_SWAP),
            (_scale(X, -1), M_GLOBAL_SIGN),
        ]
        return A, B, X, left, cands
    raise VerificationError("no solvable matrix equation")


def _inv_2x2_hard(rng):
    for _ in range(_MAX_DRAWS):
        if rng.random() < 0.6:
            A, B, X, left, cands = _draw_solve_2x2(rng)
            check(_mul(A, X) == B if left else _mul(X, A) == B, "X does not satisfy the equation")
            ans = _answer("X = ", X)
            dis = _tagged("X = ", ans["answer"], ans["answer_text"], cands, rng=rng)
            if dis is None:
                continue
            eq = "AX = B" if left else "XA = B"
            return dict(
                text=f"The matrices A = {_shown(A)} and B = {_shown(B)} satisfy {eq}. Find the matrix X. {_FORMAT_NOTE}",
                working=[f"{'X = A⁻¹B' if left else 'X = BA⁻¹'}, with A⁻¹ = {_shown(_inverse(A))}."],
                distractors=dis,
                **ans,
            )
        out = _inv_2x2_param(rng)
        if out is not None:
            return out
    raise VerificationError("no usable hard 2×2 draw")


def _param_2x2(rng):
    """A with one entry ``k`` or ``mk + n`` evaluated at k0: returns (symbolic text, matrix at k0, k0)."""
    for _ in range(_MAX_DRAWS):
        k0 = _nz(rng, -3, 4)
        m1, n1, m2, n2 = _nz(rng, -3, 3), rng.randint(-4, 4), _nz(rng, -3, 3), rng.randint(-4, 4)
        b, c = _nz(rng, -5, 5), _nz(rng, -5, 5)
        a_val, d_val = m1 * k0 + n1, m2 * k0 + n2
        M = _mat([[a_val, b], [c, d_val]])
        det = _det(M)
        if det == 0 or abs(det) > 12 or a_val == d_val or a_val == 0 or d_val == 0:
            continue
        a_txt = join_terms([(m1, "k"), (n1, "")])
        d_txt = join_terms([(m2, "k"), (n2, "")])
        return f"[[{a_txt}, {b}], [{c}, {d_txt}]]", M, k0
    raise VerificationError("no parametric 2×2")


def _inv_2x2_param(rng):
    txt, M, k0 = _param_2x2(rng)
    inv = _inverse(M)
    _check_inverse(M, inv)
    ans = _answer("A⁻¹ = ", inv)
    dis = _tagged("A⁻¹ = ", ans["answer"], ans["answer_text"], _inverse_slips_2x2(M), rng=rng)
    if dis is None:
        return None
    return dict(
        text=f"The matrix A = {txt} has a parameter k. Find A⁻¹ when k = {k0}. {_FORMAT_NOTE}",
        working=[f"At k = {k0}, A = {_shown(M)} and det(A) = {_det(M)}."],
        distractors=dis,
        **ans,
    )


def _retry(fn, rng):
    for _ in range(_MAX_DRAWS):
        out = fn(rng)
        if out is not None:
            return out
    raise VerificationError("no usable matrix draw")


@generator(
    "mat_inv_2x2",
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def mat_inv_2x2(rng: random.Random, tier: str = "easy") -> dict:
    if tier == "easy":
        return _retry(lambda r: _inv_2x2_body(r, {Fr(1), Fr(-1)}, _WHOLE_NOTE, need_fraction=False), rng)
    if tier == "medium":
        dets = {Fr(v) for v in (-7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7)}
        return _retry(lambda r: _inv_2x2_body(r, dets, _FORMAT_NOTE, need_fraction=True), rng)
    return _inv_2x2_hard(rng)


# -- 3×3 draws -------------------------------------------------------------------------------------------------------


def _draw_3x3(rng, dets, lim=3, *, zeros_ok: bool = True):
    for _ in range(_MAX_DRAWS):
        M = _mat([[rng.randint(-lim, lim) for _ in range(3)] for _ in range(3)])
        if _det(M) not in dets:
            continue
        flat = [x for row in M for x in row]
        if flat.count(0) > 3 or len(set(flat)) < 4:
            continue
        if _t(M) == M:
            continue
        return M
    raise VerificationError("no 3×3 matrix with the wanted determinant")


def _inverse_slips_3x3(M):
    det = _det(M)
    cof = _cofactors(M)
    inv = _inverse(M)
    return [
        (_scale(cof, Fr(1) / det), M_NO_TRANSPOSE),
        (_scale(_t(_minors(M)), Fr(1) / det), M_NO_SIGNS),
        (_adj(M), M_NO_DET),
        (_scale(inv, -1), M_GLOBAL_SIGN),
    ]


def _inv_3x3_body(rng, dets, note):
    M = _draw_3x3(rng, dets)
    inv = _inverse(M)
    _check_inverse(M, inv)
    ans = _answer("A⁻¹ = ", inv)
    dis = _tagged("A⁻¹ = ", ans["answer"], ans["answer_text"], _inverse_slips_3x3(M), rng=rng)
    if dis is None:
        return None
    det = _det(M)
    return dict(
        text=f"Find the inverse of A = {_shown(M)}. {note}",
        working=[f"det(A) = {det}.", "Cofactor matrix, transposed (the adjugate), divided by det(A)."],
        distractors=dis,
        **ans,
    )


def _inv_3x3_solve(rng):
    for _ in range(_MAX_DRAWS):
        A = _draw_3x3(rng, {Fr(v) for v in (-4, -3, -2, -1, 1, 2, 3, 4)})
        x = _mat([[rng.randint(-3, 4)] for _ in range(3)])
        if len({r[0] for r in x}) < 2 or sum(1 for r in x if r[0] == 0) > 1:
            continue
        b = _mul(A, x)
        inv = _inverse(A)
        _check_inverse(A, inv)
        cof = _cofactors(A)
        det = _det(A)
        cands = [
            (_mul(_scale(cof, Fr(1) / det), b), M_NO_TRANSPOSE),
            (_mul(_adj(A), b), M_NO_DET),
            (_mul(_scale(_t(_minors(A)), Fr(1) / det), b), M_NO_SIGNS),
            (_scale(x, -1), M_GLOBAL_SIGN),
        ]
        ans = _answer("x = ", x)
        dis = _tagged("x = ", ans["answer"], ans["answer_text"], cands, rng=rng)
        if dis is None:
            continue
        return dict(
            text=f"The system Ax = b has A = {_shown(A)} and b = {_shown(b)}. Use A⁻¹ to find the column x. {_FORMAT_NOTE}",
            working=[f"det(A) = {det}; x = A⁻¹b."],
            distractors=dis,
            **ans,
        )
    raise VerificationError("no solvable 3×3 system")


def _param_3x3(rng):
    """A 3×3 with one entry k: returns (text, matrix at k0, k0, position, cofactor, rest-of-det)."""
    for _ in range(_MAX_DRAWS):
        base = [[rng.randint(-3, 3) for _ in range(3)] for _ in range(3)]
        i, j = rng.randrange(3), rng.randrange(3)
        k0 = _nz(rng, -3, 4)
        base[i][j] = k0
        M = _mat(base)
        det = _det(M)
        if det == 0 or abs(det) > 12:
            continue
        flat = [x for row in base for x in row]
        if flat.count(0) > 3:
            continue
        shown = [[("k" if (r, c) == (i, j) else str(base[r][c])) for c in range(3)] for r in range(3)]
        txt = "[" + ", ".join("[" + ", ".join(row) + "]" for row in shown) + "]"
        return txt, M, k0, (i, j)
    raise VerificationError("no parametric 3×3")


def _inv_3x3_param(rng):
    txt, M, k0, _pos = _param_3x3(rng)
    inv = _inverse(M)
    _check_inverse(M, inv)
    ans = _answer("A⁻¹ = ", inv)
    dis = _tagged("A⁻¹ = ", ans["answer"], ans["answer_text"], _inverse_slips_3x3(M), rng=rng)
    if dis is None:
        return None
    return dict(
        text=f"The matrix A = {txt} has a parameter k. Find A⁻¹ when k = {k0}. {_FORMAT_NOTE}",
        working=[f"At k = {k0}, det(A) = {_det(M)}."],
        distractors=dis,
        **ans,
    )


@generator(
    "mat_inv_3x3",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def mat_inv_3x3(rng: random.Random, tier: str = "easy") -> dict:
    if tier == "easy":
        return _retry(lambda r: _inv_3x3_body(r, {Fr(1), Fr(-1)}, _WHOLE_NOTE), rng)
    if tier == "medium":
        dets = {Fr(v) for v in (-4, -3, -2, 2, 3, 4)}
        return _retry(lambda r: _inv_3x3_body(r, dets, _FORMAT_NOTE), rng)
    return _retry(lambda r: _inv_3x3_solve(r) if r.random() < 0.5 else _inv_3x3_param(r), rng)


# -- exam chains -----------------------------------------------------------------------------------------------------


def _num_dist(answer, cands):
    out, seen = [], {str(answer)}
    for value, mid in cands:
        if str(value) in seen:
            continue
        seen.add(str(value))
        out.append(Distractor(str(value), mid, value=value))
        if len(out) == 3:
            break
    return out if len(out) >= 2 else None


def _expr_dist(answer: str, cands, reader):
    out, seen = [], {answer}
    for text, mid in cands:
        if text in seen or reader(text):
            continue
        seen.add(text)
        out.append(Distractor(text, mid, value=text))
        if len(out) == 3:
            break
    return out if len(out) >= 2 else None


def _accepts_expr(answer: str):
    question = SimpleNamespace(
        answer=answer, answer_text=answer, answer_form=None, text="Find det(A) in terms of k.", parts=None
    )
    return lambda reply: bool(GRADERS["expression"](question, reply).correct)


def _det_text(coef: int, const: int) -> str:
    return join_terms([(coef, "k"), (const, "")])


def _exam_matrices(rng, n):
    """A matrix with one entry ``k``: (text, builder k -> matrix, det = C k + D with integer singular k0, C, D)."""
    for _ in range(_MAX_DRAWS):
        base = [[rng.randint(-3, 4) for _ in range(n)] for _ in range(n)]
        i, j = rng.randrange(n), rng.randrange(n)
        base[i][j] = 0

        def build(k, base=base, i=i, j=j):
            rows = [list(r) for r in base]
            rows[i][j] = k
            return _mat(rows)

        D = _det(build(Fr(0)))
        C = _det(build(Fr(1))) - D
        if C == 0 or D % C != 0:
            continue
        k0 = -D / C
        if k0 == 0 or abs(k0) > 9:
            continue
        flat = [x for r in base for x in r]
        if flat.count(0) > (3 if n == 3 else 2):
            continue
        for _ in range(30):
            k1 = Fr(rng.randint(-4, 5))
            det1 = C * k1 + D
            if k1 not in (0, k0) and 2 <= abs(det1) <= 8:
                shown = [[("k" if (r, c) == (i, j) else str(base[r][c])) for c in range(n)] for r in range(n)]
                txt = "[" + ", ".join("[" + ", ".join(row) + "]" for row in shown) + "]"
                return txt, build, int(C), int(D), int(k0), int(k1)
    raise VerificationError("no parametric matrix for the exam chain")


def _exam_chain(rng, n):
    for _ in range(_MAX_DRAWS):
        txt, build, C, D, k0, k1 = _exam_matrices(rng, n)
        # (a) det as an expression in k, checked against the matrix itself at three values of k.
        for kv in (Fr(-2), Fr(3), Fr(5, 2)):
            check(_det(build(kv)) == C * kv + D, "det expression disagrees with the matrix")
        det_text = _det_text(C, D)
        sym_k = sympy.Symbol("k")
        sym_m = sympy.Matrix(
            n,
            n,
            lambda r, c: (
                sym_k
                if build(Fr(0))[r][c] != build(Fr(1))[r][c]
                else sympy.Rational(build(Fr(0))[r][c].numerator, build(Fr(0))[r][c].denominator)
            ),
        )
        check(sympy.expand(sym_m.det() - (C * sym_k + D)) == 0, "SymPy determinant disagrees")
        dis_a = _expr_dist(
            det_text,
            [
                (_det_text(C, -D), M_DET_SIGN),
                (_det_text(-C, D), M_DET_SIGN),
                (_det_text(C, 0), "dropped-constant-term"),
                (_det_text(-C, -D), M_DET_SIGN),
            ],
            _accepts_expr(det_text),
        )
        # (b) singular k.
        check(_det(build(Fr(k0))) == 0, "A is not singular at k0")
        dis_b = _num_dist(k0, [(-k0, M_SIGN), (-D, M_NO_DIVIDE), (D, M_NO_DIVIDE), (k0 + 1, M_SIGN), (k0 - 1, M_SIGN)])
        # (c) inverse at k1.
        A = build(Fr(k1))
        inv = _inverse(A)
        _check_inverse(A, inv)
        ans_c = _answer("A⁻¹ = ", inv)
        slips = _inverse_slips_2x2(A) if n == 2 else _inverse_slips_3x3(A)
        dis_c = _tagged("A⁻¹ = ", ans_c["answer"], ans_c["answer_text"], slips, rng=rng)
        # (d) solve AX = B at k1.
        if n == 2:
            if rng.random() < 0.6:
                X = _mat([[rng.randint(-3, 4) for _ in range(2)] for _ in range(2)])
                B = _mul(A, X)
            else:
                B = _mat([[rng.randint(-5, 6) for _ in range(2)] for _ in range(2)])
                X = _mul(inv, B)
            wrong_side = _mul(B, inv)
            bad = _mul(_scale([[A[0][0], -A[0][1]], [-A[1][0], A[1][1]]], Fr(1) / _det(A)), B)
            cands_d = [
                (wrong_side, M_WRONG_SIDE),
                (_mul(_adj(A), B), M_NO_DET),
                (bad, M_NO_SWAP),
                (_scale(X, -1), M_GLOBAL_SIGN),
            ]
            B_text, d_what = _shown(B), "the matrix X"
        else:
            X = _mat([[rng.randint(-3, 4)] for _ in range(3)])
            B = _mul(A, X)
            det1 = _det(A)
            cands_d = [
                (_mul(_scale(_cofactors(A), Fr(1) / det1), B), M_NO_TRANSPOSE),
                (_mul(_adj(A), B), M_NO_DET),
                (_mul(_scale(_t(_minors(A)), Fr(1) / det1), B), M_NO_SIGNS),
                (_scale(X, -1), M_GLOBAL_SIGN),
            ]
            B_text, d_what = _shown(B), "the column x"
        check(_mul(A, X) == B, "X does not satisfy AX = B")
        label_d = "X = " if n == 2 else "x = "
        ans_d = _answer(label_d, X)
        dis_d = _tagged(label_d, ans_d["answer"], ans_d["answer_text"], cands_d, rng=rng)
        if None in (dis_a, dis_b, dis_c, dis_d):
            continue
        eq = "AX = B" if n == 2 else "Ax = b"
        return dict(
            text=f"The matrix A = {txt} has a parameter k.",
            parts=[
                dict(
                    prompt="Find det(A) in terms of k.",
                    answer_kind="expression",
                    answer=str(C * sym_k + D),
                    answer_text=det_text,
                    answer_latex="plain:" + det_text,
                    marks=1,
                    working=["Expand along a row, keeping the k term."],
                    distractors=dis_a,
                ),
                dict(
                    prompt="Find the value of k for which A has no inverse.",
                    answer_kind="number",
                    answer=k0,
                    answer_text=str(k0),
                    answer_latex=str(k0),
                    marks=2,
                    working=[f"A has no inverse when det(A) = 0: {det_text} = 0, so k = {k0}."],
                    distractors=dis_b,
                ),
                dict(
                    prompt=f"Find A⁻¹ when k = {k1}. {_FORMAT_NOTE}",
                    answer_kind="matrix",
                    marks=2,
                    working=[f"At k = {k1}, det(A) = {C * k1 + D}."],
                    distractors=dis_c,
                    **ans_c,
                ),
                dict(
                    prompt=f"For k = {k1}, solve {eq}, where {'B' if n == 2 else 'b'} = {B_text}, to find {d_what}.",
                    answer_kind="matrix",
                    marks=2,
                    working=[f"{'X = A⁻¹B' if n == 2 else 'x = A⁻¹b'}."],
                    distractors=dis_d,
                    **ans_d,
                ),
            ],
        )
    raise VerificationError("no usable matrix exam chain")


@generator("exd_x_mat_inv_2x2", level=13, presets={"exam": dict(tier="exam")})
def exd_x_mat_inv_2x2(rng: random.Random, tier: str = "exam") -> dict:
    return _exam_chain(rng, 2)


@generator("exd_x_mat_inv_3x3", level=14, presets={"exam": dict(tier="exam")})
def exd_x_mat_inv_3x3(rng: random.Random, tier: str = "exam") -> dict:
    return _exam_chain(rng, 3)
