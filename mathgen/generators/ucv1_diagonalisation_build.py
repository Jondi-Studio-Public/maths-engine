"""Generates `ucv1_diagonalisation` questions (easy, medium, hard) and the exam chain `exd_x_uc_diagonalisation`.

``ucv1-diagonalisation`` (``linear-algebra``, answer ``matrix``): the factorisation A = P D P^-1 of an integer matrix
with integer eigenvalues.

* easy: multiply out A = P D P^-1 from a supplied P (determinant 1 or -1) and diagonal D. Answer an integer matrix.
* medium: from A alone, write D with the eigenvalues in increasing order down the diagonal (a 2×2 matrix, or a 3×3
  block matrix), or write P for a stated D so that P^-1 A P = D (a 2×2 matrix, the column order fixed by D; each
  column is an eigenvector, so it may be scaled: ``column_scalar_multiple``).
* hard: A^n for a stated n by diagonalising a 2×2 matrix, so the answer is P D^n P^-1 (an integer matrix).
* exam (``exd_x_uc_diagonalisation``): a 2×2 matrix with one eigenvalue stated. (a) the other eigenvalue, (b) P with
  columns in the order of D = diag(smaller, larger), (c) P^-1 for a P the part prompt states (its columns scaled, so the
  inverse has fractions), (d) one entry of A^n, which follows through from (a) because it is a combination of the two
  eigenvalues' n-th powers.

Every matrix is built backwards from integer eigenvalues and an integer P of determinant 1 or -1. Answers are
re-derived by a second route: repeated multiplication for A^n, SymPy's matrix inverse for P D P^-1, and A P = P D for
every P.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator, join_terms, phrase, pick_distractors
from ._ucv1_linalg import (
    column,
    diagonal,
    draw_2x2,
    inverse,
    mat_str,
    matrix_distractors,
    mmul,
    mpow,
    nullspace,
    primitive,
    rows_value,
    transpose,
    unimodular,
    with_columns,
    max_abs,
)

_CAP_ENTRY = 30000


def _sym(m):
    return sympy.Matrix([[sympy.Rational(v.numerator, v.denominator) for v in row] for row in m])


def _eigenvector(a, lam):
    n = len(a)
    shifted = [[a[i][j] - (lam if i == j else 0) for j in range(n)] for i in range(n)]
    basis = nullspace(shifted)
    check(len(basis) == 1, "eigenspace is not a line")
    return primitive(basis[0])


def _eigenvector_matrix(a, lams):
    """P with the primitive eigenvector for lams[i] in column i."""
    return with_columns([_eigenvector(a, lam) for lam in lams])


def _verify_factorisation(a, p, d):
    """A P = P D (so the columns are eigenvectors) and SymPy's P D P^-1 is A."""
    check(mmul(a, p) == mmul(p, d), "A P != P D")
    check(_sym(a) == _sym(p) * _sym(d) * _sym(p).inv(), "SymPy's P D P^-1 differs from A")


def _row_of_shift(a, lam):
    """The first non-zero row of A - λI, as a vector (a row taken for an eigenvector)."""
    n = len(a)
    shifted = [[a[i][j] - (lam if i == j else 0) for j in range(n)] for i in range(n)]
    return primitive(next(row for row in shifted if any(v != 0 for v in row)))


def _plus(l1, l2):
    return diagonal([l2, l1])


# -- easy -----------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "Let P = {p} and D = {d}. Find the matrix A = PDP⁻¹.",
    "Given P = {p} and D = {d}, find A = PDP⁻¹.",
    "The matrix A satisfies A = PDP⁻¹, where P = {p} and D = {d}. Find A.",
)


def _easy(rng):
    for _ in attempts(500, "a P with at least three non-zero entries"):
        p = unimodular(rng, 2, 4)
        if sum(1 for row in p for v in row if v != 0) >= 3:
            break
    l1, l2 = rng.sample(range(-3, 5), 2)
    d = diagonal([l1, l2])
    pinv = inverse(p)
    a = mmul(mmul(p, d), pinv)
    _verify_factorisation(a, p, d)
    answer = rows_value(a)
    candidates = [
        (mmul(mmul(pinv, d), p), "diag-conjugation-reversed"),
        (mmul(p, d), "diag-forgot-inverse"),
        (mmul(mmul(p, _plus(l1, l2)), pinv), "diag-eigenvalues-swapped"),
        (mmul(mmul(p, d), transpose(p)), "diag-transpose-for-inverse"),
    ]
    answer_text = mat_str(a)
    distractors = pick_distractors(answer_text, matrix_distractors(candidates, answer), 4)
    check(len(distractors) >= 2, "too few distractors")
    working = [
        f"det P = {sympy.Integer(int(_sym(p).det()))}, so P⁻¹ = {mat_str(pinv)}.",
        f"PD = {mat_str(mmul(p, d))}.",
        f"A = PDP⁻¹ = {answer_text}.",
    ]
    return dict(
        text=phrase(rng, _EASY_STEMS, p=mat_str(p), d=mat_str(d)),
        answer_kind="matrix",
        answer=answer,
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        working=working,
        distractors=distractors,
    )


# -- medium ---------------------------------------------------------------------------------------------------

_D_STEMS = (
    "The matrix A = {a} is diagonalisable. Write down a diagonal matrix D such that A = PDP⁻¹ for some invertible "
    "matrix P, with the eigenvalues in increasing order down the diagonal of D.",
    "Find the diagonal matrix D in a diagonalisation A = PDP⁻¹ of A = {a}, with the eigenvalues of A in increasing "
    "order down the diagonal.",
)
_P_STEMS = (
    "Find an invertible matrix P such that P⁻¹AP = D, where A = {a} and D = {d}.",
    "The matrix A = {a} satisfies P⁻¹AP = D, where D = {d}. Find a possible matrix P.",
)


def _block(rng, b, c, first):
    zero, c = Fraction(0), Fraction(c)
    if first:
        return [[b[0][0], b[0][1], zero], [b[1][0], b[1][1], zero], [zero, zero, c]]
    return [[c, zero, zero], [zero, b[0][0], b[0][1]], [zero, b[1][0], b[1][1]]]


def _medium_d(rng):
    if rng.random() < 0.6:
        _p, lams, a = draw_2x2(rng, full=True)
    else:
        _p, lams2, b = draw_2x2(rng)
        c = rng.choice([v for v in range(-4, 6) if v not in lams2])
        a = _block(rng, b, c, rng.random() < 0.5)
        lams = [*lams2, c]
    ordered = sorted(lams)
    d = diagonal(ordered)
    answer = rows_value(d)
    # Independent check: SymPy's eigenvalues of A are exactly these, with multiplicity.
    sym_values = sorted(v for v, mult in _sym(a).eigenvals().items() for _ in range(mult))
    check(sym_values == ordered, "SymPy's eigenvalues differ")
    n = len(a)
    candidates = [
        (diagonal(ordered[::-1]), "diag-eigenvalues-swapped"),
        (diagonal([a[i][i] for i in range(n)]), "eigval-diagonal-entry"),
        (diagonal([-v for v in ordered]), "sign-flip-both-roots"),
        (diagonal([ordered[1], ordered[0], *ordered[2:]]), "diag-eigenvalues-swapped"),
        (diagonal(lams), "diag-eigenvalues-swapped"),
    ]
    answer_text = mat_str(d)
    distractors = pick_distractors(answer_text, matrix_distractors(candidates, answer), 4)
    check(len(distractors) >= 2, "too few distractors")
    working = [
        f"The eigenvalues of A are {', '.join(str(v) for v in ordered)}.",
        f"In increasing order down the diagonal, D = {answer_text}.",
    ]
    return dict(
        text=phrase(rng, _D_STEMS, a=mat_str(a)),
        answer_kind="matrix",
        answer=answer,
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        working=working,
        distractors=distractors,
    )


def _medium_p(rng):
    _p, lams, a = draw_2x2(rng, full=True)
    d = diagonal(lams)  # lams is in a random order, so the column order is not always increasing
    p = _eigenvector_matrix(a, lams)
    _verify_factorisation(a, p, d)
    answer = rows_value(p)
    cols = [column(p, 0), column(p, 1)]
    slipped = [primitive([cols[0][1], cols[0][0]]), cols[1]]
    flipped = []
    for lam, col in zip(lams, cols):
        shifted = [[a[i][j] + (lam if i == j else 0) for j in range(2)] for i in range(2)]
        basis = nullspace(shifted)
        flipped.append(primitive(basis[0]) if len(basis) == 1 else col)
    candidates = [
        (with_columns([cols[1], cols[0]]), "diag-columns-wrong-order"),
        (transpose(p), "diag-vectors-as-rows"),
        (with_columns(slipped), "eigvec-component-sign-slip"),
        (with_columns(flipped), "eigvec-eigenvalue-sign"),
        (with_columns([cols[0], primitive([cols[1][1], cols[1][0]])]), "eigvec-component-sign-slip"),
        (with_columns([_row_of_shift(a, lams[0]), _row_of_shift(a, lams[1])]), "eigvec-row-as-vector"),
    ]
    answer_text = mat_str(p)
    form = "column_scalar_multiple"
    distractors = pick_distractors(answer_text, matrix_distractors(candidates, answer, form), 4)
    check(len(distractors) >= 2, "too few distractors")
    working = [
        f"For λ = {lams[0]}, (A - λI)v = 0 gives v = {mat_str([[x] for x in cols[0]])}; for λ = {lams[1]}, "
        f"v = {mat_str([[x] for x in cols[1]])}.",
        f"The columns follow the order of D, so P = {answer_text}.",
    ]
    return dict(
        text=phrase(rng, _P_STEMS, a=mat_str(a), d=mat_str(d)),
        answer_kind="matrix",
        answer=answer,
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        answer_form=form,
        working=working,
        distractors=distractors,
    )


# -- hard -----------------------------------------------------------------------------------------------------

_POWER_STEMS = (
    "Diagonalise the matrix A = {a} and hence find A{power}.",
    "By diagonalising A = {a}, find A{power}.",
    "Find A{power} for A = {a}, by first diagonalising A.",
)
_SUPERSCRIPT = {0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹"}


def _super(n):
    return "".join(_SUPERSCRIPT[int(ch)] for ch in str(n))


def _draw_power(rng):
    for _ in attempts(500, "a 2×2 matrix and power with a modest result"):
        p, lams, a = draw_2x2(rng, -2, 3, cap=20, full=True, bound=3)
        top = max(abs(v) for v in lams)
        if 0 in lams or top < 2:
            continue
        n = rng.choice([3, 4, 5, 6, 8] if top <= 2 else [3, 4, 5])
        result = mmul(mmul(p, diagonal([Fraction(v) ** n for v in lams])), inverse(p))
        if max_abs(result) <= _CAP_ENTRY:
            return p, lams, a, n, result
    raise AssertionError  # pragma: no cover


def _hard(rng):
    p, lams, a, n, result = _draw_power(rng)
    # Independent route: repeated multiplication, and SymPy's matrix power.
    check(mpow(a, n) == result, "A^n by repeated multiplication differs")
    check(_sym(a) ** n == _sym(result), "SymPy's A^n differs")
    pinv = inverse(p)

    def via(powers):
        return mmul(mmul(p, diagonal(powers)), pinv)

    answer = rows_value(result)
    l1, l2 = lams
    candidates = [
        ([[v**n for v in row] for row in a], "matrix-power-entrywise"),
        (via([Fraction(l2) ** n, Fraction(l1) ** n]), "diag-eigenvalues-swapped"),
        (mpow(a, n - 1), "matrix-power-off-by-one"),
        ([[n * v for v in row] for row in a], "matrix-power-as-multiple"),
        (mmul(mmul(pinv, diagonal([Fraction(l1) ** n, Fraction(l2) ** n])), p), "diag-conjugation-reversed"),
        (diagonal([Fraction(l1) ** n, Fraction(l2) ** n]), "diag-forgot-change-of-basis"),
    ]
    answer_text = mat_str(result)
    distractors = pick_distractors(answer_text, matrix_distractors(candidates, answer), 4)
    check(len(distractors) >= 2, "too few distractors")
    working = [
        f"The eigenvalues of A are {l1} and {l2}, with eigenvectors the columns of P = {mat_str(p)}.",
        f"A{_super(n)} = PD{_super(n)}P⁻¹ with D{_super(n)} = {mat_str(diagonal([Fraction(l1) ** n, Fraction(l2) ** n]))}.",
        f"A{_super(n)} = {answer_text}.",
    ]
    return dict(
        text=phrase(rng, _POWER_STEMS, a=mat_str(a), power=_super(n)),
        answer_kind="matrix",
        answer=answer,
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        working=working,
        distractors=distractors,
    )


@generator(
    "ucv1_diagonalisation",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_diagonalisation(rng, tier="easy"):
    """Diagonalisation A = P D P^-1 of an integer matrix; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return rng.choice([_medium_d, _medium_p])(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------

_EXAM_STEMS = (
    "The matrix A = {a} has an eigenvalue {lam}.",
    "Let A = {a}. One eigenvalue of A is {lam}.",
)


@generator("exd_x_uc_diagonalisation", level=18, presets={"exam": dict(tier="exam")})
def exd_x_uc_diagonalisation(rng, tier="exam"):
    """Exam chain: the other eigenvalue, P, a stated P's inverse, and an entry of A^n."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "an exam diagonalisation"):
        _p, lams, a = draw_2x2(rng, -3, 4, cap=20, full=True, bound=3)
        if 0 in lams or abs(lams[0] - lams[1]) < 2:
            continue
        break
    small, large = sorted(lams)
    k = rng.randrange(2)
    given, other = lams[k], lams[1 - k]
    trace = int(a[0][0] + a[1][1])
    determinant = int(a[0][0] * a[1][1] - a[0][1] * a[1][0])
    check(trace == given + other and determinant == given * other, "trace/determinant disagree with the eigenvalues")
    check(sorted(_sym(a).eigenvals()) == [small, large], "SymPy's eigenvalues differ")
    d = diagonal([small, large])
    p = _eigenvector_matrix(a, [small, large])
    _verify_factorisation(a, p, d)
    scales = rng.choice([(1, 2), (2, 1), (2, 3), (3, 1), (1, -2), (2, -1), (3, 2)])
    p_stated = [[p[i][j] * scales[j] for j in range(2)] for i in range(2)]
    check(mmul(a, p_stated) == mmul(p_stated, d), "the stated P is not an eigenvector matrix")
    pinv_stated = inverse(p_stated)
    check(_sym(p_stated).inv() == _sym(pinv_stated), "SymPy's inverse of the stated P differs")
    n = rng.choice([3, 4, 5])
    row, col = rng.randrange(2), rng.randrange(2)
    power = mpow(a, n)
    entry_value = int(power[row][col])
    # Independent route: the entry is sum over eigenvalues of (P[row][k] * Pinv[k][col]) * lambda_k^n.
    coefficient = {lam: int(p_stated[row][i] * pinv_stated[i][col]) for i, lam in enumerate([small, large])}
    check(sum(c * lam**n for lam, c in coefficient.items()) == entry_value, "entry of A^n disagrees with P D^n P^-1")
    check(int(_sym(a).__pow__(n)[row, col]) == entry_value, "SymPy's A^n disagrees")

    def dist(v, tag):
        return Distractor(str(v), tag, v)

    a_dist = pick_distractors(
        str(other),
        [
            dist(given, "eigval-repeated-given"),
            dist(-other, "sign-flip-both-roots"),
            dist(trace, "eigval-trace-as-eigenvalue"),
            dist(trace + given, "eigval-trace-difference-sign"),
            dist(determinant, "eigval-determinant-as-eigenvalue"),
        ],
        3,
    )
    p_text = mat_str(p)
    form = "column_scalar_multiple"
    cols = [column(p, 0), column(p, 1)]
    p_candidates = [
        (with_columns([cols[1], cols[0]]), "diag-columns-wrong-order"),
        (transpose(p), "diag-vectors-as-rows"),
        (with_columns([primitive([cols[0][1], cols[0][0]]), cols[1]]), "eigvec-component-sign-slip"),
        (with_columns([cols[0], primitive([cols[1][1], cols[1][0]])]), "eigvec-component-sign-slip"),
    ]
    b_dist = pick_distractors(p_text, matrix_distractors(p_candidates, rows_value(p), form), 3)
    (pa, pb), (pc, pd_) = p_stated
    det_p = pa * pd_ - pb * pc
    inv_text = mat_str(pinv_stated)
    inverse_candidates = [
        ([[pd_, -pb], [-pc, pa]], "inverse-adjugate-undivided"),
        ([[pd_ / det_p, pb / det_p], [pc / det_p, pa / det_p]], "inverse-sign-pattern"),
        (transpose(p_stated), "diag-transpose-for-inverse"),
        (
            [[Fraction(1) / v if v != 0 else None for v in r] for r in p_stated]
            if all(v != 0 for r in p_stated for v in r)
            else None,
            "inverse-entrywise-reciprocal",
        ),
    ]
    c_dist = pick_distractors(inv_text, matrix_distractors(inverse_candidates, rows_value(pinv_stated)), 3)
    c_given, c_other = coefficient[given], coefficient[other]
    d_candidates = [
        (int(a[row][col]) ** n, "matrix-power-entrywise"),
        (c_given * other**n + c_other * given**n, "diag-eigenvalues-swapped"),
        (int(mpow(a, n - 1)[row][col]), "matrix-power-off-by-one"),
        (n * int(a[row][col]), "matrix-power-as-multiple"),
        (int(diagonal([small**n, large**n])[row][col]), "diag-forgot-change-of-basis"),
    ]
    d_dist = pick_distractors(str(entry_value), [dist(v, tag) for v, tag in d_candidates], 3)
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")

    ft = f"({c_given})*({given})**{n} + ({c_other})*(part_a)**{n}"
    stem = phrase(rng, _EXAM_STEMS, a=mat_str(a), lam=given)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the other eigenvalue of A.",
                answer_kind="number",
                answer=other,
                answer_text=str(other),
                marks=2,
                working=[
                    f"The eigenvalues sum to the trace {trace}, so the other is {trace} - ({given}) = {other}.",
                    f"Check: their product is {given * other}, which is det A = {determinant}.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=(
                    "Find a matrix P whose columns are eigenvectors of A, so that P⁻¹AP = D, where D is the diagonal "
                    "matrix with the eigenvalues of A in increasing order down the diagonal."
                ),
                answer_kind="matrix",
                answer=rows_value(p),
                answer_text=p_text,
                answer_latex="plain:" + p_text,
                answer_form=form,
                marks=2,
                working=[
                    f"The eigenvalues in increasing order are {small} and {large}, so D = {mat_str(d)}.",
                    f"Solving (A - λI)v = 0 for each gives the columns of P = {p_text}.",
                ],
                distractors=b_dist,
            ),
            dict(
                prompt=f"Taking P = {mat_str(p_stated)}, find P⁻¹.",
                answer_kind="matrix",
                answer=rows_value(pinv_stated),
                answer_text=inv_text,
                answer_latex="plain:" + inv_text,
                marks=2,
                working=[
                    f"det P = {det_p}, so P⁻¹ = (1/{det_p}) {mat_str([[pd_, -pb], [-pc, pa]])} = {inv_text}.",
                ],
                distractors=c_dist,
            ),
            dict(
                prompt=f"Hence find the entry in row {row + 1}, column {col + 1} of A{_super(n)}.",
                answer_kind="number",
                answer=entry_value,
                answer_text=str(entry_value),
                marks=3,
                follow_through=ft,
                working=[
                    f"A{_super(n)} = PD{_super(n)}P⁻¹, with D{_super(n)} = {mat_str(diagonal([small**n, large**n]))}.",
                    f"The entry is {join_terms([(coefficient[small], f'({small}){_super(n)}'), (coefficient[large], f'({large}){_super(n)}')])} "
                    f"= {entry_value}.",
                ],
                distractors=d_dist,
            ),
        ],
    )
