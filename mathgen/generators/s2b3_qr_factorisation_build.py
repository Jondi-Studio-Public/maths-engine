"""Generates `s2b3_qr` questions (easy, medium, hard) and the exam chain `exd_x_s2b3_qr`.

``uc2-qr-factorisation`` (``linear-algebra``, answer ``matrix``): the QR factorisation A = QR of a 3x2 or 3x3 integer
matrix with independent columns, Q having orthonormal columns and R upper triangular with a POSITIVE diagonal, which
makes both factors unique, so they are graded entry by entry.

* easy: R for a 3x2 matrix whose Gram-Schmidt lengths are whole numbers.
* medium: Q for a 3x2 matrix of the same kind (every entry a fraction with denominator 3 or 7).
* hard: one of: R for a 3x3 matrix with whole-number lengths; Q for a 3x2 matrix whose Q has square roots; R for a 3x3
  matrix whose R has square roots. Entries are exact (`sqrt(2)/2`, not 0.707).
* exam (``exd_x_s2b3_qr``): a 3x2 matrix A and a vector b. (a) r11, (b) r12 (follows through from (a)), (c) r22 (follows
  through from (b)), (d) the least-squares solution of Ax = b by back-substitution in R x = Q^T b.

Matrices with whole-number lengths are built as A = Q R from a rational orthonormal frame (the columns of an integer
matrix with orthogonal rows of the same length m, divided by m); square-root cases are drawn at random and kept when
every length is the square root of a small rational. Each answer is re-derived with SymPy's ``QRdecomposition``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, phrase
from ._s2b3_kit import (
    dot,
    frac,
    fraction_answer,
    fraction_distractors,
    mat_answer,
    mat_text,
    mat_vec,
    matrix_distractors,
    need,
    number_answer,
    to_sym,
    vec_answer,
    vec_text,
)
from ._ucv1_linalg import inverse, rank, transpose


def _int_norm_vectors(bound=8, lo=3, hi=11):
    """Integer vectors in [-bound, bound]^3 with no zero entry and a whole-number length lo..hi."""
    out = []
    for x in range(-bound, bound + 1):
        for y in range(-bound, bound + 1):
            for z in range(-bound, bound + 1):
                n2 = x * x + y * y + z * z
                n = math.isqrt(n2)
                if n * n == n2 and lo <= n <= hi and x and y and z:
                    out.append(((x, y, z), n))
    return out


_VECTORS = _int_norm_vectors()


def _gcd(v):
    g = 0
    for c in v:
        g = math.gcd(g, abs(c))
    return g


def _orthogonal_triple(rng, count):
    """`count` mutually orthogonal integer vectors with whole-number lengths, as [(vector, length)]."""
    for _ in attempts(500, "orthogonal integer vectors with whole-number lengths"):
        v1, n1 = rng.choice(_VECTORS)
        near = [(v, n) for v, n in _VECTORS if sum(a * b for a, b in zip(v, v1)) == 0 and n <= 9]
        if not near:
            continue
        v2, n2 = rng.choice(near)
        if count == 2:
            return [(v1, n1), (v2, n2)]
        cross = [v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]]
        g = _gcd(cross)
        v3 = tuple(c // g for c in cross)
        n3sq = sum(c * c for c in v3)
        n3 = math.isqrt(n3sq)
        if n3 * n3 == n3sq and n3 <= 12:
            return [(v1, n1), (v2, n2), (v3, n3)]
    raise AssertionError  # pragma: no cover


def _fm(rows):
    return [[Fraction(v) for v in row] for row in rows]


def _build_rational(rng, ncols):
    """(A, w, R): an integer matrix A = QR whose Q has columns w_i / |w_i| with whole-number lengths, and R."""
    for _ in attempts(500, "an integer matrix with whole-number Gram-Schmidt lengths"):
        triple = _orthogonal_triple(rng, ncols)
        vs = [v for v, _n in triple]
        ns = [n for _v, n in triple]
        gs = [_gcd(v) for v in vs]
        t = [[0] * ncols for _ in range(ncols)]
        for j in range(ncols):
            for i in range(j):
                t[i][j] = rng.choice((-3, -2, -1, 1, 2, 3))
        a_cols = [
            [vs[j][k] + sum(t[i][j] * (vs[i][k] // gs[i]) for i in range(j)) for k in range(3)] for j in range(ncols)
        ]
        if max(abs(x) for col in a_cols for x in col) > 12 or any(all(col[k] == 0 for col in a_cols) for k in range(3)):
            continue
        a = transpose([[Fraction(x) for x in col] for col in a_cols])
        r = [
            [
                Fraction(ns[i]) if i == j else Fraction(t[i][j] * ns[i] // gs[i]) if i < j else Fraction(0)
                for j in range(ncols)
            ]
            for i in range(ncols)
        ]
        return [[int(x) for x in row] for row in a], [list(v) for v in vs], r
    raise AssertionError  # pragma: no cover


def _sympy_qr(a):
    q, r = to_sym(_fm(a)).QRdecomposition()
    return q, r


def _entry_form(q):
    return [[q[i, j] for j in range(q.cols)] for i in range(q.rows)]


def _gram_data(a):
    """Exact Gram-Schmidt data of the columns of A: the orthogonal columns u_i (Fractions)."""
    cols = transpose(_fm(a))
    us: list[list[Fraction]] = []
    for col in cols:
        u = list(col)
        for w in us:
            c = dot(col, w) / dot(w, w)
            u = [x - c * y for x, y in zip(u, w)]
        us.append(u)
    return cols, us


def _exact_qr(a):
    """(Q, R) as exact SymPy matrices from the column Gram-Schmidt vectors."""
    cols, us = _gram_data(a)
    n = len(cols)
    root = [sympy.sqrt(sympy.Rational(dot(u, u).numerator, dot(u, u).denominator)) for u in us]
    q = sympy.zeros(3, n)
    r = sympy.zeros(n, n)
    for i in range(n):
        for k in range(3):
            q[k, i] = sympy.Rational(us[i][k].numerator, us[i][k].denominator) / root[i]
        r[i, i] = root[i]
        for j in range(i + 1, n):
            c = dot(us[i], cols[j])
            r[i, j] = sympy.Rational(c.numerator, c.denominator) / root[i]
    return q, r


def _verify(a, q, r):
    sa = to_sym(_fm(a))
    check((q * r - sa).applyfunc(sympy.expand) == sympy.zeros(*sa.shape), "QR is not A")
    check((q.T * q).applyfunc(sympy.expand) == sympy.eye(sa.cols), "Q does not have orthonormal columns")
    check(all(r[i, i].is_positive for i in range(r.rows)), "R does not have a positive diagonal")
    sq, sr = sa.QRdecomposition()
    check((sq - q).applyfunc(sympy.expand) == sympy.zeros(*q.shape), "SymPy's Q differs")
    check((sr - r).applyfunc(sympy.expand) == sympy.zeros(*r.shape), "SymPy's R differs")


def _readable(expr, radicand=30, den=12) -> bool:
    """Whether `expr` is a rational, or a rational times the square root of a small whole number."""
    if expr.is_Rational:
        return expr.q <= den and abs(expr.p) <= 60
    sq = sympy.nsimplify(expr**2)
    if not sq.is_Rational:
        return False
    coeff, rest = expr.as_coeff_Mul()
    return (
        rest.is_Pow
        and rest.exp == sympy.Rational(1, 2)
        and rest.base.is_Integer
        and rest.base <= radicand
        and coeff.q <= den
        and abs(coeff.p) <= 60
    )


def _surd_form(value: Fraction):
    """(c, k) with sqrt(value) = c sqrt(k), k squarefree, for a positive rational value."""
    pq = value.numerator * value.denominator
    square = 1
    for f in range(2, math.isqrt(pq) + 1):
        while pq % (f * f) == 0:
            pq //= f * f
            square *= f
    return Fraction(square, value.denominator), pq


def _draw_surd(rng, ncols):
    for _ in attempts(6000, "a matrix with square-root QR entries"):
        a = [[rng.randint(-3, 3) for _ in range(ncols)] for _ in range(3)]
        if rank(_fm(a)) != ncols or any(all(row[c] == 0 for row in a) for c in range(ncols)):
            continue
        if sum(1 for row in a for v in row if v == 0) > 2:
            continue
        cols, us = _gram_data(a)
        forms = [_surd_form(dot(u, u)) for u in us]
        if any(k > 15 or c.denominator > 6 for c, k in forms) or all(k == 1 for _c, k in forms):
            continue
        q, r = _exact_qr(a)
        if not all(_readable(e) for e in list(q) + list(r)) or all(e.is_Rational for e in list(q) + list(r)):
            continue
        return a, q, r
    raise AssertionError  # pragma: no cover


# -- distractors ------------------------------------------------------------------------------------------------


def _as_rows(m):
    return [[m[i, j] for j in range(m.cols)] for i in range(m.rows)]


def _r_errors(a, q, r):
    cols, us = _gram_data(a)
    n = len(cols)
    sym = lambda x: sympy.Rational(x.numerator, x.denominator)  # noqa: E731
    squared = r.copy()
    for i in range(n):
        squared[i, i] = sym(dot(us[i], us[i]))
    undivided = r.copy()
    for i in range(n):
        for j in range(i + 1, n):
            c = dot(us[i], cols[j])
            undivided[i, j] = sym(c)
    from_original = r.copy()
    for i in range(n):
        from_original[i, i] = sympy.sqrt(sym(dot(cols[i], cols[i])))
    return [
        (_as_rows(squared), "qr-diagonal-is-squared-length"),
        (_as_rows(undivided), "qr-off-diagonal-not-divided-by-length"),
        (_as_rows(r.T), "qr-r-not-upper-triangular"),
        (_as_rows(from_original), "qr-diagonal-from-original-column"),
    ]


def _q_errors(a, q, r):
    cols, us = _gram_data(a)
    n = len(cols)
    unnormalised = sympy.Matrix(
        [[sympy.Rational(us[i][k].numerator, us[i][k].denominator) for i in range(n)] for k in range(3)]
    )
    skip = q.copy()
    for i in range(1, n):
        norm = sympy.sqrt(sympy.Rational(dot(cols[i], cols[i]).numerator, dot(cols[i], cols[i]).denominator))
        for k in range(3):
            skip[k, i] = sympy.Rational(cols[i][k].numerator, cols[i][k].denominator) / norm
    flipped = q.copy()
    for k in range(3):
        flipped[k, n - 1] = -q[k, n - 1]
    return [
        (_as_rows(unnormalised), "qr-q-columns-not-normalised"),
        (_as_rows(skip), "qr-q-skips-orthogonalising"),
        (_as_rows(q.T), "qr-q-transposed"),
        (_as_rows(flipped), "qr-q-column-sign-flipped"),
    ]


_R_STEMS = (
    "Let A = {a}. Its QR factorisation A = QR has Q with orthonormal columns and R upper triangular with positive "
    "diagonal entries. Find R.",
    "The matrix A = {a} has independent columns. Find the upper triangular matrix R, with positive diagonal, in its "
    "QR factorisation A = QR.",
    "Factorise A = {a} as A = QR with Q having orthonormal columns and R upper triangular with positive diagonal "
    "entries. What is R?",
)
_Q_STEMS = (
    "Let A = {a}. Its QR factorisation A = QR has Q with orthonormal columns and R upper triangular with positive "
    "diagonal entries. Find Q.",
    "The matrix A = {a} has independent columns. Find the matrix Q with orthonormal columns in its QR factorisation "
    "A = QR, where R is upper triangular with positive diagonal.",
    "Factorise A = {a} as A = QR with Q having orthonormal columns and R upper triangular with positive diagonal "
    "entries. What is Q?",
)


def _r_question(rng, a, q, r):
    _verify(a, q, r)
    rows = _as_rows(r)
    cands = matrix_distractors(_r_errors(a, q, r), rows)
    return dict(
        text=phrase(rng, _R_STEMS, a=mat_text(a)),
        distractors=need(cands),
        working=[
            "Gram–Schmidt on the columns gives the orthonormal columns of Q; then r_ij = q_i · a_j for i ≤ j.",
            f"R = {mat_text(rows)}.",
        ],
        **mat_answer(rows),
    )


def _q_question(rng, a, q, r):
    _verify(a, q, r)
    rows = _as_rows(q)
    cands = matrix_distractors(_q_errors(a, q, r), rows)
    return dict(
        text=phrase(rng, _Q_STEMS, a=mat_text(a)),
        distractors=need(cands),
        working=[
            "Orthogonalise the columns of A by Gram–Schmidt, then divide each by its length.",
            f"Q = {mat_text(rows)}.",
        ],
        **mat_answer(rows),
    )


def _rational_case(rng, ncols):
    a, w, r = _build_rational(rng, ncols)
    q, rr = _exact_qr(a)
    check(
        _as_rows(rr) == [[sympy.Rational(int(v.numerator), int(v.denominator)) for v in row] for row in r], "R differs"
    )
    return a, q, rr


def _easy(rng):
    return _r_question(rng, *_rational_case(rng, 2))


def _medium(rng):
    return _q_question(rng, *_rational_case(rng, 2))


def _hard_rational_r(rng):
    return _r_question(rng, *_rational_case(rng, 3))


def _hard_surd_q(rng):
    return _q_question(rng, *_draw_surd(rng, 2))


def _hard_surd_r(rng):
    return _r_question(rng, *_draw_surd(rng, 3))


@generator(
    "s2b3_qr",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2b3_qr(rng, tier="easy"):
    """QR factorisation with positive diagonal; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice((_hard_rational_r, _hard_surd_q, _hard_surd_r, _hard_surd_q))(rng)


# -- exam -------------------------------------------------------------------------------------------------------


def _ints(true, candidates, *, positive=True):
    out, seen = [], {true}
    for value, tag in candidates:
        value = Fraction(value)
        if value.denominator != 1 or value == 0 or (positive and value < 0) or int(value) in seen:
            continue
        seen.add(int(value))
        out.append(Distractor(str(int(value)), tag, exact_value(int(value))))
    return out[:3]


_EXAM_STEMS = (
    "The matrix A = {a} has the QR factorisation A = QR, where Q has orthonormal columns and R is upper triangular "
    "with positive diagonal entries. Let b = {b}.",
    "Let A = {a} and b = {b}. Write A = QR with Q having orthonormal columns and R upper triangular with positive "
    "diagonal entries.",
    "The columns of A = {a} are independent, so A = QR with Q orthonormal and R upper triangular with a positive "
    "diagonal. Let b = {b}.",
)


@generator("exd_x_s2b3_qr", level=18, presets={"exam": dict(tier="exam")})
def exd_x_s2b3_qr(rng, tier="exam"):
    """A 3x2 matrix with whole-number lengths: r11, r12, r22 and the least-squares solution through R x = Q^T b."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a QR exam matrix with a readable least-squares solution"):
        a, w, r = _build_rational(rng, 2)
        b = [rng.randint(-4, 4) for _ in range(3)]
        cols = transpose(_fm(a))
        r11, r12, r22 = int(r[0][0]), int(r[0][1]), int(r[1][1])
        qtb = [dot(w[0], b) / r11, dot(w[1], b) / r22]
        x2 = qtb[1] / r22
        x1 = (qtb[0] - r12 * x2) / r11
        x = [x1, x2]
        resid = [bi - sum(Fraction(row[j]) * x[j] for j in range(2)) for bi, row in zip(b, a)]
        if (
            any(e != 0 for e in resid)
            and all(e != 0 for e in x)
            and max(e.denominator for e in x) <= 12
            and max(abs(e) for e in x) <= 12
            and r12 != 0
        ):
            break
    q, rr = _exact_qr(a)
    _verify(a, q, rr)
    check([[rr[0, 0], rr[0, 1]], [rr[1, 0], rr[1, 1]]] == [[r11, r12], [0, r22]], "R differs")
    sa = to_sym(_fm(a))
    check([frac(e) for e in sa.pinv() * to_sym(b)] == x, "SymPy's least-squares solution differs")
    n12 = int(dot(cols[0], cols[1]))
    n22 = int(dot(cols[1], cols[1]))
    n11 = int(dot(cols[0], cols[0]))
    check(n12 == r11 * r12 and n22 == r12 * r12 + r22 * r22, "the norm identities fail")

    a_dist = _ints(
        r11,
        [
            (n11, "qr-diagonal-is-squared-length"),
            (sum(abs(int(v)) for v in cols[0]), "qr-norm-is-sum-of-entries"),
            (r22, "qr-confuses-r11-and-r22"),
            (r11 + 1, "qr-norm-off-by-one"),
        ],
    )
    b_dist = fraction_distractors(
        r12,
        [
            (n12, "qr-off-diagonal-not-divided-by-length"),
            (Fraction(n12, r11 * r11), "qr-divides-by-squared-length"),
            (Fraction(n12, r22), "qr-divides-by-the-wrong-length"),
            (-r12, "sign-error-moving-term"),
        ],
    )
    c_dist = fraction_distractors(
        r22,
        [
            (n22 - r12 * r12, "qr-diagonal-is-squared-length"),
            (r12, "qr-confuses-r11-and-r22"),
            (r11, "qr-confuses-r11-and-r22"),
            (r22 + 1, "qr-norm-off-by-one"),
        ],
        positive=True,
    )
    y = qtb
    d_dist = matrix_distractors(
        [
            (y, "qr-forgot-back-substitution"),
            ([y[0] / r11, (y[1] - r12 * (y[0] / r11)) / r22], "qr-back-substitution-order-reversed"),
            ([(y[0] + r12 * x2) / r11, x2], "sign-error-moving-term"),
            ([x2, x1], "ls-coefficients-in-the-wrong-order"),
        ],
        x,
        vector=True,
        limit=3,
    )
    stem = phrase(rng, _EXAM_STEMS, a=mat_text(a), b=vec_text(b))
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find r11, the length of the first column of A.",
                marks=1,
                sensible_range=(0, None),
                working=[f"r11 = |a1| = √{n11} = {r11}."],
                distractors=need(a_dist),
                **number_answer(r11),
            ),
            dict(
                prompt="Find r12 = q1 · a2.",
                marks=2,
                follow_through=f"{n12}/part_a",
                working=[f"q1 = a1/{r11}, so r12 = (a1 · a2)/{r11} = {n12}/{r11} = {r12}."],
                distractors=need(b_dist),
                **fraction_answer(r12),
            ),
            dict(
                prompt="Find r22.",
                marks=2,
                follow_through=f"sqrt({n22} - part_b**2)",
                sensible_range=(0, None),
                working=[f"r22² = |a2|² - r12² = {n22} - {r12 * r12}, so r22 = {r22}."],
                distractors=need(c_dist),
                **fraction_answer(r22),
            ),
            dict(
                prompt="Hence find the least-squares solution of Ax = b by solving Rx = Qᵀb.",
                marks=3,
                working=[f"Qᵀb = {vec_text(qtb)}. Back-substitution gives x̂ = {vec_text(x)}."],
                distractors=need(d_dist),
                **vec_answer(x),
            ),
        ],
    )
