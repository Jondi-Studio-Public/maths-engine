"""Generates `ucv1_null_space` questions (easy, medium, hard) and the exam chain `exd_x_uc_null_space`.

``ucv1-null-space`` (``linear-algebra``, answer ``number``): rank, nullity and column space of a supplied integer
matrix, all by row reduction.

* easy: the rank of a 3×3 or 3×4 integer matrix in which at least one row is a combination of the others.
* medium: the nullity (dimension of the null space) of a 3×4, 3×5, 4×4 or 4×5 matrix, by rank-nullity: the rank is found
  by row reduction and subtracted from the number of columns.
* hard: a parameter k. Either the value of k for which a vector (b1, b2, k), with k in any position, lies in the column
  space of a 3×3 matrix of rank 2 (the consistency condition y·b = 0 for the left null vector y), or the value of k
  that makes one entry of a rank-2 matrix the one that gives it a non-trivial null space (det = 0, linear in k).
* exam (``exd_x_uc_null_space``): a 3×3 matrix of rank 2. (a) its rank, (b) its nullity (follows through from (a)),
  (c) a basis vector of the null space (a ``matrix`` answer, any non-zero multiple), (d) the k for which (b1, b2, k)
  lies in the column space.

Every matrix is built as a product B C of a 3×r and an r×n integer matrix, so its rank is known; the rank, nullity and
null vectors are re-derived by Gauss-Jordan elimination in exact fractions and by SymPy's rank() and nullspace(), and
every k is checked by asking SymPy for the rank of the augmented matrix at k and at k + 1.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, join_terms, phrase, pick_distractors, rational_text
from ._ucv1_linalg import (
    column,
    column_value,
    fm,
    mat_str,
    max_abs,
    mmul,
    nullspace,
    primitive,
    rank,
    transpose,
    vec_str,
    vector_distractors,
)

FORM = "scalar_multiple"
_MAX_ENTRY = 8


def _sym(m):
    return sympy.Matrix([[sympy.Rational(v.numerator, v.denominator) for v in row] for row in m])


def _product(rng, m, n, r):
    """An m×n integer matrix of rank exactly r, with no zero row or column and, for r > 1, no two proportional rows."""
    for _ in attempts(2000, f"a {m}×{n} matrix of rank {r}"):
        b = [[rng.randint(-2, 2) for _ in range(r)] for _ in range(m)]
        c = [[rng.randint(-2, 2) for _ in range(n)] for _ in range(r)]
        mat = mmul(fm(b), fm(c))
        if max_abs(mat) > _MAX_ENTRY or max_abs(mat) < 2:
            continue
        if any(all(v == 0 for v in row) for row in mat) or any(all(v == 0 for v in col) for col in zip(*mat)):
            continue
        if sum(1 for row in mat for v in row if v == 0) > 2:
            continue
        if rank(mat) != r:
            continue
        if r > 1 and any(rank([mat[i], mat[j]]) < 2 for i in range(m) for j in range(i + 1, m)):
            continue
        return mat
    raise AssertionError  # pragma: no cover


def _verify_rank(mat, r):
    """The rank by SymPy, and the nullity by SymPy's null space."""
    sym = _sym(mat)
    check(sym.rank() == r, "SymPy's rank differs")
    check(len(sym.nullspace()) == len(mat[0]) - r, "SymPy's nullity differs")


def _dist_int(value, tag):
    return Distractor(str(value), tag, value)


def _fraction_dist(value, tag):
    value = Fraction(value)
    return Distractor(rational_text(value), tag, exact_value(value))


# -- easy and medium ---------------------------------------------------------------------------------------------

_RANK_STEMS = (
    "What is the rank of the matrix M = {m}?",
    "Find the rank of M = {m}.",
    "Determine the rank of the matrix M = {m}.",
)
_NULLITY_STEMS = (
    "Find the dimension of the null space of the matrix M = {m}.",
    "What is the nullity of M = {m}?",
    "The linear map T: ℝ{n} → ℝ{rows} is given by T(x) = Mx, where M = {m}. Find the dimension of the kernel of T.",
)
_SUPER = {3: "³", 4: "⁴", 5: "⁵"}


def _easy(rng):
    m, n, r = rng.choice([(3, 3, 2), (3, 3, 2), (3, 3, 3), (3, 4, 2), (3, 4, 3), (3, 3, 1), (4, 3, 2), (4, 3, 3)])
    mat = _product(rng, m, n, r)
    _verify_rank(mat, r)
    candidates = [
        _dist_int(m, "rank-counts-nonzero-rows"),
        _dist_int(n - r, "rank-nullity-confused"),
        _dist_int(r - 1, "rank-off-by-one"),
        _dist_int(r + 1, "rank-off-by-one"),
        _dist_int(n, "nullity-counts-columns"),
    ]
    candidates = [c for c in candidates if 0 < int(c.text) <= min(m, n) or c.misconception != "rank-off-by-one"]
    distractors = pick_distractors(str(r), candidates, 4)
    check(len(distractors) >= 2, "too few distractors")
    working = [
        f"Row-reducing M = {mat_str(mat)} leaves {r} non-zero row{'s' if r != 1 else ''}.",
        f"So the rank of M is {r}.",
    ]
    return dict(
        text=phrase(rng, _RANK_STEMS, m=mat_str(mat)),
        answer_kind="number",
        answer=r,
        answer_text=str(r),
        answer_latex=str(r),
        working=working,
        distractors=distractors,
    )


def _medium(rng):
    m, n, r = rng.choice([(3, 4, 2), (3, 5, 2), (4, 4, 2), (4, 4, 3), (4, 5, 3), (3, 5, 3), (4, 5, 2), (4, 4, 3)])
    mat = _product(rng, m, n, r)
    _verify_rank(mat, r)
    nullity = n - r
    check(len(nullspace(mat)) == nullity, "nullspace dimension differs from n - rank")
    candidates = [
        _dist_int(r, "rank-nullity-confused"),
        _dist_int(m - r, "nullity-uses-rows"),
        _dist_int(n, "nullity-counts-columns"),
        _dist_int(nullity + 1, "rank-off-by-one"),
        _dist_int(nullity - 1, "rank-off-by-one"),
    ]
    candidates = [c for c in candidates if int(c.text) >= 0]
    distractors = pick_distractors(str(nullity), candidates, 4)
    check(len(distractors) >= 2, "too few distractors")
    working = [
        f"Row-reducing M = {mat_str(mat)} gives rank M = {r}.",
        f"M has {n} columns, so by rank-nullity the nullity is {n} - {r} = {nullity}.",
    ]
    return dict(
        text=phrase(rng, _NULLITY_STEMS, m=mat_str(mat), n=_SUPER[n], rows=_SUPER[m]),
        answer_kind="number",
        answer=nullity,
        answer_text=str(nullity),
        answer_latex=str(nullity),
        working=working,
        distractors=distractors,
    )


# -- hard -----------------------------------------------------------------------------------------------------

_COLSPACE_STEMS = (
    "Find the value of k for which the vector b = {b}ᵀ lies in the column space of M = {m}.",
    "The vector b = {b}ᵀ is in the column space of M = {m}. Find the value of k.",
    "For M = {m}, find the value of k such that the equation Mx = {b}ᵀ has a solution.",
)
_DET_STEMS = (
    "Find the value of k for which the matrix M = {m} has a non-trivial null space.",
    "Find the value of k for which Mx = 0 has a non-zero solution, where M = {m}.",
    "Find the value of k for which the matrix M = {m} is not invertible.",
)


def _rank2_3x3(rng):
    """(M, y, z): a 3×3 integer matrix of rank 2, its left null vector y and null vector z (both primitive)."""
    for _ in attempts(500, "a rank-2 3×3 matrix with small null vectors"):
        mat = _product(rng, 3, 3, 2)
        left = nullspace(transpose(mat))
        right = nullspace(mat)
        if len(left) != 1 or len(right) != 1:
            continue
        y, z = primitive(left[0]), primitive(right[0])
        if max(abs(v) for v in y + z) > 6:
            continue
        return mat, y, z
    raise AssertionError  # pragma: no cover


def _draw_colspace(rng, mat, y):
    """(t, b, k): position t of k in b, the vector b with k in it, and the integer k that puts b in the column space."""
    for _ in attempts(500, "a vector with an integer k in the column space"):
        t = rng.choice([i for i in range(3) if y[i] != 0])
        b = [rng.randint(-4, 5) for _ in range(3)]
        total = sum(y[i] * b[i] for i in range(3) if i != t)
        if total % y[t] != 0:
            continue
        k = -total // y[t]
        if k == 0 or abs(k) > 15:
            continue
        b[t] = k
        return t, b, k
    raise AssertionError  # pragma: no cover


def _draw_colspace_with_distractors(rng, mat, y, z, n=4):
    """`_draw_colspace`, redrawn until at least two distinct wrong values are on offer."""
    for _ in attempts(200, "a column-space question with two distractors"):
        t, b, k = _draw_colspace(rng, mat, y)
        distractors = pick_distractors(str(k), _colspace_distractors(y, z, b, t, k), n, tolerance=Fraction(1, 10))
        if len(distractors) >= 2:
            return t, b, k, distractors
    raise AssertionError  # pragma: no cover


def _b_text(b, t):
    return "(" + ", ".join("k" if i == t else str(v) for i, v in enumerate(b)) + ")"


def _colspace_distractors(y, z, b, t, k):
    others = [i for i in range(3) if i != t]
    total = sum(y[i] * b[i] for i in others)
    out = [_fraction_dist(-k, "sign-error-moving-term")]
    if abs(y[t]) != 1:
        out.append(_fraction_dist(-total, "colspace-forgot-multiplier"))
    if z[t] != 0:
        out.append(_fraction_dist(Fraction(-sum(z[i] * b[i] for i in others), z[t]), "colspace-null-vector-used"))
    for i in others:
        slip = Fraction(-(total - 2 * y[i] * b[i]), y[t])
        out.append(_fraction_dist(slip, "sign-error-moving-term"))
    return out


def _check_colspace(mat, b, t, k):
    """SymPy: the augmented matrix has rank 2 at k and rank 3 at k + 1."""
    good, bad = list(b), list(b)
    good[t], bad[t] = k, k + 1
    base = _sym(mat)
    check(base.row_join(sympy.Matrix(good)).rank() == 2, "b is not in the column space at k")
    check(base.row_join(sympy.Matrix(bad)).rank() == 3, "b is in the column space at k + 1")


def _hard_colspace(rng):
    mat, y, z = _rank2_3x3(rng)
    t, b, k, distractors = _draw_colspace_with_distractors(rng, mat, y, z)
    _check_colspace(mat, b, t, k)
    working = [
        f"Row-reducing Mᵀ shows the left null vector y = {vec_str(y)}, so b is in the column space exactly when y·b = 0.",
        f"y·b = {join_terms([(y[i], 'k' if i == t else '(' + str(b[i]) + ')') for i in range(3)])} = 0, so k = {k}.",
    ]
    return dict(
        text=phrase(rng, _COLSPACE_STEMS, b=_b_text(b, t), m=mat_str(mat)),
        answer_kind="number",
        answer=k,
        answer_text=str(k),
        answer_latex=str(k),
        working=working,
        distractors=distractors,
    )


def _m_text_with_k(mat, i, j):
    rows = []
    for r in range(3):
        rows.append("[" + ", ".join("k" if (r, c) == (i, j) else str(mat[r][c]) for c in range(3)) + "]")
    return "[" + ", ".join(rows) + "]"


def _sarrus_terms(entries):
    """The six signed products of a 3×3 determinant, as SymPy expressions."""
    a = entries
    return [
        a[0][0] * a[1][1] * a[2][2],
        a[0][1] * a[1][2] * a[2][0],
        a[0][2] * a[1][0] * a[2][1],
        -a[0][2] * a[1][1] * a[2][0],
        -a[0][0] * a[1][2] * a[2][1],
        -a[0][1] * a[1][0] * a[2][2],
    ]


def _det_distractors(mat, i, j, k0):
    kk = sympy.symbols("k")
    entries = [[kk if (r, c) == (i, j) else sympy.Integer(int(mat[r][c])) for c in range(3)] for r in range(3)]
    terms = _sarrus_terms(entries)

    def root(expr):
        poly = sympy.Poly(sympy.expand(expr), kk)
        if poly.degree() != 1:
            return None
        return _linear_root(poly)

    out = []
    with_k = [n for n, term in enumerate(terms) if term.has(kk)]
    for n in with_k + [n for n in range(6) if n not in with_k]:
        flipped = sum(terms) - 2 * terms[n]
        value = root(flipped)
        if value is not None:
            out.append(_fraction_dist(value, "det-sarrus-sign-slip"))
    for n in with_k + [n for n in range(6) if n not in with_k]:
        value = root(sum(terms) - terms[n])
        if value is not None:
            out.append(_fraction_dist(value, "det-sarrus-term-dropped"))
    return [_fraction_dist(-k0, "sign-error-moving-term"), *out]


def _linear_root(poly):
    a, b = poly.coeff_monomial(poly.gens[0]), poly.coeff_monomial(1)
    return Fraction(int(-b.p) * int(a.q), int(b.q) * int(a.p)) if a != 0 else None


def _hard_det(rng):
    for _ in attempts(500, "a rank-2 matrix with a non-zero cofactor"):
        mat, _y, _z = _rank2_3x3(rng)
        i, j = rng.randrange(3), rng.randrange(3)
        minor = [[mat[r][c] for c in range(3) if c != j] for r in range(3) if r != i]
        cofactor = minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0]
        k0 = int(mat[i][j])
        if cofactor != 0 and k0 != 0:
            break
    # Independent route: SymPy solves det(M(k)) = 0 and finds a single root.
    kk = sympy.symbols("k")
    sym = _sym(mat)
    sym[i, j] = kk
    roots = sympy.solve(sym.det(), kk)
    check(roots == [k0], "SymPy's root of det M(k) differs")
    check(rank(mat) == 2, "M at k is not of rank 2")
    distractors = pick_distractors(str(k0), _det_distractors(mat, i, j, k0), 4, tolerance=Fraction(1, 10))
    check(len(distractors) >= 2, "too few distractors")
    working = [
        f"Expanding det M along row {i + 1}: det M = {cofactor if (i + j) % 2 == 0 else -cofactor}(k - ({k0})).",
        f"This is zero only for k = {k0}, and M then has rank 2, so its null space is non-trivial.",
    ]
    return dict(
        text=phrase(rng, _DET_STEMS, m=_m_text_with_k(mat, i, j)),
        answer_kind="number",
        answer=k0,
        answer_text=str(k0),
        answer_latex=str(k0),
        working=working,
        distractors=distractors,
    )


@generator(
    "ucv1_null_space",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_null_space(rng, tier="easy"):
    """Rank, nullity and column space of an integer matrix; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_colspace, _hard_det, _hard_colspace])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_uc_null_space", level=18, presets={"exam": dict(tier="exam")})
def exd_x_uc_null_space(rng, tier="exam"):
    """Exam chain: rank, nullity, a null vector and the k that puts b in the column space of a 3×3 matrix."""
    check(tier == "exam", f"unknown tier {tier!r}")
    mat, y, z = _rank2_3x3(rng)
    _verify_rank(mat, 2)
    t, b, k, d_dist = _draw_colspace_with_distractors(rng, mat, y, z, 3)
    _check_colspace(mat, b, t, k)
    # The null vector by SymPy as well: M z = 0 and SymPy's null space is the line through z.
    sym_null = _sym(mat).nullspace()
    check(len(sym_null) == 1, "SymPy's null space is not a line")
    check(all(sum(mat[r][c] * z[c] for c in range(3)) == 0 for r in range(3)), "M z != 0")
    check(primitive([Fraction(int(v.p), int(v.q)) for v in sym_null[0]]) == z, "SymPy's null vector differs")

    z_text = vec_str(z)
    row0, col0 = list(mat[0]), column(mat, 0)
    z_candidates = [
        (row0, "nullvec-row-as-vector"),
        (y, "nullvec-left-null"),
        ([-z[0], *z[1:]], "nullvec-component-sign-slip"),
        (col0, "nullvec-column-as-vector"),
    ]
    c_dist = pick_distractors(z_text, vector_distractors(z_candidates, z), 3)
    a_dist = pick_distractors("2", [_dist_int(3, "rank-counts-nonzero-rows"), _dist_int(1, "rank-nullity-confused")], 3)
    b_dist = pick_distractors(
        "1",
        [
            _dist_int(2, "rank-nullity-confused"),
            _dist_int(3, "nullity-counts-columns"),
            _dist_int(0, "rank-off-by-one"),
        ],
        3,
    )
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")

    return dict(
        text=f"Let M = {mat_str(mat)}.",
        parts=[
            dict(
                prompt="Find the rank of M.",
                answer_kind="number",
                answer=2,
                answer_text="2",
                marks=2,
                working=[f"Row-reducing M = {mat_str(mat)} leaves two non-zero rows, so the rank is 2."],
                distractors=a_dist,
            ),
            dict(
                prompt="Hence write down the dimension of the null space of M.",
                answer_kind="number",
                answer=1,
                answer_text="1",
                marks=1,
                follow_through="3 - part_a",
                working=["M has 3 columns, so the nullity is 3 - 2 = 1."],
                distractors=b_dist,
            ),
            dict(
                prompt="Find a non-zero vector in the null space of M.",
                answer_kind="matrix",
                answer=column_value(z),
                answer_text=z_text,
                answer_latex="plain:" + z_text,
                answer_form=FORM,
                marks=2,
                working=[f"Solving Mx = 0 by row reduction gives the one-parameter family x = t{z_text}."],
                distractors=c_dist,
            ),
            dict(
                prompt=f"The vector b = {_b_text(b, t)}ᵀ lies in the column space of M. Find the value of k.",
                answer_kind="number",
                answer=k,
                answer_text=str(k),
                marks=3,
                working=[
                    f"The left null vector is y = {vec_str(y)}, and b is in the column space exactly when y·b = 0.",
                    f"This gives k = {k}.",
                ],
                distractors=d_dist,
            ),
        ],
    )
