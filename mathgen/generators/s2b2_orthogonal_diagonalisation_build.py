"""Generates `s2b2_orthogonal_diagonalisation` questions (easy, medium, hard) and the exam chain
`exd_x_s2b2_orthogonal_diagonalisation`.

``uc2-orthogonal-diagonalisation`` (``linear-algebra``, answer ``matrix``): a real symmetric matrix A has orthonormal
eigenvectors, so A = Q D Qᵀ with Q orthogonal. Q is only unique up to the order and sign of its columns, so every
question states a normal form: columns ordered by increasing eigenvalue, each unit column with a positive first
non-zero entry (with distinct eigenvalues that fixes Q completely).

* easy: a symmetric 2×2 matrix; the matrix Q.
* medium: a symmetric 3×3 matrix; either the unit eigenvector (normal form) for its largest eigenvalue, or QᵀAQ for a
  given orthogonal Q whose columns are eigenvectors in a shuffled order (a diagonal matrix in that order).
* hard: a symmetric 3×3 matrix; the full matrix Q, or the matrix of the orthogonal projection onto the eigenspace of
  the largest eigenvalue, or (for a repeated eigenvalue) onto its two-dimensional eigenspace, I - v vᵀ/|v|².
* exam: a 3×3 symmetric matrix with distinct eigenvalues: (a) the largest, (b) the smallest, (c) the third from the
  trace, (d) the matrix Q.

A is built as Σ λ_k v_k v_kᵀ / |v_k|² from an integer orthogonal triple v_k and integer eigenvalues, kept only when A is
an integer matrix. Every key is re-derived with SymPy (``eigenvals`` and the exact product QᵀAQ).
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, phrase, pick_distractors

from ._s2b2_kit import tidied, fm, mat_str, mmul, number_distractor, sym, transpose

_RATIONAL_TRIPLES = (
    ((1, 2, 2), (2, 1, -2), (2, -2, 1)),
    ((2, 3, 6), (3, -6, 2), (6, 2, -3)),
)
_IRRATIONAL_TRIPLES = (
    ((1, 1, 0), (1, -1, 0), (0, 0, 1)),
    ((1, 1, 1), (1, -1, 0), (1, 1, -2)),
    ((1, 0, 1), (1, 0, -1), (0, 1, 0)),
    ((2, 1, 0), (1, -2, 0), (0, 0, 1)),
)
_PAIRS_2X2 = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2))


def _num(x):
    """An int, `Fraction` or SymPy number as an exact SymPy number."""
    if isinstance(x, Fraction):
        return sympy.Rational(x.numerator, x.denominator)
    return sympy.sympify(x)


def smat(rows):
    """A matrix (any rows of ints, Fractions or SymPy numbers) as a list of rows of exact SymPy numbers."""
    return [[_num(x) for x in row] for row in rows]


def _entry_value(x):
    return exact_value(x) if x.is_Rational else str(x)


def mvalue(m):
    """The matrix as the JSON value the matrix grader reads: ints, {p, q} fractions, or text for a surd."""
    return [[_entry_value(x) for x in row] for row in m]


def _entry_text(x):
    return str(x) if x.is_Rational else str(x).replace("sqrt", "√").replace("**", "^").replace("*", "")


def mtext(m):
    """The canonical spelling of the matrix, as `mathgen.grading.matrix.matrix_text` writes it."""
    return "[" + ", ".join("[" + ", ".join(_entry_text(x) for x in row) + "]" for row in m) + "]"


def matrix_options(candidates, answer):
    """Tagged matrix options from `(matrix, tag)` pairs: none equal to `answer` (entry by entry, exactly) or repeated."""
    answer = smat(answer)
    out, seen = [], set()
    for rows, tag in candidates:
        m = smat(rows)
        text = mtext(m)
        if m == answer or text in seen:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, mvalue(m)))
    return out


def _unit(v):
    """The unit vector along the integer vector `v`, with a positive first non-zero entry, as SymPy numbers."""
    first = next(x for x in v if x != 0)
    sign = 1 if first > 0 else -1
    norm = sympy.sqrt(sum(x * x for x in v))
    return [sympy.Integer(sign * x) / norm for x in v]


def _q_rows(vectors):
    """The matrix with the unit forms of `vectors` as columns, as rows of SymPy numbers."""
    units = [_unit(v) for v in vectors]
    return [[units[c][r] for c in range(len(units))] for r in range(len(units[0]))]


def _sym_q(vectors):
    return sympy.Matrix(_q_rows(vectors))


def _norm2(v):
    return sum(x * x for x in v)


def _build_a(vectors, lams):
    """`Σ λ v vᵀ / |v|²` as a Fraction matrix."""
    n = len(vectors[0])
    a = [[Fraction(0)] * n for _ in range(n)]
    for v, lam in zip(vectors, lams):
        d = _norm2(v)
        for i in range(n):
            for j in range(n):
                a[i][j] += Fraction(lam * v[i] * v[j], d)
    return a


def _is_good(a, cap=12, max_zero=2):
    flat = [x for row in a for x in row]
    return all(x.denominator == 1 for x in flat) and max(abs(x) for x in flat) <= cap and flat.count(0) <= max_zero


def _signed_permuted(rng, triple):
    """The orthogonal triple with its coordinates permuted and signed at random and the vectors in random order."""
    n = len(triple[0])
    perm = list(range(n))
    rng.shuffle(perm)
    signs = [rng.choice([-1, 1]) for _ in range(n)]
    vectors = [tuple(signs[i] * v[perm[i]] for i in range(n)) for v in triple]
    rng.shuffle(vectors)
    return vectors


def _draw_3x3(rng, rational=True, distinct=True, chooser=None):
    """`(vectors, lams)` with vectors an orthogonal triple of integer vectors and lams their integer eigenvalues,
    sorted so lams is increasing; A = Σ λ v vᵀ/|v|² is a dense integer matrix."""
    for _ in attempts(5000, "an orthogonal triple with integer eigenvalues"):
        base = rng.choice(_RATIONAL_TRIPLES if rational else _IRRATIONAL_TRIPLES)
        vectors = _signed_permuted(rng, base)
        lams = rng.sample(range(-6, 10), 3) if distinct else None
        if chooser is not None:
            lams = chooser(rng)
        elif not distinct:
            lam, mu = rng.sample(range(-5, 8), 2)
            lams = [lam, lam, mu]
        pairs = sorted(zip(lams, vectors), key=lambda t: t[0])
        lams = [p[0] for p in pairs]
        vectors = [p[1] for p in pairs]
        if not _is_good(_build_a(vectors, lams)):
            continue
        return vectors, lams
    raise AssertionError  # pragma: no cover


def _draw_2x2(rng, accept=None):
    """`(vectors, lams, A)`: orthogonal integer vectors, the integer eigenvalues sorted ascending, and the integer
    symmetric matrix A = λ_low I + t w wᵀ whose eigenvector w = (p, q) has the larger eigenvalue λ_low + t(p² + q²)."""
    for _ in attempts(5000, "a symmetric 2×2 matrix with integer eigenvalues and orthogonal eigenvectors"):
        p, q = (1, 1) if rng.random() < 0.08 else rng.choice(_PAIRS_2X2[1:])
        q *= rng.choice([-1, 1])
        t = rng.choice([1, 1, 2, 3]) * rng.choice([1, 1, 1, -1])
        lo = rng.randint(-6, 6)
        n = p * p + q * q
        # t > 0: w has the larger eigenvalue; t < 0: w has the smaller one.
        lams = sorted([lo, lo + t * n])
        if accept is not None and not accept(lams):
            continue
        a = [[Fraction(lo * (i == j) + t * w_i * w_j) for j, w_j in enumerate((p, q))] for i, w_i in enumerate((p, q))]
        if max(abs(x) for row in a for x in row) > 12 or a[0][1] == 0:
            continue
        perp = (-q, p)
        vectors = [(p, q), perp] if t < 0 else [perp, (p, q)]
        # With t < 0, w = (p, q) has eigenvalue lo + t n < lo, i.e. the smaller one.
        return vectors, lams, a
    raise AssertionError  # pragma: no cover


def _verify_q(a, vectors, lams):
    """SymPy: A has exactly these eigenvalues and QᵀAQ is the diagonal matrix of them."""
    m, q = sym(a), _sym_q(vectors)
    check(m == m.T, "A is not symmetric")
    check(sorted(m.eigenvals().items()) == sorted((sympy.Integer(l), lams.count(l)) for l in set(lams)), "eigenvalues")
    check((q.T * m * q - sympy.diag(*lams)).applyfunc(sympy.expand) == sympy.zeros(len(lams)), "QᵀAQ is not diagonal")
    check((q.T * q).applyfunc(sympy.expand) == sympy.eye(len(lams)), "Q is not orthogonal")


def _plain_columns(vectors):
    """The vectors themselves (not normalised), as columns, each with a positive first non-zero entry."""
    cols = []
    for v in vectors:
        first = next(x for x in v if x != 0)
        cols.append([x if first > 0 else -x for x in v])
    return [[cols[c][r] for c in range(len(cols))] for r in range(len(cols[0]))]


def _transpose(rows):
    return [list(col) for col in zip(*rows)]


def _q_distractors(vectors, answer):
    n = len(vectors)
    first = list(vectors)
    first[0] = tuple(reversed(first[0])) if n == 2 else (first[0][1], first[0][0], first[0][2])
    scaled = []
    for v in vectors:
        total = sum(v)
        scaled.append([sympy.Rational(x, total) if total else sympy.Integer(x) for x in v])
    candidates = [
        (_q_rows(list(reversed(vectors))), "diag-columns-wrong-order"),
        (_plain_columns(vectors), "eigvec-not-normalised"),
        (_transpose(answer), "diag-vectors-as-rows"),
        (_q_rows(first), "eigvec-components-swapped"),
        ([[scaled[c][r] for c in range(n)] for r in range(n)], "eigvec-normalised-by-sum"),
    ]
    return matrix_options(candidates, answer)


# -- easy ---------------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "The symmetric matrix A = {m} has two distinct eigenvalues. Find the orthogonal matrix Q whose columns are unit "
    "eigenvectors of A, so that QᵀAQ is diagonal. Order the columns by increasing eigenvalue and give each column a "
    "positive first entry.",
    "Orthogonally diagonalise A = {m}: find the orthogonal matrix Q with QᵀAQ diagonal, taking the columns of Q as unit "
    "eigenvectors in order of increasing eigenvalue, each with a positive first entry.",
    "A = {m} is symmetric with two different eigenvalues. Find the matrix Q whose columns are the unit eigenvectors of "
    "A, listed by increasing eigenvalue and each with a positive first entry, so that QᵀAQ is diagonal.",
)


def _easy(rng):
    vectors, lams, a = _draw_2x2(rng)
    _verify_q(a, vectors, lams)
    rows = _q_rows(vectors)
    text = mtext(rows)
    distractors = pick_distractors(text, _q_distractors(vectors, rows), 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _EASY_STEMS, m=mat_str(a)),
        answer_kind="matrix",
        answer=mvalue(rows),
        answer_text=text,
        answer_latex="plain:" + text,
        working=[
            f"The eigenvalues are {lams[0]} and {lams[1]}.",
            f"Unit eigenvectors, in order of increasing eigenvalue, are the columns of Q = {text}.",
        ],
        distractors=distractors,
    )


# -- medium -------------------------------------------------------------------------------------------------------


def _column_text(vector):
    return "(" + ", ".join(str(x) for x in vector) + ")"


_MEDIUM_VECTOR_STEMS = (
    "The symmetric matrix A = {m} has three distinct eigenvalues. Find the unit eigenvector for the largest "
    "eigenvalue, with a positive first non-zero entry.",
    "Find the unit eigenvector of the symmetric matrix A = {m} belonging to its largest eigenvalue. Give it with a "
    "positive first non-zero entry.",
)
_MEDIUM_PRODUCT_STEMS = (
    "The columns of the orthogonal matrix Q = {q} are eigenvectors of the symmetric matrix A = {m}. Find QᵀAQ.",
    "Let A = {m} and Q = {q}. The columns of Q are eigenvectors of A. Calculate the diagonal matrix QᵀAQ.",
)


def _q_with_denominator(vectors):
    d = int(sympy.sqrt(_norm2(vectors[0])))
    return f"(1/{d})·{mat_str(fm(transpose([list(v) for v in vectors])))}", d


def _medium_vector(rng):
    vectors, lams = _draw_3x3(rng, rational=rng.random() < 0.7)
    a = _build_a(vectors, lams)
    _verify_q(a, vectors, lams)
    top = vectors[2]
    rows = [[x] for x in _unit(top)]
    total = sum(top)
    candidates = [
        (_q_rows([vectors[0]]), "eigvec-other-eigenvalue"),
        (_q_rows([vectors[1]]), "eigvec-other-eigenvalue"),
        ([[x] for x in top], "eigvec-not-normalised"),
        ([[sympy.Rational(x, total) if total else x] for x in top], "eigvec-normalised-by-sum"),
    ]
    text = mtext(rows)
    distractors = pick_distractors(text, matrix_options(candidates, rows), 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _MEDIUM_VECTOR_STEMS, m=mat_str(a)),
        answer_kind="matrix",
        answer=mvalue(rows),
        answer_text=text,
        answer_latex="plain:" + text,
        working=[
            f"The eigenvalues are {lams[0]}, {lams[1]} and {lams[2]}, so the largest is {lams[2]}.",
            f"Solving (A - {lams[2]}I)v = 0 gives v along {_column_text(top)}; dividing by its length gives {text}.",
        ],
        distractors=distractors,
    )


def _medium_product(rng):
    vectors, lams = _draw_3x3(rng, rational=True)
    a = _build_a(vectors, lams)
    order = rng.choice([[1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [2, 0, 1]])
    shown = [vectors[i] for i in order]
    shown_lams = [lams[i] for i in order]
    _verify_q(a, shown, shown_lams)
    q_text, d = _q_with_denominator(shown)
    diag_rows = [[shown_lams[i] if i == j else 0 for j in range(3)] for i in range(3)]
    q = fm([[Fraction(shown[c][r], d) for c in range(3)] for r in range(3)])
    qt = transpose(q)
    sorted_rows = [[lams[i] if i == j else 0 for j in range(3)] for i in range(3)]
    candidates = [
        (sorted_rows, "diag-eigenvalues-swapped"),
        (mmul(mmul(q, a), qt), "diag-conjugation-reversed"),
        (mmul(qt, a), "diag-forgot-inverse"),
        (a, "diag-forgot-change-of-basis"),
    ]
    text = mtext(smat(diag_rows))
    distractors = pick_distractors(text, matrix_options(candidates, diag_rows), 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _MEDIUM_PRODUCT_STEMS, m=mat_str(a), q=q_text),
        answer_kind="matrix",
        answer=mvalue(smat(diag_rows)),
        answer_text=text,
        answer_latex="plain:" + text,
        working=[
            "Each column q of Q is an eigenvector, so its eigenvalue is qᵀAq, and QᵀAQ is diagonal.",
            f"The eigenvalues in the order of the columns of Q are {shown_lams[0]}, {shown_lams[1]}, {shown_lams[2]}.",
        ],
        distractors=distractors,
    )


def _medium(rng):
    return rng.choice([_medium_vector, _medium_product])(rng)


# -- hard ---------------------------------------------------------------------------------------------------------

_HARD_Q_STEMS = (
    "The symmetric matrix A = {m} has three distinct eigenvalues. Find the orthogonal matrix Q with QᵀAQ diagonal, whose "
    "columns are unit eigenvectors in order of increasing eigenvalue, each with a positive first non-zero entry.",
    "Orthogonally diagonalise the symmetric matrix A = {m}: give the matrix Q whose columns are unit eigenvectors in "
    "order of increasing eigenvalue, each with a positive first non-zero entry.",
)
_HARD_PROJ_STEMS = (
    "The symmetric matrix A = {m} has three distinct eigenvalues. Find the matrix of the orthogonal projection of ℝ³ "
    "onto the eigenspace of its largest eigenvalue.",
    "Find the 3×3 matrix P of the orthogonal projection onto the eigenspace of the largest eigenvalue of A = {m}.",
)
_HARD_REPEATED_STEMS = (
    "The symmetric matrix A = {m} has a repeated eigenvalue. Find the matrix of the orthogonal projection of ℝ³ onto "
    "that two-dimensional eigenspace.",
    "A = {m} is symmetric and has an eigenvalue of multiplicity two. Find the matrix of the orthogonal projection onto "
    "the eigenspace of that eigenvalue.",
)


def _line_projection(v):
    """v vᵀ / (vᵀv) as SymPy numbers."""
    d = _norm2(v)
    return [[sympy.Rational(v[i] * v[j], d) for j in range(3)] for i in range(3)]


def _hard_q(rng):
    vectors, lams = _draw_3x3(rng, rational=rng.random() < 0.5)
    a = _build_a(vectors, lams)
    _verify_q(a, vectors, lams)
    rows = _q_rows(vectors)
    text = mtext(rows)
    distractors = pick_distractors(text, _q_distractors(vectors, rows), 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _HARD_Q_STEMS, m=mat_str(a)),
        answer_kind="matrix",
        answer=mvalue(rows),
        answer_text=text,
        answer_latex="plain:" + text,
        working=[
            f"The eigenvalues are {lams[0]}, {lams[1]}, {lams[2]}; the eigenvectors are along "
            + ", ".join(_column_text(v) for v in vectors)
            + ".",
            f"Normalising and placing them in columns gives Q = {text}.",
        ],
        distractors=distractors,
    )


def _hard_projection(rng):
    vectors, lams = _draw_3x3(rng, rational=rng.random() < 0.5)
    a = _build_a(vectors, lams)
    _verify_q(a, vectors, lams)
    top = vectors[2]
    rows = _line_projection(top)
    # SymPy: u uᵀ for the unit eigenvector.
    u = sympy.Matrix(_unit(top))
    check((u * u.T - sympy.Matrix(rows)).applyfunc(sympy.expand) == sympy.zeros(3), "projection differs")
    ident = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
    candidates = [
        ([[top[i] * top[j] for j in range(3)] for i in range(3)], "projection-not-normalised"),
        ([[ident[i][j] - rows[i][j] for j in range(3)] for i in range(3)], "projection-complement-confused"),
        ([[lams[2] * rows[i][j] for j in range(3)] for i in range(3)], "projection-scaled-by-eigenvalue"),
        (_line_projection(vectors[0]), "eigvec-other-eigenvalue"),
    ]
    text = mtext(rows)
    distractors = pick_distractors(text, matrix_options(candidates, rows), 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _HARD_PROJ_STEMS, m=mat_str(a)),
        answer_kind="matrix",
        answer=mvalue(smat(rows)),
        answer_text=text,
        answer_latex="plain:" + text,
        working=[
            f"The largest eigenvalue is {lams[2]}, with eigenvector along {_column_text(top)}.",
            f"The projection onto the line spanned by v is v vᵀ/(vᵀv), which is {text}.",
        ],
        distractors=distractors,
    )


def _hard_repeated(rng):
    vectors, lams = _draw_3x3(rng, rational=rng.random() < 0.5, distinct=False)
    a = _build_a(vectors, lams)
    simple = next(v for lam, v in zip(lams, vectors) if lams.count(lam) == 1)
    repeated = next(lam for lam in lams if lams.count(lam) == 2)
    line = _line_projection(simple)
    ident = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
    rows = [[ident[i][j] - line[i][j] for j in range(3)] for i in range(3)]
    m = sym(a)
    check((m - repeated * sympy.eye(3)).rank() == 1, "the repeated eigenvalue's eigenspace is not a plane")
    p = sympy.Matrix(rows)
    check((m * p - repeated * p).applyfunc(sympy.expand) == sympy.zeros(3), "A P != λ P")
    check(p * p == p and p == p.T and p.trace() == 2, "P is not a rank-2 orthogonal projection")
    candidates = [
        (line, "projection-complement-confused"),
        ([[ident[i][j] - simple[i] * simple[j] for j in range(3)] for i in range(3)], "projection-not-normalised"),
        ([[repeated * rows[i][j] for j in range(3)] for i in range(3)], "projection-scaled-by-eigenvalue"),
    ]
    text = mtext(rows)
    distractors = pick_distractors(text, matrix_options(candidates, rows), 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _HARD_REPEATED_STEMS, m=mat_str(a)),
        answer_kind="matrix",
        answer=mvalue(rows),
        answer_text=text,
        answer_latex="plain:" + text,
        working=[
            f"The eigenvalue {repeated} has a two-dimensional eigenspace, the plane orthogonal to the eigenvector "
            f"along {_column_text(simple)} of the other eigenvalue.",
            f"The projection onto that plane is I - v vᵀ/(vᵀv) = {text}.",
        ],
        distractors=distractors,
    )


def _hard(rng):
    return rng.choice([_hard_q, _hard_projection, _hard_repeated])(rng)


@generator(
    "s2b2_orthogonal_diagonalisation",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@tidied
def s2b2_orthogonal_diagonalisation(rng, tier="easy"):
    """Orthogonal diagonalisation of a real symmetric matrix; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_s2b2_orthogonal_diagonalisation", level=19, presets={"exam": dict(tier="exam")})
@tidied
def exd_x_s2b2_orthogonal_diagonalisation(rng, tier="exam"):
    """Exam chain: the largest, smallest and (from the trace) middle eigenvalue of a symmetric 3×3 matrix, then Q."""
    check(tier == "exam", f"unknown tier {tier!r}")
    vectors, lams = _draw_3x3(rng, rational=rng.random() < 0.6)
    a = _build_a(vectors, lams)
    _verify_q(a, vectors, lams)
    lo, mid, hi = lams
    trace = int(sum(a[i][i] for i in range(3)))
    check(trace == lo + mid + hi, "trace differs from the eigenvalue sum")
    rows = _q_rows(vectors)
    q_text = mtext(rows)
    diag_entries = [int(a[i][i]) for i in range(3)]
    a_dist = pick_distractors(
        str(hi),
        [
            number_distractor(mid, "eigval-other-root"),
            number_distractor(lo, "eigval-other-root"),
            number_distractor(-hi, "sign-flip-both-roots"),
            number_distractor(max(diag_entries), "eigval-diagonal-entry"),
        ],
        3,
    )
    b_dist = pick_distractors(
        str(lo),
        [
            number_distractor(mid, "eigval-other-root"),
            number_distractor(hi, "eigval-other-root"),
            number_distractor(-lo, "sign-flip-both-roots"),
            number_distractor(min(diag_entries), "eigval-diagonal-entry"),
        ],
        3,
    )
    c_dist = pick_distractors(
        str(mid),
        [
            number_distractor(trace + hi + lo, "eigval-trace-difference-sign"),
            number_distractor(trace, "eigval-trace-as-eigenvalue"),
            number_distractor(trace - hi, "eigval-trace-single-count"),
            number_distractor(hi - lo, "eigval-other-root"),
        ],
        3,
    )
    d_dist = pick_distractors(q_text, _q_distractors(vectors, rows), 3)
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")
    return dict(
        text=phrase(
            rng,
            (
                "The symmetric matrix A = {m} has three distinct eigenvalues.",
                "Let A = {m}, a real symmetric matrix with three distinct eigenvalues.",
                "A = {m} is symmetric and has three different eigenvalues.",
            ),
            m=mat_str(a),
        ),
        parts=[
            dict(
                prompt="Find the largest eigenvalue of A.",
                answer_kind="number",
                answer=hi,
                answer_text=str(hi),
                marks=2,
                working=[f"det(A - λI) = 0 has roots {lo}, {mid} and {hi}; the largest is {hi}."],
                distractors=a_dist,
                sensible_range=(-30, 30),
            ),
            dict(
                prompt="Find the smallest eigenvalue of A.",
                answer_kind="number",
                answer=lo,
                answer_text=str(lo),
                marks=2,
                working=[f"The smallest root of the characteristic polynomial is {lo}."],
                distractors=b_dist,
                sensible_range=(-30, 30),
            ),
            dict(
                prompt="Use the trace of A to find the remaining eigenvalue.",
                answer_kind="number",
                answer=mid,
                answer_text=str(mid),
                marks=1,
                follow_through=f"{trace} - part_a - part_b",
                working=[
                    f"The eigenvalues add up to the trace {trace}, so the third is {trace} - ({hi}) - ({lo}) = {mid}."
                ],
                distractors=c_dist,
            ),
            dict(
                prompt="Find the orthogonal matrix Q with QᵀAQ diagonal, whose columns are unit eigenvectors in order of "
                "increasing eigenvalue, each with a positive first non-zero entry.",
                answer_kind="matrix",
                answer=mvalue(rows),
                answer_text=q_text,
                answer_latex="plain:" + q_text,
                marks=3,
                working=[f"Solving (A - λI)v = 0 for each eigenvalue and normalising gives Q = {q_text}."],
                distractors=d_dist,
            ),
        ],
    )
