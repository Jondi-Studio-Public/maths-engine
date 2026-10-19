"""Generates `w6c_hermitian` questions (easy, medium, hard) and the exam chain `exd_x_w6c_hermitian`.

``uc2-hermitian-unitary-matrices`` (``linear-algebra``): complex 2×2 matrices with Gaussian-integer entries, the
conjugate transpose A* = Āᵀ, Hermitian matrices (H* = H, so real eigenvalues) and unitary matrices (U*U = I).

* easy: conjugate-transpose arithmetic. For a 2×2 complex matrix A, the real or imaginary part of an entry of A*A, or
  tr(A*A) = the sum of the squared moduli of the entries.
* medium: an eigenvalue of a 2×2 Hermitian matrix [[a, b + ci], [b - ci, d]], which is a root of
  λ² - (a + d)λ + (ad - b² - c²) = 0. The numbers are chosen so that the eigenvalues are integers: the larger or the
  smaller eigenvalue, or the gap between them.
* hard: one of the normalising constant N that makes (1/N)[[p, -q̄λ̄...]] unitary (the entries are drawn as a unitary
  matrix scaled by N), the real entry x that completes a unitary matrix (the norm condition alone leaves the sign
  open and orthogonality fixes it), or the real parameter k > 0 for which a Hermitian matrix with k in its off-diagonal
  entry has a stated eigenvalue.
* exam (``exd_x_w6c_hermitian``): a 2×2 Hermitian matrix. (a) the imaginary part of the (1, 2) entry of H*H, (b) the smaller eigenvalue, (c) the larger
  eigenvalue (follows through from (b), since the eigenvalues sum to the trace), (d) tr(H*H) (follows through from (b)
  and (c), since tr(H*H) = λ₁² + λ₂²).

Each key is re-derived with SymPy: ``Matrix.H`` (the conjugate transpose), ``eigenvals`` and ``det``, and every
unitary matrix is checked by U*U = I.
"""

from __future__ import annotations

import functools
from fractions import Fraction
from math import isqrt

import sympy

from ..core import attempts, check, generator, join_terms, paren_negative, phrase
from ._w6c_kit import Redraw, redrawn, value_fields, value_part

# -- Gaussian integers as (re, im) pairs ---------------------------------------------------------------------------


def g_mul(x, y):
    return (x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0])


def g_add(x, y):
    return (x[0] + y[0], x[1] + y[1])


def g_conj(x):
    return (x[0], -x[1])


def g_norm2(x):
    return x[0] * x[0] + x[1] * x[1]


def g_text(x) -> str:
    """A Gaussian integer as a learner writes it: ``3``, ``-i``, ``4i``, ``2 - 3i``."""
    re_, im = x
    if im == 0:
        return str(re_)
    mag = "i" if abs(im) == 1 else f"{abs(im)}i"
    if re_ == 0:
        return mag if im > 0 else f"-{mag}"
    return f"{re_} {'+' if im > 0 else '-'} {mag}"


def g_paren(x) -> str:
    """`g_text`, bracketed when it has two terms or a leading minus so it can sit beside another factor."""
    text = g_text(x)
    return f"({text})" if " " in text or text.startswith("-") else text


def m_text(rows) -> str:
    return "[" + ", ".join("[" + ", ".join(g_text(v) for v in row) + "]" for row in rows) + "]"


def m_star(a):
    """The conjugate transpose of a 2-D list of Gaussian integers."""
    return [[g_conj(a[j][i]) for j in range(len(a))] for i in range(len(a[0]))]


def m_mul(a, b):
    rows, inner, cols = len(a), len(b), len(b[0])
    out = []
    for i in range(rows):
        row = []
        for j in range(cols):
            total = (0, 0)
            for k in range(inner):
                total = g_add(total, g_mul(a[i][k], b[k][j]))
            row.append(total)
        out.append(row)
    return out


def sym(a):
    return sympy.Matrix([[v[0] + v[1] * sympy.I for v in row] for row in a])


UNITS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


# -- easy -----------------------------------------------------------------------------------------------------------

_EASY_REAL_STEMS = (
    "Let A = {m} and let A* be its conjugate transpose. Find the real part of the ({i}, {j}) entry of A*A.",
    "For A = {m}, find the real part of the ({i}, {j}) entry of the product A*A, where A* is the conjugate transpose of A.",
)
_EASY_IMAG_STEMS = (
    "Let A = {m} and let A* be its conjugate transpose. Find the imaginary part of the ({i}, {j}) entry of A*A.",
    "For A = {m}, find the imaginary part of the ({i}, {j}) entry of the product A*A, where A* is the conjugate transpose of A.",
)
_EASY_TRACE_STEMS = (
    "Find tr(A*A) for A = {m}, where A* is the conjugate transpose of A.",
    "A* denotes the conjugate transpose of A = {m}. Evaluate the trace of A*A.",
)


def _complex_matrix(rng, bound=3):
    for _ in attempts(500, "a complex 2×2 matrix with several complex entries"):
        a = [[(rng.randint(-bound, bound), rng.randint(-bound, bound)) for _ in range(2)] for _ in range(2)]
        if sum(1 for row in a for v in row if v[1] != 0) < 2 or sum(1 for row in a for v in row if v == (0, 0)) > 1:
            continue
        return a
    raise Redraw("no matrix")  # pragma: no cover


@redrawn
def _easy(rng):
    a = _complex_matrix(rng)
    gram = m_mul(m_star(a), a)
    check(sym(gram) == (sym(a).H * sym(a)).expand(), "SymPy's A*A differs")
    plain = m_mul([[a[j][i] for j in range(2)] for i in range(2)], a)  # AᵀA: the conjugate forgotten
    reversed_product = m_mul(a, m_star(a))  # AA*: the order reversed
    shape = rng.choice(["re", "im", "trace"])
    if shape == "trace":
        value = gram[0][0][0] + gram[1][1][0]
        check(sympy.trace(sym(gram)) == value, "SymPy's trace differs")
        check(value == sum(g_norm2(v) for row in a for v in row), "tr(A*A) is not the sum of squared moduli")
        wrong = [
            ("herm-modulus-as-square", sum(g_mul(v, v)[0] for row in a for v in row)),
            ("herm-real-parts-only", sum(v[0] * v[0] for row in a for v in row)),
            ("herm-imaginary-parts-only", sum(v[1] * v[1] for row in a for v in row)),
            ("herm-conjugate-forgotten", plain[0][0][0] + plain[1][1][0]),
        ]
        working = [
            f"tr(A*A) is the sum of the squared moduli of the entries of A: {' + '.join(str(g_norm2(v)) for row in a for v in row)} = {value}."
        ]
        return dict(text=phrase(rng, _EASY_TRACE_STEMS, m=m_text(a)), working=working, **value_fields(value, wrong))
    i, j = rng.choice([(1, 2), (2, 1), (1, 2), (2, 1), (1, 1), (2, 2)] if shape == "re" else [(1, 2), (2, 1)])
    entry = gram[i - 1][j - 1]
    part = 0 if shape == "re" else 1
    value = entry[part]
    check(value != 0 or shape == "re", "zero imaginary part")
    other = gram[j - 1][i - 1]
    wrong = [
        ("herm-conjugate-forgotten", plain[i - 1][j - 1][part]),
        ("herm-product-order-reversed", reversed_product[i - 1][j - 1][part]),
        ("herm-entry-transposed", other[part]),
        ("herm-imaginary-sign", -value),
    ]
    stems = _EASY_REAL_STEMS if shape == "re" else _EASY_IMAG_STEMS
    return dict(
        text=phrase(rng, stems, m=m_text(a), i=i, j=j),
        working=[
            f"A* = {m_text(m_star(a))}.",
            f"The ({i}, {j}) entry of A*A is {g_text(entry)}.",
        ],
        **value_fields(value, wrong),
    )


# -- medium ---------------------------------------------------------------------------------------------------------


@functools.cache
def _hermitian_pool(wide=False):
    """``(a, d, b, c, D)``: [[a, b + ci], [b - ci, d]] with c != 0, a + d even and integer eigenvalues (a + d)/2 ± D."""
    pool = []
    lo, hi, bound, cmax = (-9, 12, 8, 9) if wide else (-5, 8, 5, 6)
    for a in range(lo, hi):
        for d in range(lo, hi):
            if (a + d) % 2 or a == d:
                continue
            for b in range(-bound, bound + 1):
                for c in range(1, cmax + 1):
                    inside = ((a - d) // 2) ** 2 + b * b + c * c
                    root = isqrt(inside)
                    if root * root == inside:
                        pool.append((a, d, b, c, root))
    return pool


def _hermitian(rng, wide=False):
    a, d, b, c, root = rng.choice(_hermitian_pool(wide))
    if rng.random() < 0.5:
        c = -c
    h = [[(a, 0), (b, c)], [(b, -c), (d, 0)]]
    check(sym(h) == sym(h).H, "the matrix is not Hermitian")
    lo, hi = (a + d) // 2 - root, (a + d) // 2 + root
    check(sorted(sym(h).eigenvals()) == [lo, hi], "SymPy's eigenvalues differ")
    return h, a, d, b, c, root, lo, hi


_MEDIUM_LARGE_STEMS = (
    "Find the larger eigenvalue of the Hermitian matrix H = {m}.",
    "The matrix H = {m} is Hermitian, so its eigenvalues are real. What is the largest one?",
)
_MEDIUM_SMALL_STEMS = (
    "Find the smaller eigenvalue of the Hermitian matrix H = {m}.",
    "The matrix H = {m} is Hermitian, so its eigenvalues are real. What is the smallest one?",
)
_MEDIUM_GAP_STEMS = (
    "Find the difference between the larger and smaller eigenvalues of the Hermitian matrix H = {m}.",
    "How far apart are the two (real) eigenvalues of H = {m}?",
)


@redrawn
def _medium(rng):
    h, a, d, b, c, root, lo, hi = _hermitian(rng)
    shape = rng.choice(["hi", "lo", "gap"])
    value = {"hi": hi, "lo": lo, "gap": hi - lo}[shape]
    wrong = {
        "hi": [
            ("herm-diagonal-entry-as-eigenvalue", max(a, d)),
            ("herm-wrong-ordering", lo),
            ("herm-radius-as-eigenvalue", root),
            ("herm-mean-as-eigenvalue", (a + d) // 2),
        ],
        "lo": [
            ("herm-diagonal-entry-as-eigenvalue", min(a, d)),
            ("herm-wrong-ordering", hi),
            ("herm-radius-as-eigenvalue", -root),
            ("herm-mean-as-eigenvalue", (a + d) // 2),
        ],
        "gap": [
            ("herm-radius-as-eigenvalue", root),
            ("herm-diagonal-entry-as-eigenvalue", abs(a - d)),
            ("herm-sum-instead-of-gap", hi + lo),
        ],
    }[shape]
    stems = {"hi": _MEDIUM_LARGE_STEMS, "lo": _MEDIUM_SMALL_STEMS, "gap": _MEDIUM_GAP_STEMS}[shape]
    return dict(
        text=phrase(rng, stems, m=m_text(h)),
        working=[
            f"The characteristic polynomial is {join_terms([(1, 'λ²'), (-(a + d), 'λ')])} + ({a * d} - {b * b} - {c * c}) = {join_terms([(1, 'λ²'), (-(a + d), 'λ'), (a * d - b * b - c * c, '')])}, since (b + ci)(b - ci) = b² + c².",
            f"Its roots are {lo} and {hi}.",
        ],
        **value_fields(value, wrong),
    )


# -- hard -----------------------------------------------------------------------------------------------------------


@functools.cache
def _unitary_pool():
    """``(p, q, N)``: Gaussian integers p, q with |p|² + |q|² = N², N a whole number."""
    pool = []
    span = range(-4, 5)
    for p in [(x, y) for x in span for y in span]:
        for q in [(x, y) for x in span for y in span]:
            total = g_norm2(p) + g_norm2(q)
            root = isqrt(total)
            if root * root == total and 2 <= root <= 9 and q != (0, 0) and p != (0, 0):
                pool.append((p, q, root))
    return pool


def _scaled_unitary(rng):
    p, q, n = rng.choice(_unitary_pool())
    unit = rng.choice(UNITS)
    top_right = g_mul(unit, g_conj(q))
    m = [[p, (-top_right[0], -top_right[1])], [q, g_mul(unit, g_conj(p))]]
    u = sym(m) / n
    check((u.H * u).expand() == sympy.eye(2), "the matrix is not unitary")
    return m, p, q, n, unit


_UNITARY_N_STEMS = (
    "The matrix U = (1/N){m} is unitary, where N is a positive real number. Find N.",
    "For what positive real N is U = (1/N){m} a unitary matrix?",
)
_UNITARY_X_STEMS = (
    "The matrix U = (1/{n}){m} is unitary for a real number x. Find x.",
    "Find the real number x for which U = (1/{n}){m} is unitary.",
)
_EIGENVALUE_STEMS = (
    "The matrix H = {m} is Hermitian for a real number k > 0, and {lam} is one of its eigenvalues. Find k.",
    "H = {m}, where k is a positive real number. Given that {lam} is an eigenvalue of H, find k.",
)


@redrawn
def _hard_normaliser(rng):
    m, p, q, n, _unit = _scaled_unitary(rng)
    total = g_norm2(p) + g_norm2(q)
    check(total == n * n, "N² differs from the column norm")
    wrong = [
        ("herm-norm-squared-not-root", n * n),
        ("herm-imaginary-parts-ignored", p[0] ** 2 + q[0] ** 2),
        ("herm-square-instead-of-modulus-squared", g_mul(p, p)[0] + g_mul(q, q)[0]),
        ("herm-both-columns-summed", 2 * n * n),
    ]
    wrong = [(tag, v) for tag, v in wrong if v > 0]
    return dict(
        text=phrase(rng, _UNITARY_N_STEMS, m=m_text(m)),
        working=[
            f"U*U = I needs each column to have length 1, so N² = |{g_text(p)}|² + |{g_text(q)}|² = {g_norm2(p)} + {g_norm2(q)} = {total}.",
            f"Hence N = {n}.",
        ],
        **value_fields(n, wrong),
    )


@redrawn
def _hard_missing_entry(rng):
    pool = [t for t in _unitary_pool() if t[0][1] == 0 and t[0][0] > 0]
    p, q, n = rng.choice(pool)
    x = p[0]
    shown = [[p, (-q[0], q[1])], [q, None]]  # the second column is (-q̄, p̄) with p real
    rows = [[g_text(v) for v in shown[0]], [g_text(shown[1][0]), "x"]]
    matrix = "[" + ", ".join("[" + ", ".join(r) + "]" for r in rows) + "]"
    full = [[p, (-q[0], q[1])], [q, (x, 0)]]
    u = sym(full) / n
    check((u.H * u).expand() == sympy.eye(2), "the completed matrix is not unitary")
    wrong_signed = sym([[p, (-q[0], q[1])], [q, (-x, 0)]]) / n
    check((wrong_signed.H * wrong_signed).expand() != sympy.eye(2), "the other sign is also unitary")
    wrong = [
        ("herm-sign-from-norm-only", -x),
        ("herm-entry-squared", x * x),
        ("herm-modulus-squared-as-entry", g_norm2(q)),
        ("herm-normaliser-as-entry", n),
    ]
    return dict(
        text=phrase(rng, _UNITARY_X_STEMS, m=matrix, n=n),
        working=[
            f"The columns must be orthogonal. (conjugate transpose of column 1) × (column 2) = {g_paren(g_conj(p))}·{g_paren(shown[0][1])} + {g_paren(g_conj(q))}·x = 0 gives x = {x}.",
            f"The length condition alone only gives x² = {x * x}, so x = ±{x}; orthogonality picks x = {x}.",
        ],
        **value_fields(x, wrong),
    )


@redrawn
def _hard_parameter(rng):
    pool = _eigenvalue_pool()
    a, d, b, c, lam = rng.choice(pool)
    k = sympy.symbols("k", positive=True)
    h = sympy.Matrix([[a, b + k * sympy.I], [b - k * sympy.I, d]])
    check(sympy.solve(sympy.expand((h - lam * sympy.eye(2)).det()), k) == [c], "SymPy's k differs")
    top = f"{b} + ki" if b > 0 else ("ki" if b == 0 else f"-{-b} + ki")
    bottom = f"{b} - ki" if b > 0 else ("-ki" if b == 0 else f"-{-b} - ki")
    matrix = f"[[{a}, {top}], [{bottom}, {d}]]"
    wrong = [
        ("herm-modulus-squared-unrooted", c * c),
        ("herm-radicand-with-offdiagonal", (a - lam) * (d - lam)),
        ("herm-offdiagonal-real-part-added", c + abs(b)),
        ("herm-offdiagonal-moduli-mixed", abs(b) + c * c),
    ]
    return dict(
        text=phrase(rng, _EIGENVALUE_STEMS, m=matrix, lam=lam),
        working=[
            f"det(H - {paren_negative(lam)}I) = ({a} - {paren_negative(lam)})({d} - {paren_negative(lam)}) - (b + ki)(b - ki) = {(a - lam) * (d - lam)} - {b * b} - k² = 0.",
            f"So k² = {c * c} and, with k > 0, k = {c}.",
        ],
        **value_fields(c, wrong),
    )


@functools.cache
def _eigenvalue_pool():
    pool = []
    for a in range(-3, 7):
        for d in range(-3, 7):
            if a == d:
                continue
            for lam in range(-4, 9):
                for b in range(-3, 4):
                    inside = (a - lam) * (d - lam) - b * b
                    if inside > 0 and isqrt(inside) ** 2 == inside and (a - lam) * (d - lam) > 0:
                        pool.append((a, d, b, isqrt(inside), lam))
    return pool


@generator(
    "w6c_hermitian",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w6c_hermitian(rng, tier="easy"):
    """Conjugate transposes, Hermitian eigenvalues and unitary matrices; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if not (tier == "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    return rng.choice([_hard_normaliser, _hard_missing_entry, _hard_parameter])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_w6c_hermitian", level=19, presets={"exam": dict(tier="exam")})
@redrawn
def exd_x_w6c_hermitian(rng, tier="exam"):
    """Exam chain on a 2×2 Hermitian matrix: det H, the smaller eigenvalue, the larger one, tr(H*H)."""
    if not (tier == "exam"):
        raise ValueError(f"unknown tier {tier!r}")
    h, a, d, b, c, root, lo, hi = _hermitian(rng, wide=True)
    check(lo != 0 and hi != 0 and lo != hi, "an eigenvalue is zero")
    det = a * d - b * b - c * c
    check(sympy.expand(sym(h).det()) == det == lo * hi, "SymPy's determinant differs")
    star_h = m_mul(m_star(h), h)
    trace_hh = star_h[0][0][0] + star_h[1][1][0]
    check(trace_hh == lo * lo + hi * hi, "tr(H*H) differs from the sum of squared eigenvalues")
    trace = a + d
    entry12 = star_h[0][1]
    check(entry12 == ((trace * b), (trace * c)), "the (1, 2) entry of H*H differs from (a + d)(b + ci)")
    check(sympy.expand((sym(h).H * sym(h))[0, 1]) == entry12[0] + entry12[1] * sympy.I, "SymPy's (1, 2) entry differs")
    plain12 = m_mul([[h[j][i] for j in range(2)] for i in range(2)], h)[0][1]  # HᵀH: the conjugate forgotten
    return dict(
        text=f"The matrix H = {m_text(h)} is Hermitian.",
        parts=[
            value_part(
                "Find the imaginary part of the (1, 2) entry of H*H, where H* is the conjugate transpose of H.",
                entry12[1],
                2,
                [
                    f"Since H* = H, the (1, 2) entry of H*H is {a}({g_text(h[0][1])}) + ({g_text(h[0][1])})({d}) = {g_text(entry12)}."
                ],
                [
                    ("herm-real-part-instead-of-imaginary", entry12[0]),
                    ("herm-entry-transposed", -entry12[1]),
                    ("herm-conjugate-forgotten", plain12[1]),
                    ("herm-imaginary-sign", c * (a - d)),
                ],
            ),
            value_part(
                "Find the smaller eigenvalue of H.",
                lo,
                3,
                [f"{join_terms([(1, 'λ²'), (-trace, 'λ'), (det, '')])} = 0 has roots {lo} and {hi}."],
                [
                    ("herm-diagonal-entry-as-eigenvalue", min(a, d)),
                    ("herm-wrong-ordering", hi),
                    ("herm-radius-as-eigenvalue", -root),
                    ("herm-mean-as-eigenvalue", trace // 2),
                ],
            ),
            value_part(
                "Hence find the larger eigenvalue of H.",
                hi,
                1,
                [f"The eigenvalues sum to the trace {trace}, so the larger one is {trace} - ({lo}) = {hi}."],
                [
                    ("herm-wrong-ordering", lo),
                    ("herm-diagonal-entry-as-eigenvalue", max(a, d)),
                    ("herm-radius-as-eigenvalue", root),
                    ("herm-sum-instead-of-gap", trace),
                ],
                follow_through=f"{trace} - part_b",
            ),
            value_part(
                "Find tr(H*H), where H* is the conjugate transpose of H.",
                trace_hh,
                2,
                [f"Since H* = H, tr(H*H) = tr(H²) = λ₁² + λ₂² = {lo * lo} + {hi * hi} = {trace_hh}."],
                [
                    ("herm-square-of-sum", trace * trace),
                    ("herm-product-of-eigenvalues", lo * hi),
                    ("herm-eigenvalue-sum-of-powers", lo + hi),
                    ("herm-conjugate-forgotten", a * a + d * d + 2 * (b * b - c * c)),
                ],
                follow_through="part_b**2 + part_c**2",
            ),
        ],
    )
