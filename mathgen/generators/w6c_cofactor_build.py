"""Generates `w6c_cofactor` questions (easy, medium, hard) and the exam chain `exd_x_w6c_cofactor`.

``uc2-cofactor-expansion-4x4`` (``linear-algebra``): determinants of 4×4 integer matrices by cofactor expansion and by
row operations, and the rules for the determinant of a product, inverse, transpose and scalar multiple.

* easy: the determinant of a 4×4 matrix with a row or column that has a single non-zero entry, and a second line
  with a zero, so one expansion along the sparse line is a single 3×3 determinant.
* medium: either the determinant of a dense 4×4 matrix built as a row-permuted L·U (so row operations reduce it to
  triangular form and the row swap changes the sign), or one cofactor Cᵢⱼ of a dense 4×4 matrix.
* hard: one of the determinant of an expression in two 4×4 matrices with det A and det B given (scalar multiples,
  inverses, transposes, powers, the adjugate), the determinant after a stated sequence of row operations, or the value
  of k that makes a 4×4 matrix with k in one entry singular.
* exam (``exd_x_w6c_cofactor``): a dense 4×4 matrix. (a) det A, (b) a cofactor Cᵢⱼ, (c) the entry (A⁻¹)ⱼᵢ = Cᵢⱼ/det A
  (follows through from (a) and (b)), (d) det(3A⁻¹) = 81/det A (follows through from (a)).

Each determinant is computed by exact elimination in `Fraction`s and re-derived with SymPy's ``det``, ``cofactor`` and
``inv``; the singular matrices for the parameter question are built with a row that is an integer combination of the
others, so the value of k is known by construction and checked by SymPy's ``det``.
"""

from __future__ import annotations

import functools
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, phrase
from ._ucv1_linalg import det, fm, mat_str
from ._w6c_kit import Redraw, redrawn, value_fields, value_part

N = 4


def _sym(a):
    return sympy.Matrix(a)


def _minor(a, i, j):
    return [[a[r][c] for c in range(N) if c != j] for r in range(N) if r != i]


def _cofactor(a, i, j):
    """The cofactor Cᵢⱼ (0-based i, j) by exact elimination of the 3×3 minor."""
    return (-1) ** (i + j) * int(det(fm(_minor(a, i, j))))


def _det(a):
    return int(det(fm(a)))


def _verify_det(a, value):
    check(_sym(a).det() == value, "SymPy's determinant differs")


def _verify_cofactor(a, i, j, value):
    check(_sym(a).cofactor(i, j) == value, "SymPy's cofactor differs")


def _lu_matrix(rng, diag_choices=(1, 2, 3, -1, -2, -3), bound=9):
    """``(A, d)``: a dense integer 4×4 matrix with determinant d, built as a row permutation of L·U."""
    for _ in attempts(3000, "a dense 4×4 matrix of small entries"):
        diag = [rng.choice(diag_choices) for _ in range(N)]
        upper = [[diag[i] if i == j else (rng.randint(-2, 2) if j > i else 0) for j in range(N)] for i in range(N)]
        lower = [[1 if i == j else (rng.randint(-2, 2) if j < i else 0) for j in range(N)] for i in range(N)]
        rows = [[sum(lower[i][k] * upper[k][j] for k in range(N)) for j in range(N)] for i in range(N)]
        order = list(range(N))
        rng.shuffle(order)
        a = [rows[k] for k in order]
        flat = [v for row in a for v in row]
        if max(abs(v) for v in flat) > bound or flat.count(0) > 2:
            continue
        d = _det(a)
        if abs(d) < 2 or abs(d) > 200:
            continue
        _verify_det(a, d)
        return a, d
    raise Redraw("no dense matrix")  # pragma: no cover


def _sparse_matrix(rng):
    """``(A, row_or_col, index, d)``: a 4×4 matrix whose chosen row or column has one non-zero entry."""
    for _ in attempts(3000, "a sparse 4×4 matrix"):
        a = [[rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5]) for _ in range(N)] for _ in range(N)]
        # a second line with a zero keeps the matrix from looking like a single expansion only
        orient = rng.choice(["row", "col"])
        idx = rng.randrange(N)
        keep = rng.randrange(N)
        for t in range(N):
            if t != keep:
                if orient == "row":
                    a[idx][t] = 0
                else:
                    a[t][idx] = 0
        other = rng.choice([k for k in range(N) if k != idx])
        spot = rng.randrange(N)
        if orient == "row":
            a[other][spot] = 0
        else:
            a[spot][other] = 0
        d = _det(a)
        if d == 0 or abs(d) > 300:
            continue
        _verify_det(a, d)
        return a, orient, idx, d
    raise Redraw("no sparse matrix")  # pragma: no cover


def _alternating_wrong(a, i):
    """Expansion along row i with every sign taken as + (the alternating signs forgotten)."""
    return sum(a[i][j] * int(det(fm(_minor(a, i, j)))) for j in range(N))


_EASY_STEMS = (
    "Evaluate the determinant of A = {m}.",
    "Find det A for the matrix A = {m}.",
    "Calculate |A|, where A = {m}.",
)


@redrawn
def _easy(rng):
    a, orient, idx, d = _sparse_matrix(rng)
    keep = next(t for t in range(N) if (a[idx][t] if orient == "row" else a[t][idx]) != 0)
    i, j = (idx, keep) if orient == "row" else (keep, idx)
    entry = a[i][j]
    cof = _cofactor(a, i, j)
    check(entry * cof == d, "the single-entry expansion differs from the determinant")
    wrong_position = [(r, c) for r in range(N) for c in range(N) if (r, c) != (i, j)]
    r2, c2 = wrong_position[rng.randrange(len(wrong_position))]
    wrong = [
        ("cof-entry-omitted", cof),
        ("cof-sign-pattern-ignored", entry * (-1) ** (i + j) * cof),
        ("cof-minor-wrong-position", entry * _cofactor(a, r2, c2)),
        ("det-diagonal-product", a[0][0] * a[1][1] * a[2][2] * a[3][3]),
        ("cof-sign-pattern-ignored", -d),
    ]
    line = f"row {i + 1}" if orient == "row" else f"column {j + 1}"
    return dict(
        text=phrase(rng, _EASY_STEMS, m=mat_str(fm(a))),
        working=[
            f"{line.capitalize()} has one non-zero entry, a{i + 1}{j + 1} = {entry}, so det A = {entry} · C{i + 1}{j + 1}.",
            f"C{i + 1}{j + 1} = {'-' if (i + j) % 2 else ''}det of the 3×3 minor = {cof}, giving det A = {d}.",
        ],
        **value_fields(d, wrong),
    )


_DET_STEMS = (
    "Find the determinant of the matrix A = {m}.",
    "Evaluate det A, where A = {m}.",
    "Determine |A| for A = {m}.",
)
_COFACTOR_STEMS = (
    "Find the cofactor C{ij} of the matrix A = {m}.",
    "For A = {m}, calculate the (i, j) cofactor with i = {i} and j = {j}.",
    "Let A = {m}. What is the cofactor C{ij}, the signed minor of the entry in row {i}, column {j}?",
)


@redrawn
def _medium_det(rng):
    a, d = _lu_matrix(rng)
    wrong = [
        ("det-row-swap-sign-ignored", -d),
        ("cof-alternating-signs-ignored", _alternating_wrong(a, 0)),
        ("det-diagonal-product", a[0][0] * a[1][1] * a[2][2] * a[3][3]),
        ("cof-first-row-only-first-minor", a[0][0] * int(det(fm(_minor(a, 0, 0))))),
    ]
    return dict(
        text=phrase(rng, _DET_STEMS, m=mat_str(fm(a))),
        working=[
            "Row operations reduce A to triangular form; a row swap changes the sign of the determinant.",
            f"The determinant is the product of the pivots, with the sign of the swaps: det A = {d}.",
        ],
        **value_fields(d, wrong),
    )


@redrawn
def _medium_cofactor(rng):
    a, d = _lu_matrix(rng, bound=7)
    i, j = rng.randrange(N), rng.randrange(N)
    cof = _cofactor(a, i, j)
    check(cof != 0, "zero cofactor")
    _verify_cofactor(a, i, j, cof)
    minor = int(det(fm(_minor(a, i, j))))
    wrong = [
        ("cof-minor-without-sign", minor),
        ("cof-index-transposed", _cofactor(a, j, i)),
        ("cof-entry-included", a[i][j] * cof),
        ("cof-sign-pattern-ignored", -cof),
        ("cof-minor-wrong-position", _cofactor(a, (i + 1) % N, j)),
    ]
    return dict(
        text=phrase(rng, _COFACTOR_STEMS, m=mat_str(fm(a)), i=i + 1, j=j + 1, ij=f"{i + 1}{j + 1}"),
        working=[
            f"Delete row {i + 1} and column {j + 1}; the 3×3 minor has determinant {minor}.",
            f"C{i + 1}{j + 1} = (-1)^({i + 1}+{j + 1}) × {minor} = {cof}.",
        ],
        **value_fields(cof, wrong),
    )


# -- hard -----------------------------------------------------------------------------------------------------------

# (expression, key function of (p, q), [(misconception, function of (p, q))]) for 4×4 matrices A, B with
# det A = p and det B = q. The size n = 4 is what makes the scalar factor 16 or 81 and -A have the determinant of A.
_PROPERTY_TEMPLATES = (
    (
        "2Aᵀ",
        lambda p, q: 16 * p,
        [
            ("det-scalar-not-raised-to-size", lambda p, q: 2 * p),
            ("det-scalar-times-size", lambda p, q: 8 * p),
            ("det-transpose-changes-sign", lambda p, q: -16 * p),
        ],
    ),
    (
        "3A⁻¹B",
        lambda p, q: Fraction(81 * q, p),
        [
            ("det-scalar-not-raised-to-size", lambda p, q: Fraction(3 * q, p)),
            ("det-inverse-ignored", lambda p, q: 81 * p * q),
            ("det-scalar-times-size", lambda p, q: Fraction(12 * q, p)),
            ("det-inverse-on-wrong-factor", lambda p, q: Fraction(81 * p, q)),
        ],
    ),
    (
        "A²B⁻¹",
        lambda p, q: Fraction(p * p, q),
        [
            ("det-power-as-multiple", lambda p, q: Fraction(2 * p, q)),
            ("det-inverse-ignored", lambda p, q: p * p * q),
            ("det-inverse-on-wrong-factor", lambda p, q: Fraction(q, p * p)),
            ("det-power-ignored", lambda p, q: Fraction(p, q)),
        ],
    ),
    (
        "adj(A)",
        lambda p, q: p**3,
        [
            ("det-adjugate-exponent", lambda p, q: p**4),
            ("det-adjugate-exponent", lambda p, q: p**2),
            ("det-adjugate-is-inverse", lambda p, q: Fraction(1, p)),
            ("det-adjugate-exponent", lambda p, q: p),
            ("det-adjugate-exponent", lambda p, q: p**5),
        ],
    ),
    (
        "(2A)⁻¹",
        lambda p, q: Fraction(1, 16 * p),
        [
            ("det-scalar-not-raised-to-size", lambda p, q: Fraction(1, 2 * p)),
            ("det-inverse-ignored", lambda p, q: 16 * p),
            ("det-scalar-times-size", lambda p, q: Fraction(1, 8 * p)),
            ("det-scalar-ignored", lambda p, q: Fraction(1, p)),
        ],
    ),
    (
        "−AᵀB⁻¹",
        lambda p, q: Fraction(p, q),
        [
            ("det-negative-sign-kept", lambda p, q: Fraction(-p, q)),
            ("det-inverse-ignored", lambda p, q: p * q),
            ("det-inverse-on-wrong-factor", lambda p, q: Fraction(q, p)),
            ("det-scalar-times-size", lambda p, q: Fraction(-4 * p, q)),
        ],
    ),
    (
        "ABAᵀ",
        lambda p, q: p * p * q,
        [
            ("det-transposed-factor-dropped", lambda p, q: p * q),
            ("det-product-as-sum", lambda p, q: 2 * p + q),
            ("det-power-as-multiple", lambda p, q: 2 * p * q),
        ],
    ),
    (
        "2AB⁻¹",
        lambda p, q: Fraction(16 * p, q),
        [
            ("det-scalar-not-raised-to-size", lambda p, q: Fraction(2 * p, q)),
            ("det-inverse-ignored", lambda p, q: 16 * p * q),
            ("det-inverse-on-wrong-factor", lambda p, q: Fraction(16 * q, p)),
            ("det-scalar-times-size", lambda p, q: Fraction(8 * p, q)),
        ],
    ),
)

_PROPERTY_STEMS = (
    "A and B are 4×4 matrices with det A = {p} and det B = {q}. Find det({expr}).",
    "Let A and B be 4×4 matrices, where det A = {p} and det B = {q}. What is the determinant of {expr}?",
    "The 4×4 matrices A and B satisfy det A = {p} and det B = {q}. Evaluate det({expr}).",
)


@redrawn
def _hard_properties(rng):
    expr, key, wrong_fns = rng.choice(_PROPERTY_TEMPLATES)
    p, q = rng.choice([2, 3, 4, 5, -2, -3]), rng.choice([2, 3, -2, -4, 5])
    if p == q:
        raise Redraw("p == q")
    value = Fraction(key(p, q))
    _check_property(expr, p, q, value)
    wrong = [(tag, fn(p, q)) for tag, fn in wrong_fns]
    return dict(
        text=phrase(rng, _PROPERTY_STEMS, p=p, q=q, expr=expr),
        working=[
            "For 4×4 matrices: det(kM) = k⁴ det M, det(MN) = det M · det N, det(Mᵀ) = det M, det(M⁻¹) = 1/det M, det(adj M) = (det M)³.",
            f"So det({expr}) = {value.numerator}" + ("" if value.denominator == 1 else f"/{value.denominator}") + ".",
        ],
        **value_fields(value, wrong),
    )


@functools.lru_cache(maxsize=None)
def _property_table(p, q):
    a = sympy.diag(p, 1, 1, 1) + sympy.Matrix(4, 4, lambda r, c: 1 if (r, c) == (0, 1) else 0)
    b = sympy.diag(q, 1, 1, 1) + sympy.Matrix(4, 4, lambda r, c: 1 if (r, c) == (1, 0) else 0)
    # a and b have determinant p and q (upper and lower unit-triangular perturbations of a diagonal matrix).
    check(a.det() == p and b.det() == q, "the test matrices have the wrong determinants")
    return {
        "2Aᵀ": (2 * a.T).det(),
        "3A⁻¹B": (3 * a.inv() * b).det(),
        "A²B⁻¹": (a * a * b.inv()).det(),
        "adj(A)": a.adjugate().det(),
        "(2A)⁻¹": (2 * a).inv().det(),
        "−AᵀB⁻¹": (-a.T * b.inv()).det(),
        "ABAᵀ": (a * b * a.T).det(),
        "2AB⁻¹": (2 * a * b.inv()).det(),
    }


def _check_property(expr, p, q, value):
    """Re-derive the determinant of ``expr`` with SymPy on matrices that have the stated determinants."""
    check(_property_table(p, q)[expr] == value, f"SymPy's determinant of {expr} differs")


_OPERATIONS = (
    ("swap", "swapping rows {i} and {j}", lambda i, j, c: -1),
    ("scale", "multiplying row {i} by {c}", lambda i, j, c: c),
    ("add", "adding {c} times row {j} to row {i}", lambda i, j, c: 1),
)
_ROW_STEMS = (
    "The 4×4 matrix B is obtained from A by {ops}. If det A = {d}, find det B.",
    "Starting from a 4×4 matrix A with det A = {d}, the row operations {ops} (in that order) give B. Find det B.",
    "det A = {d} for a 4×4 matrix A. B is formed from A by {ops}. What is det B?",
)


@redrawn
def _hard_row_operations(rng):
    d = rng.choice([2, 3, 4, 5, -2, -3, -4])
    kinds = ["swap", "scale", "add"]
    rng.shuffle(kinds)
    steps, factor, texts = [], 1, []
    wrong_ignore_swap, wrong_add_scales, wrong_scale_whole, sign_only = 1, 1, 1, 1
    for kind in kinds:
        i, j = rng.sample(range(1, N + 1), 2)
        c = rng.choice([2, 3, 4, -2, -3]) if kind != "swap" else 1
        template = next(t for t in _OPERATIONS if t[0] == kind)
        factor *= template[2](i, j, c)
        texts.append(template[1].format(i=i, j=j, c=c))
        steps.append((kind, c))
        wrong_ignore_swap *= 1 if kind == "swap" else (c if kind == "scale" else 1)
        sign_only *= -1 if kind == "swap" else 1
        wrong_add_scales *= -1 if kind == "swap" else c
        wrong_scale_whole *= -1 if kind == "swap" else (c**N if kind == "scale" else 1)
    value = factor * d
    ops = ", then ".join(texts)
    wrong = [
        ("det-row-swap-sign-ignored", wrong_ignore_swap * d),
        ("det-row-addition-scales", wrong_add_scales * d),
        ("det-row-scaled-as-whole-matrix", wrong_scale_whole * d),
        ("det-row-operations-unchanged", d),
        ("det-row-scaling-ignored", sign_only * d),
    ]
    sign = -1 if "swap" in kinds else 1
    check(value == d * sign * next(c for k, c in steps if k == "scale"), "the net factor differs")
    return dict(
        text=phrase(rng, _ROW_STEMS, ops=ops, d=d),
        working=[
            "A row swap multiplies the determinant by -1, scaling a row by c multiplies it by c, and adding a multiple of one row to another changes nothing.",
            f"So det B = {value}.",
        ],
        **value_fields(value, wrong),
    )


_PARAMETER_STEMS = (
    "Find the value of k for which the matrix M = {m} is singular.",
    "For what value of k does Mx = 0 have a non-zero solution, where M = {m}?",
    "The matrix M = {m} is not invertible. Find k.",
)


@redrawn
def _hard_parameter(rng):
    for _ in attempts(2000, "a singular 4×4 matrix with an integer entry"):
        rows = [[rng.randint(-3, 4) for _ in range(N)] for _ in range(3)]
        coef = [rng.choice([-2, -1, 1, 2]) for _ in range(3)]
        last = [sum(coef[r] * rows[r][c] for r in range(3)) for c in range(N)]
        a = rows + [last]
        order = list(range(N))
        rng.shuffle(order)
        a = [a[k] for k in order]
        if _det(a) != 0:
            continue
        i, j = rng.randrange(N), rng.randrange(N)
        k0 = a[i][j]
        cof = _cofactor(a, i, j)
        flat = [v for row in a for v in row]
        if cof == 0 or k0 == 0 or max(abs(v) for v in flat) > 12:
            continue
        break
    # det(k) = C_ij (k - k0): SymPy's determinant at k0 and k0 + 1 checks the root independently.
    sym = _sym(a)
    check(sym.det() == 0, "the matrix with k = k0 is not singular")
    sym[i, j] = k0 + 1
    check(sym.det() == cof, "det M(k0 + 1) differs from the cofactor")
    base = _det([[0 if (r, c) == (i, j) else a[r][c] for c in range(N)] for r in range(N)])
    wrong = [
        ("sign-error-moving-term", -k0),
        ("cof-forgot-cofactor-divisor", -base),
        ("cof-index-transposed", Fraction(-base, _cofactor(a, j, i)) if _cofactor(a, j, i) else None),
        ("cof-minor-without-sign", Fraction(-base, int(det(fm(_minor(a, i, j))))) if (i + j) % 2 else None),
        *[
            ("cof-expansion-term-dropped", Fraction(-(base - a[i][r] * _cofactor(a, i, r)), cof))
            for r in range(N)
            if r != j
        ],
        *[
            ("cof-minor-wrong-position", Fraction(-base, c2) if (c2 := _cofactor(a, i, jj)) else None)
            for jj in range(N)
            if jj != j
        ],
    ]
    shown = [["k" if (r, c) == (i, j) else a[r][c] for c in range(N)] for r in range(N)]
    return dict(
        text=phrase(rng, _PARAMETER_STEMS, m=_mat_with_symbol(shown)),
        working=[
            f"Expanding along row {i + 1}, det M = C{i + 1}{j + 1}·k + (the rest) = {cof}(k - ({k0})).",
            f"This is zero only for k = {k0}.",
        ],
        **value_fields(k0, wrong),
    )


def _mat_with_symbol(rows):
    return "[" + ", ".join("[" + ", ".join(str(v) for v in row) + "]" for row in rows) + "]"


@generator(
    "w6c_cofactor",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w6c_cofactor(rng, tier="easy"):
    """Determinants and cofactors of 4×4 matrices; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return rng.choice([_medium_det, _medium_cofactor])(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_properties, _hard_row_operations, _hard_parameter, _hard_properties])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_w6c_cofactor", level=19, presets={"exam": dict(tier="exam")})
@redrawn
def exd_x_w6c_cofactor(rng, tier="exam"):
    """Exam chain on a dense 4×4 matrix: det A, a cofactor, an entry of the inverse, det(3A⁻¹)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a 4×4 matrix with a fractional inverse entry"):
        a, d = _lu_matrix(rng, bound=7)
        i, j = rng.randrange(N), rng.randrange(N)
        cof = _cofactor(a, i, j)
        if cof != 0 and cof % d != 0 and 81 % d != 0:
            break
    _verify_cofactor(a, i, j, cof)
    inverse_entry = Fraction(cof, d)
    sym_inverse = _sym(a).inv()
    check(sym_inverse[j, i] == sympy.Rational(cof, d), "SymPy's inverse entry differs")
    check((3 * sym_inverse).det() == sympy.Rational(81, d), "SymPy's det(3A⁻¹) differs")
    scaled = Fraction(81, d)
    other = _cofactor(a, j, i)
    return dict(
        text=f"Let A = {mat_str(fm(a))}.",
        parts=[
            value_part(
                "Find det A.",
                d,
                3,
                ["Row operations reduce A to triangular form; the product of the pivots, with the sign of any swaps, is det A."],
                [
                    ("det-row-swap-sign-ignored", -d),
                    ("cof-alternating-signs-ignored", _alternating_wrong(a, 0)),
                    ("det-diagonal-product", a[0][0] * a[1][1] * a[2][2] * a[3][3]),
                ],
            ),
            value_part(
                f"Find the cofactor C{i + 1}{j + 1} of A.",
                cof,
                2,
                [f"C{i + 1}{j + 1} = (-1)^({i + 1}+{j + 1}) × det of the minor with row {i + 1} and column {j + 1} deleted = {cof}."],
                [
                    ("cof-minor-without-sign", abs(cof) if (i + j) % 2 else -abs(cof)),
                    ("cof-index-transposed", other),
                    ("cof-entry-included", a[i][j] * cof),
                    ("cof-sign-pattern-ignored", -cof),
                ],
            ),
            value_part(
                f"Find the entry in row {j + 1}, column {i + 1} of A⁻¹.",
                inverse_entry,
                2,
                [f"A⁻¹ = adj(A)/det A, and the (row {j + 1}, column {i + 1}) entry of adj(A) is C{i + 1}{j + 1}, so it is {inverse_entry.numerator}/{inverse_entry.denominator}."],
                [
                    ("cof-adjugate-not-transposed", Fraction(other, d)),
                    ("cof-forgot-divide-by-det", Fraction(cof)),
                    ("cof-inverse-entry-inverted", Fraction(d, cof)),
                    ("cof-adjugate-not-transposed", Fraction(-cof, d)),
                ],
                follow_through="part_b/part_a",
            ),
            value_part(
                "Find det(3A⁻¹).",
                scaled,
                2,
                [f"det(3A⁻¹) = 3⁴ det(A⁻¹) = 81/det A = {scaled.numerator}/{scaled.denominator}."],
                [
                    ("det-scalar-not-raised-to-size", Fraction(3, d)),
                    ("det-inverse-ignored", Fraction(81 * d)),
                    ("det-scalar-times-size", Fraction(12, d)),
                    ("det-scalar-power-off-by-one", Fraction(27, d)),
                ],
                follow_through="81/part_a",
            ),
        ],
    )
