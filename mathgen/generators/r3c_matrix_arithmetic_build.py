"""Generates `matrix-arithmetic` questions (easy, medium, hard) and the exam chain `r3c_matrix_arithmetic_chain`.

Sums, scalar multiples and products of small matrices, written out in full as a matrix answer.

* easy: a combination such as ``2A - 3B`` of two matrices of the same size (2 by 2, 2 by 3 or 3 by 2);
* medium: a product ``AB`` where the sizes are not equal (2 by 3 times 3 by 2, 2 by 2 times 2 by 3 and so on), so the
  order of the factors and the size of the answer both matter;
* hard: one of two things that go wrong when matrices are treated like numbers: the polynomial ``A^2 - kA + mI`` (the
  identity matrix appears from the constant term), or ``(A + B)^2 - A^2 - B^2``, which is ``AB + BA`` and not ``2AB``;
* exam (``r3c_matrix_arithmetic_chain``): the determinants of two 2 by 2 matrices, their product, and the
  determinant of the product (carried forward from the first two).

Every entry of a question is a small whole number and every answer is found by direct multiplication.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, paren_negative, phrase
from ._r3c_kit import mat_mul, matrix_part, matrix_single, number_part, transpose


def _m(rng, r, c, lo=-5, hi=9, zeros=1):
    for _ in attempts(100, "a matrix with few zeros"):
        m = [[rng.randint(lo, hi) for _ in range(c)] for _ in range(r)]
        if sum(v == 0 for row in m for v in row) <= zeros:
            return m
    raise AssertionError  # pragma: no cover


def _show(m) -> str:
    return "[" + ", ".join("[" + ", ".join(str(int(v)) for v in row) + "]" for row in m) + "]"


def _ints(m):
    return [[int(v) for v in row] for row in m]


def _mul(a, b):
    return _ints(mat_mul(a, b))


def _add(a, b, ka=1, kb=1):
    return [[ka * x + kb * y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def _scale(a, k):
    return [[k * x for x in row] for row in a]


def _entry_slip(rng, m):
    out = [list(r) for r in m]
    i, j = rng.randrange(len(m)), rng.randrange(len(m[0]))
    out[i][j] += rng.choice((-2, -1, 1, 2))
    return out


def _det2(m):
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def _easy(rng):
    for _ in attempts(200, "an easy matrix combination"):
        r, c = rng.choice([(2, 2), (2, 3), (3, 2)])
        A, B = _m(rng, r, c), _m(rng, r, c)
        ka, kb = rng.choice([(2, 3), (3, 2), (4, 3), (3, 4), (5, 2)])
        sign = rng.choice((-1, 1))
        key = _add(A, B, ka, sign * kb)
        op = "-" if sign < 0 else "+"
        text = f"Let A = {_show(A)} and B = {_show(B)}. Write the matrix {ka}A {op} {kb}B."
        slips = [
            (_add(A, B, ka, -sign * kb), "r3c-ma-sign-slip"),
            (_add(A, B, 1, sign), "r3c-ma-scalar-forgotten"),
            (_add(A, B, ka, sign), "r3c-ma-scalar-one-matrix"),
            (_add(A, B, kb, sign * ka), "r3c-ma-scalars-swapped"),
            (_entry_slip(rng, key), "r3c-ma-entry-slip"),
        ]
        working = [
            f"{ka}A = {_show(_scale(A, ka))}.",
            f"{kb}B = {_show(_scale(B, kb))}.",
            f"{ka}A {op} {kb}B = {_show(key)}.",
        ]
        q = matrix_single(text, key, working, slips)
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _product_slips(rng, A, B):
    key = _mul(A, B)
    out = []
    if len(B[0]) == len(A) and len(A[0]) == len(B):
        out.append((_mul(B, A), "r3c-ma-order-reversed"))
    if len(A) == len(B) and len(A[0]) == len(B[0]):
        out.append(([[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))], "r3c-ma-entrywise-product"))
    out.append((_ints(transpose(key)), "r3c-ma-product-transposed"))
    # one product term left out of one entry
    i, j = rng.randrange(len(A)), rng.randrange(len(B[0]))
    short = [list(r) for r in key]
    k = rng.randrange(len(B))
    short[i][j] -= A[i][k] * B[k][j]
    out.append((short, "r3c-ma-term-missing"))
    # row-by-row sum instead of row-by-column
    out.append((_entry_slip(rng, key), "r3c-ma-entry-slip"))
    out.append((_entry_slip(rng, key), "r3c-ma-entry-slip"))
    return out


def _medium(rng):
    for _ in attempts(300, "a medium matrix product"):
        r, k, c = rng.choice([(2, 3, 2), (2, 2, 3), (3, 2, 2), (2, 3, 3), (3, 3, 2), (2, 2, 2)])
        A, B = _m(rng, r, k, -4, 8), _m(rng, k, c, -4, 8)
        key = _mul(A, B)
        text = f"Let A = {_show(A)} and B = {_show(B)}. Write the product AB."
        working = [
            f"A is {r} by {k} and B is {k} by {c}, so AB is {r} by {c}.",
            "Each entry is a row of A times a column of B, added.",
            f"AB = {_show(key)}.",
        ]
        q = matrix_single(text, key, working, _product_slips(rng, A, B))
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _hard_polynomial(rng):
    for _ in attempts(200, "a matrix polynomial"):
        A = _m(rng, 2, 2, -3, 6, 0)
        k, m = rng.randint(2, 4), rng.randint(2, 5)
        A2 = _mul(A, A)
        key = [[A2[i][j] - k * A[i][j] + (m if i == j else 0) for j in range(2)] for i in range(2)]
        text = phrase(
            rng,
            [
                "Let A = {A}. Write the matrix A² - {k}A + {m}I, where I is the 2 by 2 identity matrix.",
                "The matrix A = {A}, and I is the 2 by 2 identity matrix. Find A² - {k}A + {m}I.",
                "Given A = {A}, evaluate A² - {k}A + {m}I as a single matrix, where I is the 2 by 2 identity matrix.",
            ],
            A=_show(A),
            k=k,
            m=m,
        )
        slips = [
            ([[A2[i][j] - k * A[i][j] + m for j in range(2)] for i in range(2)], "r3c-ma-constant-as-all-entries"),
            ([[A2[i][j] - k * A[i][j] for j in range(2)] for i in range(2)], "r3c-ma-identity-dropped"),
            (
                [[A[i][j] ** 2 - k * A[i][j] + (m if i == j else 0) for j in range(2)] for i in range(2)],
                "r3c-ma-squares-entries",
            ),
            (_entry_slip(rng, key), "r3c-ma-entry-slip"),
        ]
        working = [
            f"A² = {_show(A2)}, found by multiplying A by itself (not by squaring each entry).",
            f"{k}A = {_show(_scale(A, k))} and {m}I = {_show([[m, 0], [0, m]])}.",
            f"A² - {k}A + {m}I = {_show(key)}.",
        ]
        q = matrix_single(text, key, working, slips)
        if q:
            return q
    raise AssertionError  # pragma: no cover


def _hard_expansion(rng):
    for _ in attempts(300, "a matrix expansion"):
        A, B = _m(rng, 2, 2, -3, 6, 0), _m(rng, 2, 2, -3, 6, 0)
        AB, BA = _mul(A, B), _mul(B, A)
        if AB == BA:
            continue
        key = _add(AB, BA)
        S = _add(A, B)
        text = phrase(
            rng,
            [
                "Let A = {A} and B = {B}. Write the matrix (A + B)² - A² - B².",
                "The matrices are A = {A} and B = {B}. Find (A + B)² - A² - B².",
                "Given A = {A} and B = {B}, evaluate (A + B)² - A² - B² as a single matrix.",
            ],
            A=_show(A),
            B=_show(B),
        )
        slips = [
            (_scale(AB, 2), "r3c-ma-commutes"),
            (_scale(BA, 2), "r3c-ma-commutes"),
            ([[0, 0], [0, 0]], "r3c-ma-expansion-cancels"),
            (_mul(S, S), "r3c-ma-no-subtraction"),
            (_entry_slip(rng, key), "r3c-ma-entry-slip"),
        ]
        working = [
            "(A + B)² = A² + AB + BA + B², and AB is not the same as BA for matrices.",
            f"AB = {_show(AB)} and BA = {_show(BA)}.",
            f"(A + B)² - A² - B² = AB + BA = {_show(key)}.",
        ]
        q = matrix_single(text, key, working, slips)
        if q:
            return q
    raise AssertionError  # pragma: no cover


@generator(
    "r3c_matrix_arithmetic",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3c_matrix_arithmetic(rng, tier="easy"):
    """Matrix sums, multiples and products; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard_polynomial(rng) if rng.random() < 0.5 else _hard_expansion(rng)


@generator("r3c_matrix_arithmetic_chain", level=12, presets={"exam": dict(tier="exam")})
def r3c_matrix_arithmetic_chain(rng, tier="exam"):
    """Exam chain: det A, det B, the product AB, and det(AB)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a determinant chain"):
        A, B = _m(rng, 2, 2, -4, 7, 0), _m(rng, 2, 2, -4, 7, 0)
        da, db = _det2(A), _det2(B)
        AB = _mul(A, B)
        dab = _det2(AB)
        if da in (0, 1, -1) or db in (0, 1, -1) or da == db or dab != da * db or abs(dab) > 400:
            continue
        stem = phrase(
            rng,
            [
                "Let A = {A} and B = {B}.",
                "Two matrices are A = {A} and B = {B}.",
                "Consider the matrices A = {A} and B = {B}.",
            ],
            A=_show(A),
            B=_show(B),
        )
        part_a = number_part(
            "Find the determinant of A.",
            da,
            1,
            [
                f"{paren_negative(A[0][0])} x {paren_negative(A[1][1])} - {paren_negative(A[0][1])} x {paren_negative(A[1][0])} = {da}."
            ],
            [
                (A[0][0] * A[1][1] + A[0][1] * A[1][0], "r3c-ma-determinant-sign"),
                (A[0][1] * A[1][0] - A[0][0] * A[1][1], "r3c-ma-determinant-reversed"),
                (A[0][0] * A[1][1], "r3c-ma-determinant-one-diagonal"),
                (A[0][0] + A[1][1] - A[0][1] - A[1][0], "r3c-ma-determinant-adds"),
            ],
            sensible=(None, None),
        )
        part_b = number_part(
            "Find the determinant of B.",
            db,
            1,
            [
                f"{paren_negative(B[0][0])} x {paren_negative(B[1][1])} - {paren_negative(B[0][1])} x {paren_negative(B[1][0])} = {db}."
            ],
            [
                (B[0][0] * B[1][1] + B[0][1] * B[1][0], "r3c-ma-determinant-sign"),
                (B[0][1] * B[1][0] - B[0][0] * B[1][1], "r3c-ma-determinant-reversed"),
                (B[0][0] * B[1][1], "r3c-ma-determinant-one-diagonal"),
                (B[0][0] + B[1][1] - B[0][1] - B[1][0], "r3c-ma-determinant-adds"),
            ],
            sensible=(None, None),
        )
        part_c = matrix_part(
            "Write the product AB.",
            AB,
            3,
            [f"AB = {_show(AB)}."],
            _product_slips(rng, A, B),
        )
        part_d = number_part(
            "Find the determinant of AB.",
            dab,
            2,
            [f"{da} x {db} = {dab}, and this agrees with the determinant of the matrix AB."],
            [
                (da + db, "r3c-ma-determinant-adds-factors"),
                (da * db + da, "r3c-ma-determinant-slip"),
                (-dab, "r3c-ma-determinant-reversed"),
                (da * db - db, "r3c-ma-determinant-slip"),
                (da - db, "r3c-ma-determinant-adds-factors"),
            ],
            follow="part_a*part_b",
            sensible=(None, None),
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
    raise AssertionError  # pragma: no cover
