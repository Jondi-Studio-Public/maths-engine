"""LU factorisation of a 3x3 matrix: `r4b_lu_factorisation` (easy, medium, hard) and the exam chain
`exd_x_r4b_lu_factorisation`.

``uc2-lu-factorisation-3x3`` (``numerical-and-computational-mathematics``). Every question fixes the convention in
its wording so the answer is unique: Doolittle's method, L unit lower triangular (ones on its diagonal), U upper
triangular, no row interchanges.

* easy: the matrix is built from whole-number multipliers, so U has whole entries; the answer is U (a matrix).
* medium: whole-number matrix with leading minors that are not ±1, so the multipliers are fractions; the answer is L
  (a matrix of fractions).
* hard: the same kind of matrix with a right-hand side b; the answer is the intermediate column y with Ly = b.
* exam: a 3x3 matrix in four parts, all scalars: ``l₂₁``, ``u₂₂``, ``l₃₂`` and ``det A`` as the product of the
  diagonal of U, each part built on the learner's earlier values.

L and U are built in exact `Fraction` arithmetic by Doolittle's rule and checked two ways: ``L·U = A`` and SymPy's
``LUdecomposition``.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator
from ..grading import GRADERS
from ..grading.matrix import matrix_text
from ._r4b_kit import (
    Redraw,
    fraction_part,
    fr,
    with_redraw,
)
from ._ucv1_texsay import say_marked

M_SIGN = "num2-lu-multiplier-sign-wrong"
M_PARTIAL = "num2-lu-second-column-not-eliminated"
M_UPPER_OF_A = "num2-lu-upper-triangle-of-a-taken"
M_SWAPPED = "num2-lu-l-and-u-swapped"
M_TRANSPOSED = "num2-lu-factor-transposed"
M_NO_DIVIDE = "num2-lu-multiplier-not-divided-by-pivot"
M_INVERTED = "num2-lu-multiplier-inverted"
M_L_SIGN = "num2-lu-l-entries-negated"
M_FORWARD_SKIP = "num2-lu-forward-substitution-term-missed"
M_BACK = "num2-lu-solved-with-u-instead-of-l"
M_B_DIRECT = "num2-lu-right-hand-side-copied"
M_PIVOT_SIGN = "num2-lu-pivot-update-sign"
M_PIVOT_OLD = "num2-lu-pivot-left-unchanged"
M_DIAG_SUM = "num2-lu-determinant-as-diagonal-sum"
M_DET_NO_FIRST = "num2-lu-determinant-first-pivot-missed"
M_DET_A_DIAG = "num2-lu-determinant-from-diagonal-of-a"
M_L32_NO_UPDATE = "num2-lu-second-multiplier-without-first-column-term"
M_L32_UNSCALED = "num2-lu-second-multiplier-wrong-pivot"
M_U33_PIVOT = "num2-lu-last-pivot-without-second-multiplier"


def mat(rows):
    return [[Fraction(v) for v in row] for row in rows]


def mmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a):
    return [list(r) for r in zip(*a)]


def doolittle(a):
    """(L, U) by Doolittle's rule with no row interchanges, in exact arithmetic."""
    n = len(a)
    lo = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    up = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            up[i][j] = Fraction(a[i][j]) - sum(lo[i][k] * up[k][j] for k in range(i))
        check(up[i][i] != 0, "a zero pivot: the matrix needs a row interchange")
        for j in range(i + 1, n):
            lo[j][i] = (Fraction(a[j][i]) - sum(lo[j][k] * up[k][i] for k in range(i))) / up[i][i]
    return lo, up


def check_lu(a, lo, up):
    check(mmul(lo, up) == mat(a), "L times U is not A")
    sl, su, perm = sympy.Matrix(a).LUdecomposition()
    check(not perm, "SymPy needed a row interchange")
    ref_l = [[Fraction(int(sl[i, j].p), int(sl[i, j].q)) for j in range(3)] for i in range(3)]
    ref_u = [[Fraction(int(su[i, j].p), int(su[i, j].q)) for j in range(3)] for i in range(3)]
    check(ref_l == lo and ref_u == up, "SymPy's LU disagrees")


def rows_of(m):
    return [[exact_value(x) for x in row] for row in m]


def shown(m) -> str:
    return matrix_text(rows_of(m))


def answer_dict(label, m):
    rows = rows_of(m)
    text = f"{label}{matrix_text(rows)}"
    return dict(answer=rows, answer_text=text, answer_latex="plain:" + text), rows, text


def accepts(rows, text, reply) -> bool:
    question = SimpleNamespace(answer=rows, answer_text=text, answer_form=None, text="", parts=None)
    return bool(GRADERS["matrix"](question, reply).correct)


def tagged(label, rows, text, cands, limit=4):
    out, seen = [], {text}
    for m, tag in cands:
        if m is None:
            continue
        t = f"{label}{matrix_text(rows_of(m))}"
        if t in seen or accepts(rows, text, t):
            continue
        seen.add(t)
        out.append(Distractor(t, tag, value=rows_of(m)))
        if len(out) == limit:
            break
    return out


def _eliminate(a, sign=1, partial=False):
    """Row reduction to upper triangular form, with the multiplier applied with a chosen sign, or stopping early."""
    m = mat(a)
    n = len(m)
    for c in range(n - 1 if not partial else 1):
        if m[c][c] == 0:
            return None  # the wrong-sign elimination hit a zero pivot: not a plausible reply
        for r in range(c + 1, n):
            mult = m[r][c] / m[c][c]
            m[r] = [m[r][j] - sign * mult * m[c][j] for j in range(n)]
    return m


def draw_integer_multipliers(rng):
    for _ in attempts(2000, "an integer-multiplier matrix"):
        l21, l31, l32 = (rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(3))
        lo = mat([[1, 0, 0], [l21, 1, 0], [l31, l32, 1]])
        up = mat(
            [
                [rng.randint(1, 4), rng.randint(-4, 4), rng.randint(-4, 4)],
                [0, rng.choice([-3, -2, -1, 1, 2, 3, 4]), rng.randint(-4, 4)],
                [0, 0, rng.choice([-3, -2, -1, 1, 2, 3, 4])],
            ]
        )
        a = mmul(lo, up)
        if max(abs(v) for row in a for v in row) <= 20 and up[0][1] != 0 and up[0][2] != 0 and up[1][2] != 0:
            return [[int(v) for v in row] for row in a]
    raise AssertionError("unreachable")


def draw_fraction_multipliers(rng, lo_val=-6, hi_val=6):
    for _ in attempts(5000, "a matrix with fractional multipliers"):
        a = [[rng.randint(lo_val, hi_val) for _ in range(3)] for _ in range(3)]
        if a[0][0] == 0 or a[0][0] in (1, -1):
            continue
        det2 = a[0][0] * a[1][1] - a[0][1] * a[1][0]
        if det2 == 0:
            continue
        try:
            lo, up = doolittle(a)
        except Exception:
            continue
        mults = [lo[1][0], lo[2][0], lo[2][1]]
        if sum(1 for v in mults if v.denominator > 1) < 2 or any(v == 0 for v in mults):
            continue
        if any(v.denominator > 40 or abs(v) > 12 for v in mults):
            continue
        if any(abs(up[i][j]) > 200 or up[i][j].denominator > 40 for i in range(3) for j in range(3)):
            continue
        if any(a[i][j] == 0 for i in range(3) for j in range(3)):
            continue
        return a
    raise AssertionError("unreachable")


CONVENTION = (
    "Use Doolittle's method with no row interchanges, so that «A = LU» where «L» is unit lower triangular "
    "(ones on its diagonal) and «U» is upper triangular."
)


@generator(
    "r4b_lu_factorisation",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4b_lu_factorisation(rng, tier="easy"):
    return with_redraw(lambda: _single(rng, tier), "an LU question with enough distractors")


def _single(rng, tier):
    if tier == "easy":
        a = draw_integer_multipliers(rng)
        lo, up = doolittle(a)
        check_lu(a, lo, up)
        ans, rows, text = answer_dict("U = ", up)
        cands = [
            (_eliminate(a, sign=-1), M_SIGN),
            (_eliminate(a, partial=True), M_PARTIAL),
            (mat([[a[i][j] if j >= i else 0 for j in range(3)] for i in range(3)]), M_UPPER_OF_A),
            (lo, M_SWAPPED),
            (transpose(up), M_TRANSPOSED),
        ]
        dist = tagged("U = ", rows, text, cands)
        if len(dist) < 3:
            raise Redraw
        stem = say_marked(
            rng.choice(
                [
                    f"Factorise «A = {shown(mat(a))}». {CONVENTION} Give the matrix «U», with whole-number entries.",
                    f"Find the LU factorisation of «A = {shown(mat(a))}». {CONVENTION} What is the matrix «U»? "
                    "Its entries are whole numbers.",
                    f"Let «A = {shown(mat(a))}». {CONVENTION} Compute «U», whose entries are all whole numbers.",
                ]
            )
        )
        return dict(
            text=stem[0],
            latex=stem[1],
            answer_kind="matrix",
            distractors=dist,
            **ans,
            working=[
                "Eliminate below each pivot, recording the multipliers in L; the reduced matrix is U.",
                f"U = {shown(up)}",
            ],
        )
    a = draw_fraction_multipliers(rng)
    lo, up = doolittle(a)
    check_lu(a, lo, up)
    if tier == "medium":
        ans, rows, text = answer_dict("L = ", lo)
        cands = [
            (_neg_lower(lo), M_L_SIGN),
            (_with_inverted_multipliers(lo), M_INVERTED),
            (_with_unscaled_multipliers(a, lo), M_NO_DIVIDE),
            (up, M_SWAPPED),
            (transpose(lo), M_TRANSPOSED),
        ]
        dist = tagged("L = ", rows, text, cands)
        if len(dist) < 3:
            raise Redraw
        stem = say_marked(
            rng.choice(
                [
                    f"Factorise «A = {shown(mat(a))}». {CONVENTION} Give the matrix «L» with whole numbers or fractions.",
                    f"Find the LU factorisation of «A = {shown(mat(a))}». {CONVENTION} What is the matrix «L»? "
                    "Its entries are whole numbers or fractions.",
                    f"Let «A = {shown(mat(a))}». {CONVENTION} Compute «L», giving each entry as a whole number or fraction.",
                ]
            )
        )
        return dict(
            text=stem[0],
            latex=stem[1],
            answer_kind="matrix",
            distractors=dist,
            **ans,
            working=["Each multiplier is the entry being eliminated divided by its pivot.", f"L = {shown(lo)}"],
        )
    # hard: forward substitution
    for _ in attempts(500, "a right-hand side"):
        b = [rng.randint(-8, 9) for _ in range(3)]
        y = _forward(lo, b)
        if all(v.denominator > 1 or v != 0 for v in y) and sum(1 for v in y if v.denominator > 1) >= 1:
            break
    ycol = [[v] for v in y]
    ans, rows, text = answer_dict("y = ", ycol)
    check(mmul(lo, ycol) == [[Fraction(v)] for v in b], "Ly is not b")
    cands = [
        ([[Fraction(v)] for v in b], M_B_DIRECT),
        ([[v] for v in _forward(_neg_lower(lo), b)], M_L_SIGN),
        ([[v] for v in _forward_skip(lo, b)], M_FORWARD_SKIP),
        ([[v] for v in _back(up, b)], M_BACK),
        ([[v] for v in _forward(_flip_multipliers(lo), b)], M_INVERTED),
    ]
    dist = tagged("y = ", rows, text, cands)
    if len(dist) < 3:
        raise Redraw
    sys_a, sys_b = shown(mat(a)), shown([[v] for v in b])
    stem = say_marked(
        rng.choice(
            [
                f"A system «Ax = b» has «A = {sys_a}» and «b = {sys_b}». {CONVENTION} "
                "Find the column «y» that solves «Ly = b», the first stage of solving the system by this factorisation.",
                f"To solve «Ax = b» with «A = {sys_a}» and «b = {sys_b}», first factorise «A». {CONVENTION} "
                "Then solve «Ly = b» by forward substitution and give the column «y».",
                f"Let «A = {sys_a}» and «b = {sys_b}». {CONVENTION} Find the column «y» satisfying «Ly = b», "
                "the first step in solving «Ax = b».",
            ]
        )
    )
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="matrix",
        distractors=dist,
        **ans,
        working=[f"L = {shown(lo)}.", f"Forward substitution on Ly = b gives y = {shown(ycol)}."],
    )


def _forward(lo, b):
    y = []
    for i in range(3):
        y.append(Fraction(b[i]) - sum(lo[i][k] * y[k] for k in range(i)))
    return y


def _forward_skip(lo, b):
    """Forward substitution that forgets the term from y_1 in the third row."""
    y = [Fraction(b[0])]
    y.append(Fraction(b[1]) - lo[1][0] * y[0])
    y.append(Fraction(b[2]) - lo[2][1] * y[1])
    return y


def _back(up, b):
    x = [Fraction(0)] * 3
    for i in (2, 1, 0):
        x[i] = (Fraction(b[i]) - sum(up[i][k] * x[k] for k in range(i + 1, 3))) / up[i][i]
    return x


def _neg_lower(lo):
    return [[-lo[i][j] if i > j else lo[i][j] for j in range(3)] for i in range(3)]


def _flip_multipliers(lo):
    return [[(1 / lo[i][j] if i > j else lo[i][j]) for j in range(3)] for i in range(3)]


def _with_inverted_multipliers(lo):
    return _flip_multipliers(lo)


def _with_unscaled_multipliers(a, lo):
    """Each multiplier left as the raw entry below the pivot (no division), the other entries as in L."""
    out = [row[:] for row in lo]
    out[1][0] = Fraction(a[1][0])
    out[2][0] = Fraction(a[2][0])
    return out


# ---------------------------------------------------------------------------------------------------------------
# exam chain


@generator("exd_x_r4b_lu_factorisation", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4b_lu_factorisation(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam LU question")


def _rt(v) -> str:
    v = fr(v)
    return f"Rational({v.numerator}, {v.denominator})"


def _exam(rng):
    for _ in attempts(5000, "an exam LU draw"):
        a = draw_fraction_multipliers(rng)
        lo, up = doolittle(a)
        det = up[0][0] * up[1][1] * up[2][2]
        if abs(det) > 2000 or det == 0 or up[1][1].denominator == 1 or lo[2][1].denominator == 1:
            continue
        break
    check_lu(a, lo, up)
    check(sympy.Matrix(a).det() == sympy.Rational(det.numerator, det.denominator), "determinant disagrees")
    l21, u22, l32 = lo[1][0], up[1][1], lo[2][1]
    l31 = lo[2][0]
    u12, u13, u23 = up[0][1], up[0][2], up[1][2]
    a11 = a[0][0]
    stem, latex = say_marked(
        rng.choice(
            [
                f"Let «A = {shown(mat(a))}». {CONVENTION}",
                f"Consider «A = {shown(mat(a))}». {CONVENTION}",
                f"The matrix «A = {shown(mat(a))}» is to be factorised. {CONVENTION}",
            ]
        )
    )
    p_a = fraction_part(
        "Find the multiplier «l₂₁».",
        l21,
        [
            (M_INVERTED, Fraction(a11, a[1][0])),
            (M_L_SIGN, -l21),
            (M_NO_DIVIDE, Fraction(a[1][0])),
            ("num2-lu-multiplier-from-wrong-row", Fraction(a[2][0], a11)),
        ],
        1,
        working=[f"l₂₁ = a₂₁/a₁₁ = {a[1][0]}/{a11} = {l21}."],
    )
    p_b = fraction_part(
        "Using your «l₂₁», find the pivot «u₂₂».",
        u22,
        [
            (M_PIVOT_SIGN, Fraction(a[1][1]) + l21 * a[0][1]),
            (M_PIVOT_OLD, Fraction(a[1][1])),
            ("num2-lu-pivot-uses-wrong-entry", Fraction(a[1][1]) - l21 * a[0][2]),
            ("num2-lu-pivot-multiplier-dropped", Fraction(a[1][1]) - a[0][1]),
        ],
        2,
        ft=f"{a[1][1]} - part_a*({a[0][1]})",
        working=[f"u₂₂ = a₂₂ − l₂₁u₁₂ = {a[1][1]} − ({l21})({a[0][1]}) = {u22}."],
    )
    p_c = fraction_part(
        "Using your «u₂₂», find the multiplier «l₃₂».",
        l32,
        [
            (M_L32_NO_UPDATE, Fraction(a[2][1]) / u22),
            (M_L32_UNSCALED, (Fraction(a[2][1]) - l31 * u12) / Fraction(a[1][1])),
            (M_INVERTED, u22 / (Fraction(a[2][1]) - l31 * u12)),
            (M_L_SIGN, -l32),
        ],
        2,
        ft=f"({a[2][1]} - {_rt(l31)}*({a[0][1]}))/part_b",
        working=[f"l₃₁ = {l31}, so l₃₂ = (a₃₂ − l₃₁u₁₂)/u₂₂ = {l32}."],
    )
    u33 = up[2][2]
    p_d = fraction_part(
        "Using your «l₃₂», complete the factorisation and find the determinant of «A» as the product of the diagonal of «U».",
        det,
        [
            (M_DIAG_SUM, up[0][0] + u22 + u33),
            (M_DET_NO_FIRST, u22 * u33),
            (M_DET_A_DIAG, Fraction(a[0][0] * a[1][1] * a[2][2])),
            (M_U33_PIVOT, a11 * u22 * (Fraction(a[2][2]) - l31 * u13)),
        ],
        2,
        ft=f"({a11})*part_b*({a[2][2]} - {_rt(l31)}*({u13}) - part_c*({a[1][2]} - part_a*({a[0][2]})))",
        working=[f"u₃₃ = {u33}, so det A = {a11} × {u22} × {u33} = {det}."],
    )
    for part in (p_a, p_b, p_c, p_d):
        part["prompt"], part["prompt_latex"] = say_marked(part["prompt"])
    return dict(text=stem, latex=latex, parts=[p_a, p_b, p_c, p_d])
