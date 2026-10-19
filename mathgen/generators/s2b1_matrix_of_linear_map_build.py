"""Generates `s2b1_matrix_of_linear_map` questions (easy, medium, hard) and the exam chain `exd_x_s2b1_map_matrix`.

``uc2-matrix-of-linear-map`` (``linear-algebra``, answer ``matrix``): the matrix of a linear map relative to supplied
bases, for maps on R^n and on polynomial or function spaces, and the similarity A' = P^-1 A P. The matrix of T from
the basis B to the basis C has as its j-th column the C-coordinates of T(b_j), so it is C^-1 A B where A is the
matrix of T in the standard bases. It is unique, so it is compared exactly.

* easy: T(x, y) given by a formula, and a non-standard basis B of R^2 used for domain and codomain.
* medium: a map on quadratics given by a formula (p' + c p, x p' - c p, p(x + s) - p(x), or p(1) x + p'(0)) relative
  to a non-standard basis B of P_2; or differentiation from P_3 with the shifted basis 1, (x - a), (x - a)^2,
  (x - a)^3 to P_2 with the standard basis.
* hard: a map from P_2 to R^2 given by two evaluation functionals relative to a non-standard basis of each space
  (a 2x3 matrix); the derivative operator plus a multiple of the identity on span{e^(kx), x e^(kx), x^2 e^(kx)} with the
  basis listed in reverse order; or a similarity in R^3, either A' = P^-1 A P for the basis given by the columns of P,
  or the standard matrix A = P M P^-1 recovered from the matrix M of T relative to that basis.
* exam (``exd_x_s2b1_map_matrix``): T(p) = a p + b p' + c x p' on P_2. (a) the trace of the standard matrix, (b) the
  matrix relative to a non-standard basis B, (c) the trace of that matrix (follows through from (a): similar matrices
  have equal trace), (d) its determinant.

The hand route is C^-1 A B in exact fractions; the independent route applies the formula to each basis element with
SymPy and solves for its coordinates in the target basis.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator, join_terms, phrase, pick_distractors
from ._s2b1_linalg import (
    plain_stem,
    draw_cols,
    int_distractors,
    mat_answer,
    X,
    mat_distractors,
    poly_coeffs,
    poly_expr,
    poly_text,
    std_matrix,
    sym,
    sym_value,
    vector_text,
)
from ._ucv1_linalg import det, fm, inverse, mat_str, max_abs, mmul, transpose, unimodular, with_columns

LEVEL = 19
_X = X


_poly = poly_expr
_coeffs = poly_coeffs
_std_matrix = std_matrix


def _map_matrix(a, bcols, ccols):
    """[T]_{B->C} = C^-1 A B (hand route)."""
    return mmul(inverse(with_columns(ccols)), mmul(a, with_columns(bcols)))


def _independent_check(images, ccols, expected):
    """SymPy: solve C x = T(b_j) for each j and compare with the columns of `expected`."""
    cmat = sym(with_columns(ccols))
    for j, image in enumerate(images):
        sol = cmat.LUsolve(
            sympy.Matrix([sympy.Rational(Fraction(v).numerator, Fraction(v).denominator) for v in image])
        )
        check(
            [sym_value(v) for v in sol] == [expected[i][j] for i in range(len(expected))],
            "SymPy's matrix column differs",
        )


def _candidates(a, bcols, ccols, m, square=True):
    b, c = with_columns(bcols), with_columns(ccols)
    cands = [
        (a, "map-matrix-standard-used"),
        (mmul(a, b), "map-matrix-forgot-output-basis"),
        (mmul(inverse(c), a), "map-matrix-forgot-input-basis"),
        (transpose(m), "map-matrix-transposed"),
    ]
    if square:
        cands.append((mmul(mmul(b, a), inverse(c)), "diag-conjugation-reversed"))
    return cands


def _finish(text, m, cands, working, n=4):
    answer = mat_answer(m)
    distractors = pick_distractors(answer["answer_text"], mat_distractors(cands, m), n)
    if len(distractors) < 3:
        return None
    return dict(text=text, working=working, distractors=distractors, **answer)


# -- easy ----------------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(500, "a plane map whose matrix in the basis differs from its standard matrix"):
        a = [[Fraction(rng.randint(-4, 4)) for _ in range(2)] for _ in range(2)]
        bcols = draw_cols(rng, 2, -3, 3, 1, 2)
        m = _map_matrix(a, bcols, bcols)
        if (
            all(a[i][j] != 0 for i in range(2) for j in range(2))
            and m != a
            and max_abs(m) <= 14
            and bcols != [[1, 0], [0, 1]]
        ):
            break
    images = [[sum(a[i][j] * bcols[k][j] for j in range(2)) for i in range(2)] for k in range(2)]
    _independent_check(images, bcols, m)
    f = (
        f"T(x, y) = ({join_terms([(int(a[0][0]), 'x'), (int(a[0][1]), 'y')])}, "
        f"{join_terms([(int(a[1][0]), 'x'), (int(a[1][1]), 'y')])})"
    )
    stem = phrase(
        rng,
        (
            "The linear map T: ℝ² → ℝ² is given by {f}. Find the matrix of T relative to the basis B = "
            "{{{b1}, {b2}}} (used for both domain and codomain).",
            "Let T: ℝ² → ℝ² be the linear map {f}, and let B = {{{b1}, {b2}}} be a basis of ℝ². Find the matrix [T]_B "
            "of T relative to B, whose j-th column is the coordinate vector of T(bⱼ) relative to B.",
        ),
        f=f,
        b1=vector_text(bcols[0]),
        b2=vector_text(bcols[1]),
    )
    working = [
        f"Compute T(b₁) = {vector_text(images[0])} and T(b₂) = {vector_text(images[1])}, then write each in terms of b₁ and b₂.",
        f"The coordinate vectors are the columns of [T]_B = {mat_str(m)}.",
    ]
    return _finish(stem, m, _candidates(a, bcols, bcols, m), working)


# -- medium: maps on polynomials -----------------------------------------------------------------------------------------


def _families(rng):
    """(description, apply) pairs for maps P_2 -> P_2, drawn with random parameters."""
    c = rng.choice([-3, -2, -1, 1, 2, 3])
    s = rng.choice([-2, -1, 1, 2])
    d = rng.choice([-2, -1, 1, 2])
    fam = [
        (join_terms([(1, "p'(x)"), (c, "p(x)")]), lambda p: sympy.diff(p, _X) + c * p),
        (join_terms([(1, "x p'(x)"), (-c, "p(x)")]), lambda p: _X * sympy.diff(p, _X) - c * p),
        (
            f"p(x {'+' if s > 0 else '-'} {abs(s)}) - p(x)",
            lambda p: sympy.expand(p.subs(_X, _X + s) - p),
        ),
        (
            join_terms([(1, "p'(x)"), (d, "x p'(x)"), (c, "p(1)")]),
            lambda p: sympy.diff(p, _X) + d * _X * sympy.diff(p, _X) + c * p.subs(_X, 1),
        ),
    ]
    return fam


def _medium_poly_map(rng):
    description, apply = rng.choice(_families(rng))
    for _ in attempts(500, "a quadratic map with a non-standard basis"):
        bcols = draw_cols(rng, 3, -2, 2, 1, 1)
        a = _std_matrix(apply, 3, 2)
        m = _map_matrix(a, bcols, bcols)
        if m != a and max_abs(m) <= 14:
            break
    images = [_coeffs(apply(_poly(col)), 2) for col in bcols]
    _independent_check(images, bcols, m)
    stem = phrase(
        rng,
        (
            "Let P₂ be the space of polynomials of degree at most 2 and let T: P₂ → P₂ be the linear map "
            "T(p)(x) = {f}. Find the matrix of T relative to the basis B = {{{b1}, {b2}, {b3}}} (for both domain and "
            "codomain).",
            "The map T: P₂ → P₂ is linear, with T(p)(x) = {f}. Find [T]_B, the matrix of T relative to the basis B = "
            "{{{b1}, {b2}, {b3}}} of P₂, where column j holds the coordinates of T(bⱼ).",
        ),
        f=description,
        b1=poly_text(bcols[0]),
        b2=poly_text(bcols[1]),
        b3=poly_text(bcols[2]),
    )
    working = [
        "Apply T to each polynomial of B and write the result as a combination of b₁, b₂, b₃ (compare coefficients).",
        f"The coordinate columns give [T]_B = {mat_str(m)}.",
    ]
    return _finish(stem, m, _candidates(a, bcols, bcols, m), working)


def _shifted_text(a: int) -> str:
    return f"(x - {a})" if a > 0 else f"(x + {-a})"


def _medium_derivative(rng):
    a = rng.choice([-3, -2, -1, 1, 2, 3])
    bcols = [_coeffs((_X - a) ** k, 3) for k in range(4)]
    ccols = [[Fraction(int(i == j)) for i in range(3)] for j in range(3)]
    amat = _std_matrix(lambda p: sympy.diff(p, _X), 4, 2)
    m = _map_matrix(amat, bcols, ccols)
    images = [_coeffs(sympy.diff((_X - a) ** k, _X), 2) for k in range(4)]
    _independent_check(images, ccols, m)
    s = _shifted_text(a)
    stem = phrase(
        rng,
        (
            "Differentiation D: P₃ → P₂ sends p to p'. Find the matrix of D relative to the basis {{1, {s}, {s}², {s}³}} of "
            "P₃ and the basis {{1, x, x²}} of P₂.",
            "Let D(p) = p' define a linear map from P₃, the cubics, to P₂. Find the matrix of D from the basis "
            "B = {{1, {s}, {s}², {s}³}} of P₃ to the basis C = {{1, x, x²}} of P₂.",
        ),
        s=s,
    )
    cands = [
        (transpose(m), "map-matrix-transposed"),
        (
            [[row[0], *[v / (k + 1) if k < 3 else v for k, v in enumerate(row[1:])]] for row in m],
            "map-matrix-derivative-factor-dropped",
        ),
        (_map_matrix(amat, [_coeffs((_X + a) ** k, 3) for k in range(4)], ccols), "taylor-centre-sign-error"),
        (amat, "map-matrix-standard-used"),
    ]
    working = [
        f"D({s}ᵏ) = k{s}ᵏ⁻¹; expand each in powers of x to get the C-coordinates.",
        f"The columns give the 3×4 matrix {mat_str(m)}.",
    ]
    return _finish(stem, m, cands, working)


# -- hard ------------------------------------------------------------------------------------------------------------------


def _hard_functionals(rng):
    """T: P_2 -> R^2, T(p) = (L1(p), L2(p)), relative to a basis of each space (a 2x3 matrix)."""
    pool = [
        ("p(1)", lambda p: p.subs(_X, 1)),
        ("p(-1)", lambda p: p.subs(_X, -1)),
        ("p(2)", lambda p: p.subs(_X, 2)),
        ("p'(0)", lambda p: sympy.diff(p, _X).subs(_X, 0)),
        ("p'(1)", lambda p: sympy.diff(p, _X).subs(_X, 1)),
        ("p(0) + p'(1)", lambda p: p.subs(_X, 0) + sympy.diff(p, _X).subs(_X, 1)),
    ]
    for _ in attempts(500, "two evaluation functionals with a 2x3 matrix"):
        (t1, f1), (t2, f2) = rng.sample(pool, 2)
        bcols = draw_cols(rng, 3, -2, 2, 1, 2)
        ccols = draw_cols(rng, 2, -2, 3, 1, 2)
        a = [[Fraction(int(f(_X**j))) for j in range(3)] for f in (f1, f2)]
        m = _map_matrix(a, bcols, ccols)
        if max_abs(m) <= 20 and m != a and sym(a).rank() == 2:
            break
    images = [[Fraction(int(f(_poly(col)))) for f in (f1, f2)] for col in bcols]
    _independent_check(images, ccols, m)
    stem = phrase(
        rng,
        (
            "The map T: P₂ → ℝ² is linear, with T(p) = ({t1}, {t2}). Find the matrix of T relative to the basis "
            "B = {{{b1}, {b2}, {b3}}} of P₂ and the basis C = {{{c1}, {c2}}} of ℝ².",
            "Let T: P₂ → ℝ² be the linear map T(p) = ({t1}, {t2}). Find the 2×3 matrix whose j-th column is the "
            "coordinate vector, relative to C = {{{c1}, {c2}}}, of T(bⱼ), where B = {{{b1}, {b2}, {b3}}} is a basis of P₂.",
        ),
        t1=t1,
        t2=t2,
        b1=poly_text(bcols[0]),
        b2=poly_text(bcols[1]),
        b3=poly_text(bcols[2]),
        c1=vector_text(ccols[0]),
        c2=vector_text(ccols[1]),
    )
    cands = [
        (a, "map-matrix-standard-used"),
        (mmul(a, with_columns(bcols)), "map-matrix-forgot-output-basis"),
        (mmul(inverse(with_columns(ccols)), a), "map-matrix-forgot-input-basis"),
        (transpose(m), "map-matrix-transposed"),
    ]
    working = [
        "Evaluate T on each polynomial of B to get vectors of ℝ², then write each as a combination of c₁ and c₂.",
        f"The coordinate columns give [T] = {mat_str(m)}, a 2×3 matrix.",
    ]
    return _finish(stem, m, cands, working)


def _hard_exponential(rng):
    lam = rng.choice([-3, -2, -1, 1, 2, 3])
    shift = rng.choice([-2, -1, 1, 2])
    # D + shift*I on the span of e^(lam x), x e^(lam x), x^2 e^(lam x), basis listed in the order x^2 e, x e, e.
    e = sympy.exp(lam * _X)
    basis = [_X**2 * e, _X * e, e]
    images = [sympy.expand(sympy.diff(b, _X) + shift * b) for b in basis]
    m = [[Fraction(0)] * 3 for _ in range(3)]
    for j, image in enumerate(images):
        coeff = sympy.Poly(sympy.simplify(image / e), _X)
        for i in range(3):
            m[i][j] = Fraction(int(coeff.coeff_monomial(_X ** (2 - i))))
    # Independent: rebuild each image from the columns of m.
    for j in range(3):
        rebuilt = sum(int(m[i][j]) * basis[i] for i in range(3))
        check(sympy.simplify(rebuilt - images[j]) == 0, "SymPy's exponential matrix differs")
    op = f"T(f) = f' {'+' if shift > 0 else '-'} {abs(shift)}f"
    ex = f"e^({lam}x)" if lam != 1 else "eˣ"
    names = [f"x²{ex}", f"x{ex}", ex]
    stem = phrase(
        rng,
        (
            "On the space of functions spanned by B = {{{b1}, {b2}, {b3}}} (in this order) the linear map is {op}. Find the "
            "matrix of T relative to B.",
            "Let V be the space of real functions with basis B = {{{b1}, {b2}, {b3}}}, listed in this order, and let T: V → V be "
            "T(f) = f' {sgn} {mag}f. Find the matrix [T]_B.",
        ),
        op=op,
        sgn="+" if shift > 0 else "-",
        mag=abs(shift),
        b1=names[0],
        b2=names[1],
        b3=names[2],
    )
    swapped = [row[::-1] for row in m[::-1]]
    cands = [
        (swapped, "map-matrix-basis-order-ignored"),
        (transpose(m), "map-matrix-transposed"),
        ([[m[i][j] if i != j else m[i][j] - shift for j in range(3)] for i in range(3)], "map-matrix-term-dropped"),
        (
            [
                [m[i][j] / (2 - j) if (i != j and 2 - j != 0 and m[i][j] != 0) else m[i][j] for j in range(3)]
                for i in range(3)
            ],
            "map-matrix-derivative-factor-dropped",
        ),
    ]
    working = [
        f"T(x²{ex}) = 2x{ex} + ({lam + shift})x²{ex}, T(x{ex}) = {ex} + ({lam + shift})x{ex}, T({ex}) = ({lam + shift}){ex}.",
        f"Reading the coordinates relative to B (x²{ex} first) gives [T]_B = {mat_str(m)}.",
    ]
    return _finish(stem, m, cands, working)


def _hard_similarity(rng):
    forward = rng.random() < 0.6
    for _ in attempts(500, "a 3x3 similarity with bounded entries"):
        a = [[Fraction(rng.randint(-3, 3)) for _ in range(3)] for _ in range(3)]
        pcols = draw_cols(rng, 3, -2, 2, 1, 2)
        p = with_columns(pcols)
        mm = mmul(mmul(inverse(p), a), p)
        if forward:
            target, given = mm, a
        else:
            target, given = a, mm
        if max_abs(mm) <= 14 and mm != a and sum(1 for row in a for v in row if v == 0) <= 3 and max_abs(given) <= 14:
            break
    if forward:
        check(sym(p) * sym(mm) == sym(a) * sym(p), "P A' != A P")
        stem = phrase(
            rng,
            (
                "The linear map T: ℝ³ → ℝ³ has matrix A = {a} relative to the standard basis. Find its matrix relative to the basis "
                "B = {{{b1}, {b2}, {b3}}}, that is, A' = P⁻¹AP where the columns of P are the vectors of B.",
                "Relative to the standard basis of ℝ³ the map T has matrix A = {a}. Find the matrix of T relative to the basis "
                "B = {{{b1}, {b2}, {b3}}}.",
            ),
            a=mat_str(a),
            b1=vector_text(pcols[0]),
            b2=vector_text(pcols[1]),
            b3=vector_text(pcols[2]),
        )
        cands = [
            (mmul(mmul(p, a), inverse(p)), "diag-conjugation-reversed"),
            (mmul(a, p), "map-matrix-forgot-input-basis"),
            (mmul(inverse(p), a), "diag-forgot-inverse"),
            (transpose(target), "map-matrix-transposed"),
            (a, "map-matrix-standard-used"),
        ]
        working = [
            "A' = P⁻¹AP, where P has the vectors of B as its columns.",
            f"Multiplying out gives A' = {mat_str(target)}.",
        ]
    else:
        check(sym(p) * sym(mm) == sym(a) * sym(p), "P M != A P")
        stem = phrase(
            rng,
            (
                "The linear map T: ℝ³ → ℝ³ has matrix M = {m} relative to the basis B = {{{b1}, {b2}, {b3}}}. Find the matrix of "
                "T relative to the standard basis of ℝ³.",
                "Relative to the basis B = {{{b1}, {b2}, {b3}}} of ℝ³, the map T has matrix M = {m}. Find the standard matrix "
                "A of T, the matrix with T(x) = Ax for every x in ℝ³.",
            ),
            m=mat_str(mm),
            b1=vector_text(pcols[0]),
            b2=vector_text(pcols[1]),
            b3=vector_text(pcols[2]),
        )
        cands = [
            (mmul(mmul(inverse(p), mm), p), "diag-conjugation-reversed"),
            (mmul(mm, inverse(p)), "diag-forgot-inverse"),
            (mmul(p, mm), "map-matrix-forgot-input-basis"),
            (transpose(target), "map-matrix-transposed"),
            (mm, "map-matrix-standard-used"),
        ]
        working = [
            "M = P⁻¹AP, so A = PMP⁻¹ where P has the vectors of B as its columns.",
            f"Multiplying out gives A = {mat_str(target)}.",
        ]
    return _finish(stem, target, cands, working)


def _retry(rng, build):
    for _ in attempts(200, "a question with enough distractors"):
        result = build(rng)
        if result is not None:
            return result
    raise AssertionError  # pragma: no cover


@generator(
    "s2b1_matrix_of_linear_map",
    level=LEVEL,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@plain_stem
def s2b1_matrix_of_linear_map(rng, tier="easy"):
    """Matrix of a linear map relative to given bases; the tiers are described in the module docstring."""
    if tier == "easy":
        return _retry(rng, _easy)
    if tier == "medium":
        return _retry(rng, rng.choice([_medium_poly_map, _medium_poly_map, _medium_derivative]))
    check(tier == "hard", f"unknown tier {tier!r}")
    return _retry(rng, rng.choice([_hard_functionals, _hard_exponential, _hard_similarity, _hard_similarity]))


# -- exam ----------------------------------------------------------------------------------------------------------------------


def _exam_try(rng):
    alpha, beta, gamma = rng.randint(-3, 3), rng.choice([-3, -2, -1, 1, 2, 3]), rng.choice([-2, -1, 1, 2])
    diag = [alpha, alpha + gamma, alpha + 2 * gamma]
    if 0 in diag:
        return None

    def apply(p):
        return alpha * p + beta * sympy.diff(p, _X) + gamma * _X * sympy.diff(p, _X)

    a = _std_matrix(apply, 3, 2)
    check(
        a == fm([[alpha, beta, 0], [0, alpha + gamma, 2 * beta], [0, 0, alpha + 2 * gamma]]), "standard matrix differs"
    )
    bcols = unimodular(rng, 3, 2)
    bcols = [list(map(int, col)) for col in transpose(bcols)]
    m = _map_matrix(a, bcols, bcols)
    if m == a or max_abs(m) > 30:
        return None
    images = [_coeffs(apply(_poly(col)), 2) for col in bcols]
    _independent_check(images, bcols, m)
    trace = sum(diag)
    detv = diag[0] * diag[1] * diag[2]
    check(sum(m[i][i] for i in range(3)) == trace, "trace of the similar matrix differs")
    check(det(m) == detv, "determinant of the similar matrix differs")

    a_dist = pick_distractors(
        str(trace),
        int_distractors(
            [
                (3 * alpha + 2 * gamma, "map-matrix-derivative-factor-dropped"),
                (3 * alpha, "map-matrix-term-dropped"),
                (trace + beta, "map-matrix-trace-includes-offdiagonal"),
                (3 * alpha - 3 * gamma, "sign-error-moving-term"),
            ],
            trace,
        ),
        3,
    )
    cands = _candidates(a, bcols, bcols, m)
    b_ans = mat_answer(m)
    b_dist = pick_distractors(b_ans["answer_text"], mat_distractors(cands, m), 3)
    c_dist = pick_distractors(
        str(trace),
        int_distractors(
            [
                (
                    sum(a[i][i] for i in range(3)) + sum(m[i][j] for i in range(3) for j in range(3) if i != j),
                    "map-matrix-trace-includes-offdiagonal",
                ),
                (int(sum(m[i][i] for i in range(3)) - m[0][0] + m[0][1]), "map-matrix-trace-includes-offdiagonal"),
                (-trace, "sign-error-moving-term"),
                (3 * alpha, "map-matrix-term-dropped"),
            ],
            trace,
        ),
        3,
    )
    d_dist = pick_distractors(
        str(detv),
        int_distractors(
            [
                (diag[0] * diag[1] * (alpha + gamma), "map-matrix-derivative-factor-dropped"),
                (trace, "map-matrix-det-for-trace"),
                (-detv, "det-sarrus-sign-slip"),
                (diag[0] + diag[1] + diag[2] + detv, "map-matrix-det-for-trace"),
            ],
            detv,
        ),
        3,
    )
    if not all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)):
        return None
    f = join_terms([(alpha, "p(x)"), (beta, "p'(x)"), (gamma, "x p'(x)")])
    stem = (
        f"Let T: P₂ → P₂ be the linear map T(p)(x) = {f}, where P₂ is the space of polynomials of degree at most 2. Let "
        f"S = {{1, x, x²}} and B = {{{poly_text(bcols[0])}, {poly_text(bcols[1])}, {poly_text(bcols[2])}}}, both bases of P₂."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the trace of the matrix of T relative to S.",
                answer_kind="number",
                answer=trace,
                answer_text=str(trace),
                marks=3,
                working=[
                    f"T(xᵏ) has leading term ({join_terms([(alpha, ''), (gamma, 'k')])})xᵏ, so the matrix is upper triangular with diagonal {diag}; the trace is {trace}."
                ],
                distractors=a_dist,
            ),
            dict(
                prompt="Find the matrix of T relative to B (for both domain and codomain).",
                marks=4,
                working=[f"[T]_B = P⁻¹AP with the coefficient vectors of B as the columns of P: {mat_str(m)}."],
                distractors=b_dist,
                **b_ans,
            ),
            dict(
                prompt="Use your answer to (a) to write down the trace of the matrix in (b).",
                answer_kind="number",
                answer=trace,
                answer_text=str(trace),
                marks=1,
                follow_through="part_a",
                working=[
                    "Matrices of the same map in different bases are similar, and similar matrices have equal trace."
                ],
                distractors=c_dist,
            ),
            dict(
                prompt="Find the determinant of the matrix in (b).",
                answer_kind="number",
                answer=detv,
                answer_text=str(detv),
                marks=2,
                working=[f"The determinant is the same in every basis: {diag[0]} × {diag[1]} × {diag[2]} = {detv}."],
                distractors=d_dist,
            ),
        ],
    )


@generator("exd_x_s2b1_map_matrix", level=LEVEL, presets={"exam": dict(tier="exam")})
@plain_stem
def exd_x_s2b1_map_matrix(rng, tier="exam"):
    """Exam chain: trace of the standard matrix, the matrix in another basis, and the invariants trace and determinant."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _retry(rng, _exam_try)
