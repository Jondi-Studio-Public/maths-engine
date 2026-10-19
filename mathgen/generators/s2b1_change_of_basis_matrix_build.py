"""Generates `s2b1_change_of_basis_matrix` questions (easy, medium, hard) and the exam chain `exd_x_s2b1_change_basis`.

``uc2-change-of-basis-matrix`` (``linear-algebra``, answer ``matrix``): the transition matrix P between two bases and
converting coordinates with it. P is the matrix with [v]_C = P [v]_B, so its columns are the C-coordinates of the
vectors of B, and P = C^-1 B when C and B hold the basis vectors in their columns. It is unique, so it is compared
exactly.

* easy: two non-standard bases of R^2; the transition matrix from B to C.
* medium: two non-standard bases of R^3 or of the quadratics P_2; the transition matrix from B to C.
* hard: one of three situations that need the direction and the order of the factors to be understood: convert
  coordinates when the matrix supplied runs the other way (the transition matrix from C to B is given, [v]_B is
  given, find [v]_C); compose two supplied transitions (B to C and C to D give B to D); the transition between the two
  shifted bases 1, (x - a), (x - a)^2 and 1, (x - b), (x - b)^2 of P_2.
* exam (``exd_x_s2b1_change_basis``): two bases of R^2, C of determinant 1 or -1. (a) det P, (b) P, (c) [v]_C for a v
  given by its B-coordinates, (d) the (1, 2) entry of the transition matrix from C back to B, which follows through
  from (a) because it is -P12 / det P.

Each matrix is built from integer bases, and each answer is re-derived with SymPy (``C * P == B``, and for the
conversions the vector rebuilt in both bases).
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, join_terms, phrase, pick_distractors, rational_text
from ._s2b1_linalg import (
    plain_stem,
    draw_cols,
    mat_answer,
    mat_distractors,
    poly_text,
    sym,
    vec_answer,
    vec_distractors,
    vector_text,
)
from ._ucv1_linalg import apply, det, fm, inverse, mat_str, max_abs, mmul, transpose, unimodular, vec_str, with_columns

LEVEL = 19

_DEFINITION = "the matrix P with [v]_C = P [v]_B for every v"


def _transition(bcols, ccols):
    """P = C^-1 B (hand route), checked against SymPy: C P = B."""
    b, c = with_columns(bcols), with_columns(ccols)
    p = mmul(inverse(c), b)
    check(sym(c) * sym(p) == sym(b), "C P differs from B")
    return p


def _p_candidates(bcols, ccols, p):
    b, c = with_columns(bcols), with_columns(ccols)
    return [
        (inverse(p), "change-basis-direction-reversed"),
        (transpose(p), "change-basis-transposed"),
        (mmul(b, inverse(c)), "change-basis-product-order"),
        (b, "change-basis-forgot-target-inverse"),
        (transpose(inverse(p)), "change-basis-direction-reversed"),
    ]


def _matrix_dict(text, p, candidates, working, n=4):
    answer = mat_answer(p)
    distractors = pick_distractors(answer["answer_text"], mat_distractors(candidates, p), n)
    if len(distractors) < 3:
        return None
    return dict(text=text, working=working, distractors=distractors, **answer)


# -- easy and medium ---------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "In ℝ² let B = {{{b1}, {b2}}} and C = {{{c1}, {c2}}}; both are bases. Find the transition matrix from B to C, "
    "that is, " + _DEFINITION + ".",
    "The sets B = {{{b1}, {b2}}} and C = {{{c1}, {c2}}} are two bases of ℝ². Write down " + _DEFINITION + ", where "
    "[v]_B and [v]_C are the coordinate vectors of v relative to B and to C.",
    "Two bases of ℝ² are B = {{{b1}, {b2}}} and C = {{{c1}, {c2}}}. Find the matrix that turns B-coordinates into "
    "C-coordinates, that is, " + _DEFINITION + ".",
)

_EXAM_STEMS = (
    "In ℝ² let B = {{{b1}, {b2}}} and C = {{{c1}, {c2}}} be bases, and let P be the transition matrix from B to C, "
    + _DEFINITION
    + ".",
    "The sets B = {{{b1}, {b2}}} and C = {{{c1}, {c2}}} are two bases of ℝ². Write P for the transition matrix from "
    "B to C, " + _DEFINITION + ".",
    "Let B = {{{b1}, {b2}}} and C = {{{c1}, {c2}}} be bases of ℝ², and let P be the matrix that turns B-coordinates "
    "into C-coordinates, " + _DEFINITION + ".",
)


def _not_standard(cols):
    n = len(cols)
    return not all(sorted(map(abs, c)) == [0] * (n - 1) + [1] for c in cols)


def _easy(rng):
    for _ in attempts(500, "two R^2 bases with a fractional transition"):
        bcols = draw_cols(rng, 2, -3, 4, 1, 4)
        ccols = draw_cols(rng, 2, -3, 4, 2, 4)
        p = _transition(bcols, ccols)
        if _not_standard(bcols) and _not_standard(ccols) and bcols != ccols and max_abs(p) <= 8:
            break
    working = [
        "The columns of P are the C-coordinates of the vectors of B, so P = C⁻¹B where the columns of B and C are the basis vectors.",
        f"C⁻¹B = {mat_str(p)}.",
    ]
    slots = dict(b1=vector_text(bcols[0]), b2=vector_text(bcols[1]), c1=vector_text(ccols[0]), c2=vector_text(ccols[1]))
    return _matrix_dict(phrase(rng, _EASY_STEMS, **slots), p, _p_candidates(bcols, ccols, p), working)


_MED_STEMS_V = (
    "In ℝ³ let B = {{{b1}, {b2}, {b3}}} and C = {{{c1}, {c2}, {c3}}}; both are bases. Find the transition matrix from "
    "B to C, that is, " + _DEFINITION + ".",
    "Two bases of ℝ³ are B = {{{b1}, {b2}, {b3}}} and C = {{{c1}, {c2}, {c3}}}. Find " + _DEFINITION + ", where "
    "[v]_B and [v]_C are the coordinate vectors of v relative to B and to C.",
)
_MED_STEMS_P = (
    "In P₂, the space of polynomials of degree at most 2, let B = {{{b1}, {b2}, {b3}}} and C = {{{c1}, {c2}, {c3}}}; "
    "both are bases. Find the transition matrix from B to C, that is, " + _DEFINITION + ".",
    "Two bases of P₂ are B = {{{b1}, {b2}, {b3}}} and C = {{{c1}, {c2}, {c3}}}. Find "
    + _DEFINITION
    + ", where [v]_B and [v]_C "
    "are the coordinate vectors of v relative to B and to C.",
)


def _medium(rng):
    space = rng.choice(["R3", "P2"])
    for _ in attempts(500, "two 3-dimensional bases"):
        bcols = draw_cols(rng, 3, -2, 3, 1, 3)
        ccols = draw_cols(rng, 3, -2, 3, 1, 2)
        p = _transition(bcols, ccols)
        if (
            _not_standard(bcols)
            and _not_standard(ccols)
            and max_abs(p) <= 9
            and any(v.denominator > 1 for r in p for v in r)
        ):
            break
    show = vector_text if space == "R3" else poly_text
    slots = {f"b{i + 1}": show(bcols[i]) for i in range(3)} | {f"c{i + 1}": show(ccols[i]) for i in range(3)}
    where = "the vectors" if space == "R3" else "the polynomials (coefficients from the constant term up)"
    working = [
        f"Write {where} of B and C as columns; the columns of P are the C-coordinates of the vectors of B, so P = C⁻¹B.",
        f"Row reducing [C | B] gives P = {mat_str(p)}.",
    ]
    stem = phrase(rng, _MED_STEMS_V if space == "R3" else _MED_STEMS_P, **slots)
    return _matrix_dict(stem, p, _p_candidates(bcols, ccols, p), working)


# -- hard ----------------------------------------------------------------------------------------------------------------


def _hard_convert(rng):
    """[v]_B and the transition matrix from C to B are supplied; the answer is [v]_C = P^-1 [v]_B."""
    n = 3
    for _ in attempts(500, "an invertible integer P with an integer solution"):
        p = unimodular(rng, n, 2) if rng.random() < 0.5 else fm(draw_cols(rng, n, -2, 3, 2, 2))
        u = [rng.choice([-3, -2, -1, 1, 2, 3, 4]) for _ in range(n)]
        vb = apply(p, u)
        if max_abs(p) <= 4 and det(p) != 0 and max(abs(x) for x in vb) <= 18 and max_abs(fm([u])) >= 2:
            break
    # Independent route: SymPy solves P x = [v]_B.
    check(
        list(sym(p).LUsolve(sympy.Matrix([int(x) for x in vb]))) == [sympy.Integer(x) for x in u],
        "SymPy's [v]_C differs",
    )
    ans = vec_answer([Fraction(x) for x in u])
    pt_inv = mmul(inverse(transpose(p)), [[x] for x in vb])
    candidates = [
        (apply(p, vb), "change-basis-direction-reversed"),
        ([r[0] for r in pt_inv], "change-basis-transposed"),
        (apply(transpose(p), vb), "change-basis-transposed"),
        (list(vb), "coords-standard-used"),
        (list(reversed(u)), "coords-order-swapped"),
    ]
    distractors = pick_distractors(ans["answer_text"], vec_distractors(candidates, u), 4)
    if len(distractors) < 3:
        return None
    stem = phrase(
        rng,
        (
            "B and C are bases of ℝ³. The transition matrix from C to B, the matrix with [v]_B = P [v]_C for every v, "
            "is P = {p}. A vector v has [v]_B = {vb}. Find [v]_C.",
            "Let B and C be two bases of ℝ³, and let P = {p} be the matrix with [v]_B = P [v]_C for every v. If "
            "[v]_B = {vb}, find the coordinate vector [v]_C.",
        ),
        p=mat_str(p),
        vb=vector_text(vb),
    )
    working = [
        "P turns C-coordinates into B-coordinates, so [v]_C = P⁻¹[v]_B: solve P x = [v]_B.",
        f"Row reduction gives [v]_C = {vec_str(u)}.",
    ]
    return dict(text=stem, working=working, distractors=distractors, **ans)


def _hard_compose(rng):
    n = rng.choice([2, 3])
    for _ in attempts(500, "two transitions whose product is not their other product"):
        a = unimodular(rng, n, 3) if rng.random() < 0.5 else fm(draw_cols(rng, n, -2, 3, 1, 3))
        m = unimodular(rng, n, 3) if rng.random() < 0.5 else fm(draw_cols(rng, n, -2, 3, 1, 3))
        prod = mmul(m, a)
        if mmul(a, m) != prod and max_abs(prod) <= 18 and max_abs(a) <= 5 and max_abs(m) <= 5:
            break
    check(sym(prod) == sym(m) * sym(a), "SymPy's product differs")
    candidates = [
        (mmul(a, m), "change-basis-product-order"),
        (inverse(prod), "change-basis-direction-reversed"),
        (transpose(prod), "change-basis-transposed"),
        (mmul(inverse(m), a), "change-basis-direction-reversed"),
        (mmul(m, inverse(a)), "change-basis-direction-reversed"),
    ]
    stem = phrase(
        rng,
        (
            "B, C and D are bases of ℝ{n}. The transition matrix M from X to Y is defined by [v]_Y = M [v]_X. The "
            "transition matrix from B to C is P = {a} and the transition matrix from C to D is Q = {m}. Find the "
            "transition matrix from B to D.",
            "Let B, C and D be bases of ℝ{n}. Going from B to C the coordinates change by P = {a}, and going from C to D "
            "they change by Q = {m} (coordinates in the new basis = matrix times coordinates in the old). Find the "
            "matrix that takes [v]_B to [v]_D.",
        ),
        n="²" if n == 2 else "³",
        a=mat_str(a),
        m=mat_str(m),
    )
    working = [
        "[v]_C = P[v]_B and [v]_D = Q[v]_C, so [v]_D = QP[v]_B.",
        f"The matrix is QP = {mat_str(prod)} (the later transition goes on the left).",
    ]
    return _matrix_dict(stem, prod, candidates, working)


def _shift(a: int) -> str:
    return f"(x - {a})" if a > 0 else f"(x + {-a})" if a < 0 else "x"


def _hard_shifted(rng):
    for _ in attempts(100, "two different centres"):
        a, b = rng.randint(-3, 3), rng.randint(-3, 3)
        if a != b and (a != 0 or b != 0):
            break
    d = b - a
    x = sympy.symbols("x")
    # Columns: the C-coordinates of 1, (x - a), (x - a)^2, found by expanding about b with SymPy.
    cols = []
    for k in range(3):
        expanded = sympy.Poly(sympy.expand((x - a) ** k), x)
        shifted = sympy.Poly(sympy.expand(expanded.as_expr().subs(x, x + b)), x)
        cols.append([int(shifted.coeff_monomial(x**j)) for j in range(3)])
    p = fm(transpose(fm(cols)))
    hand = [[1, d, d * d], [0, 1, 2 * d], [0, 0, 1]]
    check(p == fm(hand), "the shifted transition differs from the binomial expansion")
    candidates = [
        (inverse(p), "change-basis-direction-reversed"),
        (transpose(p), "change-basis-transposed"),
        (fm([[1, -d, d * d], [0, 1, -2 * d], [0, 0, 1]]), "taylor-centre-sign-error"),
        (fm([[1, d, d], [0, 1, d], [0, 0, 1]]), "taylor-derivative-not-divided"),
        (fm([[1, 0, 0], [d, 1, 0], [d * d, 2 * d, 1]]), "change-basis-transposed"),
    ]
    stem = phrase(
        rng,
        (
            "Let B = {{1, {sa}, {sa}²}} and C = {{1, {sb}, {sb}²}}, both bases of P₂. Find the transition matrix from B to C, "
            "that is, " + _DEFINITION + ".",
            "In P₂ take the bases B = {{1, {sa}, {sa}²}} and C = {{1, {sb}, {sb}²}}. Find " + _DEFINITION + ", where "
            "[v]_B and [v]_C are the coordinate vectors of v relative to B and to C.",
        ),
        sa=_shift(a),
        sb=_shift(b),
    )
    working = [
        f"Write each vector of B in powers of {_shift(b)}: {_shift(a)} = {_shift(b)} + ({d}), "
        f"and {_shift(a)}² = {join_terms([(1, _shift(b) + '²'), (2 * d, _shift(b)), (d * d, '')])}.",
        f"The columns of P are the coefficient vectors (in order 1, {_shift(b)}, {_shift(b)}²): P = {mat_str(p)}.",
    ]
    return _matrix_dict(stem, p, candidates, working)


def _retry(rng, build):
    """`build(rng)` again until it returns a question (it returns None when too few distractors survive)."""
    for _ in attempts(100, "a question with enough distractors"):
        result = build(rng)
        if result is not None:
            return result
    raise AssertionError  # pragma: no cover


@generator(
    "s2b1_change_of_basis_matrix",
    level=LEVEL,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@plain_stem
def s2b1_change_of_basis_matrix(rng, tier="easy"):
    """Transition matrix between bases and coordinate conversion; the tiers are described in the module docstring."""
    if tier == "easy":
        return _retry(rng, _easy)
    if tier == "medium":
        return _retry(rng, _medium)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _retry(rng, rng.choice([_hard_convert, _hard_compose, _hard_shifted]))


# -- exam ----------------------------------------------------------------------------------------------------------------


def _exam_try(rng):
    bcols = draw_cols(rng, 2, -3, 4, 4, 12)
    ccols = draw_cols(rng, 2, -3, 4, 2, 3)
    db, dc = int(sympy.Matrix(bcols).det()), int(sympy.Matrix(ccols).det())
    if db % dc != 0 or not (_not_standard(bcols) and _not_standard(ccols)):
        return None
    p = _transition(bcols, ccols)
    det_p = det(p)
    if abs(det_p) < 2 or max_abs(p) > 9 or p[0][1] == 0:
        return None
    check(
        det_p == Fraction(int(sympy.Matrix(bcols).det()), int(sympy.Matrix(ccols).det())),
        "det P differs from det B / det C",
    )
    coords_b = [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(2)]
    v = apply(with_columns(bcols), coords_b)
    coords_c = apply(p, coords_b)
    check(apply(with_columns(ccols), coords_c) == v, "[v]_C does not rebuild v")
    p_inv = inverse(p)
    entry = p_inv[0][1]
    check(entry == -p[0][1] / det_p, "the (1, 2) entry of P^-1 differs from -P12 / det P")
    check(Fraction(int(sym(p_inv)[0, 1].p), int(sym(p_inv)[0, 1].q)) == entry, "SymPy's inverse entry differs")

    det_val = int(det_p)
    a_dist = pick_distractors(
        str(det_val),
        [
            Distractor(str(-det_val), "det-sarrus-sign-slip", -det_val),
            Distractor(
                str(int(sympy.Matrix(bcols).det())),
                "change-basis-forgot-target-inverse",
                int(sympy.Matrix(bcols).det()),
            ),
            Distractor(
                str(int(sympy.Matrix(ccols).det() * sympy.Matrix(bcols).det())),
                "change-basis-product-order",
                int(sympy.Matrix(ccols).det() * sympy.Matrix(bcols).det()),
            ),
        ],
        3,
    )
    b_ans = mat_answer(p)
    b_dist = pick_distractors(b_ans["answer_text"], mat_distractors(_p_candidates(bcols, ccols, p), p), 3)
    c_ans = vec_answer(coords_c)
    c_cands = [
        (apply(inverse(p), coords_b), "change-basis-direction-reversed"),
        (apply(transpose(p), coords_b), "change-basis-transposed"),
        (list(coords_b), "coords-standard-used"),
        (list(reversed(coords_c)), "coords-order-swapped"),
    ]
    c_dist = pick_distractors(c_ans["answer_text"], vec_distractors(c_cands, coords_c), 3)
    d_text = rational_text(entry)
    d_cands = [
        (p[0][1], "change-basis-direction-reversed"),
        (-p[0][1], "inverse-adjugate-undivided"),
        (p[0][1] / det_p, "inverse-sign-pattern"),
        (p_inv[1][0], "change-basis-transposed"),
        (Fraction(1) / p[0][1] if p[0][1] else None, "inverse-entrywise-reciprocal"),
    ]
    d_dist = []
    for value, tag in d_cands:
        if value is None:
            continue
        d_dist.append(Distractor(rational_text(value), tag, exact_value(Fraction(value))))
    d_dist = pick_distractors(d_text, d_dist, 3)
    if not all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)):
        return None
    ft = f"-({rational_text(p[0][1])})/part_a"
    stem = phrase(
        rng,
        _EXAM_STEMS,
        b1=vector_text(bcols[0]),
        b2=vector_text(bcols[1]),
        c1=vector_text(ccols[0]),
        c2=vector_text(ccols[1]),
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the determinant of P.",
                answer_kind="number",
                answer=det_val,
                answer_text=str(det_val),
                marks=2,
                working=[f"P = C⁻¹B, so det P = det B / det C = {det_val}."],
                distractors=a_dist,
            ),
            dict(
                prompt="Find P.",
                marks=3,
                working=[f"The columns of P are the C-coordinates of the vectors of B: P = {mat_str(p)}."],
                distractors=b_dist,
                **b_ans,
            ),
            dict(
                prompt=f"A vector v has [v]_B = {vector_text(coords_b)}. Find [v]_C.",
                marks=2,
                working=[f"[v]_C = P[v]_B = {vec_str(coords_c)}."],
                distractors=c_dist,
                **c_ans,
            ),
            dict(
                prompt="Find the entry in row 1, column 2 of the transition matrix from C to B.",
                answer_kind="fraction",
                answer=exact_value(entry),
                answer_text=d_text,
                marks=2,
                follow_through=ft,
                working=[f"The matrix from C to B is P⁻¹, whose (1, 2) entry is -P₁₂ / det P = {d_text}."],
                distractors=d_dist,
            ),
        ],
    )


@generator("exd_x_s2b1_change_basis", level=LEVEL, presets={"exam": dict(tier="exam")})
@plain_stem
def exd_x_s2b1_change_basis(rng, tier="exam"):
    """Exam chain: det P, P, converted coordinates and an entry of the inverse transition."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "an exam chain with an integer determinant and enough distractors"):
        result = _exam_try(rng)
        if result is not None:
            return result
    raise AssertionError  # pragma: no cover
