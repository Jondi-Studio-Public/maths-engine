"""Generates `s2b1_coordinates_in_basis` questions (easy, medium, hard) and the exam chain `exd_x_s2b1_coordinates`.

``uc2-coordinates-in-basis`` (``linear-algebra``, answer ``matrix``): the coordinate vector of a vector, polynomial
or matrix relative to a supplied non-standard basis. The coordinates are unique, so the answer is compared exactly.

* easy: a vector of R^3 relative to a basis of three integer vectors (coordinates are whole numbers).
* medium: a quadratic polynomial relative to a basis of three quadratics (at least one coordinate is a fraction, because
  the basis determinant is 2 or 3 in absolute value).
* hard: one of three situations that need more than solving a square system by sight: a cubic polynomial relative to
  the shifted basis 1, (x - a), (x - a)^2, (x - a)^3 (its Taylor coefficients at a); a 2x2 matrix relative to a basis
  of four 2x2 matrices; a vector of a plane in R^4 relative to a basis of that plane (four equations, two unknowns,
  consistent).
* exam (``exd_x_s2b1_coordinates``): three vectors of R^3 (or three quadratics) in which one entry of the third is a
  parameter k. (a) the k for which they fail to be a basis, (b) the determinant at a stated k (follows through from
  (a)), (c) the coordinates of a vector at that k, (d) one standard component of the vector with stated coordinates.

Every basis is drawn with an integer matrix of non-zero determinant. The coordinates are found by Gauss-Jordan
elimination in exact fractions and re-derived with SymPy (`LUsolve`, `linsolve`, or a Taylor expansion), and the
vector is rebuilt from the coordinates.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import VerificationError, attempts, check, generator, phrase, pick_distractors
from ._s2b1_linalg import (
    plain_stem,
    det_root_distractors,
    draw_cols,
    int_distractors,
    mat_str,
    poly_text,
    power_text,
    sym,
    sym_value,
    vec_answer,
    vec_distractors,
)
from ._ucv1_linalg import det, fm, inverse, mmul, rank, transpose, vec_str, with_columns
from ..core import join_terms

LEVEL = 19


def _vec_text(vec) -> str:
    return "(" + ", ".join(str(int(v)) for v in vec) + ")"


def _solve(cols, target):
    """Coordinates of `target` in the basis `cols` (hand route), checked against SymPy."""
    basis = with_columns(cols)
    coords = [row[0] for row in mmul(inverse(basis), [[Fraction(t)] for t in target])]
    check(
        sym(basis).LUsolve(
            sympy.Matrix([sympy.Rational(Fraction(t).numerator, Fraction(t).denominator) for t in target])
        )
        == sympy.Matrix([sympy.Rational(c.numerator, c.denominator) for c in coords]),
        "SymPy's coordinates differ",
    )
    return coords


def _dist_set(coords, target, cols, extra=()):
    """The coordinate errors every basis question shares, best first."""
    n = len(coords)
    rows_basis = [list(r) for r in transpose(with_columns(cols))]  # basis vectors written as rows -> columns again
    candidates = [
        (list(reversed(coords)), "coords-order-swapped"),
        (list(target), "coords-standard-used"),
        (_solve(rows_basis, target) if det(fm(rows_basis)) != 0 else None, "coords-basis-as-rows"),
        ([-coords[0], *coords[1:]], "coords-sign-slip"),
        ([*coords[:-1], -coords[-1]], "coords-sign-slip"),
        *extra,
    ]
    del n
    return candidates


def _finish(rng, stem_patterns, slots, coords, target, cols, working, extra=(), n_dist=4):
    candidates = _dist_set(coords, target, cols, extra)
    answer = vec_answer(coords)
    distractors = pick_distractors(answer["answer_text"], vec_distractors(candidates, coords), n_dist)
    check(len(distractors) >= 3, "too few distractors")
    return dict(
        text=phrase(rng, stem_patterns, **slots),
        working=working,
        distractors=distractors,
        **answer,
    )


# -- easy: R^3 --------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "The vectors b₁ = {b1}, b₂ = {b2} and b₃ = {b3} form a basis B of ℝ³. Find the coordinate vector of v = {v} "
    "relative to B, written (c₁, c₂, c₃) where v = c₁b₁ + c₂b₂ + c₃b₃.",
    "In ℝ³ take the basis B = {{{b1}, {b2}, {b3}}}. Find the coordinates (c₁, c₂, c₃) of v = {v} relative to B, "
    "that is, the numbers with v = c₁b₁ + c₂b₂ + c₃b₃.",
    "Relative to the basis B = {{{b1}, {b2}, {b3}}} of ℝ³, write v = {v} as v = c₁b₁ + c₂b₂ + c₃b₃. Find the "
    "coordinate vector (c₁, c₂, c₃).",
)


def _easy(rng):
    for _ in attempts(500, "an R^3 basis question"):
        cols = draw_cols(rng, 3, -2, 3, 2, 6)
        coords = [rng.choice([-3, -2, -1, 1, 2, 3, 4]) for _ in range(3)]
        if len(set(coords)) < 2:
            continue
        v = [sum(coords[j] * cols[j][i] for j in range(3)) for i in range(3)]
        if max(abs(x) for x in v) <= 14 and v != [coords[0], coords[1], coords[2]]:
            break
    solved = _solve(cols, v)
    check(solved == [Fraction(c) for c in coords], "coordinates differ from the chosen ones")
    working = [
        f"Solve c₁{_vec_text(cols[0])} + c₂{_vec_text(cols[1])} + c₃{_vec_text(cols[2])} = {_vec_text(v)}, "
        "a 3×3 linear system for the coordinates.",
        f"Row reduction gives (c₁, c₂, c₃) = {vec_str(coords)}.",
    ]
    return _finish(
        rng,
        _EASY_STEMS,
        dict(b1=_vec_text(cols[0]), b2=_vec_text(cols[1]), b3=_vec_text(cols[2]), v=_vec_text(v)),
        solved,
        v,
        cols,
        working,
    )


# -- medium: P_2 ---------------------------------------------------------------------------------------------------

_MEDIUM_STEMS = (
    "The polynomials p₁ = {p1}, p₂ = {p2} and p₃ = {p3} form a basis B of the space P₂ of polynomials of degree at "
    "most 2. Find the coordinates (c₁, c₂, c₃) of q = {q} relative to B, where q = c₁p₁ + c₂p₂ + c₃p₃.",
    "Let B = {{{p1}, {p2}, {p3}}}, a basis of P₂. Write q = {q} as c₁p₁ + c₂p₂ + c₃p₃ and give the coordinate vector "
    "(c₁, c₂, c₃) of q relative to B.",
)


def _medium(rng):
    for _ in attempts(500, "a P_2 basis question"):
        cols = draw_cols(rng, 3, -2, 2, 2, 3)
        q = [rng.randint(-5, 5) for _ in range(3)]
        if q[2] == 0 or q[0] == 0:
            continue
        coords = _solve(cols, q)
        if any(c.denominator > 1 for c in coords) and all(c != 0 for c in coords):
            break
    # Rebuild q from the coordinates as the independent check.
    rebuilt = [sum(coords[j] * cols[j][i] for j in range(3)) for i in range(3)]
    check(rebuilt == [Fraction(x) for x in q], "the coordinates do not rebuild q")
    working = [
        "Compare coefficients of 1, x and x² in q = c₁p₁ + c₂p₂ + c₃p₃ to get a 3×3 linear system.",
        f"Solving it gives (c₁, c₂, c₃) = {vec_str(coords)}.",
    ]
    return _finish(
        rng,
        _MEDIUM_STEMS,
        dict(p1=poly_text(cols[0]), p2=poly_text(cols[1]), p3=poly_text(cols[2]), q=poly_text(q)),
        coords,
        q,
        cols,
        working,
    )


# -- hard ------------------------------------------------------------------------------------------------------------


def _shifted(a: int) -> str:
    return f"(x - {a})" if a > 0 else f"(x + {-a})"


def _hard_taylor(rng):
    x = sympy.symbols("x")
    for _ in attempts(500, "a cubic with a non-trivial Taylor expansion"):
        a = rng.choice([-3, -2, -1, 1, 2, 3])
        q = [rng.randint(-6, 6) for _ in range(3)] + [rng.choice([-3, -2, -1, 1, 2, 3])]
        poly = sum(c * x**k for k, c in enumerate(q))
        taylor = sympy.Poly(sympy.expand(poly.subs(x, x + a)), x).all_coeffs()[::-1]
        coords = [Fraction(int(c)) for c in taylor]
        if all(c != 0 for c in coords) and coords != [Fraction(c) for c in q] and max(abs(c) for c in coords) <= 60:
            break
    # Independent route: rebuild q from sum c_k (x - a)^k.
    rebuilt = sympy.expand(sum(int(c) * (x - a) ** k for k, c in enumerate(coords)))
    check(sympy.Poly(rebuilt - poly, x).is_zero, "the shifted coordinates do not rebuild q")
    derivs = [Fraction(int(sympy.diff(poly, x, k).subs(x, a))) for k in range(4)]
    other = sympy.Poly(sympy.expand(poly.subs(x, x - a)), x).all_coeffs()[::-1]
    extra = [
        (derivs, "taylor-derivative-not-divided"),
        ([Fraction(int(c)) for c in other], "taylor-centre-sign-error"),
    ]
    basis = f"B = {{1, {_shifted(a)}, {_shifted(a)}², {_shifted(a)}³}}"
    stem = phrase(
        rng,
        (
            "Let P₃ be the space of polynomials of degree at most 3, with the basis {basis}. Find the coordinates "
            "(c₀, c₁, c₂, c₃) of q = {q} relative to B, where q = c₀ + c₁{s} + c₂{s}² + c₃{s}³.",
            "The set {basis} is a basis of P₃. Write q = {q} in terms of this basis and give its coordinate "
            "vector (c₀, c₁, c₂, c₃) relative to B.",
        ),
        basis=basis,
        q=poly_text(q),
        s=_shifted(a),
    )
    answer = vec_answer(coords)
    candidates = [
        (list(q), "coords-standard-used"),
        *extra,
        (list(reversed(coords)), "coords-order-swapped"),
        ([c if k % 2 == 0 else -c for k, c in enumerate(coords)], "coords-sign-slip"),
    ]
    distractors = pick_distractors(answer["answer_text"], vec_distractors(candidates, coords), 4)
    check(len(distractors) >= 3, "too few distractors")
    working = [
        f"Put x = ({join_terms([(1, 'x'), (-a, '')])}) + ({a}) in q and expand in powers of ({join_terms([(1, 'x'), (-a, '')])}); equivalently take c_k = q⁽ᵏ⁾({a}) / k!.",
        f"The coefficients are (c₀, c₁, c₂, c₃) = {vec_str(coords)}.",
    ]
    return dict(text=stem, working=working, distractors=distractors, **answer)


def _hard_matrix(rng):
    for _ in attempts(2000, "a basis of M_2"):
        mats = [[rng.randint(-1, 2) for _ in range(4)] for _ in range(4)]
        d = abs(det(fm(mats)))
        v = [rng.randint(-5, 5) for _ in range(4)]
        if not 1 <= d <= 3 or any(all(e == 0 for e in m) for m in mats):
            continue
        coords = _solve(mats, v)
        if any(c.denominator > 1 for c in coords) and sum(1 for c in coords if c == 0) <= 1:
            break

    def m2(entries):
        return mat_str([entries[:2], entries[2:]])

    check(
        [sum(coords[j] * mats[j][i] for j in range(4)) for i in range(4)] == [Fraction(e) for e in v],
        "the coordinates do not rebuild the matrix",
    )
    stem = phrase(
        rng,
        (
            "The matrices B₁ = {b1}, B₂ = {b2}, B₃ = {b3} and B₄ = {b4} form a basis B of the space of 2×2 real "
            "matrices. Find the coordinates (c₁, c₂, c₃, c₄) of A = {a} relative to B, where A = c₁B₁ + c₂B₂ + c₃B₃ + c₄B₄.",
            "Let B = {{B₁, B₂, B₃, B₄}} be the basis of the space of 2×2 real matrices with B₁ = {b1}, B₂ = {b2}, "
            "B₃ = {b3} and B₄ = {b4}. Find the coordinate vector of A = {a} relative to B.",
        ),
        b1=m2(mats[0]),
        b2=m2(mats[1]),
        b3=m2(mats[2]),
        b4=m2(mats[3]),
        a=m2(v),
    )
    answer = vec_answer(coords)
    candidates = _dist_set(coords, v, mats)
    distractors = pick_distractors(answer["answer_text"], vec_distractors(candidates, coords), 4)
    check(len(distractors) >= 3, "too few distractors")
    working = [
        "Compare the four entries of A = c₁B₁ + c₂B₂ + c₃B₃ + c₄B₄ to get a 4×4 linear system.",
        f"Solving it gives (c₁, c₂, c₃, c₄) = {vec_str(coords)}.",
    ]
    return dict(text=stem, working=working, distractors=distractors, **answer)


def _hard_plane(rng):
    for _ in attempts(2000, "a plane basis in R^4"):
        cols = [[rng.randint(-2, 3) for _ in range(4)] for _ in range(2)]
        coords = [rng.choice([-3, -2, -1, 1, 2, 3]), rng.choice([-3, -2, -1, 1, 2, 3])]
        v = [coords[0] * cols[0][i] + coords[1] * cols[1][i] for i in range(4)]
        if rank(fm(cols)) == 2 and max(abs(x) for x in v) <= 12 and sum(1 for x in v if x == 0) <= 1:
            break
    solution = sympy.linsolve((sym(transpose(fm(cols))), sympy.Matrix([int(x) for x in v])), sympy.symbols("c1 c2"))
    check(list(solution) == [tuple(sympy.Integer(c) for c in coords)], "the plane coordinates differ")
    b1, b2 = _vec_text(cols[0]), _vec_text(cols[1])
    stem = phrase(
        rng,
        (
            "The vectors b₁ = {b1} and b₂ = {b2} form a basis B of a plane W in ℝ⁴, and v = {v} lies in W. Find the "
            "coordinates (c₁, c₂) of v relative to B, where v = c₁b₁ + c₂b₂.",
            "Let W be the plane in ℝ⁴ with basis B = {{{b1}, {b2}}}. The vector v = {v} lies in W. Find its "
            "coordinate vector (c₁, c₂) relative to B.",
        ),
        b1=b1,
        b2=b2,
        v=_vec_text(v),
    )
    answer = vec_answer([Fraction(c) for c in coords])
    cf = [Fraction(c) for c in coords]
    candidates = [
        (list(reversed(cf)), "coords-order-swapped"),
        ([Fraction(v[0]), Fraction(v[1])], "coords-standard-used"),
        ([-cf[0], cf[1]], "coords-sign-slip"),
        ([cf[0], -cf[1]], "coords-sign-slip"),
        ([cf[0] + cf[1], cf[0] - cf[1]], "coords-sign-slip"),
    ]
    distractors = pick_distractors(answer["answer_text"], vec_distractors(candidates, cf), 4)
    check(len(distractors) >= 3, "too few distractors")
    working = [
        "The equation v = c₁b₁ + c₂b₂ is four equations in two unknowns; they are consistent because v is in W.",
        f"Any two independent equations give (c₁, c₂) = {vec_str(cf)}, and the other two check.",
    ]
    return dict(text=stem, working=working, distractors=distractors, **answer)


@generator(
    "s2b1_coordinates_in_basis",
    level=LEVEL,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@plain_stem
def s2b1_coordinates_in_basis(rng, tier="easy"):
    """Coordinates of a vector, polynomial or matrix relative to a given basis; tiers in the module docstring."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    for _ in attempts(60, "a basis question with enough distractors"):
        try:
            if tier == "easy":
                return _easy(rng)
            if tier == "medium":
                return _medium(rng)
            return rng.choice([_hard_taylor, _hard_matrix, _hard_plane, _hard_taylor])(rng)
        except VerificationError:
            continue
    raise AssertionError("unreachable")  # pragma: no cover


# -- exam -------------------------------------------------------------------------------------------------------------


def _k_vec_text(entries, k_index, sign, space) -> str:
    """The third basis element, with ±k in position k_index."""
    if space == "R3":
        return (
            "("
            + ", ".join((("-k" if sign < 0 else "k") if i == k_index else str(e)) for i, e in enumerate(entries))
            + ")"
        )
    terms = [
        (sign if i == k_index else e, "k" + power_text("x", i) if i == k_index else power_text("x", i))
        for i, e in enumerate(entries)
    ]
    return join_terms(list(reversed(terms)))


def _plain_text(vec, space) -> str:
    return _vec_text(vec) if space == "R3" else poly_text(vec)


@generator("exd_x_s2b1_coordinates", level=LEVEL, presets={"exam": dict(tier="exam")})
@plain_stem
def exd_x_s2b1_coordinates(rng, tier="exam"):
    """Exam chain: when is a family with a parameter a basis, then coordinates and components at a chosen value."""
    check(tier == "exam", f"unknown tier {tier!r}")
    space = rng.choice(["R3", "P2"])
    for _ in attempts(200, "an exam chain with enough distractors"):
        result = _exam_try(rng, space)
        if result is not None:
            return result
    raise AssertionError  # pragma: no cover


def _exam_try(rng, space):
    k_sym = sympy.symbols("k")
    for _ in attempts(4000, "an exam basis with an integer bad parameter"):
        cols = draw_cols(rng, 3, -2, 3, 1, 9)[:2]
        third = [rng.randint(-2, 3) for _ in range(3)]
        k_index = rng.randrange(3)
        sign = rng.choice([1, -1])
        entries = [[sympy.Integer(v) for v in c] for c in (cols[0], cols[1])]
        u3 = [sympy.Integer(v) for v in third]
        u3[k_index] = sign * k_sym
        matrix = sympy.Matrix([entries[0], entries[1], u3]).T
        poly = sympy.Poly(sympy.expand(matrix.det()), k_sym)
        if poly.degree() != 1:
            continue
        d1, d0 = int(poly.coeff_monomial(k_sym)), int(poly.coeff_monomial(1))
        if d0 % d1 != 0:
            continue
        k0 = -d0 // d1
        m = rng.choice([v for v in range(-3, 5) if v not in (0, k0)])
        third_m = list(third)
        third_m[k_index] = sign * m
        basis_m = [list(cols[0]), list(cols[1]), third_m]
        if k0 == 0 or abs(k0) > 9 or abs(det(fm(basis_m))) == 0:
            continue
        target = [rng.randint(-6, 6) for _ in range(3)]
        coords = _solve(basis_m, target)
        if sum(1 for c in coords if c.denominator > 1) < 1:
            continue
        break
    check(sympy.solve(matrix.det(), k_sym) == [k0], "SymPy's bad k differs")
    check(int(matrix.det().subs(k_sym, m)) == d1 * (m - k0), "determinant at k = m differs")
    check(int(det(fm(basis_m))) * 1 == d1 * (m - k0), "the hand determinant differs")
    q_coords = [rng.randint(-3, 3) for _ in range(3)]
    if all(c == 0 for c in q_coords):
        q_coords = [1, -1, 2]
    comp = rng.randrange(3)
    w_value = sum(q_coords[j] * basis_m[j][comp] for j in range(3))
    comp_name = {"R3": ("first", "second", "third"), "P2": ("constant term", "coefficient of x", "coefficient of x²")}[
        space
    ][comp]

    first_two = [_plain_text(cols[0], space), _plain_text(cols[1], space)]
    third_text = _k_vec_text(third, k_index, sign, space)
    if space == "R3":
        stem = f"In ℝ³ let u₁ = {first_two[0]}, u₂ = {first_two[1]} and u₃ = {third_text}, where k is a real number."
        w_prompt = (
            f"A vector w has coordinates ({', '.join(str(c) for c in q_coords)}) relative to (u₁, u₂, u₃) when "
            f"k = {m}. Find the {comp_name} component of w in the standard basis."
        )
        v_prompt = f"Take k = {m}, so that (u₁, u₂, u₃) is a basis. Find the coordinates of v = {_vec_text(target)} relative to it."
    else:
        stem = f"In P₂ let p₁ = {first_two[0]}, p₂ = {first_two[1]} and p₃ = {third_text}, where k is a real number."
        w_prompt = (
            f"A polynomial r has coordinates ({', '.join(str(c) for c in q_coords)}) relative to (p₁, p₂, p₃) when "
            f"k = {m}. Find the {comp_name} of r."
        )
        v_prompt = f"Take k = {m}, so that (p₁, p₂, p₃) is a basis. Find the coordinates of q = {poly_text(target)} relative to it."

    sarrus_entries = [[sympy.Integer(v) for v in basis_col] for basis_col in (cols[0], cols[1])]
    third_sym = list(u3)
    # Rows of the matrix whose columns are u1, u2, u3.
    rows = [[sarrus_entries[0][i], sarrus_entries[1][i], third_sym[i]] for i in range(3)]
    a_cands = [
        (-k0, "sign-error-moving-term"),
        *[(r, t) for r, t in det_root_distractors(rows, k_sym, k0)],
        (Fraction(-d0), "colspace-forgot-multiplier") if abs(d1) != 1 else (None, ""),
    ]
    a_dist = pick_distractors(str(k0), int_distractors(a_cands, k0), 3)
    det_m = d1 * (m - k0)
    b_dist = pick_distractors(
        str(det_m),
        int_distractors(
            [
                (d1 * (m + k0), "sign-error-moving-term"),
                (m - k0, "colspace-forgot-multiplier"),
                (d1 * m, "det-sarrus-term-dropped"),
                (-det_m, "det-sarrus-sign-slip"),
            ],
            det_m,
        ),
        3,
    )
    c_answer = vec_answer(coords)
    rows_basis = [list(r) for r in transpose(with_columns(basis_m))]
    c_cands = _dist_set(coords, target, basis_m)
    c_dist = pick_distractors(c_answer["answer_text"], vec_distractors(c_cands, coords), 3)
    d_dist = pick_distractors(
        str(w_value),
        int_distractors(
            [
                (q_coords[comp], "coords-standard-used"),
                (sum(q_coords[2 - j] * basis_m[j][comp] for j in range(3)), "coords-order-swapped"),
                (sum(q_coords[j] * rows_basis[j][comp] for j in range(3)), "coords-basis-as-rows"),
                (sum(q_coords[j] * basis_m[j][comp] for j in range(3) if j != 2), "coords-standard-used"),
            ],
            w_value,
        ),
        3,
    )
    if not all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)):
        return None
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the value of k for which these three do not form a basis.",
                answer_kind="number",
                answer=k0,
                answer_text=str(k0),
                marks=3,
                working=[
                    f"Expanding the determinant gives {d1}({join_terms([(1, 'k'), (-k0, '')])}) up to sign, which is zero only at k = {k0}."
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=f"Find the determinant of the 3×3 matrix whose columns are the standard coordinates of the three when k = {m}.",
                answer_kind="number",
                answer=det_m,
                answer_text=str(det_m),
                marks=1,
                follow_through=f"{d1}*({m} - part_a)",
                working=[f"The determinant is {d1}({join_terms([(1, 'k'), (-k0, '')])}), so at k = {m} it is {det_m}."],
                distractors=b_dist,
            ),
            dict(
                prompt=v_prompt,
                marks=3,
                working=[f"Solve the 3×3 system for the coordinates: {vec_str(coords)}."],
                distractors=c_dist,
                **c_answer,
            ),
            dict(
                prompt=w_prompt,
                answer_kind="number",
                answer=w_value,
                answer_text=str(w_value),
                marks=2,
                working=[f"Form the combination of the basis elements with these coordinates: the entry is {w_value}."],
                distractors=d_dist,
            ),
        ],
    )


__all__ = ["s2b1_coordinates_in_basis", "exd_x_s2b1_coordinates", "sym_value"]
