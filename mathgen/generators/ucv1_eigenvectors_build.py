"""Generates `ucv1_eigenvectors` questions (easy, medium, hard) and the exam chain `exd_x_uc_eigenvectors`.

``ucv1-eigenvectors`` (``linear-algebra``, answer ``matrix``): find an eigenvector of a supplied integer matrix. The
answer is a column vector graded with ``answer_form = "scalar_multiple"``, so any non-zero multiple is right.

* easy: a 2×2 matrix with integer eigenvalues and one eigenvalue stated.
* medium: a 2×2 matrix and the eigenvalue only described ("the larger eigenvalue", so the characteristic equation comes
  first), or a 3×3 upper-triangular matrix (back substitution, never the unit vector), or a 3×3 block matrix
  diag(B, c) or diag(c, B) with an eigenvalue of the 2×2 block B.
* hard: a dense 3×3 matrix with its eigenvalues stated (a full 3×3 elimination), or a defective 3×3 matrix with a
  repeated eigenvalue whose eigenspace is one-dimensional, so the eigenvector is the only solution of rank 2.
* exam (``exd_x_uc_eigenvectors``): a 2×2 matrix with distinct integer eigenvalues. (a) the larger eigenvalue,
  (b) the smaller (follows through from (a) by the trace), (c) an eigenvector for the larger, (d) one for the smaller.

Every matrix is built backwards from integer eigenvalues and an integer matrix P of determinant 1 or -1 (so
A = P D P^-1 is an integer matrix and its eigenvectors are P's columns). The answer is re-derived by Gauss-Jordan
elimination in exact fractions, then checked three ways: A v = λ v in integers, SymPy's null space of A - λI, and
the column of P it must be a multiple of.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator, paren_negative, phrase, pick_distractors
from ..grading.matrix import matrices_match
from ._ucv1_linalg import (
    apply,
    column,
    column_value,
    diagonal,
    draw_2x2,
    fm,
    identity,
    inverse,
    madd,
    mat_str,
    max_abs,
    mmul,
    nullspace,
    primitive,
    rank,
    unimodular,
    vec_str,
    vector_distractors,
)

FORM = "scalar_multiple"
_CAP_3X3 = 14


# -- construction -------------------------------------------------------------------------------------------


def _shift(a, lam):
    """A - λI."""
    return madd(a, identity(len(a)), -lam)


def _eigenvector(a, lam):
    """The eigenvector of `a` for `lam` by Gauss-Jordan elimination (the eigenspace must be one-dimensional)."""
    basis = nullspace(_shift(a, lam))
    check(len(basis) == 1, f"eigenspace of {lam} is {len(basis)}-dimensional")
    return primitive(basis[0])


def _verify_eigenvector(a, lam, vec):
    """Two independent checks: A v = λ v exactly, and SymPy's null space of A - λI is the line through v."""
    check(apply(a, vec) == [lam * v for v in vec], "A v != λ v")
    m = sympy.Matrix([[int(v) for v in row] for row in a]) - lam * sympy.eye(len(a))
    sym = m.nullspace()
    check(len(sym) == 1, "SymPy finds a different eigenspace dimension")
    check(
        matrices_match(column_value(vec), [[sympy.nsimplify(c)] for c in sym[0]], FORM),
        "SymPy's eigenvector is not a multiple of the answer",
    )


def _first_nonzero_row(m):
    return next((row for row in m if any(v != 0 for v in row)), None)


def _negated_first(vec):
    return [-vec[0], *vec[1:]]


def _swapped(vec):
    out = list(vec)
    out[0], out[-1] = out[-1], out[0]
    return out


def _flipped_sign_vector(a, lam):
    """The eigenvector one would get from A + λI (the eigenvalue's sign wrong), or None if there is none."""
    basis = nullspace(_shift(a, -lam))
    return basis[0] if len(basis) == 1 else None


def _finish(rng, text, a, lam, vec, candidates, working):
    answer_text = vec_str(vec)
    distractors = pick_distractors(answer_text, vector_distractors(candidates, vec), 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=text,
        answer_kind="matrix",
        answer=column_value(vec),
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        answer_form=FORM,
        working=working,
        distractors=distractors,
    )


# -- easy and medium ----------------------------------------------------------------------------------------

_EASY_STEMS = (
    "The matrix A = {a} has an eigenvalue λ = {lam}. Find an eigenvector of A corresponding to this eigenvalue.",
    "One eigenvalue of A = {a} is {lam}. Find a corresponding eigenvector.",
    "Find an eigenvector of the matrix A = {a} corresponding to the eigenvalue {lam}.",
)
_NAMED_STEMS = (
    "Find an eigenvector of the matrix A = {a} corresponding to its {which} eigenvalue.",
    "The matrix A = {a} has two distinct real eigenvalues. Find an eigenvector corresponding to the {which} one.",
)
_TRIANGULAR_STEMS = (
    "The matrix A = {a} has eigenvalue λ = {lam}. Find a corresponding eigenvector.",
    "Find an eigenvector of A = {a} corresponding to the eigenvalue {lam}.",
)
_BLOCK_STEMS = (
    "The matrix A = {a} has an eigenvalue λ = {lam}. Find an eigenvector of A corresponding to it.",
    "Find an eigenvector of A = {a} corresponding to the eigenvalue λ = {lam}.",
)


def _candidates_2x2(a, lam, vec, other_vec):
    m = _shift(a, lam)
    row = _first_nonzero_row(m)
    p, q = row[0], row[1]
    return [
        (other_vec, "eigvec-other-eigenvalue"),
        ([q, p], "eigvec-component-sign-slip"),
        (row, "eigvec-row-as-vector"),
        (_flipped_sign_vector(a, lam), "eigvec-eigenvalue-sign"),
        ([vec[1], vec[0]], "eigvec-components-swapped"),
    ]


def _working_2x2(a, lam, vec):
    m = _shift(a, lam)
    return [
        f"A - {lam}I = {mat_str(m)}." if lam >= 0 else f"A + {-lam}I = {mat_str(m)}.",
        f"Solving (A - λI)v = 0 gives v = {vec_str(vec)}.",
        f"Check: Av = {vec_str(apply(a, vec))} = {lam}v.",
    ]


def _easy(rng):
    p, lams, a = draw_2x2(rng, full=rng.random() < 0.7)
    k = rng.randrange(2)
    lam, other = lams[k], lams[1 - k]
    vec = _eigenvector(a, lam)
    _verify_eigenvector(a, lam, vec)
    check(matrices_match(column_value(vec), column_value(column(p, k)), FORM), "not the column of P")
    other_vec = _eigenvector(a, other)
    text = phrase(rng, _EASY_STEMS, a=mat_str(a), lam=lam)
    return _finish(rng, text, a, lam, vec, _candidates_2x2(a, lam, vec, other_vec), _working_2x2(a, lam, vec))


def _medium_named(rng):
    p, lams, a = draw_2x2(rng, full=True)
    which = rng.choice(["larger", "smaller"])
    k = lams.index(max(lams) if which == "larger" else min(lams))
    lam, other = lams[k], lams[1 - k]
    vec = _eigenvector(a, lam)
    _verify_eigenvector(a, lam, vec)
    check(matrices_match(column_value(vec), column_value(column(p, k)), FORM), "not the column of P")
    other_vec = _eigenvector(a, other)
    trace, determinant = a[0][0] + a[1][1], a[0][0] * a[1][1] - a[0][1] * a[1][0]
    working = [
        f"det(A - λI) = {_signed_quadratic(int(trace), int(determinant))} = 0, so λ = {lams[0]} or {lams[1]}.",
        f"The {which} eigenvalue is {lam}.",
        *_working_2x2(a, lam, vec),
    ]
    text = phrase(rng, _NAMED_STEMS, a=mat_str(a), which=which)
    return _finish(rng, text, a, lam, vec, _candidates_2x2(a, lam, vec, other_vec), working)


def _triangular_solution_lines(vec, k):
    """The back-substitution line for a triangular matrix, true to the working: free variable x(k+1) = 1 gives a
    vector with that entry 1, which is then scaled to the whole-number eigenvector the question asks for."""
    pivot = vec[k]
    base = [Fraction(v) / Fraction(pivot) for v in vec]
    first = f"Setting the free variable x{k + 1} = 1 and back-substituting gives {vec_str(base)}."
    if base == [Fraction(v) for v in vec]:
        return [first]
    return [first, f"Scaling to clear fractions (any non-zero multiple is an eigenvector) gives v = {vec_str(vec)}."]


def _medium_triangular(rng):
    for _ in attempts(500, "an upper-triangular 3×3 with a non-trivial eigenvector"):
        diag = rng.sample(range(-3, 5), 3)
        up = [rng.randint(-3, 3) for _ in range(3)]
        a = fm([[diag[0], up[0], up[1]], [0, diag[1], up[2]], [0, 0, diag[2]]])
        k = rng.choice([1, 2])
        lam = diag[k]
        vec = _eigenvector(a, lam)
        if all(v == 0 for v in vec[:k]):
            continue
        break
    _verify_eigenvector(a, lam, vec)
    others = [i for i in range(3) if i != k]
    other_vecs = [_eigenvector(a, diag[i]) for i in others]
    unit = [1 if i == k else 0 for i in range(3)]
    candidates = [
        (unit, "eigvec-unit-vector-triangular"),
        (other_vecs[0], "eigvec-other-eigenvalue"),
        (_negated_first(vec), "eigvec-component-sign-slip"),
        (_swapped(vec), "eigvec-components-swapped"),
        (other_vecs[1], "eigvec-other-eigenvalue"),
    ]
    working = [
        f"A - {lam}I = {mat_str(_shift(a, lam))}." if lam >= 0 else f"A + {-lam}I = {mat_str(_shift(a, lam))}.",
        *_triangular_solution_lines(vec, k),
        f"Check: Av = {vec_str(apply(a, vec))} = {lam}v.",
    ]
    text = phrase(rng, _TRIANGULAR_STEMS, a=mat_str(a), lam=lam)
    return _finish(rng, text, a, lam, vec, candidates, working)


def _medium_block(rng):
    _p, lams, b = draw_2x2(rng)
    c = Fraction(rng.choice([v for v in range(-4, 6) if v not in lams]))
    k = rng.randrange(2)
    lam, other = lams[k], lams[1 - k]
    first = rng.random() < 0.5  # block B on the top left, c on the bottom right
    zero = Fraction(0)
    if first:
        a = [[b[0][0], b[0][1], zero], [b[1][0], b[1][1], zero], [zero, zero, c]]
    else:
        a = [[c, zero, zero], [zero, b[0][0], b[0][1]], [zero, b[1][0], b[1][1]]]
    block_vec = _eigenvector(b, lam)
    other_block_vec = _eigenvector(b, other)

    def pad(v, filler=0):
        return [*v, filler] if first else [filler, *v]

    vec = primitive(pad(block_vec))
    _verify_eigenvector(a, lam, vec)
    unit = [0, 0, 1] if first else [1, 0, 0]
    candidates = [
        (pad(block_vec, 1), "eigvec-padded-block"),
        (pad(other_block_vec), "eigvec-other-eigenvalue"),
        (unit, "eigvec-unit-vector-triangular"),
        (_negated_first(vec), "eigvec-component-sign-slip"),
        (_swapped(vec), "eigvec-components-swapped"),
    ]
    working = [
        f"The matrix is block diagonal, so an eigenvector for λ = {lam} comes from the 2×2 block {mat_str(b)}.",
        f"For that block, (B {'-' if lam >= 0 else '+'} {abs(lam)}I)u = 0 gives u = {vec_str(block_vec)}.",
        f"Padding with 0 for the other block gives v = {vec_str(vec)}.",
    ]
    text = phrase(rng, _BLOCK_STEMS, a=mat_str(a), lam=lam)
    return _finish(rng, text, a, lam, vec, candidates, working)


# -- hard -----------------------------------------------------------------------------------------------------

_DENSE_STEMS = (
    "The matrix A = {a} has eigenvalues {lams}. Find an eigenvector of A corresponding to the eigenvalue {lam}.",
    "Given that the eigenvalues of A = {a} are {lams}, find an eigenvector corresponding to λ = {lam}.",
)
_DEFECTIVE_STEMS = (
    "The matrix A = {a} has characteristic polynomial {charpoly}, and the eigenspace for λ = {lam} is "
    "one-dimensional. Find an eigenvector of A corresponding to λ = {lam}.",
    "A = {a} has characteristic polynomial {charpoly}. Given that the eigenspace for λ = {lam} is one-dimensional, "
    "find an eigenvector corresponding to λ = {lam}.",
)


def _factor_text(lam):
    return f"(λ - {lam})" if lam > 0 else ("λ" if lam == 0 else f"(λ + {-lam})")


def _list_text(values):
    values = [str(v) for v in values]
    return ", ".join(values[:-1]) + " and " + values[-1]


def _dense_zero_count(a):
    return sum(1 for row in a for v in row if v == 0)


def _hard_dense(rng):
    for _ in attempts(500, "a dense 3×3 matrix with integer eigenvalues"):
        p = unimodular(rng, 3, 2)
        lams = rng.sample(range(-3, 5), 3)
        a = mmul(mmul(p, diagonal(lams)), inverse(p))
        if max_abs(a) > _CAP_3X3 or _dense_zero_count(a) > 1:
            continue
        break
    k = rng.randrange(3)
    lam = lams[k]
    vec = _eigenvector(a, lam)
    _verify_eigenvector(a, lam, vec)
    check(matrices_match(column_value(vec), column_value(column(p, k)), FORM), "not the column of P")
    others = [_eigenvector(a, lams[i]) for i in range(3) if i != k]
    m = _shift(a, lam)
    candidates = [
        (others[0], "eigvec-other-eigenvalue"),
        (_flipped_sign_vector(a, lam), "eigvec-eigenvalue-sign"),
        (_first_nonzero_row(m), "eigvec-row-as-vector"),
        (_swapped(vec), "eigvec-components-swapped"),
        (others[1], "eigvec-other-eigenvalue"),
    ]
    working = [
        f"A - {lam}I = {mat_str(m)}." if lam >= 0 else f"A + {-lam}I = {mat_str(m)}.",
        f"Row-reducing, the system (A - λI)v = 0 has a one-parameter family of solutions, v = {vec_str(vec)}.",
        f"Check: Av = {vec_str(apply(a, vec))} = {lam}v.",
    ]
    text = phrase(rng, _DENSE_STEMS, a=mat_str(a), lams=_list_text(sorted(lams)), lam=lam)
    return _finish(rng, text, a, lam, vec, candidates, working)


def _hard_defective(rng):
    for _ in attempts(500, "a defective 3×3 matrix"):
        p = unimodular(rng, 3, 2)
        lam, mu = rng.sample(range(-3, 4), 2)
        jordan = fm([[lam, 1, 0], [0, lam, 0], [0, 0, mu]])
        a = mmul(mmul(p, jordan), inverse(p))
        if max_abs(a) > _CAP_3X3 or _dense_zero_count(a) > 3:
            continue
        break
    m = _shift(a, lam)
    check(rank(m) == 2, "the eigenspace is not one-dimensional")
    vec = _eigenvector(a, lam)
    _verify_eigenvector(a, lam, vec)
    check(matrices_match(column_value(vec), column_value(column(p, 0)), FORM), "not the first column of P")
    generalised = column(p, 1)
    check(apply(m, generalised) == column(p, 0), "P e2 is not a generalised eigenvector")
    candidates = [
        (generalised, "eigvec-generalised"),
        (column(p, 2), "eigvec-other-eigenvalue"),
        (_flipped_sign_vector(a, lam), "eigvec-eigenvalue-sign"),
        (_first_nonzero_row(m), "eigvec-row-as-vector"),
        (_swapped(vec), "eigvec-components-swapped"),
    ]
    charpoly = f"{_factor_text(lam)}²{_factor_text(mu)}"
    working = [
        f"A - {lam}I = {mat_str(m)}." if lam >= 0 else f"A + {-lam}I = {mat_str(m)}.",
        f"This has rank 2, so the null space is a line: v = {vec_str(vec)}.",
        f"Check: Av = {vec_str(apply(a, vec))} = {lam}v.",
    ]
    text = phrase(rng, _DEFECTIVE_STEMS, a=mat_str(a), charpoly=charpoly, lam=lam)
    return _finish(rng, text, a, lam, vec, candidates, working)


@generator(
    "ucv1_eigenvectors",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_eigenvectors(rng, tier="easy"):
    """One eigenvector of a supplied integer matrix; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return rng.choice([_medium_named, _medium_triangular, _medium_block, _medium_named])(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_dense, _hard_defective])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


def _signed_quadratic(trace, determinant):
    """λ² - tλ + d as text."""
    t = f" - {trace}λ" if trace > 0 else (f" + {-trace}λ" if trace < 0 else "")
    d = f" + {determinant}" if determinant > 0 else (f" - {-determinant}" if determinant < 0 else "")
    return f"λ²{t}{d}"


_EXAM_STEMS = [
    "The matrix A = {a} has two distinct real eigenvalues.",
    "Consider the 2×2 matrix A = {a}. Its characteristic equation has two distinct real roots.",
    "A linear map on the plane has matrix A = {a} in the standard basis. A has two distinct real eigenvalues.",
]
_EXAM_PROMPTS_A = [
    "Find the larger eigenvalue of A.",
    "Solve the characteristic equation of A and state the larger eigenvalue.",
    "Determine the larger of the two eigenvalues of A.",
]
_EXAM_PROMPTS_C = [
    "Find an eigenvector of A corresponding to the larger eigenvalue.",
    "Give a non-zero vector v with Av = λv, where λ is the larger eigenvalue.",
]


@generator("exd_x_uc_eigenvectors", level=18, presets={"exam": dict(tier="exam")})
def exd_x_uc_eigenvectors(rng, tier="exam"):
    """Exam chain: the eigenvalues of a 2×2 matrix, then an eigenvector for each."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a 2×2 matrix with non-zero eigenvalues"):
        p, lams, a = draw_2x2(rng, -5, 6, full=True)
        if 0 not in lams and abs(lams[0] - lams[1]) >= 2:
            break
    small, large = sorted(lams)
    trace = a[0][0] + a[1][1]
    determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    check(trace == small + large and determinant == small * large, "trace/determinant disagree with the eigenvalues")
    # Independent route: the roots of the characteristic polynomial from SymPy.
    lam_sym = sympy.symbols("lambda")
    roots = sorted(sympy.roots(lam_sym**2 - int(trace) * lam_sym + int(determinant), lam_sym))
    check(roots == [small, large], "SymPy's roots of the characteristic polynomial differ")
    vec_large, vec_small = _eigenvector(a, large), _eigenvector(a, small)
    _verify_eigenvector(a, large, vec_large)
    _verify_eigenvector(a, small, vec_small)

    trace_i, det_i = int(trace), int(determinant)

    def dist(value, tag):
        return Distractor(str(value), tag, value)

    a_dist = pick_distractors(
        str(large),
        [
            dist(small, "eigval-other-root"),
            dist(-large, "sign-flip-both-roots"),
            dist(trace_i, "eigval-trace-as-eigenvalue"),
            dist(det_i, "eigval-determinant-as-eigenvalue"),
            dist(a[0][0].numerator, "eigval-diagonal-entry"),
        ],
        3,
    )
    b_dist = pick_distractors(
        str(small),
        [
            dist(large, "eigval-other-root"),
            dist(-small, "sign-flip-both-roots"),
            dist(trace_i + large, "eigval-trace-difference-sign"),
            dist(trace_i, "eigval-trace-as-eigenvalue"),
            dist(a[1][1].numerator, "eigval-diagonal-entry"),
        ],
        3,
    )
    other_large = [vec_small, "eigvec-other-eigenvalue"]
    large_candidates = [
        tuple(other_large),
        ([_shift(a, large)[0][1], _shift(a, large)[0][0]], "eigvec-component-sign-slip"),
        (_flipped_sign_vector(a, large), "eigvec-eigenvalue-sign"),
        (_first_nonzero_row(_shift(a, large)), "eigvec-row-as-vector"),
        (_swapped(vec_large), "eigvec-components-swapped"),
    ]
    small_candidates = [
        (vec_large, "eigvec-other-eigenvalue"),
        ([_shift(a, small)[0][1], _shift(a, small)[0][0]], "eigvec-component-sign-slip"),
        (_flipped_sign_vector(a, small), "eigvec-eigenvalue-sign"),
        (_first_nonzero_row(_shift(a, small)), "eigvec-row-as-vector"),
        (_swapped(vec_small), "eigvec-components-swapped"),
    ]
    c_text, d_text = vec_str(vec_large), vec_str(vec_small)
    c_dist = pick_distractors(c_text, vector_distractors(large_candidates, vec_large), 3)
    d_dist = pick_distractors(d_text, vector_distractors(small_candidates, vec_small), 3)
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")

    # The wording is chosen last, so the matrix and every answer are the same as for a single fixed wording.
    stem = rng.choice(_EXAM_STEMS).format(a=mat_str(a))
    prompt_a = rng.choice(_EXAM_PROMPTS_A)
    prompt_c = rng.choice(_EXAM_PROMPTS_C)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=prompt_a,
                answer_kind="number",
                answer=large,
                answer_text=str(large),
                marks=2,
                working=[
                    f"det(A - λI) = {_signed_quadratic(trace_i, det_i)} = 0, so λ = {small} or λ = {large}.",
                    f"The larger eigenvalue is {large}.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt="Hence, or otherwise, find the smaller eigenvalue of A.",
                answer_kind="number",
                answer=small,
                answer_text=str(small),
                marks=1,
                follow_through=f"({trace_i}) - part_a",
                working=[
                    f"The eigenvalues sum to the trace {trace_i}, so the smaller is {trace_i} - {paren_negative(large)} = {small}."
                ],
                distractors=b_dist,
            ),
            dict(
                prompt=prompt_c,
                answer_kind="matrix",
                answer=column_value(vec_large),
                answer_text=c_text,
                answer_latex="plain:" + c_text,
                answer_form=FORM,
                marks=2,
                working=_working_2x2(a, large, vec_large),
                distractors=c_dist,
            ),
            dict(
                prompt="Find an eigenvector of A corresponding to the smaller eigenvalue.",
                answer_kind="matrix",
                answer=column_value(vec_small),
                answer_text=d_text,
                answer_latex="plain:" + d_text,
                answer_form=FORM,
                marks=2,
                working=_working_2x2(a, small, vec_small),
                distractors=d_dist,
            ),
        ],
    )
