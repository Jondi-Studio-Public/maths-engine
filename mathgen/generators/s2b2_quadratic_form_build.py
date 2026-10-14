"""Generates `s2b2_quadratic_form_definiteness` questions (easy, medium, hard) and the exam chain
`exd_x_s2b2_quadratic_form_definiteness`.

``uc2-quadratic-form-definiteness`` (``linear-algebra``): a quadratic form q(x) = xᵀAx with A symmetric is read through
the eigenvalues of A: they give its definiteness (all positive: positive definite, and so on), its signature and its
form in principal axes.

* easy: the symmetric matrix A of a 3-variable quadratic form, where the cross-term coefficients are twice the
  off-diagonal entries (so half-integers appear). Answer ``matrix``.
* medium: the diagonal matrix D of the eigenvalues, in increasing order, of the matrix of a quadratic form in two or
  three variables. Answer ``matrix``.
* hard (``matrix``): the matrix PᵀAP of the form after a linear substitution x = Py, or the matrix of a sum of
  squares of linear forms, which needs the products of the coefficients.
* exam (2 variables): (a) the matrix A, (b) the larger eigenvalue, (c) the other from the trace, (d) the
  classification, as a multiple-choice part paired with the three typed parts before it.

The forms come from integer symmetric matrices with integer eigenvalues (orthogonal triples, as in
``s2b2_orthogonal_diagonalisation_build``); the keys are re-derived with SymPy (``eigenvals``, ``is_positive_definite``,
and the expansion of xᵀAx).
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator, join_terms, latex_answer, phrase, pick_distractors
from ._s2b2_kit import tidied, fm, mat_str, number_distractor, sym
from .s2b2_orthogonal_diagonalisation_build import _build_a, _draw_2x2, _draw_3x3, matrix_options, mtext, mvalue, smat

_VARS3 = ("x", "y", "z")
_VARS2 = ("x", "y")
_SUP2 = "²"


def _monomials(names):
    n = len(names)
    out = [((i, i), names[i] + _SUP2) for i in range(n)]
    out += [((i, j), names[i] + names[j]) for i in range(n) for j in range(i + 1, n)]
    return out


def _poly_text(a, names, k_at=None):
    """The quadratic form xᵀAx as text, ordered x², xy, xz, y², yz, z²; the diagonal entry `k_at` is written as k."""
    n = len(names)
    terms = []
    for i in range(n):
        for j in range(i, n):
            coef = a[i][j] if i == j else 2 * a[i][j]
            mono = names[i] + _SUP2 if i == j else names[i] + names[j]
            if i == j and i == k_at:
                terms.append((1, "k" + mono))
            else:
                terms.append((int(coef), mono))
    return join_terms(terms)


def _check_poly(a, names, text_terms=None):
    """SymPy: xᵀAx expanded equals the polynomial the stem states (rebuilt from the monomials)."""
    xs = sympy.symbols(" ".join(names))
    xs = (xs,) if len(names) == 1 else xs
    vec = sympy.Matrix(xs)
    expanded = sympy.expand((vec.T * sym(a) * vec)[0])
    built = sum((a[i][i] * xs[i] ** 2 for i in range(len(names))), sympy.Integer(0)) + sum(
        (2 * a[i][j] * xs[i] * xs[j] for i in range(len(names)) for j in range(i + 1, len(names))), sympy.Integer(0)
    )
    check(sympy.expand(built - expanded) == 0, "the polynomial differs from xᵀAx")


# -- easy ---------------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "The quadratic form q(x, y, z) = {q} can be written q = xᵀAx with A a symmetric matrix. Find A.",
    "Write q(x, y, z) = {q} as xᵀAx, where A is symmetric. Find the matrix A.",
)


def _easy(rng):
    for _ in attempts(1000, "a 3-variable form with half-integer matrix entries"):
        diag = [rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]) for _ in range(3)]
        cross = [rng.randint(-7, 7) for _ in range(3)]
        if sum(c % 2 for c in cross) < 2 or sum(1 for c in cross if c == 0) > 1:
            continue
        break
    c01, c02, c12 = cross
    a = [
        [Fraction(diag[0]), Fraction(c01, 2), Fraction(c02, 2)],
        [Fraction(c01, 2), Fraction(diag[1]), Fraction(c12, 2)],
        [Fraction(c02, 2), Fraction(c12, 2), Fraction(diag[2])],
    ]
    q = join_terms([(diag[0], "x²"), (c01, "xy"), (c02, "xz"), (diag[1], "y²"), (c12, "yz"), (diag[2], "z²")])
    # SymPy: expanding xᵀAx returns the polynomial.
    x, y, z = sympy.symbols("x y z")
    v = sympy.Matrix([x, y, z])
    expected = diag[0] * x**2 + c01 * x * y + c02 * x * z + diag[1] * y**2 + c12 * y * z + diag[2] * z**2
    check(sympy.expand((v.T * sym(a) * v)[0] - expected) == 0, "xᵀAx differs from q")

    answer = smat(a)
    unhalved = [
        [Fraction(diag[0]), Fraction(c01), Fraction(c02)],
        [Fraction(c01), Fraction(diag[1]), Fraction(c12)],
        [Fraction(c02), Fraction(c12), Fraction(diag[2])],
    ]
    upper = [
        [Fraction(diag[0]), Fraction(c01), Fraction(c02)],
        [Fraction(0), Fraction(diag[1]), Fraction(c12)],
        [Fraction(0), Fraction(0), Fraction(diag[2])],
    ]
    all_halved = [[x_ / 2 for x_ in row] for row in unhalved]
    misplaced = [
        [Fraction(diag[0]), Fraction(c01, 2), Fraction(c12, 2)],
        [Fraction(c01, 2), Fraction(diag[1]), Fraction(c02, 2)],
        [Fraction(c12, 2), Fraction(c02, 2), Fraction(diag[2])],
    ]
    candidates = [
        (unhalved, "qf-cross-term-not-halved"),
        (upper, "qf-matrix-not-symmetric"),
        (all_halved, "qf-diagonal-halved"),
        (misplaced, "qf-cross-terms-misplaced"),
    ]
    answer_text = mtext(answer)
    distractors = pick_distractors(answer_text, matrix_options(candidates, answer), 4)
    check(len(distractors) >= 3, "too few distractors")
    return dict(
        text=phrase(rng, _EASY_STEMS, q=q),
        answer_kind="matrix",
        answer=mvalue(answer),
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        working=[
            "The diagonal entries of A are the coefficients of x², y² and z².",
            "Each cross term is split equally between two symmetric entries: the coefficient of xy is 2a₁₂, and so on.",
            f"So A = {answer_text}.",
        ],
        distractors=distractors,
    )


# -- medium -------------------------------------------------------------------------------------------------------

_MEDIUM_STEMS = (
    "The quadratic form q({vars}) = {q} has a symmetric matrix A. Find the diagonal matrix D whose diagonal entries "
    "are the eigenvalues of A in increasing order.",
    "Find the diagonal matrix D that has the eigenvalues of the matrix of q({vars}) = {q} on its diagonal, in "
    "increasing order.",
)


def _diag(values):
    n = len(values)
    return [[Fraction(values[i]) if i == j else Fraction(0) for j in range(n)] for i in range(n)]


def _eigen_candidates(a, lams):
    n = len(lams)
    diagonal = sorted(int(a[i][i]) for i in range(n))
    negated = sorted(-v for v in lams)
    doubled = sym([[2 * v if i != j else v for j, v in enumerate(row)] for i, row in enumerate(a)])
    out = [
        (diagonal, "qf-diagonal-read-as-eigenvalues"),
        (negated, "sign-flip-both-roots"),
        (list(reversed(lams)), "eigval-order-reversed"),
    ]
    for i in range(n):
        flipped = list(lams)
        flipped[i] = -flipped[i]
        out.append((sorted(flipped), "sign-flip-one-root"))
    roots = doubled.eigenvals()
    if all(r.is_Integer for r in roots) and sum(roots.values()) == n:
        out.append((sorted(int(r) for r, m in roots.items() for _ in range(m)), "qf-cross-term-not-halved"))
    return out


def _medium(rng):
    if rng.random() < 0.5:
        _vectors, lams, a = _draw_2x2(rng)
        names = _VARS2
    else:
        vectors, lams = _draw_3x3(rng, rational=rng.random() < 0.7)
        a = _build_a(vectors, lams)
        names = _VARS3
    _check_poly(a, names)
    check(
        sorted(sym(a).eigenvals().items()) == sorted((sympy.Integer(l), lams.count(l)) for l in set(lams)),
        "eigenvalues",
    )
    answer = smat(_diag(lams))
    answer_text = mtext(answer)
    candidates = [(_diag(values), tag) for values, tag in _eigen_candidates(a, lams)]
    distractors = pick_distractors(answer_text, matrix_options(candidates, answer), 4)
    check(len(distractors) >= 3, "too few distractors")
    return dict(
        text=phrase(rng, _MEDIUM_STEMS, vars=", ".join(names), q=_poly_text(a, names)),
        answer_kind="matrix",
        answer=mvalue(answer),
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        working=[
            f"The matrix of q is A = {mat_str(a)}.",
            f"Its characteristic polynomial has roots {', '.join(map(str, lams))}, so D = {answer_text}.",
        ],
        distractors=distractors,
    )


# -- hard ---------------------------------------------------------------------------------------------------------

_SUBST_STEMS = (
    "Substituting {subst} into the quadratic form q({old}) = {q} gives a quadratic form in {new}. Find the symmetric "
    "matrix B of that form.",
    "The quadratic form q({old}) = {q} is rewritten in the variables {new} by {subst}. Find the symmetric matrix of "
    "the rewritten form.",
)
_SQUARES_STEMS = (
    "Find the symmetric matrix A with q = xᵀAx for q({old}) = {q}.",
    "The quadratic form q({old}) = {q} is a sum of squares. Find its symmetric matrix A.",
)


def _small_symmetric(rng, n):
    for _ in attempts(500, "a symmetric matrix with several non-zero entries"):
        a = [[0] * n for _ in range(n)]
        for i in range(n):
            a[i][i] = rng.choice([-3, -2, -1, 1, 2, 3, 4])
            for j in range(i + 1, n):
                a[i][j] = a[j][i] = rng.choice([-2, -1, 0, 1, 2, 3])
        if sum(1 for i in range(n) for j in range(i + 1, n) if a[i][j] != 0) >= (1 if n == 2 else 2):
            return a
    raise AssertionError  # pragma: no cover


def _substitution_matrix(rng, n):
    identity = [[int(i == j) for j in range(n)] for i in range(n)]
    for _ in attempts(500, "an invertible substitution with several non-zero entries"):
        p = [[rng.choice([-2, -1, 0, 1, 1, 2, 3]) for _ in range(n)] for _ in range(n)]
        if sympy.Matrix(p).det() != 0 and sum(1 for row in p for v in row if v) >= n + 2 and p != identity:
            return p
    raise AssertionError  # pragma: no cover


def _hard_substitution(rng):
    n = rng.choice([2, 2, 3])
    names = _VARS2 if n == 2 else _VARS3
    new = ("u", "v") if n == 2 else ("u", "v", "w")
    a = _small_symmetric(rng, n)
    p = _substitution_matrix(rng, n)
    ma, mp = sympy.Matrix(a), sympy.Matrix(p)
    b = mp.T * ma * mp
    # SymPy: substituting x = Py into the polynomial xᵀAx and expanding gives yᵀBy.
    xs = sympy.symbols(" ".join(names))
    ys = sympy.symbols(" ".join(new))
    xv, yv = sympy.Matrix(xs), sympy.Matrix(ys)
    substituted = sympy.expand((xv.T * ma * xv)[0].subs(dict(zip(xs, list(mp * yv))), simultaneous=True))
    check(sympy.expand(substituted - (yv.T * b * yv)[0]) == 0, "the substituted form differs from yᵀBy")
    subst = ", ".join(f"{names[i]} = {join_terms([(p[i][j], new[j]) for j in range(n) if p[i][j]])}" for i in range(n))
    candidates = [
        (mp * ma * mp.T, "qf-congruence-order-reversed"),
        (mp.inv() * ma * mp, "qf-congruence-as-similarity"),
        (ma * mp, "qf-substitution-one-sided"),
        (mp.T * ma, "qf-substitution-one-sided"),
        (mp.T * mp, "qf-substitution-matrix-reported"),
        (ma, "qf-substitution-matrix-reported"),
    ]
    answer = smat(b.tolist())
    answer_text = mtext(answer)
    distractors = pick_distractors(answer_text, matrix_options([(c.tolist(), t) for c, t in candidates], answer), 4)
    check(len(distractors) >= 3, "too few distractors")
    return dict(
        text=phrase(rng, _SUBST_STEMS, old=", ".join(names), new=", ".join(new), q=_poly_text(a, names), subst=subst),
        answer_kind="matrix",
        answer=mvalue(answer),
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        working=[
            f"q = xᵀAx with A = {mat_str(a)}, and the substitution is x = Py with P = {mat_str(p)}.",
            f"Then q = yᵀ(PᵀAP)y, so B = PᵀAP = {answer_text}.",
        ],
        distractors=distractors,
    )


def _linear_form_text(coeffs, names):
    return "(" + join_terms([(c, names[i]) for i, c in enumerate(coeffs) if c]) + ")²"


def _hard_squares(rng):
    n = rng.choice([2, 3, 3])
    names = _VARS2 if n == 2 else _VARS3
    count = 2 if n == 3 else rng.choice([2, 2, 3])
    for _ in attempts(500, "linear forms whose squares have several cross terms"):
        forms = [[rng.choice([-2, -1, 0, 1, 1, 2, 3]) for _ in range(n)] for _ in range(count)]
        if all(sum(1 for c in f if c) >= 2 for f in forms) and len({tuple(f) for f in forms}) == count:
            break
    a = [[sum(f[i] * f[j] for f in forms) for j in range(n)] for i in range(n)]
    xs = sympy.symbols(" ".join(names))
    xv = sympy.Matrix(xs)
    expanded = sum((sum(f[i] * xs[i] for i in range(n)) ** 2 for f in forms), sympy.Integer(0))
    check(sympy.expand(expanded - (xv.T * sympy.Matrix(a) * xv)[0]) == 0, "the sum of squares differs from xᵀAx")
    q = " + ".join(_linear_form_text(f, names) for f in forms)
    dropped = [[sum(f[i] ** 2 for f in forms) if i == j else 0 for j in range(n)] for i in range(n)]
    undivided = [[a[i][j] * (2 if i != j else 1) for j in range(n)] for i in range(n)]
    one_form = [[forms[0][i] * forms[0][j] for j in range(n)] for i in range(n)]
    answer = smat(a)
    answer_text = mtext(answer)
    candidates = [
        (dropped, "qf-squares-cross-terms-dropped"),
        (undivided, "qf-squares-cross-terms-not-halved"),
        (one_form, "qf-squares-cross-terms-dropped"),
    ]
    distractors = pick_distractors(answer_text, matrix_options(candidates, answer), 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _SQUARES_STEMS, old=", ".join(names), q=q),
        answer_kind="matrix",
        answer=mvalue(answer),
        answer_text=answer_text,
        answer_latex="plain:" + answer_text,
        working=[
            "Each square (c₁x + c₂y + ...)² contributes the matrix ccᵀ, where c lists its coefficients.",
            f"Adding them, A = {answer_text}.",
        ],
        distractors=distractors,
    )


def _hard(rng):
    return rng.choice([_hard_substitution, _hard_substitution, _hard_squares])(rng)


@generator(
    "s2b2_quadratic_form_definiteness",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@tidied
def s2b2_quadratic_form_definiteness(rng, tier="easy"):
    """Matrix, eigenvalues, definiteness and principal axes of a quadratic form; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------

_CLASSES = (
    "positive definite",
    "positive semi-definite",
    "negative definite",
    "negative semi-definite",
    "indefinite",
)


def _classify(lams):
    if all(v > 0 for v in lams):
        return _CLASSES[0]
    if all(v >= 0 for v in lams):
        return _CLASSES[1]
    if all(v < 0 for v in lams):
        return _CLASSES[2]
    if all(v <= 0 for v in lams):
        return _CLASSES[3]
    return _CLASSES[4]


_CLASS_TAGS = {
    ("positive definite", "positive semi-definite"): "qf-semidefinite-vs-definite",
    ("positive definite", "negative definite"): "qf-sign-reversed",
    ("positive definite", "negative semi-definite"): "qf-sign-reversed",
    ("positive definite", "indefinite"): "qf-indefinite-assumed",
}


def _class_tag(truth, option):
    if option == "indefinite":
        return "qf-indefinite-assumed"
    if truth.split()[-1] == option.split()[-1] or truth.replace("semi-", "") == option.replace("semi-", ""):
        return "qf-semidefinite-vs-definite"
    if "semi" in truth and "semi" not in option and option.split()[0] == truth.split()[0]:
        return "qf-semidefinite-vs-definite"
    return "qf-sign-reversed"


def _exam_lams(rng):
    target = rng.choice(_CLASSES)

    def accept(ls):
        return _classify(ls) == target

    return accept


@generator("exd_x_s2b2_quadratic_form_definiteness", level=19, presets={"exam": dict(tier="exam")})
@tidied
def exd_x_s2b2_quadratic_form_definiteness(rng, tier="exam"):
    """Exam chain: the matrix of a 2-variable form, its two eigenvalues (the second from the trace) and its class."""
    check(tier == "exam", f"unknown tier {tier!r}")
    accept = _exam_lams(rng)
    _vectors, lams, a = _draw_2x2(rng, accept=accept)
    _check_poly(a, _VARS2)
    lo, hi = lams
    trace = int(a[0][0] + a[1][1])
    check(trace == lo + hi, "trace differs from the eigenvalue sum")
    truth = _classify(lams)
    check(truth == sympy_class(sym(a)), "SymPy's class differs")
    answer_rows = smat(a)
    answer_text = mtext(answer_rows)
    b = int(2 * a[0][1])
    q = _poly_text(a, _VARS2)
    unhalved = [[int(a[0][0]), b], [b, int(a[1][1])]]
    upper = [[int(a[0][0]), b], [0, int(a[1][1])]]
    a_dist = pick_distractors(
        answer_text,
        matrix_options(
            [
                (unhalved, "qf-cross-term-not-halved"),
                (upper, "qf-matrix-not-symmetric"),
                ([[int(a[0][0]), int(a[0][0]) and b // 2 or 0], [b // 2, int(a[1][1])]], "qf-cross-terms-misplaced"),
                ([[int(a[0][0]), -int(a[0][1])], [-int(a[0][1]), int(a[1][1])]], "sign-error-moving-term"),
            ],
            answer_rows,
        ),
        3,
    )
    b_dist = pick_distractors(
        str(hi),
        [
            number_distractor(lo, "eigval-other-root"),
            number_distractor(-hi, "sign-flip-both-roots"),
            number_distractor(max(int(a[0][0]), int(a[1][1])), "qf-diagonal-read-as-eigenvalues"),
            number_distractor(trace, "eigval-trace-as-eigenvalue"),
        ],
        3,
    )
    c_dist = pick_distractors(
        str(lo),
        [
            number_distractor(trace + hi, "eigval-trace-difference-sign"),
            number_distractor(trace, "eigval-trace-as-eigenvalue"),
            number_distractor(hi, "eigval-repeated-given"),
            number_distractor(int(a[0][0] * a[1][1] - a[0][1] ** 2) - hi, "eigval-determinant-as-eigenvalue"),
            number_distractor(-lo, "sign-flip-both-roots"),
            number_distractor(trace - lo, "eigval-other-root"),
        ],
        3,
    )
    options = [c for c in _CLASSES if c != truth]
    d_dist = [Distractor(c, _class_tag(truth, c), c) for c in options]
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist)), "too few distractors")
    return dict(
        text=f"The quadratic form q(x, y) = {q} can be written q = xᵀAx with A a symmetric matrix.",
        parts=[
            dict(
                prompt="Find A.",
                answer_kind="matrix",
                answer=mvalue(answer_rows),
                answer_text=answer_text,
                answer_latex="plain:" + answer_text,
                marks=2,
                working=[
                    f"The diagonal entries are the coefficients of x² and y²; the off-diagonal entries are half the xy coefficient {b}."
                ],
                distractors=a_dist,
            ),
            dict(
                prompt="Find the larger eigenvalue of A.",
                answer_kind="number",
                answer=hi,
                answer_text=str(hi),
                marks=2,
                working=[f"det(A - λI) = 0 has roots {lo} and {hi}, so the larger is {hi}."],
                distractors=b_dist,
                sensible_range=(-30, 30),
            ),
            dict(
                prompt="Use the trace of A to find the other eigenvalue.",
                answer_kind="number",
                answer=lo,
                answer_text=str(lo),
                marks=1,
                follow_through=f"{trace} - part_b",
                working=[f"The eigenvalues add up to the trace {trace}, so the other is {trace} - ({hi}) = {lo}."],
                distractors=c_dist,
            ),
            dict(
                prompt="Classify q: is it positive definite, positive semi-definite, negative definite, negative "
                "semi-definite or indefinite?",
                answer_kind="mcq",
                answer=truth,
                answer_text=truth,
                marks=2,
                working=[f"The eigenvalues are {lo} and {hi}, so q is {truth}."],
                distractors=d_dist,
            ),
        ],
    )


def sympy_class(m):
    """The definiteness class of a symmetric SymPy matrix, from its (real, exact) eigenvalues."""
    values = list(m.eigenvals())
    return _classify(values)
