"""Generates `s2b1_kernel_and_image` questions (easy, medium, hard) and the exam chain `exd_x_s2b1_kernel_image`.

``uc2-kernel-and-image`` (``linear-algebra``, answer ``number``): the dimension of the kernel and image of a linear map
that is *not* handed over as a matrix: maps on polynomial spaces and on 2x2 matrices, given by a formula or by
conditions. (``ucv1-null-space`` already covers the rank, nullity and column space of a supplied integer matrix.)

* easy: a map defined by one idea that can be read off without a table of numbers: p(x) - p(-x), p'' on P_4, the
  values at three nodes on P_4, (p(1), p'(1)) on P_3, X -/+ X^T on 2x2 matrices. Ask for the dimension of the kernel or of
  the image.
* medium: maps that need the matrix of the map or a structural argument: p(x + s) - p(x), x p' - k p, p'' - c p',
  X -> AX for a rank-1 or rank-2 A, X -> AX - XA. Ask for the rank or the nullity.
* hard: a parameter. The value of k that gives T_k(p) = (L1, L2, L3 + k L4) on P_2 a non-trivial kernel (a determinant
  linear in k); the larger k for which X -> AX - kX on 2x2 matrices is not injective (an eigenvalue); the dimension
  of the kernel of the Sylvester map X -> AX - XB on 2x2 matrices (the number of eigenvalues A and B share, since each
  of A and B has two distinct eigenvalues).
* exam (``exd_x_s2b1_kernel_image``): T_k: P_2 -> R^3 with k in one functional. (a) the k for which T_k is not
  injective, (b) the dimension of the image there, (c) the dimension of the kernel (follows through from (b) by
  rank-nullity), (d) the constant
  term of the kernel vector whose x^2 coefficient is 1 (a fraction).

Every dimension is found from the exact matrix of the map by Gauss-Jordan elimination and re-derived with SymPy's
`rank()` and `nullspace()`; each k is checked by asking SymPy for the rank at k and at k + 1.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, join_terms, phrase, pick_distractors, rational_text
from ._s2b1_linalg import (
    X,
    det_root_distractors,
    int_distractors,
    poly_expr,
    std_matrix,
    sym,
)
from ._ucv1_linalg import draw_2x2, fm, nullspace, rank

LEVEL = 19
_ENTRIES = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _matrix_map(apply):
    """The 4x4 matrix of a map M_2 -> M_2 in the basis E11, E12, E21, E22."""
    cols = []
    for i, j in _ENTRIES:
        e = sympy.zeros(2, 2)
        e[i, j] = 1
        image = apply(e)
        cols.append([Fraction(int(image[r, c])) for r, c in _ENTRIES])
    return [list(row) for row in zip(*cols)]


def _verify_dims(a, nullity, rnk):
    """SymPy re-derivation of rank and nullity of the exact matrix `a`."""
    s = sym(a)
    check(s.rank() == rnk, "SymPy's rank differs")
    check(len(s.nullspace()) == nullity, "SymPy's nullity differs")
    check(nullity + rnk == len(a[0]), "rank-nullity fails")


def _dims(a):
    a = fm(a)
    rnk = rank(a)
    nullity = len(a[0]) - rnk
    check(len(nullspace(a)) == nullity, "hand nullspace differs from n - rank")
    _verify_dims(a, nullity, rnk)
    return nullity, rnk


# -- the map pools ---------------------------------------------------------------------------------------------------


def _signed(sign, term):
    return f"{term}" if sign > 0 else f"-{term}"


def _sub(n: int) -> str:
    return "P" + "₀₁₂₃₄₅₆₇₈₉"[n]


def _easy_maps(rng):
    """(description, domain, codomain, matrix) for the easy pool."""
    sign = rng.choice([1, -1])
    n = rng.choice([3, 4, 5])
    nodes = sorted(rng.sample([-2, -1, 0, 1, 2, 3], rng.choice([2, 3])))
    a = rng.choice([-2, -1, 1, 2])
    cases = []
    # p(x) -/+ p(-x) on P_n
    ap = lambda p: sympy.expand(p - p.subs(X, -X)) if sign < 0 else sympy.expand(p + p.subs(X, -X))  # noqa: E731
    cases.append((f"T(p)(x) = p(x) {'-' if sign < 0 else '+'} p(-x)", _sub(n), _sub(n), std_matrix(ap, n + 1, n)))
    # second derivative on P_n
    cases.append(("T(p)(x) = p''(x)", _sub(n), _sub(n - 2), std_matrix(lambda p: sympy.diff(p, X, 2), n + 1, n - 2)))
    # values at nodes on P_n
    node_text = ", ".join(f"p({v})" for v in nodes)
    rows = [[Fraction(v**j) for j in range(n + 1)] for v in nodes]
    cases.append((f"T(p) = ({node_text})", _sub(n), f"ℝ{'²³'[len(nodes) - 2]}", rows))
    # (p(a), p'(a)) on P_n
    rows2 = [
        [Fraction(a**j) for j in range(n + 1)],
        [Fraction(j * a ** (j - 1)) if j else Fraction(0) for j in range(n + 1)],
    ]
    cases.append((f"T(p) = (p({a}), p'({a}))", _sub(n), "ℝ²", rows2))
    # p(x) - p(a) on P_n
    cases.append((f"T(p)(x) = p(x) - p({a})", _sub(n), _sub(n), std_matrix(lambda p: p - p.subs(X, a), n + 1, n)))
    # X -/+ X^T on 2x2 matrices, and X -> tr(X) I
    cases.append(
        (
            f"T(X) = X {'-' if sign < 0 else '+'} Xᵀ",
            "M₂(ℝ)",
            "M₂(ℝ)",
            _matrix_map(lambda m: m - m.T if sign < 0 else m + m.T),
        )
    )
    cases.append(
        (
            "T(X) = tr(X)·I, where I is the 2×2 identity matrix",
            "M₂(ℝ)",
            "M₂(ℝ)",
            _matrix_map(lambda m: m.trace() * sympy.eye(2)),
        )
    )
    return cases


def _medium_maps(rng):
    s = rng.choice([-2, -1, 1, 2])
    k = rng.choice([0, 1, 2, 3, 4, 5])
    c = rng.choice([-2, -1, 1, 2, 3])
    cases = []
    cases.append(
        (
            f"T(p)(x) = p(x {'+' if s > 0 else '-'} {abs(s)}) - p(x)",
            "P₃",
            "P₃",
            std_matrix(lambda p: sympy.expand(p.subs(X, X + s) - p), 4, 3),
        )
    )
    cases.append(
        (
            f"T(p)(x) = {join_terms([(1, 'x p' + chr(39) + '(x)'), (-k, 'p(x)')])}",
            "P₃",
            "P₃",
            std_matrix(lambda p: X * sympy.diff(p, X) - k * p, 4, 3),
        )
    )
    cases.append(
        (
            f"T(p)(x) = p''(x) {'-' if c > 0 else '+'} {abs(c)}p'(x)",
            "P₄",
            "P₄",
            std_matrix(lambda p: sympy.diff(p, X, 2) - c * sympy.diff(p, X), 5, 4),
        )
    )
    # X -> AX for a 2x2 integer A of rank 1 or 2
    for _ in attempts(200, "a 2x2 matrix of rank 1 or 2"):
        am = sympy.Matrix(2, 2, lambda i, j: rng.randint(-3, 3))
        if am.rank() >= 1 and am != sympy.zeros(2, 2):
            break
    cases.append(
        (
            f"T(X) = AX, where A = [[{am[0, 0]}, {am[0, 1]}], [{am[1, 0]}, {am[1, 1]}]]",
            "M₂(ℝ)",
            "M₂(ℝ)",
            _matrix_map(lambda m: am * m),
        )
    )
    for _ in attempts(200, "a non-scalar 2x2 matrix"):
        bm = sympy.Matrix(2, 2, lambda i, j: rng.randint(-3, 3))
        if bm[0, 1] != 0 or bm[1, 0] != 0 or bm[0, 0] != bm[1, 1]:
            break
    cases.append(
        (
            f"T(X) = AX - XA, where A = [[{bm[0, 0]}, {bm[0, 1]}], [{bm[1, 0]}, {bm[1, 1]}]]",
            "M₂(ℝ)",
            "M₂(ℝ)",
            _matrix_map(lambda m: bm * m - m * bm),
        )
    )
    return cases


# -- question assembly ---------------------------------------------------------------------------------------------------

_DOMAIN_WORDS = {
    "P₀": "constant polynomials",
    "P₁": "polynomials of degree at most 1",
    "P₂": "polynomials of degree at most 2",
    "P₃": "polynomials of degree at most 3",
    "P₄": "polynomials of degree at most 4",
    "P₅": "polynomials of degree at most 5",
    "M₂(ℝ)": "2×2 real matrices",
}


def _space_note(domain, codomain):
    notes = []
    for sp in (domain, codomain):
        if sp in _DOMAIN_WORDS and sp not in [n.split(" is")[0] for n in notes]:
            notes.append(f"{sp} is the space of {_DOMAIN_WORDS[sp]}")
    return ("; ".join(notes) + ".") if notes else ""


def _count_question(rng, case, ask):
    description, domain, codomain, matrix = case
    nullity, rnk = _dims(matrix)
    n_in, n_out = len(matrix[0]), len(matrix)
    note = _space_note(domain, codomain)
    quantity = {
        "ker": "the dimension of the kernel of T",
        "im": "the dimension of the image of T",
        "rank": "the rank of T",
    }[ask]
    answer = nullity if ask == "ker" else rnk
    stem = phrase(
        rng,
        (
            "Let T: {d} → {c} be the linear map {desc}. {note} Find {q}.",
            "The map T: {d} → {c} is linear, with {desc}. {note} What is {q}?",
        ),
        d=domain,
        c=codomain,
        desc=description,
        note=note,
        q=quantity,
    ).replace("  ", " ")
    if ask == "ker":
        cands = [
            (rnk, "rank-nullity-confused"),
            (n_in, "nullity-counts-columns"),
            (n_out - rnk, "nullity-uses-rows"),
            (0, "kernel-assumed-trivial"),
            (nullity + 1, "rank-off-by-one"),
            (nullity - 1, "rank-off-by-one"),
        ]
        working = [f"The matrix of T has rank {rnk}, so by rank-nullity dim ker T = {n_in} - {rnk} = {nullity}."]
    else:
        cands = [
            (nullity, "rank-nullity-confused"),
            (n_out, "image-assumed-whole-codomain"),
            (n_in, "nullity-counts-columns"),
            (rnk + 1, "rank-off-by-one"),
            (rnk - 1, "rank-off-by-one"),
        ]
        working = [
            f"The image is spanned by the columns of the matrix of T, which has rank {rnk}; dim ker T = {nullity} and {nullity} + {rnk} = {n_in}."
        ]
    cands = [(v, t) for v, t in cands if v >= 0]
    distractors = pick_distractors(str(answer), int_distractors(cands, answer), 4)
    if len(distractors) < 3:
        return None
    return dict(
        text=stem,
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=distractors,
    )


def _retry(rng, build):
    for _ in attempts(200, "a question with enough distractors"):
        result = build(rng)
        if result is not None:
            return result
    raise AssertionError  # pragma: no cover


def _easy(rng):
    return _count_question(rng, rng.choice(_easy_maps(rng)), rng.choice(["ker", "im"]))


def _medium(rng):
    return _count_question(rng, rng.choice(_medium_maps(rng)), rng.choice(["ker", "rank", "im"]))


# -- hard ------------------------------------------------------------------------------------------------------------------

_FUNCTIONALS = [
    ("p(0)", lambda p: p.subs(X, 0)),
    ("p(1)", lambda p: p.subs(X, 1)),
    ("p(-1)", lambda p: p.subs(X, -1)),
    ("p(2)", lambda p: p.subs(X, 2)),
    ("p'(0)", lambda p: sympy.diff(p, X).subs(X, 0)),
    ("p'(1)", lambda p: sympy.diff(p, X).subs(X, 1)),
    ("p'(-1)", lambda p: sympy.diff(p, X).subs(X, -1)),
    ("p(0) + p(1)", lambda p: p.subs(X, 0) + p.subs(X, 1)),
]


def _functional_row(f):
    return [sympy.Integer(int(f(X**j))) for j in range(3)]


def _draw_k_map(rng):
    """(texts, rows(k), k0, d1): T_k(p) = (L1, L2, L3 + k L4) with det linear in k and an integer root k0 != 0."""
    k = sympy.symbols("k")
    for _ in attempts(2000, "functionals whose determinant has an integer root in k"):
        (t1, f1), (t2, f2), (t3, f3), (t4, f4) = rng.sample(_FUNCTIONALS, 4)
        rows = [
            _functional_row(f1),
            _functional_row(f2),
            [a + k * b for a, b in zip(_functional_row(f3), _functional_row(f4))],
        ]
        det = sympy.expand(sympy.Matrix(rows).det())
        poly = sympy.Poly(det, k)
        if poly.degree() != 1:
            continue
        d1, d0 = int(poly.coeff_monomial(k)), int(poly.coeff_monomial(1))
        if d0 % d1 != 0 or d0 == 0:
            continue
        k0 = -d0 // d1
        if abs(k0) > 8:
            continue
        at = sympy.Matrix(rows).subs(k, k0)
        if at.rank() != 2:
            continue
        return (t1, t2, t3, t4), rows, k0, d1, (f1, f2, f3, f4)
    raise AssertionError  # pragma: no cover


def _k_text(texts):
    t1, t2, t3, t4 = texts
    return f"({t1}, {t2}, {t3} + k·{t4 if ' ' not in t4 else '(' + t4 + ')'})"


def _hard_parameter(rng):
    texts, rows, k0, d1, _fs = _draw_k_map(rng)
    k = sympy.symbols("k")
    # Independent: the rank is 2 at k0 and 3 at k0 + 1.
    check(sympy.Matrix(rows).subs(k, k0).rank() == 2, "T_k0 has full rank")
    check(sympy.Matrix(rows).subs(k, k0 + 1).rank() == 3, "T_{k0+1} is not injective")
    check(sympy.solve(sympy.Matrix(rows).det(), k) == [k0], "SymPy's root differs")
    cands = [(-k0, "sign-error-moving-term"), (k0 + 1, "rank-off-by-one")]
    cands += [(Fraction(r), t) for r, t in det_root_distractors(rows, k, k0)]
    distractors = pick_distractors(str(k0), int_distractors(cands, k0), 4)
    if len(distractors) < 3:
        return None
    stem = phrase(
        rng,
        (
            "For a real number k let T_k: P₂ → ℝ³ be the linear map T_k(p) = {f}, where P₂ is the space of polynomials of degree at "
            "most 2. Find the value of k for which T_k has a non-trivial kernel.",
            "Let P₂ be the space of polynomials of degree at most 2, and for each real k define the linear map T_k: P₂ → ℝ³ by "
            "T_k(p) = {f}. For which value of k is T_k not injective?",
        ),
        f=_k_text(texts),
    )
    working = [
        "T_k is injective exactly when its 3×3 matrix (in the basis 1, x, x²) is invertible, so set its determinant to zero.",
        f"The determinant is {d1}(k - ({k0})) up to the order of rows, so k = {k0}; there the rank is 2.",
    ]
    return dict(
        text=stem,
        answer_kind="number",
        answer=k0,
        answer_text=str(k0),
        answer_latex=str(k0),
        working=working,
        distractors=distractors,
    )


def _hard_eigen(rng):
    p, (l1, l2), a = draw_2x2(rng, -4, 5, 12, full=True, bound=2)
    k = sympy.symbols("k")
    larger = max(l1, l2)
    am = sympy.Matrix([[int(v) for v in row] for row in a])
    check(sorted(sympy.solve((am - k * sympy.eye(2)).det(), k)) == sorted([l1, l2]), "SymPy's eigenvalues differ")
    # T_k has the 4x4 matrix (A - kI) ⊗ I; its nullity is 2 at an eigenvalue and 0 elsewhere.
    rows = _matrix_map(lambda m: am * m - larger * m)
    check(_dims(rows)[0] == 2, "T at the larger eigenvalue does not have a 2-dimensional kernel")
    trace, determinant = int(am.trace()), int(am.det())
    cands = [
        (min(l1, l2), "eigval-other-root"),
        (-larger, "sign-error-moving-term"),
        (trace, "eigval-trace-as-eigenvalue"),
        (int(a[0][0]), "eigval-diagonal-entry"),
        (determinant, "eigval-determinant-as-eigenvalue"),
    ]
    distractors = pick_distractors(str(larger), int_distractors(cands, larger), 4)
    if len(distractors) < 3:
        return None
    stem = phrase(
        rng,
        (
            "For a real number k let T_k: M₂(ℝ) → M₂(ℝ) be the linear map T_k(X) = AX - kX, where A = [[{a}, {b}], [{c}, {d}]] and "
            "M₂(ℝ) is the space of 2×2 real matrices. T_k fails to be injective for exactly two values of k. Find the larger one.",
            "Let A = [[{a}, {b}], [{c}, {d}]] and, for each real k, let T_k(X) = AX - kX define a linear map on the space M₂(ℝ) of "
            "2×2 real matrices. For two values of k the map T_k is not injective. What is the larger of them?",
        ),
        a=int(a[0][0]),
        b=int(a[0][1]),
        c=int(a[1][0]),
        d=int(a[1][1]),
    )
    working = [
        "T_k(X) = (A - kI)X is injective exactly when A - kI is invertible, that is, unless k is an eigenvalue of A.",
        f"det(A - kI) = (k - ({l1}))(k - ({l2})), so the larger value is k = {larger}.",
    ]
    return dict(
        text=stem,
        answer_kind="number",
        answer=larger,
        answer_text=str(larger),
        answer_latex=str(larger),
        working=working,
        distractors=distractors,
    )


def _hard_sylvester(rng):
    shared = rng.choice([0, 1, 1, 2])
    for _ in attempts(500, "two matrices with the required shared eigenvalues"):
        pool = rng.sample(range(-3, 5), 4)
        ev_a = pool[:2]
        if shared == 2:
            ev_b = list(reversed(ev_a)) if rng.random() < 0.5 else list(ev_a)
        elif shared == 1:
            ev_b = [ev_a[rng.randrange(2)], pool[2]]
            rng.shuffle(ev_b)
        else:
            ev_b = pool[2:]
        if len(set(ev_b)) != 2 or len(set(ev_a)) != 2:
            continue
        a = _conjugate(rng, ev_a)
        b = _conjugate(rng, ev_b)
        if a is not None and b is not None:
            break
    am, bm = sympy.Matrix([[int(v) for v in r] for r in a]), sympy.Matrix([[int(v) for v in r] for r in b])
    matrix = _matrix_map(lambda m: am * m - m * bm)
    nullity, rnk = _dims(matrix)
    count = sum(1 for x in ev_a for y in ev_b if x == y)
    check(nullity == count, "kernel dimension differs from the number of shared eigenvalue pairs")
    cands = [
        (rnk, "rank-nullity-confused"),
        (4, "nullity-counts-columns"),
        (2 if count != 2 else 1, "kernel-assumed-by-eigenvalue-count"),
        (0, "kernel-assumed-trivial"),
        (count + 1, "rank-off-by-one"),
        (1, "rank-off-by-one"),
    ]
    distractors = pick_distractors(str(nullity), int_distractors(cands, nullity), 4)
    if len(distractors) < 3:
        return None
    stem = phrase(
        rng,
        (
            "Let A = {a} and B = {b}, and let T: M₂(ℝ) → M₂(ℝ) be the linear map T(X) = AX - XB on the space of 2×2 real matrices. "
            "Find the dimension of the kernel of T.",
            "The map T(X) = AX - XB on the space M₂(ℝ) of 2×2 real matrices has A = {a} and B = {b}. What is dim ker T?",
        ),
        a=_text(a),
        b=_text(b),
    )
    working = [
        "AX = XB has a non-zero solution X = u wᵀ whenever u is an eigenvector of A and w is an eigenvector of Bᵀ for the same eigenvalue.",
        f"A has eigenvalues {ev_a[0]} and {ev_a[1]}, B has {ev_b[0]} and {ev_b[1]}; each shared eigenvalue adds one dimension, so dim ker T = {nullity}.",
    ]
    return dict(
        text=stem,
        answer_kind="number",
        answer=nullity,
        answer_text=str(nullity),
        answer_latex=str(nullity),
        working=working,
        distractors=distractors,
    )


def _text(m):
    return "[[" + f"{int(m[0][0])}, {int(m[0][1])}], [{int(m[1][0])}, {int(m[1][1])}" + "]]"


def _conjugate(rng, evs):
    """An integer 2x2 matrix with eigenvalues `evs`, not diagonal and not triangular, with small entries."""
    from ._ucv1_linalg import diagonal, inverse, mmul, max_abs, unimodular

    for _ in attempts(400, "an integer matrix with the given eigenvalues"):
        p = unimodular(rng, 2, 3)
        m = mmul(mmul(p, diagonal(evs)), inverse(p))
        if max_abs(m) <= 16 and m[0][1] != 0 and m[1][0] != 0:
            return m
    return None


@generator(
    "s2b1_kernel_and_image",
    level=LEVEL,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def s2b1_kernel_and_image(rng, tier="easy"):
    """Dimension of the kernel or image of a linear map on polynomials or 2x2 matrices; tiers in the module docstring."""
    if tier == "easy":
        return _retry(rng, _easy)
    if tier == "medium":
        return _retry(rng, _medium)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _retry(rng, rng.choice([_hard_parameter, _hard_parameter, _hard_eigen, _hard_sylvester]))


# -- exam ----------------------------------------------------------------------------------------------------------------------


def _exam_try(rng):
    texts, rows, k0, d1, _fs = _draw_k_map(rng)
    k = sympy.symbols("k")
    at = sympy.Matrix(rows).subs(k, k0)
    check(at.rank() == 2 and len(at.nullspace()) == 1, "T_k0 does not have a line as kernel")
    kernel = at.nullspace()[0]
    if kernel[2] == 0:
        return None
    normalised = kernel / kernel[2]  # coefficients (c0, c1, 1): the kernel polynomial with x^2 coefficient 1
    const = Fraction(int(normalised[0].p), int(normalised[0].q))
    check(all(sum(rows[r][c].subs(k, k0) * normalised[c] for c in range(3)) == 0 for r in range(3)), "T_k0 (q) != 0")
    hand = nullspace([[Fraction(int(v)) for v in row] for row in at.tolist()])
    check(len(hand) == 1, "hand kernel is not a line")
    a_dist = pick_distractors(
        str(k0),
        int_distractors(
            [
                (-k0, "sign-error-moving-term"),
                (k0 + 1, "rank-off-by-one"),
                *[(Fraction(r), t) for r, t in det_root_distractors(rows, k, k0)],
            ],
            k0,
        ),
        3,
    )
    b_dist = pick_distractors(
        "2",
        int_distractors([(1, "rank-off-by-one"), (3, "image-assumed-whole-codomain"), (0, "rank-off-by-one")], 2),
        3,
    )
    c_dist = pick_distractors(
        "1",
        int_distractors([(3, "nullity-counts-columns"), (0, "kernel-assumed-trivial")], 1),
        3,
    )
    d_text = rational_text(const)
    d_dist = pick_distractors(
        d_text,
        [
            Distractor(rational_text(-const), "sign-error-moving-term", exact_value(-const)),
            *(
                [
                    Distractor(
                        rational_text(Fraction(1) / const),
                        "kernel-coefficient-inverted",
                        exact_value(Fraction(1) / const),
                    )
                ]
                if const != 0
                else []
            ),
            Distractor(
                rational_text(Fraction(int(kernel[0].p), int(kernel[0].q))),
                "kernel-not-normalised",
                exact_value(Fraction(int(kernel[0].p), int(kernel[0].q))),
            ),
            Distractor(
                rational_text(Fraction(int(normalised[1].p), int(normalised[1].q))),
                "kernel-wrong-coefficient",
                exact_value(Fraction(int(normalised[1].p), int(normalised[1].q))),
            ),
        ],
        3,
    )
    if not all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)):
        return None
    stem = (
        f"For a real number k let T_k: P₂ → ℝ³ be the linear map T_k(p) = {_k_text(texts)}, where P₂ is the space of polynomials "
        "of degree at most 2."
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the value of k for which T_k is not injective.",
                answer_kind="number",
                answer=k0,
                answer_text=str(k0),
                marks=3,
                working=[f"The determinant of the matrix of T_k vanishes only at k = {k0}."],
                distractors=a_dist,
            ),
            dict(
                prompt="For this value of k, find the dimension of the image of T_k.",
                answer_kind="number",
                answer=2,
                answer_text="2",
                marks=2,
                working=[
                    "The matrix of T_k has two independent rows at this k and not three, so the image has dimension 2."
                ],
                distractors=b_dist,
            ),
            dict(
                prompt="Use rank-nullity to find the dimension of the kernel of T_k for this value of k.",
                answer_kind="number",
                answer=1,
                answer_text="1",
                marks=1,
                follow_through="3 - part_b",
                working=["dim ker T_k = dim P₂ - dim im T_k = 3 - 2 = 1."],
                distractors=c_dist,
            ),
            dict(
                prompt="The kernel of T_k is spanned by exactly one polynomial whose x² coefficient is 1. Find its constant term.",
                answer_kind="fraction",
                answer=exact_value(const),
                answer_text=d_text,
                marks=3,
                working=[f"Solving T_k(a + bx + x²) = 0 for k = {k0} gives a = {d_text}."],
                distractors=d_dist,
            ),
        ],
    )


@generator("exd_x_s2b1_kernel_image", level=LEVEL, presets={"exam": dict(tier="exam")})
def exd_x_s2b1_kernel_image(rng, tier="exam"):
    """Exam chain: the k with a non-trivial kernel, its nullity, the rank by rank-nullity, and a normalised kernel vector."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _retry(rng, _exam_try)


__all__ = ["s2b1_kernel_and_image", "exd_x_s2b1_kernel_image", "poly_expr", "join_terms"]
