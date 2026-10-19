"""Generates `s2b2_diagonalisability_test` questions (easy, medium, hard) and the exam chain
`exd_x_s2b2_diagonalisability_test`.

``uc2-diagonalisability-test`` (``linear-algebra``): is a 3×3 integer matrix diagonalisable? The test is that, for every
eigenvalue, the geometric multiplicity (the dimension of the eigenspace, 3 - rank(A - λI)) equals the algebraic
multiplicity (the multiplicity as a root of the characteristic polynomial).

* easy: for a stated eigenvalue of a dense 3×3 matrix, its geometric multiplicity (1 or 2), from the rank of A - λI;
  the algebraic multiplicity is 2 or 3, so the answer is never just the multiplicity of the root. Answer ``number``.
* medium: a parameter k in one off-diagonal entry of a matrix whose eigenvalues λ, μ, λ do not depend on k; the one
  value of k for which it is diagonalisable (the two rows of A - λI must become proportional). Answer ``number``.
* hard: k on the diagonal, so the eigenvalues k, λ, μ collide at k = λ and at k = μ; A is defective at exactly one of
  them, and the value of k for which A is not diagonalisable is asked for. Answer ``number``.
* exam: a dense matrix with a repeated eigenvalue: (a) that eigenvalue, (b) the third eigenvalue from the trace,
  (c) rank(A - λI), (d) the geometric multiplicity, 3 - rank.

Matrices are built as a Jordan matrix conjugated by a unimodular P (easy, exam) or as a triangular matrix with a
permutation similarity (medium, hard). Multiplicities are re-derived with SymPy (``eigenvals``, ``rank``,
``is_diagonalizable``).
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator, phrase, pick_distractors
from ._s2b2_kit import (
    tidied,
    conjugate,
    jordan_matrix,
    mat_str,
    number_distractor,
    permute,
    rank,
    sym,
)

_LAMBDAS = range(-3, 5)


def _sym_structure(a, lam):
    """(algebraic, geometric) multiplicity of `lam` for the integer matrix `a`, by SymPy."""
    m = sym(a)
    alg = m.eigenvals().get(sympy.Integer(lam), 0)
    geo = 3 - (m - lam * sympy.eye(3)).rank()
    return int(alg), int(geo)


# -- easy ---------------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "Find the geometric multiplicity of the eigenvalue λ = {lam} of A = {m}.",
    "The matrix A = {m} has λ = {lam} as an eigenvalue. Find the dimension of the eigenspace of A for λ = {lam}.",
    "For A = {m}, find the geometric multiplicity of the eigenvalue λ = {lam}.",
)

_PATTERNS = {
    (2, 1): lambda lam, mu: [(lam, 2), (mu, 1)],
    (2, 2): lambda lam, mu: [(lam, 1), (lam, 1), (mu, 1)],
    (3, 2): lambda lam, mu: [(lam, 2), (lam, 1)],
    (3, 1): lambda lam, mu: [(lam, 3)],
}


def _easy(rng):
    key = rng.choice([(2, 1), (2, 2), (3, 2), (3, 1), (2, 1), (2, 2)])
    lam, mu = rng.sample(_LAMBDAS, 2)
    blocks = _PATTERNS[key](lam, mu)
    rng.shuffle(blocks)
    a, _p = conjugate(rng, jordan_matrix(blocks))
    check(_sym_structure(a, lam) == key, "SymPy's multiplicities differ from the construction")
    alg, geo = key
    rk = 3 - geo
    candidates = [
        number_distractor(alg, "geo-equals-alg"),
        number_distractor(rk, "rank-nullity-confused"),
        number_distractor(1, "geo-always-one"),
        number_distractor(geo + 1, "rank-off-by-one"),
        number_distractor(geo - 1, "rank-off-by-one"),
        number_distractor(alg + 1, "alg-multiplicity-off-by-one"),
        number_distractor(3, "geo-reported-as-matrix-size"),
    ]
    text = str(geo)
    distractors = pick_distractors(text, candidates, 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _EASY_STEMS, m=mat_str(a), lam=lam),
        answer_kind="number",
        answer=geo,
        answer_text=text,
        working=[
            f"A - {lam}I has rank {rk}, so by rank-nullity the eigenspace has dimension 3 - {rk} = {geo}.",
            f"(The algebraic multiplicity of {lam} is {alg}; the geometric multiplicity {geo} cannot exceed it.)",
        ],
        distractors=distractors,
    )


# -- medium and hard: a parameter k -----------------------------------------------------------------------------


def _matrix_text(a, k_at=None):
    """A matrix (Fractions) as the stem spells it, with the letter k in entry `k_at` = (row, column)."""
    rows = []
    for i, row in enumerate(a):
        rows.append("[" + ", ".join("k" if (i, j) == k_at else str(int(v)) for j, v in enumerate(row)) + "]")
    return "[" + ", ".join(rows) + "]"


def _k_position(order, slot):
    """Where the entry in the triangular matrix's position `slot` lands after `permute(.., order)`."""
    inverse = {old: new for new, old in enumerate(order)}
    return inverse[slot[0]], inverse[slot[1]]


def _triangular_family(rng):
    """(lam, mu, a, b, c, m): integers with a c = b m, all of a, b, c, m non-zero."""
    for _ in attempts(500, "a, b, c with a c = b m"):
        lam, mu = rng.sample(_LAMBDAS, 2)
        m = mu - lam
        a, c = (rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(2))
        if (a * c) % m:
            continue
        b = a * c // m
        if b == 0 or abs(b) > 9:
            continue
        return lam, mu, a, b, c, m
    raise AssertionError  # pragma: no cover


_MEDIUM_STEMS = (
    "The matrix A = {m} depends on a real number k. Find the value of k for which A is diagonalisable.",
    "For exactly one value of k the matrix A = {m} is diagonalisable. Find it.",
    "Find the value of k for which A = {m} has three linearly independent eigenvectors.",
)


def _is_diagonalisable(a):
    """Rank test on an integer matrix with integer eigenvalues: every eigenvalue has geometric multiplicity equal to
    its algebraic multiplicity, found from char-poly roots by SymPy."""
    return sym(a).is_diagonalizable()


def _medium(rng):
    lam, mu, a, b, c, m = _triangular_family(rng)
    slot_name = rng.choice(["a", "b", "c"])
    # a c = b m: whichever of a, b, c carries k is the one this equation solves for.
    solved = {"a": Fraction(b * m, c), "b": Fraction(a * c, m), "c": Fraction(b * m, a)}[slot_name]
    k0 = int(solved)
    check(solved == k0 and k0 == {"a": a, "b": b, "c": c}[slot_name], "k is not the dropped entry")
    t = [[lam, a, b], [0, mu, c], [0, 0, lam]]
    slot = {"a": (0, 1), "b": (0, 2), "c": (1, 2)}[slot_name]
    order = rng.choice([[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [2, 0, 1]])
    shown = permute(t, order)
    stem_matrix = _matrix_text(shown, _k_position(order, slot))
    # SymPy: diagonalisable at k0, not at the neighbouring integers.
    for k, expected in ((k0, True), (k0 + 1, False), (k0 - 1, False)):
        entries = {"a": a, "b": b, "c": c}
        entries[slot_name] = k
        tk = permute([[lam, entries["a"], entries["b"]], [0, mu, entries["c"]], [0, 0, lam]], order)
        check(_is_diagonalisable(tk) == expected, f"diagonalisability at k={k} differs from the construction")
    wrong = []
    # Dropping the eigenvalue gap m from a c = b m.
    gap_dropped = {"a": Fraction(b, c), "b": Fraction(a * c), "c": Fraction(b, a)}[slot_name]
    wrong.append((gap_dropped, "diag-condition-gap-dropped"))
    # The gap attached to the wrong side: a m = b c.
    swapped = {"a": Fraction(b * c, m), "b": Fraction(a * m, c), "c": Fraction(a * m, b)}[slot_name]
    wrong.append((swapped, "diag-condition-wrong-ratio"))
    wrong.append((-k0, "sign-error-moving-term"))
    wrong.append((0, "parameter-zero-assumed"))
    candidates = [number_distractor(v, tag) for v, tag in wrong if Fraction(v).denominator == 1]
    distractors = pick_distractors(str(k0), candidates, 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _MEDIUM_STEMS, m=stem_matrix),
        answer_kind="number",
        answer=k0,
        answer_text=str(k0),
        answer_latex=str(k0),
        working=[
            f"The eigenvalues {lam}, {mu}, {lam} do not involve k, so the repeated eigenvalue {lam} needs a "
            "two-dimensional eigenspace.",
            f"That means A - {lam}I has rank 1: its two non-zero rows must be proportional, which gives k = {k0}.",
        ],
        distractors=distractors,
    )


_HARD_STEMS = (
    "The matrix A = {m} depends on a real number k. Find the value of k for which A is not diagonalisable.",
    "For exactly one value of k the matrix A = {m} is not diagonalisable. Find it.",
    "Find the value of k for which A = {m} cannot be diagonalised.",
)


def _diagonal_family(rng):
    """(lam, mu, a, b, c, bad): T = [[k, a, b], [0, lam, c], [0, 0, mu]], defective exactly at k = bad."""
    for _ in attempts(1000, "a matrix defective at exactly one of two values of k"):
        lam, mu = rng.sample(_LAMBDAS, 2)
        gap = lam - mu
        c = rng.choice([-3, -2, -1, 1, 2, 3])
        if rng.random() < 0.5:
            # Defective at k = lam (a != 0); diagonalisable at k = mu, which needs a c = b (lam - mu).
            a = rng.choice([-3, -2, -1, 1, 2, 3])
            if (a * c) % gap:
                continue
            b = a * c // gap
            bad = lam
        else:
            # Diagonalisable at k = lam (a = 0); defective at k = mu, which needs b != 0.
            a = 0
            b = rng.choice([-3, -2, -1, 1, 2, 3])
            bad = mu
        if abs(b) > 9:
            continue
        return lam, mu, a, b, c, bad
    raise AssertionError  # pragma: no cover


def _diagonal_parts(rng):
    lam, mu, a, b, c, bad = _diagonal_family(rng)
    order = rng.choice([[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [2, 0, 1]])
    t = [[bad, a, b], [0, lam, c], [0, 0, mu]]
    shown = permute([[0, a, b], [0, lam, c], [0, 0, mu]], order)
    position = _k_position(order, (0, 0))
    stem_matrix = _matrix_text(shown, position)
    other = mu if bad == lam else lam
    for k, expected in ((bad, False), (other, True), (bad + 7, True)):
        tk = permute([[k, a, b], [0, lam, c], [0, 0, mu]], order)
        check(_is_diagonalisable(tk) == expected, f"diagonalisability at k={k} differs from the construction")
    del t
    return lam, mu, a, b, c, bad, other, stem_matrix


def _hard(rng):
    lam, mu, a, b, c, bad, other, stem_matrix = _diagonal_parts(rng)
    candidates = [
        number_distractor(other, "diag-parameter-other-collision"),
        number_distractor(-bad, "sign-error-moving-term"),
        number_distractor(0, "parameter-zero-assumed"),
        number_distractor(lam + mu - bad, "diag-parameter-other-collision"),
        number_distractor(-other, "sign-error-moving-term"),
    ]
    distractors = pick_distractors(str(bad), candidates, 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _HARD_STEMS, m=stem_matrix),
        answer_kind="number",
        answer=bad,
        answer_text=str(bad),
        answer_latex=str(bad),
        working=[
            f"The eigenvalues are k, {lam} and {mu}, so A can fail to be diagonalisable only where two of them "
            f"coincide: k = {lam} or k = {mu}.",
            f"At k = {other} the eigenspace of the repeated eigenvalue is two-dimensional, so A is diagonalisable.",
            f"At k = {bad} it is only one-dimensional, so A is not diagonalisable.",
        ],
        distractors=distractors,
    )


@generator(
    "s2b2_diagonalisability_test",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@tidied
def s2b2_diagonalisability_test(rng, tier="easy"):
    """Algebraic against geometric multiplicity for a 3×3 matrix; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_s2b2_diagonalisability_test", level=19, presets={"exam": dict(tier="exam")})
@tidied
def exd_x_s2b2_diagonalisability_test(rng, tier="exam"):
    """Exam chain: the repeated eigenvalue, the third eigenvalue from the trace, rank(A - λI) and the geometric
    multiplicity of a dense 3×3 matrix with a double eigenvalue."""
    check(tier == "exam", f"unknown tier {tier!r}")
    lam, mu = rng.sample([v for v in _LAMBDAS if v != 0], 1) + rng.sample(_LAMBDAS, 1)
    for _ in attempts(100, "distinct eigenvalues"):
        if mu != lam:
            break
        mu = rng.choice(_LAMBDAS)
    defective = rng.random() < 0.55
    blocks = [(lam, 2), (mu, 1)] if defective else [(lam, 1), (lam, 1), (mu, 1)]
    rng.shuffle(blocks)
    a, _p = conjugate(rng, jordan_matrix(blocks))
    trace = int(sum(a[i][i] for i in range(3)))
    check(trace == 2 * lam + mu, "trace differs from the sum of the eigenvalues")
    m = sym(a)
    check(m.eigenvals() == {sympy.Integer(lam): 2, sympy.Integer(mu): 1}, "SymPy's eigenvalues differ")
    rk = int((m - lam * sympy.eye(3)).rank())
    check(rk == (2 if defective else 1), "SymPy's rank differs")
    geo = 3 - rk
    diag_entries = [int(a[i][i]) for i in range(3)]

    a_dist = pick_distractors(
        str(lam),
        [
            number_distractor(mu, "eigval-other-root"),
            number_distractor(-lam, "sign-flip-both-roots"),
            number_distractor(diag_entries[0], "eigval-diagonal-entry"),
            number_distractor(diag_entries[1], "eigval-diagonal-entry"),
            number_distractor(diag_entries[2], "eigval-diagonal-entry"),
            number_distractor(trace, "eigval-trace-as-eigenvalue"),
            number_distractor(-mu, "sign-flip-both-roots"),
        ],
        3,
    )
    b_dist = pick_distractors(
        str(mu),
        [
            number_distractor(trace - lam, "eigval-trace-single-count"),
            number_distractor(trace + 2 * lam, "eigval-trace-difference-sign"),
            number_distractor(lam, "eigval-repeated-given"),
            number_distractor(trace, "eigval-trace-as-eigenvalue"),
            number_distractor(-mu, "sign-flip-both-roots"),
            number_distractor(trace + lam, "eigval-trace-difference-sign"),
        ],
        3,
    )
    c_dist = pick_distractors(
        str(rk),
        [
            number_distractor(geo, "rank-nullity-confused"),
            number_distractor(3, "rank-counts-nonzero-rows"),
            number_distractor(rk + 1 if rk < 3 else rk - 1, "rank-off-by-one"),
            number_distractor(0, "rank-off-by-one"),
        ],
        3,
    )
    d_dist = pick_distractors(
        str(geo),
        [
            number_distractor(rk, "rank-nullity-confused"),
            number_distractor(2, "geo-equals-alg"),
            number_distractor(3, "nullity-counts-columns"),
            number_distractor(1, "geo-always-one"),
        ],
        3,
    )
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")
    return dict(
        text=phrase(
            rng,
            (
                "Let A = {m}. The characteristic polynomial of A has a repeated root.",
                "The matrix A = {m} has a repeated eigenvalue.",
                "Consider A = {m}, whose characteristic polynomial has exactly one repeated root.",
            ),
            m=mat_str(a),
        ),
        parts=[
            dict(
                prompt="Find the repeated eigenvalue λ of A.",
                answer_kind="number",
                answer=lam,
                answer_text=str(lam),
                marks=3,
                working=[f"det(A - λI) = -(λ - {lam})²(λ - {mu}), so the repeated eigenvalue is λ = {lam}."],
                distractors=a_dist,
                sensible_range=(-20, 20),
            ),
            dict(
                prompt="Use the trace of A to find the third eigenvalue.",
                answer_kind="number",
                answer=mu,
                answer_text=str(mu),
                marks=1,
                follow_through=f"{trace} - 2*part_a",
                working=[f"The eigenvalues add up to the trace {trace}, so the third is {trace} - 2({lam}) = {mu}."],
                distractors=b_dist,
            ),
            dict(
                prompt="Find the rank of A - λI for the repeated eigenvalue λ.",
                answer_kind="number",
                answer=rk,
                answer_text=str(rk),
                marks=2,
                working=[f"Row-reducing A - {lam}I leaves {rk} non-zero row{'s' if rk != 1 else ''}."],
                distractors=c_dist,
            ),
            dict(
                prompt="Find the geometric multiplicity of the repeated eigenvalue.",
                answer_kind="number",
                answer=geo,
                answer_text=str(geo),
                marks=1,
                follow_through="3 - part_c",
                working=[f"The eigenspace is the null space of A - {lam}I, of dimension 3 - {rk} = {geo}."],
                distractors=d_dist,
                sensible_range=(0, 4),
            ),
        ],
    )
