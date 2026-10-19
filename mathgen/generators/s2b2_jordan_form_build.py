"""Generates `s2b2_jordan_form` questions (easy, medium, hard) and the exam chain `exd_x_s2b2_jordan_form`.

``uc2-jordan-form`` (``linear-algebra``, answer ``matrix``): the Jordan normal form J of an integer matrix with a
repeated eigenvalue. J is only unique up to the order of its blocks, so every question states the normal form: 1s
directly above the diagonal, blocks ordered by increasing eigenvalue and, within one eigenvalue, the larger block
first. Then J is a single matrix and is graded exactly.

* easy: a 2×2 matrix with a repeated eigenvalue that is not a multiple of the identity, or a 3×3 matrix with
  eigenvalues λ, λ, μ (defective or not).
* medium: a 3×3 matrix with one triple eigenvalue, whose Jordan form has one block of size 3 or blocks of size 2 and 1.
* hard: the Jordan form of A² for a 3×3 matrix A = P J P⁻¹ with eigenvalues λ, λ, μ and one block of size 2 at λ.
  Squaring a block J(λ; 2) gives the block J(λ²; 2) when λ ≠ 0, but when λ = 0 the block collapses (J(0; 2)² = 0) and
  A² is diagonalisable.
* exam: a 3×3 matrix with one triple eigenvalue: (a) λ, (b) rank(A - λI), (c) the size of the largest Jordan block,
  rank + 1, (d) J.

Every matrix is A = P J P⁻¹ for a unimodular P. The block structure is re-derived from ranks of powers of A - λI
(``nullity`` of the powers, by SymPy) rather than from J, and A² is checked with SymPy for the hard tier.
"""

from __future__ import annotations

import sympy

from ..core import attempts, check, generator, phrase, pick_distractors
from ._s2b2_kit import (
    tidied,
    conjugate,
    jordan_matrix,
    mat_str,
    matrix_distractors,
    number_distractor,
    rows_value,
    sym,
)

_LAMBDAS = list(range(-4, 6))
_FORM = (
    "Give J with the 1s directly above the diagonal, the blocks in order of increasing eigenvalue and, for equal "
    "eigenvalues, the larger block first."
)

_STEMS = (
    "Find the Jordan normal form J of A = {m}. " + _FORM,
    "The matrix A = {m} is similar to a Jordan matrix J. Find J. " + _FORM,
    "Write down the Jordan matrix J that A = {m} is similar to. " + _FORM,
)


def _blocks_for(structure):
    """Jordan blocks in the stated normal-form order: by eigenvalue ascending, then larger first."""
    return sorted(structure, key=lambda b: (b[0], -b[1]))


def _block_sizes_by_sympy(a, lam):
    """Sizes of the Jordan blocks of eigenvalue `lam`, largest first, from nullities of powers of A - λI (SymPy)."""
    n = a.shape[0]
    shifted = a - lam * sympy.eye(n)
    nullities = [0]
    power = sympy.eye(n)
    for _ in range(n):
        power = power * shifted
        nullities.append(n - power.rank())
    sizes = []
    for k in range(n, 0, -1):
        # The number of blocks of size at least k is nullity(k) - nullity(k-1).
        at_least = nullities[k] - nullities[k - 1]
        at_least_next = (nullities[k + 1] - nullities[k]) if k < n else 0
        sizes += [k] * (at_least - at_least_next)
    return sizes


def _verify(a, blocks):
    """SymPy: the block sizes for each eigenvalue, from ranks of (A - λI)^k, match the claimed Jordan blocks."""
    m = sym(a)
    for lam in {b[0] for b in blocks}:
        claimed = sorted((size for value, size in blocks if value == lam), reverse=True)
        check(_block_sizes_by_sympy(m, lam) == claimed, f"SymPy's block sizes for λ = {lam} differ")
    check(
        sorted(m.eigenvals().items())
        == sorted((sympy.Integer(l), sum(s for v, s in blocks if v == l)) for l in {b[0] for b in blocks}),
        "eigenvalues",
    )


def _candidates(blocks, answer):
    """Tagged wrong Jordan matrices for the block list `blocks` (in normal-form order)."""
    out = []
    n = sum(size for _l, size in blocks)
    lam_list = [lam for lam, size in blocks for _ in range(size)]
    # 1. The diagonal matrix of eigenvalues: the matrix assumed diagonalisable.
    from ._s2b2_kit import diagonal

    out.append((diagonal(sorted(lam_list)), "jordan-diagonal-assumed"))
    # 2. The 1s below the diagonal.
    lower = [row[:] for row in jordan_matrix(blocks)]
    lower = [[lower[j][i] for j in range(n)] for i in range(n)]
    out.append((lower, "jordan-one-below-diagonal"))
    # 3. Blocks in the wrong order: reversed, or larger block last.
    out.append((jordan_matrix(list(reversed(blocks))), "jordan-blocks-misordered"))
    out.append((jordan_matrix(sorted(blocks, key=lambda b: (b[0], b[1]))), "jordan-blocks-misordered"))
    # 4. Block size read from the algebraic multiplicity (one block per eigenvalue) or split into 1×1 blocks.
    merged = {}
    for lam, size in blocks:
        merged[lam] = merged.get(lam, 0) + size
    out.append((jordan_matrix(sorted(merged.items())), "jordan-block-size-from-multiplicity"))
    # 5. One fewer or one more 1 in the largest block.
    biggest = max(range(len(blocks)), key=lambda i: blocks[i][1])
    split = list(blocks)
    lam, size = split[biggest]
    if size > 1:
        split[biggest : biggest + 1] = [(lam, size - 1), (lam, 1)]
        out.append((jordan_matrix(_blocks_for(split)), "jordan-block-size-off-by-one"))
    # 6. The eigenvalues with the wrong sign.
    out.append((jordan_matrix([(-lam, size) for lam, size in blocks]), "sign-flip-both-roots"))
    return [(mat, tag) for mat, tag in out if mat is not None]


def _answer(blocks):
    jm = jordan_matrix(blocks)
    return jm, rows_value(jm)


def _question(rng, stems, a, blocks, working_extra):
    jm, answer_rows = _answer(blocks)
    text = mat_str(jm)
    distractors = pick_distractors(text, matrix_distractors(_candidates(blocks, answer_rows), answer_rows), 4)
    check(len(distractors) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, stems, m=mat_str(a)),
        answer_kind="matrix",
        answer=answer_rows,
        answer_text=text,
        answer_latex="plain:" + text,
        working=working_extra + [f"So J = {text}."],
        distractors=distractors,
    )


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy(rng):
    if rng.random() < 0.4:
        lam = rng.choice(_LAMBDAS)
        blocks = [(lam, 2)]
        a, _p = conjugate(rng, jordan_matrix(blocks), cap=12)
        working = [
            f"det(A - λI) = (λ - {lam})², and A - {lam}I is not the zero matrix, so the eigenspace has dimension 1.",
            "One eigenvector means one Jordan block, of size 2.",
        ]
    else:
        lam, mu = rng.sample(_LAMBDAS, 2)
        defective = rng.random() < 0.65
        blocks = _blocks_for([(lam, 2), (mu, 1)] if defective else [(lam, 1), (lam, 1), (mu, 1)])
        a, _p = conjugate(rng, jordan_matrix(blocks))
        working = [
            f"The eigenvalues are {lam} (repeated) and {mu}.",
            f"A - {lam}I has rank {2 if defective else 1}, so the eigenspace of {lam} has dimension {1 if defective else 2}"
            + (": one block of size 2." if defective else ": two blocks of size 1."),
        ]
    _verify(a, blocks)
    return _question(rng, _STEMS, a, blocks, working)


# -- medium -------------------------------------------------------------------------------------------------------


def _medium(rng):
    lam = rng.choice(_LAMBDAS)
    big = rng.random() < 0.5
    blocks = [(lam, 3)] if big else [(lam, 2), (lam, 1)]
    a, _p = conjugate(rng, jordan_matrix(blocks), cap=14)
    _verify(a, blocks)
    rank = 2 if big else 1
    working = [
        f"The only eigenvalue is {lam}, with algebraic multiplicity 3.",
        f"A - {lam}I has rank {rank}, so the eigenspace has dimension {3 - rank}: there {'is 1 Jordan block' if big else 'are 2 Jordan blocks'}.",
        "A single block has size 3."
        if big
        else "Two blocks of total size 3 are a block of size 2 and a block of size 1.",
    ]
    return _question(rng, _STEMS, a, blocks, working)


# -- hard ---------------------------------------------------------------------------------------------------------

_HARD_STEMS = (
    "The matrix A = {m} has a repeated eigenvalue. Find the Jordan normal form of A². " + _FORM,
    "Find the Jordan normal form J of the matrix A² for A = {m}. " + _FORM,
    "For A = {m}, which has a repeated eigenvalue, find the Jordan matrix that A² is similar to. " + _FORM,
)


def _hard(rng):
    for _ in attempts(500, "eigenvalues whose squares are distinct"):
        lam = 0 if rng.random() < 0.4 else rng.choice([v for v in _LAMBDAS if v])
        mu = rng.choice(_LAMBDAS)
        if mu != lam and mu * mu != lam * lam:
            break
    a, _p = conjugate(rng, jordan_matrix(_blocks_for([(lam, 2), (mu, 1)])))
    _verify(a, [(lam, 2), (mu, 1)])
    a_sq = sym(a) ** 2
    if lam == 0:
        blocks = _blocks_for([(0, 1), (0, 1), (mu * mu, 1)])
        working = [
            f"A has eigenvalue 0 with a single Jordan block of size 2 and eigenvalue {mu}.",
            "A block J(0; 2) has square zero, so A² has a two-dimensional kernel and the eigenvalue 0 splits into two blocks of size 1.",
        ]
    else:
        blocks = _blocks_for([(lam * lam, 2), (mu * mu, 1)])
        working = [
            f"A has a block of size 2 for the eigenvalue {lam} and a block of size 1 for {mu}.",
            f"Squaring a block with eigenvalue {lam} ≠ 0 keeps its size, so A² has a block of size 2 for {lam * lam}.",
        ]
    # SymPy: the block structure of A² itself, from ranks of powers.
    for value in {b[0] for b in blocks}:
        claimed = sorted((size for v, size in blocks if v == value), reverse=True)
        check(_block_sizes_by_sympy(a_sq, value) == claimed, f"SymPy's block sizes of A² for {value} differ")
    jm, answer_rows = _answer(blocks)
    text = mat_str(jm)
    # The classic error: the square of the Jordan matrix J itself, [[λ², 2λ], [0, λ²]], which is not in normal form.
    j_sq = sym(jordan_matrix(_blocks_for([(lam, 2), (mu, 1)]))) ** 2
    squared_j = [[int(j_sq[i, j]) for j in range(3)] for i in range(3)]
    naive = jordan_matrix(_blocks_for([(lam * lam, 2), (mu * mu, 1)]))
    candidates = (
        [(squared_j, "jordan-block-squared-not-renormalised"), (naive, "jordan-block-size-kept-for-zero")]
        if lam == 0
        else [(squared_j, "jordan-block-squared-not-renormalised")]
    )
    candidates += _candidates(blocks, answer_rows)
    distractors = pick_distractors(text, matrix_distractors(candidates, answer_rows), 4)
    check(len(distractors) >= 3, "too few distractors")
    return dict(
        text=phrase(rng, _HARD_STEMS, m=mat_str(a)),
        answer_kind="matrix",
        answer=answer_rows,
        answer_text=text,
        answer_latex="plain:" + text,
        working=working + [f"So the Jordan form of A² is {text}."],
        distractors=distractors,
    )


@generator(
    "s2b2_jordan_form",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@tidied
def s2b2_jordan_form(rng, tier="easy"):
    """Jordan normal form of an integer matrix with a repeated eigenvalue; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_s2b2_jordan_form", level=19, presets={"exam": dict(tier="exam")})
@tidied
def exd_x_s2b2_jordan_form(rng, tier="exam"):
    """Exam chain: for a 3×3 matrix with one triple eigenvalue, λ, rank(A - λI), the largest block size and J."""
    check(tier == "exam", f"unknown tier {tier!r}")
    lam = rng.choice([v for v in _LAMBDAS if v])
    big = rng.random() < 0.5
    blocks = [(lam, 3)] if big else [(lam, 2), (lam, 1)]
    a, _p = conjugate(rng, jordan_matrix(blocks), cap=14)
    _verify(a, blocks)
    trace = int(sum(a[i][i] for i in range(3)))
    check(trace == 3 * lam, "trace differs from 3λ")
    rk = 2 if big else 1
    check(int((sym(a) - lam * sympy.eye(3)).rank()) == rk, "SymPy's rank differs")
    size = rk + 1
    jm, answer_rows = _answer(blocks)
    j_text = mat_str(jm)
    diag_entries = [int(a[i][i]) for i in range(3)]
    a_dist = pick_distractors(
        str(lam),
        [
            number_distractor(trace, "eigval-trace-as-eigenvalue"),
            number_distractor(-lam, "sign-flip-both-roots"),
            number_distractor(diag_entries[0], "eigval-diagonal-entry"),
            number_distractor(diag_entries[1], "eigval-diagonal-entry"),
            number_distractor(trace - lam, "eigval-trace-single-count"),
        ],
        3,
    )
    b_dist = pick_distractors(
        str(rk),
        [
            number_distractor(3 - rk, "rank-nullity-confused"),
            number_distractor(3, "rank-counts-nonzero-rows"),
            number_distractor(0, "rank-off-by-one"),
            number_distractor(rk + 1 if rk == 1 else rk - 1, "rank-off-by-one"),
        ],
        3,
    )
    c_dist = pick_distractors(
        str(size),
        [
            number_distractor(rk, "jordan-block-size-rank-confused"),
            number_distractor(3 - rk, "jordan-block-count-as-size"),
            number_distractor(3, "jordan-block-size-from-multiplicity"),
            number_distractor(1, "jordan-diagonal-assumed"),
        ],
        3,
    )
    d_dist = pick_distractors(j_text, matrix_distractors(_candidates(blocks, answer_rows), answer_rows), 3)
    check(all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")
    return dict(
        text=phrase(
            rng,
            (
                "The matrix A = {m} has a single eigenvalue, of algebraic multiplicity 3.",
                "Let A = {m}, a matrix whose only eigenvalue has algebraic multiplicity 3.",
                "A = {m} has exactly one eigenvalue, repeated three times.",
            ),
            m=mat_str(a),
        ),
        parts=[
            dict(
                prompt="Find the eigenvalue λ of A.",
                answer_kind="number",
                answer=lam,
                answer_text=str(lam),
                marks=2,
                working=[
                    f"The eigenvalues add up to the trace {trace}, and all three are equal, so λ = {trace}/3 = {lam}."
                ],
                distractors=a_dist,
                sensible_range=(-20, 20),
            ),
            dict(
                prompt="Find the rank of A - λI.",
                answer_kind="number",
                answer=rk,
                answer_text=str(rk),
                marks=2,
                working=[f"Row-reducing A - {lam}I leaves {rk} non-zero row{'s' if rk != 1 else ''}."],
                distractors=b_dist,
            ),
            dict(
                prompt="Hence find the size of the largest Jordan block of A.",
                answer_kind="number",
                answer=size,
                answer_text=str(size),
                marks=1,
                follow_through="part_b + 1",
                working=[
                    f"There are 3 - {rk} = {3 - rk} Jordan block{'s' if 3 - rk != 1 else ''} of total size 3, so the largest has size {size}."
                ],
                distractors=c_dist,
                sensible_range=(0, 4),
            ),
            dict(
                prompt="Write down the Jordan normal form J of A, with the 1s directly above the diagonal and the larger "
                "block first.",
                answer_kind="matrix",
                answer=answer_rows,
                answer_text=j_text,
                answer_latex="plain:" + j_text,
                marks=2,
                working=[f"The block sizes are {', '.join(str(s) for _l, s in blocks)}, so J = {j_text}."],
                distractors=d_dist,
            ),
        ],
    )
