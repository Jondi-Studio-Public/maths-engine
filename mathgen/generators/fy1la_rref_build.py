"""Generates `fy1la_rref` questions (easy, medium, hard) and the exam chain `exd_x_fy1la_rref`.

``ucv1-row-reduction-rref`` (``linear-algebra``, answer ``matrix``): reduce an integer matrix to reduced row echelon
form (RREF) and read the rank, the free variables and a solution off it.

* easy: the augmented matrix of three equations in three unknowns with one solution; the RREF is [I | c].
* medium: a 3×4 matrix of rank 2 (one row is a combination of the others), so the RREF has a zero row and two free
  columns after the pivots in columns 1 and 2.
* hard: rank 2 with the pivots NOT in the first two columns (a zero first column, or a pivot that skips a column), so the
  reduction needs a row swap and the pivot positions have to be found rather than assumed.
* exam (``exd_x_fy1la_rref``): a homogeneous system with a 3×4 coefficient matrix of rank 2. (a) the rank, (b) the RREF,
  (c) the number of free variables (follows through from (a)), (d) the solution with stated values of the free variables.

Every matrix is built as `E R` from a target RREF `R` and a unimodular integer matrix `E`, so the RREF of the stated
matrix is `R`. It is re-derived by Gauss-Jordan elimination in exact fractions and by SymPy's `rref()`.
"""

from __future__ import annotations

from ..core import VerificationError, attempts, check, generator, phrase
from ._fy1la_kit import (
    apply,
    echelon,
    fm,
    mat_str,
    matrix_distractors,
    matrix_from_rref,
    matrix_part,
    num_distractors,
    number_part,
    rank,
    rows_value,
    rref,
    sym,
    vec_str,
    vector_distractors,
)

_NOTE = "Give the matrix with whole-number entries."


def _targets(rng, tier):
    """A target RREF `R` (rows of ints) for the tier."""

    def nz(lo=-4, hi=4):
        return rng.choice([v for v in range(lo, hi + 1) if v != 0])

    if tier == "easy":
        return [[1, 0, 0, nz()], [0, 1, 0, rng.randint(-4, 4)], [0, 0, 1, nz()]]
    if tier == "medium":
        return [[1, 0, nz(), rng.randint(-4, 4)], [0, 1, rng.randint(-4, 4), nz()], [0, 0, 0, 0]]
    shape = rng.choice(["skip", "skip", "zero-first", "late"])
    if shape == "skip":
        return [[1, nz(), 0, rng.randint(-4, 4)], [0, 0, 1, nz()], [0, 0, 0, 0]]
    if shape == "zero-first":
        return [[0, 1, 0, nz()], [0, 0, 1, rng.randint(-4, 4)], [0, 0, 0, 0]]
    return [[1, nz(), nz(), 0], [0, 0, 0, 1], [0, 0, 0, 0]]


def _verify(a, target):
    """The RREF by SymPy, and equality with the target the matrix was built from."""
    reduced, pivots = rref(a)
    check(reduced == fm(target), "the reduction does not reproduce the target")
    ref, ref_pivots = sym(a).rref()
    check(list(ref_pivots) == pivots, "SymPy's pivot columns differ")
    check(ref == sym(target), "SymPy's RREF differs")
    return reduced, pivots


def _rref_candidates(a, reduced, pivots):
    """Wrong matrices a learner reaches, in priority order: stopping at echelon form, adding where they should
    subtract, scaling the rows and stopping, and one arithmetic slip."""
    rows, cols = len(a), len(a[0])
    flipped = [row[:] for row in reduced]
    for i, pc in enumerate(pivots):
        for c in range(cols):
            if c not in pivots and c > pc:
                flipped[i][c] = -flipped[i][c]
    scaled = []
    for row in a:
        lead = next((v for v in row if v != 0), None)
        scaled.append([v / lead for v in row] if lead is not None else row[:])
    slip = [row[:] for row in reduced]
    spots = [(i, c) for i in range(len(pivots)) for c in range(cols) if c not in pivots and slip[i][c] != 0]
    if spots:
        i, c = spots[-1]
        slip[i][c] += 1 if slip[i][c] <= 0 else -1
    slip_other = [row[:] for row in reduced]
    if spots:
        i, c = spots[0]
        slip_other[i][c] += -1 if slip_other[i][c] <= 0 else 1
    check(rows == 3, "unexpected shape")
    return [
        (echelon(a), "rref-stopped-at-echelon"),
        (flipped, "rref-adds-instead-of-subtracts"),
        (scaled, "rref-rows-scaled-only"),
        (slip, "row-operation-arithmetic-slip"),
        (slip_other, "row-operation-arithmetic-slip"),
    ]


_STEMS = {
    "easy": (
        "The augmented matrix of a system of three equations in x, y and z is {m}. Reduce it to reduced row echelon form. {n}",
        "Reduce the augmented matrix {m} of a system of three equations in x, y and z to reduced row echelon form. {n}",
        "Use row operations to bring the augmented matrix {m} of a system in x, y and z to reduced row echelon form. {n}",
    ),
    "medium": (
        "Find the reduced row echelon form of M = {m}. {n}",
        "Reduce M = {m} to reduced row echelon form. {n}",
        "Row-reduce the matrix M = {m} completely, to reduced row echelon form. {n}",
    ),
    "hard": (
        "Find the reduced row echelon form of M = {m}. {n}",
        "Row-reduce M = {m} to reduced row echelon form. {n}",
        "Use row operations to find the reduced row echelon form of M = {m}. {n}",
    ),
}


def _single(rng, tier):
    for _ in attempts(200, "a reduction question with distractors"):
        target = _targets(rng, tier)
        a = matrix_from_rref(rng, target)
        if a is None:
            continue
        reduced, pivots = _verify(a, target)
        answer_rows = [[int(v) for v in row] for row in reduced]
        dist = matrix_distractors(_rref_candidates(a, reduced, pivots), [[v for v in r] for r in reduced])
        if dist is None:
            continue
        r = len(pivots)
        text = mat_str(reduced)
        working = [
            f"Row-reduce {mat_str(a)}: swap rows if a pivot is zero, scale each pivot row to a leading 1, then clear "
            "the entries above and below each pivot.",
            f"The pivots are in column{'s' if r > 1 else ''} {' and '.join(str(p + 1) for p in pivots)}, giving {text}.",
        ]
        return dict(
            text=phrase(rng, _STEMS[tier], m=mat_str(a), n=_NOTE),
            answer_kind="matrix",
            answer=rows_value(answer_rows),
            answer_text=text,
            answer_latex="plain:" + text,
            working=working,
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


@generator(
    "fy1la_rref",
    level=16,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1la_rref(rng, tier="easy"):
    """Reduce an integer matrix to reduced row echelon form; the tiers are described in the module docstring."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return _single(rng, tier)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_fy1la_rref", level=16, presets={"exam": dict(tier="exam")})
def exd_x_fy1la_rref(rng, tier="exam"):
    """Exam chain: rank, RREF, number of free variables and a solution of a homogeneous 3×4 system of rank 2."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "an exam reduction chain"):
        pivots_wanted = rng.choice([(0, 1), (0, 2)])
        nz = [v for v in range(-4, 5) if v != 0]
        if pivots_wanted == (0, 1):
            target = [[1, 0, rng.choice(nz), rng.choice(nz)], [0, 1, rng.choice(nz), rng.choice(nz)], [0, 0, 0, 0]]
        else:
            target = [[1, rng.choice(nz), 0, rng.choice(nz)], [0, 0, 1, rng.choice(nz)], [0, 0, 0, 0]]
        a = matrix_from_rref(rng, target)
        if a is None:
            continue
        reduced, pivots = _verify(a, target)
        check(rank(a) == 2 and tuple(pivots) == pivots_wanted, "unexpected rank or pivots")
        free = [c for c in range(4) if c not in pivots]
        f1, f2 = free
        # (d) x_f1 = 1, x_f2 = 0, pivot variables read from the reduced rows.
        sol = [0, 0, 0, 0]
        sol[f1] = 1
        for i, pc in enumerate(pivots):
            sol[pc] = -reduced[i][f1]
        check(all(v == 0 for v in apply(a, sol)), "the solution does not satisfy Ax = 0")
        other = [0, 0, 0, 0]
        other[f2] = 1
        for i, pc in enumerate(pivots):
            other[pc] = -reduced[i][f2]

        dist_a = num_distractors(
            2, [(3, "rank-counts-all-rows"), (4, "rank-counts-all-columns"), (1, "rank-misses-a-pivot")]
        )
        dist_b = matrix_distractors(_rref_candidates(a, reduced, pivots), reduced)
        dist_c = num_distractors(
            2,
            [(1, "free-variables-from-rows"), (3, "free-variables-miscounted"), (4, "free-variables-all-columns")],
        )
        wrong_pivots = [0, 0, 0, 0]
        wrong_pivots[f1] = 1
        mixed = [0, 0, 0, 0]
        mixed[f1] = 1
        for i, pc in enumerate(pivots):
            mixed[pc] = -reduced[i][f2]
        dist_d = vector_distractors(
            [
                (
                    [1 if c == f1 else (reduced[pivots.index(c)][f1] if c in pivots else 0) for c in range(4)],
                    "rref-free-column-read-unsigned",
                ),
                (other, "free-variable-values-swapped"),
                (wrong_pivots, "pivot-variables-left-zero"),
                (mixed, "free-column-mixed-up"),
            ],
            sol,
        )
        if None in (dist_a, dist_b, dist_c, dist_d):
            continue
        names = [f"x{c + 1}" for c in range(4)]
        reduced_text = mat_str(reduced)
        return dict(
            text=phrase(
                rng,
                (
                    "A homogeneous system Ax = 0 in the unknowns x1, x2, x3 and x4 has coefficient matrix A = {a}.",
                    "Consider the homogeneous system Ax = 0 in the unknowns x1, x2, x3 and x4, where A = {a}.",
                    "The coefficient matrix of a homogeneous system Ax = 0 in x1, x2, x3 and x4 is A = {a}.",
                ),
                a=mat_str(a),
            ),
            parts=[
                number_part(
                    "Find the rank of A.",
                    2,
                    dist_a,
                    marks=1,
                    working=[f"Row-reducing A leaves two non-zero rows ({reduced_text}), so the rank is 2."],
                    sensible_range=(0, 4),
                ),
                matrix_part(
                    "Reduce A to reduced row echelon form. " + _NOTE,
                    reduced,
                    dist_b,
                    marks=3,
                    working=[f"Pivots in columns {pivots[0] + 1} and {pivots[1] + 1}: {reduced_text}."],
                ),
                number_part(
                    "How many free variables does the system have?",
                    2,
                    dist_c,
                    marks=1,
                    follow_through="4 - part_a",
                    working=["There are 4 unknowns and 2 pivots, so 4 - 2 = 2 free variables."],
                    sensible_range=(0, 5),
                ),
                matrix_part(
                    f"Find the solution of the system with {names[f1]} = 1 and {names[f2]} = 0, as a column (x1, x2, x3, x4).",
                    sol,
                    dist_d,
                    marks=2,
                    vector=True,
                    working=[
                        f"The free variables are {names[f1]} and {names[f2]}; each pivot variable is minus the "
                        f"matching entry of the reduced row times {names[f1]}, giving {vec_str(sol)}."
                    ],
                ),
            ],
        )
    raise VerificationError("unreachable")  # pragma: no cover
