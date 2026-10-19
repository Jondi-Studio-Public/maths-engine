"""Generates `fy1la_general_solution` questions (easy, medium, hard) and the exam chain `exd_x_fy1la_general_solution`.

``ucv1-general-solution-parametric`` (``linear-algebra``, answer ``matrix``): the solutions of a consistent linear
system with free variables are `x = p + s u + t v`. The answers asked for are vectors, which grade robustly, and one
expression in the parameters.

* easy: two equations in x, y and z (one free variable). The solution in which z has a stated value.
* medium: three equations in x, y and z, one of which is redundant, so the solutions are a line `p + t d`. A direction
  vector d of the line (any non-zero multiple is accepted).
* hard: three equations in x1 to x4 (two free variables, one redundant equation). The solution in which the two free
  variables have stated values.
* exam (``exd_x_fy1la_general_solution``): a consistent 3×4 system of rank 2. (a) the rank, (b) the number of
  parameters in the general solution (follows through from (a)), (c) the particular solution in which the free variables
  are zero, (d) one pivot variable as an expression in the parameters s and t.

Each system is built as `E T` from a target augmented RREF `T` and a unimodular `E`, so its general solution is known.
The solution is re-derived by exact elimination and by SymPy, and substituted back into every equation.
"""

from __future__ import annotations

import sympy

from ..core import VerificationError, attempts, check, generator, join_terms, phrase
from ._fy1la_kit import (
    apply,
    expression_distractors,
    expression_part,
    fm,
    max_abs,
    mmul,
    num_distractors,
    number_part,
    rank,
    rref,
    sym,
    system_text,
    unimodular,
    vec_str,
    vector_distractors,
    matrix_part,
    column_value,
)

_TXT_NOTE = "Give the solution as a column (x, y, z)."


def _nz(rng, lo=-4, hi=4):
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def _system_from(rng, target, rows, cap=9):
    """(A, b) with `[A | b] = E target` for a unimodular E; entries within `cap`, no equation repeated or all zero."""
    base = fm(target)
    for _ in attempts(3000, "a system with the target reduced form"):
        aug = mmul(unimodular(rng, rows, 2), base)
        if max_abs(aug) > cap or max_abs(aug) < 2:
            continue
        if any(all(v == 0 for v in row[:-1]) for row in aug):
            continue
        keys = {tuple(row) for row in aug}
        if len(keys) < rows or any(tuple(-v for v in row) in keys for row in aug):
            continue
        if any(rank([r1[:-1], r2[:-1]]) < 2 for i, r1 in enumerate(aug) for r2 in aug[i + 1 :]):
            continue
        return [row[:-1] for row in aug], [row[-1] for row in aug]
    raise VerificationError("unreachable")  # pragma: no cover - attempts() raises first


def _solution(target, pivots, free_values):
    """The solution of the system whose augmented RREF is `target`, with the free columns set to `free_values`."""
    n = len(target[0]) - 1
    free = [c for c in range(n) if c not in pivots]
    x = [0] * n
    for c, v in zip(free, free_values):
        x[c] = v
    for i, pc in enumerate(pivots):
        x[pc] = target[i][n] - sum(target[i][c] * x[c] for c in free)
    return x


def _verify(a, b, target, pivots):
    """Reduce [A | b] by hand and by SymPy, and substitute a solution back."""
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    reduced, found = rref(aug)
    check(reduced == fm(target) and found == list(pivots), "the reduction does not reproduce the target")
    ref, ref_pivots = sym(aug).rref()
    check(ref == sym(target) and list(ref_pivots) == list(pivots), "SymPy's reduction differs")
    check(rank(a) == len(pivots) and rank(aug) == len(pivots), "unexpected rank")


def _check_solution(a, b, x):
    check(apply(a, x) == [v for v in b], "the solution does not satisfy the system")


_NAMES3 = ("x", "y", "z")
_NAMES4 = ("x1", "x2", "x3", "x4")


def _easy(rng):
    for _ in attempts(300, "a one-free-variable question"):
        a_, d_ = _nz(rng), _nz(rng)
        target = [[1, 0, a_, rng.randint(-4, 4)], [0, 1, d_, rng.randint(-4, 4)]]
        a, b = _system_from(rng, target, 2)
        _verify(a, b, target, (0, 1))
        t0 = rng.choice([-2, -1, 1, 2, 3])
        sol = _solution(target, (0, 1), [t0])
        _check_solution(a, b, sol)
        sym_x = sympy.Matrix(a).gauss_jordan_solve(sympy.Matrix(b))[0]
        free = sym_x.free_symbols.pop()
        check([v for v in sym_x.subs(free, t0)] == [sympy.Integer(int(v)) for v in sol], "SymPy's solution differs")
        c, e = target[0][3], target[1][3]
        answer = [int(v) for v in sol]
        dist = vector_distractors(
            [
                ([c + a_ * t0, e + d_ * t0, t0], "rref-free-column-read-unsigned"),
                ([c, e, t0], "free-variable-not-substituted"),
                ([answer[0], answer[1], 0], "free-variable-value-dropped"),
                ([c - a_ * t0, e - d_ * t0, -t0], "free-variable-sign-slip"),
            ],
            answer,
        )
        if dist is None:
            continue
        stems = (
            "The system {s} has infinitely many solutions. Find the solution in which z = {t}. " + _TXT_NOTE,
            "Solve {s}, given that z = {t}. " + _TXT_NOTE,
        )
        text = vec_str(answer)
        return dict(
            text=phrase(rng, stems, s=system_text(a, b, _NAMES3[:3]), t=t0),
            answer_kind="matrix",
            answer=column_value(answer),
            answer_text=text,
            answer_latex="plain:" + text,
            working=[
                f"Row-reducing the augmented matrix gives x = {join_terms([(c, ''), (-a_, 'z')])} and y = {join_terms([(e, ''), (-d_, 'z')])}.",
                f"With z = {t0}: x = {answer[0]} and y = {answer[1]}.",
            ],
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


def _medium(rng):
    for _ in attempts(300, "a line-of-solutions question"):
        a_, d_ = _nz(rng, -3, 3), _nz(rng, -3, 3)
        target = [[1, 0, a_, rng.randint(-4, 4)], [0, 1, d_, rng.randint(-4, 4)], [0, 0, 0, 0]]
        a, b = _system_from(rng, target, 3)
        _verify(a, b, target, (0, 1))
        direction = [-a_, -d_, 1]
        _check_solution(a, [0, 0, 0], direction)
        particular = _solution(target, (0, 1), [0])
        _check_solution(a, b, particular)
        null = sym(a).nullspace()
        check(len(null) == 1 and null[0].cross(sympy.Matrix(direction)).norm() == 0, "SymPy's direction differs")
        dist = vector_distractors(
            [
                ([a_, d_, 1], "rref-free-column-read-unsigned"),
                (particular, "particular-solution-as-direction"),
                ([-a_, -d_, 0], "direction-free-component-zero"),
                ([int(v) for v in a[0]], "direction-row-as-vector"),
            ],
            direction,
            "scalar_multiple",
        )
        if dist is None:
            continue
        stems = (
            "The solutions of the system {s} lie on a line. Find a direction vector of the line. Give it as a column (x, y, z).",
            "All the solutions of the system {s} lie on one line. Find a vector parallel to that line. Give it as a column (x, y, z).",
        )
        text = vec_str(direction)
        return dict(
            text=phrase(rng, stems, s=system_text(a, b, _NAMES3)),
            answer_kind="matrix",
            answer=column_value(direction),
            answer_text=text,
            answer_latex="plain:" + text,
            answer_form="scalar_multiple",
            working=[
                f"Row-reducing gives x = {join_terms([(target[0][3], ''), (-a_, 'z')])} and "
                f"y = {join_terms([(target[1][3], ''), (-d_, 'z')])}, with z free.",
                f"So the solutions are {vec_str(particular)} + z{vec_str(direction)}, and the direction vector is {text}.",
            ],
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


def _hard_target(rng, pivots):
    nz = _nz
    if pivots == (0, 1):
        return [
            [1, 0, nz(rng), nz(rng), rng.randint(-4, 4)],
            [0, 1, nz(rng), rng.randint(-4, 4), nz(rng)],
            [0, 0, 0, 0, 0],
        ]
    return [[1, nz(rng), 0, nz(rng), rng.randint(-4, 4)], [0, 0, 1, nz(rng), nz(rng)], [0, 0, 0, 0, 0]]


def _two_free(rng, pivots):
    """(target, a, b, free columns, values for them, solution, candidate wrong solutions)."""
    target = _hard_target(rng, pivots)
    a, b = _system_from(rng, target, 3)
    _verify(a, b, target, pivots)
    return target, a, b


def _hard(rng):
    for _ in attempts(300, "a two-free-variable question"):
        pivots = (0, 1)
        target, a, b = _two_free(rng, pivots)
        s0, t0 = rng.choice([-2, -1, 1, 2, 3]), rng.choice([-2, -1, 1, 2, 3])
        sol = [int(v) for v in _solution(target, pivots, [s0, t0])]
        _check_solution(a, b, sol)
        wrong_sign = [int(v) for v in _solution_unsigned(target, pivots, [s0, t0])]
        swapped = [int(v) for v in _solution(target, pivots, [t0, s0])]
        ignore_t = [int(v) for v in _solution(target, pivots, [s0, 0])]
        dist = vector_distractors(
            [
                (wrong_sign, "rref-free-column-read-unsigned"),
                (swapped, "free-variable-values-swapped"),
                (ignore_t, "free-variable-value-dropped"),
                ([target[0][4], target[1][4], s0, t0], "free-variable-not-substituted"),
            ],
            sol,
        )
        if dist is None:
            continue
        stems = (
            "The system {s} is consistent. Find the solution in which x3 = {s0} and x4 = {t0}. Give it as a column (x1, x2, x3, x4).",
            "Solve {s}, given that x3 = {s0} and x4 = {t0}. Give the answer as a column (x1, x2, x3, x4).",
        )
        text = vec_str(sol)
        return dict(
            text=phrase(rng, stems, s=system_text(a, b, _NAMES4), s0=s0, t0=t0),
            answer_kind="matrix",
            answer=column_value(sol),
            answer_text=text,
            answer_latex="plain:" + text,
            working=[
                f"Row-reducing [A | b] gives x1 = "
                f"{join_terms([(target[0][4], ''), (-target[0][2], 'x3'), (-target[0][3], 'x4')])} and "
                f"x2 = {join_terms([(target[1][4], ''), (-target[1][2], 'x3'), (-target[1][3], 'x4')])}.",
                f"With x3 = {s0} and x4 = {t0}: x1 = {sol[0]}, x2 = {sol[1]}.",
            ],
            distractors=dist,
        )
    raise VerificationError("unreachable")  # pragma: no cover


def _solution_unsigned(target, pivots, free_values):
    """The slip of adding the free-column entries instead of subtracting them."""
    n = len(target[0]) - 1
    free = [c for c in range(n) if c not in pivots]
    x = [0] * n
    for c, v in zip(free, free_values):
        x[c] = v
    for i, pc in enumerate(pivots):
        x[pc] = target[i][n] + sum(target[i][c] * x[c] for c in free)
    return x


@generator(
    "fy1la_general_solution",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1la_general_solution(rng, tier="easy"):
    """Solutions of a consistent system with free variables; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


def _param_text(const, coefs) -> str:
    return join_terms([(const, ""), (coefs[0], "s"), (coefs[1], "t")])


@generator("exd_x_fy1la_general_solution", level=17, presets={"exam": dict(tier="exam")})
def exd_x_fy1la_general_solution(rng, tier="exam"):
    """Exam chain: rank, number of parameters, a particular solution and a pivot variable in terms of s and t."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "an exam general-solution chain"):
        pivots = rng.choice([(0, 1), (0, 2)])
        target, a, b = _two_free(rng, pivots)
        free = [c for c in range(4) if c not in pivots]
        particular = [int(v) for v in _solution(target, pivots, [0, 0])]
        _check_solution(a, b, particular)
        # x_free[0] = s, x_free[1] = t: the first pivot variable is c - u s - v t.
        row = target[0]
        const, u, v = int(row[4]), int(row[free[0]]), int(row[free[1]])
        pivot_name = f"x{pivots[0] + 1}"
        s_sym, t_sym = sympy.symbols("s t")
        params = {free[0]: s_sym, free[1]: t_sym}
        expected = const - u * s_sym - v * t_sym
        check(sympy.expand(expected - _sympy_pivot(a, b, pivots, params)) == 0, "the pivot expression differs")
        answer_expr = sympy.expand(expected)
        text = _param_text(const, (-u, -v))
        dist_a = num_distractors(
            2, [(3, "rank-counts-all-rows"), (4, "rank-counts-all-columns"), (1, "rank-misses-a-pivot")]
        )
        dist_b = num_distractors(
            2, [(1, "free-variables-from-rows"), (3, "free-variables-miscounted"), (4, "free-variables-all-columns")]
        )
        dist_c = vector_distractors(
            [
                ([-v_ for v_ in particular], "solution-sign-slip"),
                ([int(x) for x in b] + [0], "rhs-read-as-solution"),
                (_solution(target, pivots, [1, 1]), "free-variables-set-to-one"),
                (_solution_unsigned(target, pivots, [0, 0]), "free-variable-sign-slip"),
            ],
            particular,
        )
        prompt_d = f"Let {_NAMES4[free[0]]} = s and {_NAMES4[free[1]]} = t. Write {pivot_name} in terms of s and t."
        dist_d = expression_distractors(
            str(answer_expr),
            [
                (_param_text(const, (u, v)), "rref-free-column-read-unsigned"),
                (_param_text(const, (-v, -u)), "free-variable-values-swapped"),
                (_param_text(0, (-u, -v)), "particular-solution-omitted"),
                (_param_text(-const, (-u, -v)), "solution-sign-slip"),
            ],
            prompt_d,
            n=2,
        )
        if None in (dist_a, dist_b, dist_c, dist_d):
            continue
        reduced = target
        return dict(
            text=f"The system {system_text(a, b, _NAMES4)} is consistent.",
            parts=[
                number_part(
                    "Find the rank of the coefficient matrix.",
                    2,
                    dist_a,
                    marks=1,
                    working=["Row-reducing the augmented matrix leaves two non-zero rows, so the rank is 2."],
                    sensible_range=(0, 4),
                ),
                number_part(
                    "How many parameters does the general solution have?",
                    2,
                    dist_b,
                    marks=1,
                    follow_through="4 - part_a",
                    working=["There are 4 unknowns and the rank is 2, so there are 4 - 2 = 2 parameters."],
                    sensible_range=(0, 5),
                ),
                matrix_part(
                    "Find the solution in which both free variables are zero. Give it as a column (x1, x2, x3, x4).",
                    particular,
                    dist_c,
                    marks=2,
                    vector=True,
                    working=[
                        f"Setting the free variables to 0, each pivot variable equals its entry in the last column: {vec_str(particular)}."
                    ],
                ),
                expression_part(
                    prompt_d,
                    answer_expr,
                    text,
                    dist_d,
                    marks=3,
                    working=[
                        f"Row {1} of the reduced matrix {reduced[0]} reads {pivot_name} + ({u})s + ({v})t = {const}, so {pivot_name} = {text}."
                    ],
                ),
            ],
        )
    raise VerificationError("unreachable")  # pragma: no cover


def _sympy_pivot(a, b, pivots, params):
    """The first pivot variable as a SymPy expression in the parameters, from the target's first row (second route:
    solve the original system with SymPy and read the same variable)."""
    n = 4
    free = [c for c in range(n) if c not in pivots]
    pivot_block = sympy.Matrix([[int(a[r][c]) for c in pivots] for r in range(3)])
    rhs = sympy.Matrix([int(b[r]) - sum(int(a[r][c]) * params[c] for c in free) for r in range(3)])
    solution, _ = pivot_block.gauss_jordan_solve(rhs)  # raises if the system is inconsistent or under-determined
    return solution[0]
