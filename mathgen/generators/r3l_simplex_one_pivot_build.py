"""Generates `uc2-simplex-one-pivot` questions (easy, medium, hard) and the exam chain `exd_x_r3l_simplex_one_pivot`.

``uc2-simplex-one-pivot`` (``optimisation``, answer ``matrix``): the simplex tableau for a maximisation problem
max z = c.x subject to A x <= b, x >= 0. The tableau has one row per constraint, with the columns of the decision
variables, then the slack variables, then the right-hand side; the last row is the objective row, which holds -c under
the decision variables. Every stem states the pivot rule, so each answer is a single tableau of exact fractions:
the entering column is the most negative entry of the objective row (lowest column on a tie) and the pivot row is the
smallest ratio RHS / entry over positive entries (lowest row on a tie). The generators only accept instances in which
neither choice involves a tie.

* easy: write the initial tableau for the linear programme.
* medium: the tableau after one pivot, starting from the initial tableau (supplied).
* hard: the tableau after one pivot, starting from a tableau in the middle of a run (supplied), so the entries are
  already fractions and the basic columns are not the original slack columns.
* exam (``exd_x_r3l_simplex_one_pivot``): from the linear programme, (a) the initial tableau, (b) the smallest ratio
  of the first pivot, (c) the tableau after that pivot, (d) the value of z after it (follows through from (b)).

The tableau width is kept to six columns or fewer (at most five variables, plus the right-hand side) because the matrix grader
reads at most six columns. Every pivot is recomputed with SymPy's exact rational matrices (row operations as an elementary matrix product) and
compared with the Fraction implementation in `_r3l_kit`.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import check, generator, pick_distractors
from ._r3l_kit import (
    PIVOT_RULE,
    draw_lp,
    entering_column,
    initial_tableau,
    leaving_row,
    linear_text,
    num,
    pivot,
    ratios,
    sub,
)
from ._ucv1_linalg import mat_str, matrix_distractors, rows_value


def _names(n, m):
    return [sub("x", j + 1) for j in range(n)] + [sub("s", i + 1) for i in range(m)]


def _lp_text(a, b, c) -> str:
    n = len(c)
    xs = [sub("x", j + 1) for j in range(n)]
    cons = "; ".join(f"{linear_text(row, xs)} ≤ {r}" for row, r in zip(a, b))
    return f"Maximise z = {linear_text(c, xs)} subject to {cons}, and {', '.join(xs)} ≥ 0."


def _obj_eq(c, xs) -> str:
    """`z − 6x₁ − 8x₂ = 0`."""
    return "z " + linear_text([-v for v in c], xs).replace("−", "− ", 1) + " = 0"


def _layout(n, m, c) -> str:
    cols = ", ".join(_names(n, m) + ["RHS"])
    xs = [sub("x", j + 1) for j in range(n)]
    return (
        f"Write a simplex tableau with columns {cols}: one row for each constraint, in order, then the objective row "
        f"for {_obj_eq(c, xs)}."
    )


def _tableau_layout(n, m, c) -> str:
    cols = ", ".join(_names(n, m) + ["RHS"])
    xs = [sub("x", j + 1) for j in range(n)]
    return (
        f"Its columns are {cols}; the rows are the {m} constraints in order, then the objective row "
        f"({_obj_eq(c, xs)} to begin with)."
    )


def _sym(tab):
    return sympy.Matrix([[sympy.Rational(v.numerator, v.denominator) for v in row] for row in tab])


def _verify_pivot(before, r, col, after):
    """SymPy: the same row operations written as an elementary matrix applied on the left."""
    rows = len(before)
    e = sympy.eye(rows)
    p = _sym(before)[r, col]
    for i in range(rows):
        e[i, r] = -_sym(before)[i, col] / p if i != r else 1 / p
    check(e * _sym(before) == _sym(after), "SymPy's pivot differs")
    check(_sym(after)[:, col] == sympy.Matrix([1 if i == r else 0 for i in range(rows)]), "pivot column is not a unit")


def _scale_free_pivot(tab, r, c):
    """Eliminate with the pivot row left unscaled (the slip of never dividing by the pivot)."""
    out = [list(row) for row in tab]
    p = tab[r][c]
    for i, row in enumerate(tab):
        if i != r and row[c] != 0:
            f = row[c] / p
            out[i] = [v - f * w for v, w in zip(row, tab[r])]
    return out


def _added_pivot(tab, r, c):
    """Scale the pivot row, then add (rather than subtract) multiples of it to clear the column."""
    p = tab[r][c]
    scaled = [v / p for v in tab[r]]
    out = [list(row) for row in tab]
    out[r] = scaled
    for i, row in enumerate(tab):
        if i != r and row[c] != 0:
            out[i] = [v + row[c] * w for v, w in zip(row, scaled)]
    return out


def _pivot_candidates(tab, r, c):
    """(tableau, misconception) for the slips on a pivot of `tab`, most tempting first."""
    out = []
    positives = [i for i, row in enumerate(tab[:-1]) if row[c] > 0 and i != r]
    if positives:
        wrong_row = max(positives, key=lambda i: tab[i][-1] / tab[i][c])
        out.append((pivot(tab, wrong_row, c), "r3l-simplex-leaving-row-wrong"))
    obj = tab[-1][:-1]
    other = [j for j in range(len(obj)) if obj[j] < 0 and j != c]
    if other:
        j = min(other, key=lambda j: (obj[j], j))
        row = leaving_row(tab, j)
        if row is not None:
            out.append((pivot(tab, row, j), "r3l-simplex-entering-column-wrong"))
    out.append((_scale_free_pivot(tab, r, c), "r3l-simplex-pivot-row-unscaled"))
    out.append((_added_pivot(tab, r, c), "r3l-simplex-row-operation-sign"))
    done = pivot(tab, r, c)
    stale = [list(x) for x in done]
    stale[-1] = list(tab[-1])
    out.append((stale, "r3l-simplex-objective-row-not-updated"))
    return out


def _matrix_dists(cands, answer, n=3):
    rows = rows_value(answer)
    dists = matrix_distractors([(mat, tag) for mat, tag in cands], rows)
    return dists[:n]


def _pick_pivot_instance(rng, n, m, which):
    """Data for a pivot question: (a, b, c, before, after, (r, col))."""
    a, b, c, tabs, pivots = draw_lp(rng, n, m)
    k = {"first": 0, "second": 1}[which]
    return a, b, c, tabs[k], tabs[k + 1], pivots[k]


# -- single-answer tiers ---------------------------------------------------------------------------------------------


def _easy(rng):
    n, m = rng.choice([(2, 2), (2, 3), (3, 2)])
    a, b, c, _tabs, _pivots = draw_lp(rng, n, m)
    tab = initial_tableau(a, b, c)
    xs = [sub("x", j + 1) for j in range(n)]
    cands = []
    plus = [list(r) for r in tab]
    plus[-1] = [-v for v in tab[-1]]
    cands.append((plus, "r3l-simplex-objective-row-sign"))
    flipped = [list(r) for r in tab]
    for i, row in enumerate(tab[:-1]):
        flipped[i] = row[:n] + list(reversed(row[n : n + m])) + row[-1:]
    cands.append((flipped, "r3l-simplex-slack-misplaced"))
    swapped = [[*reversed(r[:n]), *r[n:]] for r in tab[:-1]] + [tab[-1]]
    cands.append((swapped, "r3l-simplex-columns-swapped"))
    answer = rows_value(tab)
    distractors = _matrix_dists(cands, tab)
    check(len(distractors) >= 2, "too few tableau distractors")
    text = f"{_lp_text(a, b, c)} Introduce a slack variable for each constraint. {_layout(n, m, c)} Give the initial tableau."
    return dict(
        text=text,
        answer_kind="matrix",
        answer=answer,
        answer_text=mat_str(tab),
        answer_latex="plain:" + mat_str(tab),
        working=[
            "Each constraint gains a slack variable, so the constraint rows are [A | I | b].",
            f"The objective row holds the negatives of the profits: {_obj_eq(c, xs)} gives −c under the decision variables and 0 elsewhere.",
            f"The tableau is {mat_str(tab)}.",
        ],
        distractors=distractors,
    )


def _pivot_question(rng, which, n, m):
    a, b, c, before, after, (r, col) = _pick_pivot_instance(rng, n, m, which)
    _verify_pivot(before, r, col, after)
    cands = _pivot_candidates(before, r, col)
    distractors = _matrix_dists(cands, after)
    check(len(distractors) >= 2, "too few tableau distractors")
    return a, b, c, before, after, (r, col), distractors


def _medium(rng):
    n, m = rng.choice([(2, 2), (2, 3), (3, 2)])
    a, b, c, before, after, (r, col), distractors = _pivot_question(rng, "first", n, m)
    xs = [sub("x", j + 1) for j in range(n)]
    text = (
        f"{_lp_text(a, b, c)} Its initial simplex tableau is {mat_str(before)}. {_tableau_layout(n, m, c)} {PIVOT_RULE} "
        "Perform one pivot and give the new tableau, with every entry an exact fraction."
    )
    names = _names(n, m)
    return dict(
        text=text,
        answer_kind="matrix",
        answer=rows_value(after),
        answer_text=mat_str(after),
        answer_latex="plain:" + mat_str(after),
        working=[
            f"The most negative objective entry is {num(before[-1][col])}, in the {names[col]} column.",
            f"The ratios are {', '.join(num(x) if x is not None else '–' for x in ratios(before, col))}; the smallest is in row {r + 1}, so the pivot is {num(before[r][col])}.",
            f"Divide row {r + 1} by the pivot and clear the rest of the column to get {mat_str(after)}.",
        ],
        distractors=distractors,
    )


def _hard(rng):
    n, m = rng.choice([(3, 2), (2, 3), (3, 2)])
    a, b, c, before, after, (r, col), distractors = _pivot_question(rng, "second", n, m)
    names = _names(n, m)
    text = (
        f"A simplex run on a maximisation problem with {n} decision variables and {m} constraints has reached the tableau "
        f"{mat_str(before)}. Its columns are {', '.join(names + ['RHS'])}; the rows are the {m} constraints in order, then the objective row. {PIVOT_RULE} "
        "Perform the next pivot and give the new tableau, with every entry an exact fraction."
    )
    return dict(
        text=text,
        answer_kind="matrix",
        answer=rows_value(after),
        answer_text=mat_str(after),
        answer_latex="plain:" + mat_str(after),
        working=[
            f"The most negative objective entry is {num(before[-1][col])}, in the {names[col]} column.",
            f"The ratios are {', '.join(num(x) if x is not None else '–' for x in ratios(before, col))}; the smallest is in row {r + 1}, so the pivot is {num(before[r][col])}.",
            f"Divide row {r + 1} by the pivot and clear the rest of the column to get {mat_str(after)}.",
        ],
        distractors=distractors,
    )


@generator(
    "r3l_simplex_one_pivot",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3l_simplex_one_pivot(rng, tier="easy"):
    """The simplex tableau: initial tableau, one pivot, or one pivot mid-run; see the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------------


@generator("exd_x_r3l_simplex_one_pivot", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r3l_simplex_one_pivot(rng, tier="exam"):
    """Exam chain: initial tableau, smallest ratio, tableau after the first pivot, z after it."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n, m = rng.choice([(2, 3), (3, 2), (2, 2)])
    a, b, c, tabs, pivots = draw_lp(rng, n, m)
    first, second = tabs[0], tabs[1]
    r, col = pivots[0]
    _verify_pivot(first, r, col, second)
    theta = second[r][-1]  # the pivot row's RHS after scaling is the smallest ratio
    check(theta == first[r][-1] / first[r][col], "ratio differs")
    coef = -first[-1][col]
    z1 = second[-1][-1]
    check(z1 == coef * theta, "z after the pivot is coef * theta")
    xs = [sub("x", j + 1) for j in range(n)]
    names = _names(n, m)
    text = f"{_lp_text(a, b, c)} Introduce a slack variable for each constraint. {PIVOT_RULE}"
    # part (a): the initial tableau
    plus = [list(x) for x in first]
    plus[-1] = [-v for v in first[-1]]
    flipped = [row[:n] + list(reversed(row[n : n + m])) + row[-1:] for row in first[:-1]] + [first[-1]]
    a_dist = _matrix_dists(
        [(plus, "r3l-simplex-objective-row-sign"), (flipped, "r3l-simplex-slack-misplaced")], first, 3
    )
    # part (b): the smallest ratio
    rs = [x for x in ratios(first, col) if x is not None]
    b_cands = [
        Distractor(num(x), "r3l-simplex-leaving-row-wrong", exact_value(x))
        for x in sorted(rs, reverse=True)
        if x != theta
    ]
    b_cands.append(Distractor(num(coef), "r3l-simplex-ratio-for-entering-value", exact_value(coef)))
    b_cands.append(Distractor(num(first[r][-1]), "r3l-simplex-ratio-numerator-only", exact_value(first[r][-1])))
    b_cands.append(Distractor(num(first[r][col]), "r3l-simplex-ratio-denominator-only", exact_value(first[r][col])))
    b_cands.append(
        Distractor(
            num(first[r][col] / first[r][-1]), "r3l-simplex-ratio-inverted", exact_value(first[r][col] / first[r][-1])
        )
    )
    b_dist = pick_distractors(num(theta), b_cands, 3)
    # part (c): the tableau after the pivot
    c_dist = _matrix_dists(_pivot_candidates(first, r, col), second, 3)
    # part (d): z after the pivot
    d_cands = [
        Distractor(num(coef), "r3l-simplex-ratio-for-entering-value", exact_value(coef)),
        Distractor(num(theta), "r3l-simplex-ratio-for-entering-value", exact_value(theta)),
        Distractor(num(first[-1][col] * theta), "r3l-simplex-objective-sign", exact_value(first[-1][col] * theta)),
        Distractor(num(coef + theta), "r3l-simplex-ratio-for-entering-value", exact_value(coef + theta)),
    ]
    d_dist = pick_distractors(num(z1), d_cands, 3)
    check(
        all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)),
        f"too few distractors {[len(x) for x in (a_dist, b_dist, c_dist, d_dist)]}",
    )
    return dict(
        text=text,
        parts=[
            dict(
                prompt=_layout(n, m, c) + " Give the initial tableau.",
                answer_kind="matrix",
                answer=rows_value(first),
                answer_text=mat_str(first),
                answer_latex="plain:" + mat_str(first),
                marks=2,
                working=[f"The tableau is {mat_str(first)}."],
                distractors=a_dist,
            ),
            dict(
                prompt="In the first pivot, find the smallest ratio in the minimum ratio test.",
                answer_kind="fraction",
                answer=exact_value(theta),
                answer_text=num(theta),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"The entering column is {names[col]} (entry {num(first[-1][col])}). The ratios are "
                    f"{', '.join(num(x) for x in rs)}, and the smallest is {num(theta)}."
                ],
                distractors=b_dist,
            ),
            dict(
                prompt="Perform the first pivot and give the new tableau, with every entry an exact fraction.",
                answer_kind="matrix",
                answer=rows_value(second),
                answer_text=mat_str(second),
                answer_latex="plain:" + mat_str(second),
                marks=3,
                working=[f"Pivoting on row {r + 1}, column {col + 1} gives {mat_str(second)}."],
                distractors=c_dist,
            ),
            dict(
                prompt="What is the value of z at the basic solution after this pivot?",
                answer_kind="fraction",
                answer=exact_value(z1),
                answer_text=num(z1),
                marks=1,
                follow_through=f"{num(coef)}*part_b"
                if coef.denominator == 1
                else f"Rational({coef.numerator}, {coef.denominator})*part_b",
                sensible_range=(0, None),
                working=[
                    f"z rises by {num(coef)} for each unit of {names[col]}, and {names[col]} = {num(theta)}, so z = {num(z1)}."
                ],
                distractors=d_dist,
            ),
        ],
    )
