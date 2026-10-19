"""Shared pieces for the ``r3c_*`` generators (matrix models: transition, Leslie, dominance, permutation, arithmetic, costing).

Not a generator module (the leading underscore keeps ``mathgen.generators`` from importing it as one).

* ``number_part`` / ``fraction_part`` / ``matrix_part`` / ``ordered_part`` and the ``*_single`` builders turn
  ``(value, misconception id)`` pairs into tagged ``Distractor``s that the real grader has been asked about and
  rejected, and into the dicts a generator returns for one question or for one part of an exam chain;
* the small matrix helpers (``mat_mul``, ``col``, ``transpose`` ...) work on lists of rows of ``Fraction``.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import (
    Question,
    attempts,
    check,
    latex_answer,
    numeric_answer,
    pick_distractors,
    rational_text,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

__all__ = [
    "attempts",
    "check",
    "col",
    "fraction_part",
    "fraction_single",
    "mat_mul",
    "mat_pow",
    "matrix_cells",
    "matrix_part",
    "matrix_single",
    "number_part",
    "number_single",
    "num_text",
    "ordered_part",
    "ordered_single",
    "rejected",
    "transpose",
    "vec",
]

Matrix = list[list[Fraction]]


# ---------------------------------------------------------------- matrix helpers


def mat_mul(a, b) -> Matrix:
    check(len(a[0]) == len(b), "matrix sizes do not multiply")
    return [
        [sum((Fraction(a[i][k]) * b[k][j] for k in range(len(b))), Fraction(0)) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def mat_pow(a, n: int) -> Matrix:
    out = [[Fraction(int(i == j)) for j in range(len(a))] for i in range(len(a))]
    for _ in range(n):
        out = mat_mul(a, out)
    return out


def transpose(a) -> Matrix:
    return [[Fraction(v) for v in r] for r in zip(*a)]


def col(values) -> Matrix:
    """A column vector (a matrix of one column) from a list of numbers."""
    return [[Fraction(v)] for v in values]


def vec(m) -> list[Fraction]:
    """The entries of a one-column or one-row matrix as a flat list."""
    return [Fraction(v) for r in m for v in r]


# ---------------------------------------------------------------- spelling


def num_text(value, dp: int = 0) -> str:
    """``value`` written to ``dp`` decimal places, rounded half up ("4", "2.5", "-0.30")."""
    if dp == 0:
        return str(int(round_half_up(Fraction(value), 0)))
    return stated_precision_string(round_half_up(Fraction(value), dp), dp)


def cell_text(value) -> str:
    """One matrix entry: a whole number, a short decimal, or a fraction."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    den = value.denominator
    for d in (10, 100, 1000):
        if d % den == 0:
            places = len(str(d)) - 1
            return stated_precision_string(value, places).rstrip("0")
    return f"{value.numerator}/{den}"


def matrix_cells(rows) -> str:
    """The canonical spelling of a matrix answer: ``[[1, 0], [0, 1]]`` (a column vector is ``[[1], [2]]``)."""
    return "[" + ", ".join("[" + ", ".join(cell_text(v) for v in row) + "]" for row in rows) + "]"


def _cell_value(value):
    value = Fraction(value)
    return int(value) if value.denominator == 1 else exact_value(value)


def _matrix_answer(rows):
    return [[_cell_value(v) for v in row] for row in rows]


# ---------------------------------------------------------------- the real grader


def _stub(kind: str, answer, answer_text: str) -> Question:
    return Question(
        topic="r3c-check",
        level=1,
        seed=0,
        text="",
        latex="",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        answer_kind=kind,
    )


def rejected(kind: str, answer, answer_text: str, reply: str) -> bool:
    """True when the live grader of ``kind`` marks ``reply`` wrong for the keyed ``answer``."""
    from ..grading import grade

    return not grade(kind, _stub(kind, answer, answer_text), reply).correct


# ---------------------------------------------------------------- number


def number_distractors(key, wrong, *, dp: int = 0, lo=None, want: int = 3):
    key_text = num_text(key, dp)
    tolerance = stated_precision_tolerance(dp) if dp else 0
    answer = numeric_answer(round_half_up(Fraction(key), dp))
    cands = []
    for value, tag in wrong:
        if value is None:
            continue
        value = Fraction(value)
        if lo is not None and value < lo:
            continue
        shown = num_text(value, dp)
        cands.append(Distractor(shown, tag, numeric_answer(Fraction(shown))))
    picked = pick_distractors(key_text, cands, want + 3, tolerance=tolerance)
    out = [d for d in picked if rejected("number", answer, key_text, d.text)]
    return out[:want]


def _number_dict(key, dp, dis):
    key_text = num_text(key, dp)
    return dict(
        answer_kind="number",
        answer=numeric_answer(round_half_up(Fraction(key), dp)),
        answer_text=key_text,
        answer_latex=latex_answer(key_text),
        distractors=dis,
    )


def number_single(text, key, working, wrong, *, dp: int = 0, lo=None, want: int = 3):
    dis = number_distractors(key, wrong, dp=dp, lo=lo, want=want)
    if len(dis) < want:
        return None
    return dict(text=text, working=working, **_number_dict(key, dp, dis))


def number_part(prompt, key, marks, working, wrong, *, follow=None, dp: int = 0, lo=None, sensible=(None, None)):
    dis = number_distractors(key, wrong, dp=dp, lo=lo)
    if len(dis) < 3:
        return None
    part = dict(prompt=prompt, marks=marks, working=working, **_number_dict(key, dp, dis))
    if sensible is not None:
        part["sensible_range"] = sensible
    if follow:
        part["follow_through"] = follow
    return part


# ---------------------------------------------------------------- fraction


def fraction_distractors(key, wrong, *, want: int = 3):
    key = Fraction(key)
    key_text = rational_text(key)
    cands, seen = [], {key_text}
    for value, tag in wrong:
        if value is None:
            continue
        value = Fraction(value)
        shown = rational_text(value)
        if value == key or shown in seen:
            continue
        seen.add(shown)
        cands.append(Distractor(shown, tag, exact_value(value)))
    out = [d for d in cands if rejected("fraction", exact_value(key), key_text, d.text)]
    return out[:want]


def _fraction_dict(key, dis):
    key = Fraction(key)
    text = rational_text(key)
    return dict(
        answer_kind="fraction",
        answer=exact_value(key),
        answer_text=text,
        answer_latex=latex_answer(text),
        distractors=dis,
    )


def fraction_single(text, key, working, wrong, *, want: int = 3):
    dis = fraction_distractors(key, wrong, want=want)
    if len(dis) < want:
        return None
    return dict(text=text, working=working, **_fraction_dict(key, dis))


def fraction_part(prompt, key, marks, working, wrong, *, follow=None, sensible=(0, 1)):
    dis = fraction_distractors(key, wrong)
    if len(dis) < 3:
        return None
    part = dict(prompt=prompt, marks=marks, working=working, **_fraction_dict(key, dis))
    if sensible is not None:
        part["sensible_range"] = sensible
    if follow:
        part["follow_through"] = follow
    return part


# ---------------------------------------------------------------- matrix


def matrix_distractors(rows, wrong, *, want: int = 3):
    """Tagged matrix distractors from ``(rows, misconception id)``; each is rejected by the matrix grader."""
    answer = _matrix_answer(rows)
    answer_text = matrix_cells(rows)
    out, seen = [], {answer_text}
    for value, tag in wrong:
        if value is None:
            continue
        t = matrix_cells(value)
        if t in seen or not rejected("matrix", answer, answer_text, t):
            continue
        seen.add(t)
        out.append(Distractor(t, tag, _matrix_answer(value)))
        if len(out) == want:
            break
    return out


def _matrix_dict(rows, dis):
    text = matrix_cells(rows)
    return dict(
        answer_kind="matrix",
        answer=_matrix_answer(rows),
        answer_text=text,
        answer_latex="plain:" + text,
        distractors=dis,
    )


def matrix_single(text, rows, working, wrong, *, want: int = 3):
    dis = matrix_distractors(rows, wrong, want=want)
    if len(dis) < want:
        return None
    return dict(text=text, working=working, **_matrix_dict(rows, dis))


def matrix_part(prompt, rows, marks, working, wrong, *, want: int = 3):
    dis = matrix_distractors(rows, wrong, want=want)
    if len(dis) < want:
        return None
    return dict(prompt=prompt, marks=marks, working=working, **_matrix_dict(rows, dis))


# ---------------------------------------------------------------- ordered list of words


def ordered_distractors(items, wrong, *, want: int = 3):
    answer_text = ", ".join(items)
    cands = []
    for value, tag in wrong:
        if value is None:
            continue
        t = ", ".join(value)
        if t != answer_text and rejected("ordered_list", list(items), answer_text, t):
            cands.append(Distractor(t, tag, list(value)))
    return pick_distractors(answer_text, cands, want)


def _ordered_dict(items, dis):
    text = ", ".join(items)
    return dict(
        answer_kind="ordered_list", answer=list(items), answer_text=text, answer_latex="plain:" + text, distractors=dis
    )


def ordered_single(text, items, working, wrong, *, want: int = 3):
    dis = ordered_distractors(items, wrong, want=want)
    if len(dis) < want:
        return None
    return dict(text=text, working=working, **_ordered_dict(items, dis))


def ordered_part(prompt, items, marks, working, wrong, *, want: int = 3):
    dis = ordered_distractors(items, wrong, want=want)
    if len(dis) < want:
        return None
    return dict(prompt=prompt, marks=marks, working=working, **_ordered_dict(items, dis))


# ---------------------------------------------------------------- transition models (columns sum to 1)


def random_column(rng, n: int, j: int, *, stay=(4, 8)) -> list[Fraction]:
    """Column ``j`` of a transition matrix in tenths: the diagonal entry (staying) is ``stay`` tenths at least and
    the rest is shared out among the other states, at least one of which is non-zero."""
    kept = rng.randint(*stay)
    left = 10 - kept
    others = [i for i in range(n) if i != j]
    shares = {i: 0 for i in others}
    for _ in range(left):
        shares[rng.choice(others)] += 1
    column = [Fraction(0)] * n
    column[j] = Fraction(kept, 10)
    for i, s in shares.items():
        column[i] = Fraction(s, 10)
    return column


def random_transition(rng, n: int, *, stay=(4, 8)) -> Matrix:
    """An ``n`` by ``n`` matrix whose columns are random distributions in tenths (column ``j`` is where state ``j``
    goes next), so ``T S`` moves a state vector one step on."""
    columns = [random_column(rng, n, j, stay=stay) for j in range(n)]
    return [[columns[j][i] for j in range(n)] for i in range(n)]


def percent(value) -> str:
    return f"{int(Fraction(value) * 100)}%"


def names_text(items) -> str:
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def solve_linear(a, b):
    """The exact solution of the square system ``a x = b`` (lists of ``Fraction``), or None when it is singular."""
    n = len(a)
    m = [[Fraction(v) for v in row] + [Fraction(b[i])] for i, row in enumerate(a)]
    for c in range(n):
        pivot = next((r for r in range(c, n) if m[r][c] != 0), None)
        if pivot is None:
            return None
        m[c], m[pivot] = m[pivot], m[c]
        m[c] = [v / m[c][c] for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] for i in range(n)]


def steady_state(T) -> list[Fraction] | None:
    """The distribution ``s`` (non-negative, adding to 1) with ``T s = s``, or None when it is not unique."""
    n = len(T)
    a = [[Fraction(T[i][j]) - (1 if i == j else 0) for j in range(n)] for i in range(n)]
    a[-1] = [Fraction(1)] * n
    s = solve_linear(a, [0] * (n - 1) + [1])
    if s is None or any(v < 0 for v in s):
        return None
    return s
