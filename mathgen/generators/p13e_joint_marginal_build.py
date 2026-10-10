"""Generates `ucv1-joint-distribution-marginal` questions: a marginal probability read from a joint distribution,
answered as a fraction.

Reads a finite joint PROBABILITY table over X in {x1, x2}, Y in {y1, y2, y3}
(six cells sharing one common denominator n, summing to n by construction --
see `_draw_table`) and computes a marginal or a conditional probability from
it. This is the missing piece between two shipping skills that it builds on:
`conditional-probability-counts` computes P(A|B)
from four RAW COUNTS in a 2x2 table (one division, no marginals); `finite-
joint-covariance` presents a joint PROBABILITY table over a 2x2 grid but its
final operation is covariance (a sum of products), never a marginal or
conditional. This skill starts from a probability table (like the
covariance skill's own presentation idiom) and returns a marginal/conditional
(like the counts skill's own operations, but starting already-normalised).

The skill `name` ("Compute marginals from a joint distribution") and its plural contract ("one marginal
probability and one conditional probability") are answered ACROSS the skill's four tiers, not inside one
question: easy is a marginal, medium/hard/exam are conditionals.

Tiers:
- easy: P(X = x1) -- sum the three cells of the x1 row.
- medium: P(Y = y2 | X = x1) -- a cell over its OWN row's marginal sum.
- hard: P(X = x2 | Y = y3) -- a cell over its OWN COLUMN's marginal sum (the
  student must recognise which axis to sum, since the conditioning variable
  has flipped from X in medium to Y in hard -- this is why the table is 2x3
  rather than a 2x2 square, which cannot pose a row-vs-column choice: any
  single cell of a 2x2 grid has exactly one other cell in its row and exactly
  one other cell in its column, so "sum the row" and "sum the column" differ
  only by which two numbers are added, not by which AXIS to select).
- exam: one cell is hidden from the rendered table; the learner must recover
  it (`hidden = n - sum(other five)`) before computing a marginal or
  conditional (drawn) of the ROW that contains the hidden cell -- guaranteeing
  the recovered cell is always a term in the final computation regardless of
  which of the six cells was hidden.

Independent check for the exam tier's own construction: `_draw_table` always
holds every one of the six TRUE cell values, so hiding one from the rendered
text never removes it from `cells` -- the omission is a presentation choice,
and the oracle re-parses the
five visible cells straight out of `question.text`, solves for the sixth
independently, and never imports this module.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_distinct,
    check_pairwise_distinct,
    generator,
    phrase,
    pick_distractors,
)

_MAX_ATTEMPTS = 200

_CELL_HI = {"easy": 8, "medium": 15, "hard": 20, "exam": 15}

_X_LABELS = ("x1", "x2")
_Y_LABELS = ("y1", "y2", "y3")


def _fraction_payload(value: Fraction) -> tuple[dict, str]:
    value = Fraction(value)
    text = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    return {"p": value.numerator, "q": value.denominator}, text


def _draw_table(rng, cell_hi):
    """Draw all six cell counts directly as positive integers and set the shared denominator n to their sum --
    every cell is >= 1 by construction, so the six probabilities always sum to exactly n/n = 1 with no
    rejection needed. It also matches `finite_joint_covariance`'s own idiom of drawing every cell weight and
    letting the total fall out as their sum."""
    cells = {(xi, yi): rng.randint(1, cell_hi) for xi in range(2) for yi in range(3)}
    n = sum(cells.values())
    return cells, n


def _row_sum(cells, xi):
    return sum(cells[(xi, yj)] for yj in range(3))


def _col_sum(cells, yi):
    return sum(cells[(xi, yi)] for xi in range(2))


def _table_text(cells, n, omit=None):
    parts = []
    for xi in range(2):
        for yj in range(3):
            if omit is not None and (xi, yj) == omit:
                continue
            parts.append(f"P(X={_X_LABELS[xi]},Y={_Y_LABELS[yj]})={cells[(xi, yj)]}/{n}")
    return "; ".join(parts)


def _offer(answer: Fraction, value, offered: list, formatted: list, label: str):
    """Append `value` iff it genuinely differs from the answer and from every
    value already offered -- several misconceptions can land on the same
    value, and only one should ever be shown. Every value actually offered
    is proven pairwise distinct, unconditionally."""
    value = Fraction(value)
    if value == answer or any(value == v for v in offered):
        return
    check_distinct(answer, value, msg=f"{label} collided with the answer")
    check_pairwise_distinct(*offered, value, msg=f"{label} collided with another candidate")
    offered.append(value)
    formatted.append(_fraction_payload(value)[1])


_EASY_PATTERNS = [
    "A joint distribution over X (values x1, x2) and Y (values y1, y2, y3) is given by: {table}. Find P(X = x1).",
    "The table gives every joint probability for X in x1, x2 and Y in y1, y2, y3: {table}. Compute the marginal probability P(X = x1).",
    "Given the joint probability table {table}, find the marginal probability P(X = x1).",
]

_MEDIUM_PATTERNS = [
    "A joint distribution over X (values x1, x2) and Y (values y1, y2, y3) is given by: {table}. Find P(Y = y2 | X = x1).",
    "The table gives every joint probability for X in x1, x2 and Y in y1, y2, y3: {table}. Compute the conditional probability P(Y = y2 | X = x1).",
    "Given the joint probability table {table}, find P(Y = y2 | X = x1).",
]

_HARD_PATTERNS = [
    "A joint distribution over X (values x1, x2) and Y (values y1, y2, y3) is given by: {table}. Find P(X = x2 | Y = y3).",
    "The table gives every joint probability for X in x1, x2 and Y in y1, y2, y3: {table}. Compute the conditional probability P(X = x2 | Y = y3).",
    "Given the joint probability table {table}, find P(X = x2 | Y = y3).",
]

_EXAM_PATTERNS = [
    "A joint distribution over X (values x1, x2) and Y (values y1, y2, y3) has these probabilities, with one entry missing (all six probabilities sum to 1): {table}. First recover the missing probability, then find {query}.",
    "The table below gives five of the six joint probabilities for X in x1, x2 and Y in y1, y2, y3 (all six probabilities sum to 1); one entry is missing: {table}. Recover the missing probability, then compute {query}.",
    "Given a joint probability table with one entry missing (the six probabilities sum to 1): {table}. Find the missing probability first, then compute {query}.",
]


def _p13e_joint_easy(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        cells, n = _draw_table(rng, _CELL_HI["easy"])
        hx = 0  # x1
        answer = Fraction(_row_sum(cells, hx), n)
        check(0 < answer < 1, "easy: marginal probability out of (0,1)")

        offered, formatted = [], []
        # numerator-only: offers a single cell of the row instead of the sum
        # of all three ("forgets to sum the row").
        _offer(answer, Fraction(cells[(hx, 0)], n), offered, formatted, "numerator-only")
        _offer(answer, 1 - answer, offered, formatted, "complement-slip")

        if len(formatted) < 1:
            continue
        answer_text = _fraction_payload(answer)[1]
        distractors = pick_distractors(answer_text, formatted)
        if len(distractors) < 1:
            continue

        table = _table_text(cells, n)
        text = phrase(rng, _EASY_PATTERNS, table=table)
        working = [
            f"P(X = x1) sums the x1 row: ({cells[(0, 0)]} + {cells[(0, 1)]} + {cells[(0, 2)]})/{n} = {answer_text}."
        ]
        return dict(
            text=text,
            latex=text,
            answer=_fraction_payload(answer)[0],
            answer_text=answer_text,
            answer_latex=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("easy: could not draw a usable question")


def _p13e_joint_medium(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        cells, n = _draw_table(rng, _CELL_HI["medium"])
        hx, hy = 0, 1  # x1, y2
        row = _row_sum(cells, hx)
        answer = Fraction(cells[(hx, hy)], row)
        check(cells[(hx, hy)] <= row, "medium: conditional numerator exceeds its own denominator")

        offered, formatted = [], []
        # forgets-division / treats-joint-as-conditional: offers the raw joint probability
        # P(X=x1,Y=y2) where the conditional was asked -- the two named misconceptions
        # collapse to this one value here.
        _offer(answer, Fraction(cells[(hx, hy)], n), offered, formatted, "forgets-division")
        _offer(answer, 1 - answer, offered, formatted, "complement-slip")

        if len(formatted) < 1:
            continue
        answer_text = _fraction_payload(answer)[1]
        distractors = pick_distractors(answer_text, formatted)
        if len(distractors) < 1:
            continue

        table = _table_text(cells, n)
        text = phrase(rng, _MEDIUM_PATTERNS, table=table)
        working = [f"P(Y = y2 | X = x1) = P(X=x1,Y=y2)/P(X=x1) = {cells[(hx, hy)]}/{row} = {answer_text}."]
        return dict(
            text=text,
            latex=text,
            answer=_fraction_payload(answer)[0],
            answer_text=answer_text,
            answer_latex=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("medium: could not draw a usable question")


def _p13e_joint_hard(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        cells, n = _draw_table(rng, _CELL_HI["hard"])
        hx, hy = 1, 2  # x2, y3
        col = _col_sum(cells, hy)
        row = _row_sum(cells, hx)
        answer = Fraction(cells[(hx, hy)], col)
        check(cells[(hx, hy)] <= col, "hard: conditional numerator exceeds its own denominator")

        offered, formatted = [], []
        _offer(answer, Fraction(cells[(hx, hy)], n), offered, formatted, "forgets-division")
        # wrong-axis: sums the ROW instead of the COLUMN for the denominator.
        _offer(answer, Fraction(cells[(hx, hy)], row), offered, formatted, "wrong-axis")
        _offer(answer, 1 - answer, offered, formatted, "complement-slip")

        if len(formatted) < 1:
            continue
        answer_text = _fraction_payload(answer)[1]
        distractors = pick_distractors(answer_text, formatted)
        if len(distractors) < 1:
            continue

        table = _table_text(cells, n)
        text = phrase(rng, _HARD_PATTERNS, table=table)
        working = [f"P(X = x2 | Y = y3) = P(X=x2,Y=y3)/P(Y=y3) = {cells[(hx, hy)]}/{col} = {answer_text}."]
        return dict(
            text=text,
            latex=text,
            answer=_fraction_payload(answer)[0],
            answer_text=answer_text,
            answer_latex=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("hard: could not draw a usable question")


def _query_text(hx, hy, query_marginal):
    if query_marginal:
        return f"P(X = {_X_LABELS[hx]})"
    return f"P(Y = {_Y_LABELS[hy]} | X = {_X_LABELS[hx]})"


def _quantity(cells, n, hx, hy, query_marginal):
    if query_marginal:
        return Fraction(_row_sum(cells, hx), n)
    return Fraction(cells[(hx, hy)], _row_sum(cells, hx))


def _p13e_joint_exam(rng):
    for _attempt in range(_MAX_ATTEMPTS):
        cells, n = _draw_table(rng, _CELL_HI["exam"])
        hx, hy = rng.randrange(2), rng.randrange(3)
        query_marginal = rng.random() < 0.5

        hidden = cells[(hx, hy)]
        known_sum = sum(v for k, v in cells.items() if k != (hx, hy))
        recovered = n - known_sum
        check(recovered == hidden, "exam: recovered cell mismatch")
        check(recovered > 0, "exam: recovered cell not positive")

        answer = _quantity(cells, n, hx, hy, query_marginal)

        offered, formatted = [], []
        # numerator-only / forgets-division: offers the raw recovered cell
        # over n instead of properly summing/dividing -- the same formula
        # a learner who does nothing with the recovered value beyond reading
        # it as "the" probability would produce, for either query type.
        _offer(
            answer, Fraction(hidden, n), offered, formatted, "numerator-only" if query_marginal else "forgets-division"
        )
        _offer(answer, 1 - answer, offered, formatted, "complement-slip")
        # wrong-missing-cell: recovers the hidden cell with the WRONG sign
        # (`sum of the other five - 1` instead of `1 - sum`, i.e. -recovered
        # in the numerator space) and carries THAT through the same formula.
        wrong_hidden = known_sum - n  # = -recovered
        wrong_cells = dict(cells)
        wrong_cells[(hx, hy)] = wrong_hidden
        wrong_row = _row_sum(wrong_cells, hx)
        if wrong_row != 0:
            try:
                wrong_answer = _quantity(wrong_cells, n, hx, hy, query_marginal)
                # A "probability" outside the open interval (0, 1) is
                # trivially eliminable on sight (every real
                # probability the learner sees is a proper fraction), so
                # only offer this misconception when it lands strictly
                # between 0 and 1. On the CONDITIONAL half of exam this is
                # never satisfiable: `wrong_answer = -r / (row_others - r)`
                # (r = recovered cell, row_others = the other two cells in
                # its row) is negative whenever row_others > r (excluded by
                # this guard) and strictly > 1 whenever row_others < r
                # (r / (r - row_others) with r - row_others < r) -- also
                # excluded. So `wrong-missing-cell` only ever fires on the
                # MARGINAL half of exam; stated honestly rather than hidden
                # when a particular draw makes it collide or be unexercisable.
                if 0 < wrong_answer < 1:
                    _offer(answer, wrong_answer, offered, formatted, "wrong-missing-cell")
            except ZeroDivisionError:
                pass

        if len(formatted) < 1:
            continue
        answer_text = _fraction_payload(answer)[1]
        distractors = pick_distractors(answer_text, formatted)
        if len(distractors) < 1:
            continue

        table = _table_text(cells, n, omit=(hx, hy))
        query = _query_text(hx, hy, query_marginal)
        text = phrase(rng, _EXAM_PATTERNS, table=table, query=query)
        row = _row_sum(cells, hx)
        if query_marginal:
            working = [
                f"Missing cell = {n} - {known_sum} = {recovered} (as {recovered}/{n}).",
                f"P(X = {_X_LABELS[hx]}) sums its row = {row}/{n} = {answer_text}.",
            ]
        else:
            working = [
                f"Missing cell = {n} - {known_sum} = {recovered} (as {recovered}/{n}).",
                f"P(Y = {_Y_LABELS[hy]} | X = {_X_LABELS[hx]}) = {hidden}/{row} = {answer_text}.",
            ]
        return dict(
            text=text,
            latex=text,
            answer=_fraction_payload(answer)[0],
            answer_text=answer_text,
            answer_latex=answer_text,
            working=working,
            distractors=distractors,
        )
    raise RuntimeError("exam: could not draw a usable question")


@generator(
    "p13e_joint_distribution_marginal",
    level=15,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p13e_joint_distribution_marginal(rng, tier="easy"):
    if tier == "easy":
        return _p13e_joint_easy(rng)
    if tier == "medium":
        return _p13e_joint_medium(rng)
    if tier == "hard":
        return _p13e_joint_hard(rng)
    if tier == "exam":
        return _p13e_joint_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
