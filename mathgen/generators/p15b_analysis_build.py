"""Generates a `functional-and-linear-analysis` question: which row of a stated (epsilon, N) Cauchy witness table
is inconsistent with a given finite sequence, or whether all rows are consistent.

Sits in `functional-and-linear-analysis`. The skill's contract: "For a
supplied finite sequence and tolerance table, decide whether the stated
Cauchy witness is internally consistent." This widens to a
genuinely multi-option question: which row of the supplied (epsilon, N)
witness table is inconsistent with the sequence, or are all rows
consistent -- five classes, recomputed exactly with Fraction, never a fixed
decimal approximation.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, rational_text

_NUM_ROWS = 4

_INTRO_PHRASINGS = [
    "The finite sequence has terms {seq_text}.",
    "Consider the finite sequence with terms {seq_text}.",
    "A finite sequence is given by {seq_text}.",
]

_INSTRUCTION_PHRASINGS = [
    "Which row's claim is inconsistent with the sequence, or are all rows consistent? "
    "Give your answer as one of the listed options.",
    "Exactly one row's claim may fail to match the sequence, or none may. "
    "Identify the inconsistent row, or state that all rows are consistent.",
    "Decide, row by row, whether the claimed epsilon actually bounds the tail spread. "
    "Name the one inconsistent row, or say all rows are consistent.",
]


def _sequence(b: int, k: int) -> list[Fraction]:
    """A_i = 1 / b^i for i = 0 .. k - 1: a strictly decreasing rational
    sequence, so the tail max-difference over any window [N, k - 1] is
    exactly a_N - a_{k-1} (the largest term in the window minus the
    smallest)."""
    return [Fraction(1, b**i) for i in range(k)]


def _tail_spread(seq: list[Fraction], n_index: int) -> Fraction:
    """D(N) = max_{m, n >= N} |a_m - a_n| over the supplied FINITE sequence
    -- exact, since `seq` is strictly decreasing, this is simply
    seq[N] - seq[-1]."""
    return seq[n_index] - seq[-1]


def _draw_rows(
    rng: random.Random,
    seq: list[Fraction],
    k: int,
    b: int,
    false_row: int | None,
    near_miss: bool,
    epsilon_as_sum: bool,
):
    """Draw `_NUM_ROWS` distinct claimed indices N_i in [1, k - 2] and, for
    each, an epsilon that makes the row's claim TRUE or FALSE exactly as
    directed by `false_row` (None => every row true, i.e. the "all
    consistent" class). For the true rows the margin is either generous or,
    at `near_miss`, the smallest possible nonzero Fraction step -- so a
    learner cannot eyeball the comparison and must add exact fractions."""
    check(k - 2 >= _NUM_ROWS, "completeness-cauchy-check: sequence too short for four distinct claimed indices")
    candidates = list(range(1, k - 1))
    rng.shuffle(candidates)
    n_indices = sorted(candidates[:_NUM_ROWS])

    margin_small = Fraction(1, b ** (k + 2))
    margin_big = Fraction(1, b**2)

    rows = []
    for row_idx, n_index in enumerate(n_indices):
        spread = _tail_spread(seq, n_index)
        if row_idx == false_row:
            epsilon = spread  # D(N) >= epsilon exactly -> inconsistent
            consistent = False
        else:
            margin = margin_small if near_miss else margin_big
            epsilon = spread + margin  # D(N) < epsilon strictly -> consistent
            consistent = True
        check(
            (spread < epsilon) == consistent,
            "completeness-cauchy-check: constructed row does not match its intended consistency",
        )
        rows.append(dict(n_index=n_index, epsilon=epsilon, consistent=consistent, spread=spread))

    if epsilon_as_sum:
        for row in rows:
            eps = row["epsilon"]
            # split into two DIFFERENT fractions that sum exactly back to
            # eps -- exercised through Fraction, never a float
            # approximation.
            part_a = eps / 3
            part_b = eps - part_a
            check(part_a != part_b, "completeness-cauchy-check: epsilon-sum parts must differ")
            check(part_a + part_b == eps, "completeness-cauchy-check: epsilon-sum parts do not recombine to epsilon")
            row["epsilon_display_parts"] = (part_a, part_b)

    return rows


@generator(
    "p15b_completeness_cauchy_check",
    level=20,
    presets={
        "easy": dict(b=2, k=7, near_miss=False, epsilon_as_sum=False),
        "medium": dict(b=3, k=7, near_miss=False, epsilon_as_sum=False),
        "hard": dict(b=2, k=8, near_miss=True, epsilon_as_sum=False),
        "exam": dict(b=3, k=8, near_miss=True, epsilon_as_sum=True),
    },
)
def p15b_completeness_cauchy_check(rng, b=2, k=7, near_miss=False, epsilon_as_sum=False):
    """For a supplied finite sequence and (epsilon, N) tolerance table,
    decide which row's Cauchy-tail claim is inconsistent with the sequence,
    or whether all rows are consistent (the skill's contract, widened
    to a genuinely multi-option question: five
    classes, drawn uniformly, recomputed exactly with Fraction).

    easy/medium: a single geometric sequence 1/b^i, generous margins, epsilon
    given as one already-reduced fraction. hard: the SAME family with the
    smallest possible nonzero margin (a near-miss the learner cannot
    eyeball). exam: the SAME near-miss margins, with epsilon now given as a
    SUM of two fractions that must be added exactly before the comparison
    can be made at all.
    """
    seq = _sequence(b, k)
    target = rng.choice(["none", 0, 1, 2, 3])
    false_row = None if target == "none" else target
    rows = _draw_rows(rng, seq, k, b, false_row, near_miss, epsilon_as_sum)

    # Independent re-verification: recompute the tail spread and the
    # consistency verdict for EVERY row directly from the sequence, never
    # trusting the constructor's own bookkeeping.
    inconsistent_rows = []
    for i, row in enumerate(rows):
        recomputed_spread = _tail_spread(seq, row["n_index"])
        check(recomputed_spread == row["spread"], "completeness-cauchy-check: tail spread mismatch on reverification")
        recomputed_consistent = recomputed_spread < row["epsilon"]
        check(
            recomputed_consistent == row["consistent"],
            "completeness-cauchy-check: consistency verdict mismatch on reverification",
        )
        if not recomputed_consistent:
            inconsistent_rows.append(i)
    if target == "none":
        check(len(inconsistent_rows) == 0, "completeness-cauchy-check: 'all consistent' draw has an inconsistent row")
        answer_text = "All rows are consistent."
    else:
        check(inconsistent_rows == [target], "completeness-cauchy-check: exactly one row must be inconsistent")
        answer_text = f"Row {target + 1} is inconsistent."

    seq_text = ", ".join(f"a_{i} = {rational_text(v)}" for i, v in enumerate(seq))
    row_lines = []
    for i, row in enumerate(rows):
        if epsilon_as_sum:
            p_a, p_b = row["epsilon_display_parts"]
            eps_text = f"{rational_text(p_a)} + {rational_text(p_b)}"
        else:
            eps_text = rational_text(row["epsilon"])
        row_lines.append(
            f"Row {i + 1}: epsilon = {eps_text}, N = {row['n_index']} (claims |a_m - a_n| < epsilon for all m, n >= N)"
        )
    rows_text = "; ".join(row_lines)

    intro = rng.choice(_INTRO_PHRASINGS).format(seq_text=seq_text)
    instruction = rng.choice(_INSTRUCTION_PHRASINGS)
    text = (
        f"{intro} The witness table below states, for each row, "
        f"a tolerance epsilon and an index N: {rows_text}. {instruction}"
    )
    latex = rf"\text{{{intro} }}\text{{{rows_text}. }}\text{{{instruction}}}"

    options = ["All rows are consistent."] + [f"Row {i + 1} is inconsistent." for i in range(_NUM_ROWS)]
    check(len(set(options)) == len(options), "completeness-cauchy-check: option collision")
    check(answer_text in options, "completeness-cauchy-check: answer is not one of the fixed options")
    distractors = [o for o in options if o != answer_text]
    check(len(distractors) == _NUM_ROWS, "completeness-cauchy-check: expected four distractors")

    if target == "none":
        working = [
            "Every row's epsilon exceeds the sequence's true tail spread a_N - a_(k-1) at its "
            "claimed N, so every row's claim holds: all rows are consistent."
        ]
    else:
        row = rows[target]
        working = [
            f"Row {target + 1}: the true tail spread a_{row['n_index']} - a_{k - 1} = "
            f"{rational_text(row['spread'])} is NOT less than the claimed epsilon = "
            f"{rational_text(row['epsilon'])}, so row {target + 1}'s claim fails."
        ]

    return dict(
        text=text,
        latex=latex,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=rf"\text{{{answer_text}}}",
        working=working,
        distractors=distractors,
    )
