"""Exam-tier questions for Pascal's triangle and its properties.

Entries are indexed from zero in both row and position, so position k of row n holds C(n, k) and row n sums
to 2^n. The easier tiers look an entry up or apply the addition rule once. Here the row is NOT stated: the learner
must first recover it from a property of the triangle and then use a second property to answer.

  row_sum      -- position 2 of the row is T = n(n - 1)/2. Find the sum of the whole row (2^n).
  largest      -- the row sums to 2^n. Find its largest entry, the middle one C(n, n // 2).
  next_entry   -- position 2 of the row is T. Find the entry at position 3 of the same row.

The row index is drawn first, every stated number is derived from it with ``math.comb``, and the key is
re-derived by searching for the unique row that matches the stated clue.
"""

from __future__ import annotations

import math

from ..core import check, check_distinct, generator, phrase, pick_distractors

_ROW_SUM_NOTE = "The first entry of every row is at position 0."


def _row_from_position_two(t: int) -> int:
    """The unique n >= 2 with C(n, 2) == t, found by search."""
    rows = [n for n in range(2, 40) if math.comb(n, 2) == t]
    check(len(rows) == 1, "the position-2 clue must identify exactly one row")
    return rows[0]


@generator("exd_x_s4_pascal", level=12, presets={"exam": dict(tier="exam")})
def exd_x_s4_pascal(rng, tier="exam"):
    variant = rng.choice(("row_sum", "largest", "next_entry"))
    n = rng.randint(6, 13)
    t = math.comb(n, 2)
    if variant == "row_sum":
        answer = 2**n
        text = phrase(
            rng,
            (
                "In Pascal's triangle the entry at position 2 of a certain row is {t}. "
                "What is the sum of all the entries in that row? " + _ROW_SUM_NOTE,
                "One row of Pascal's triangle has {t} at position 2. Find the total of all the numbers in this "
                "row. " + _ROW_SUM_NOTE,
                "The third number in a row of Pascal's triangle (position 2) is {t}. "
                "What do all the numbers in that row add up to?",
            ),
            t=t,
        )
        m = _row_from_position_two(t)
        check(2**m == answer, "key disagrees with the row found by search")
        # Misconceptions: row index off by one in either direction; the two end 1s left out of the sum.
        cands = [2 ** (n - 1), 2 ** (n + 1), 2**n - 2]
    elif variant == "largest":
        s = 2**n
        answer = math.comb(n, n // 2)
        text = phrase(
            rng,
            (
                "The entries in one row of Pascal's triangle add up to {s}. What is the largest entry in that row?",
                "A row of Pascal's triangle has a total of {s}. Find the greatest number in this row.",
                "The numbers in a row of Pascal's triangle sum to {s}. What is the biggest of them?",
            ),
            s=s,
        )
        m = next(r for r in range(1, 40) if 2**r == s)
        check(max(math.comb(m, k) for k in range(m + 1)) == answer, "key disagrees with the row found by search")
        # Misconceptions: row index off by one in either direction; the largest entry taken as half the total.
        cands = [math.comb(n + 1, (n + 1) // 2), math.comb(n - 1, (n - 1) // 2), s // 2]
    else:
        answer = math.comb(n, 3)
        text = phrase(
            rng,
            (
                "In Pascal's triangle the entry at position 2 of a certain row is {t}. "
                "What is the entry at position 3 of the same row? " + _ROW_SUM_NOTE,
                "A row of Pascal's triangle has {t} at position 2. Find the number at position 3 in that row. "
                + _ROW_SUM_NOTE,
                "The third number in a row of Pascal's triangle (position 2) is {t}. What is the fourth number "
                "in the same row?",
            ),
            t=t,
        )
        m = _row_from_position_two(t)
        check(math.comb(m, 3) == answer, "key disagrees with the row found by search")
        # Misconceptions: row index off by one in either direction; position off by one; the stated entry
        # repeated as the answer.
        cands = [math.comb(n - 1, 3), math.comb(n + 1, 3), math.comb(n, 4), t]
    check(m == n, "the recovered row must be the row the question was built from")

    answer_text = str(answer)
    distractors = pick_distractors(answer_text, [str(c) for c in cands if c != answer], n=3)
    check_distinct(answer, *[int(d) for d in distractors])
    check(len(distractors) >= 3, "need three principled distractors")
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"C(n, 2) = {t} gives n = {n}." if variant != "largest" else f"2^n = {2**n} gives n = {n}.",
            f"Answer: {answer}.",
        ],
        distractors=distractors,
    )
