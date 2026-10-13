"""Independent oracle for `senior-use-an-appropriately-percentaged-two-way-frequency-table`.

Reads only `question.text`: it parses the one-line table, works out from the wording of the question which
variable defines the groups being compared (so which total each count is divided by), recovers any missing
count from the row and column totals, and computes every percentage to one decimal place half-up. It checks
every given cell, every blank, the canonical `answer_text` and the real `table` grader. It does not import the
generator module.
"""

from __future__ import annotations

import os
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.grading.typed import grade_table

TOPIC = "senior_percentaged_two_way_table"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("SENIOR_PCT_ORACLE_SEEDS", "150"))

_TABLE_RE = re.compile(r"Table \((?P<head>.+?)\): (?P<body>.+)\.$")
_GROUP_PATTERNS = [
    r"percentage of the total of its (.+?) group",
    r"groups defined by (.+?)[,.] ",
    r"depends on the (.+?)(?: for |\. )",
]


def _round1(x: Fraction) -> str:
    tenths = (x * 10 + Fraction(1, 2)).__floor__()
    return f"{tenths // 10}.{tenths % 10}"


def _group_variable(stem):
    found = {m.group(1) for pat in _GROUP_PATTERNS for m in [re.search(pat, stem)] if m}
    assert len(found) == 1, (stem, found)
    return found.pop()


def _parse(text):
    m = _TABLE_RE.search(text)
    assert m, text
    head = m.group("head").split(" — ")
    rows = [r.split(" — ") for r in m.group("body").split("; ")]
    return head, rows


def _solve(text):
    stem = text.split("Table (")[0]
    head, rows = _parse(text)
    row_var, col_var = head[0].split(" / ")
    group = _group_variable(stem)
    assert group in (row_var, col_var)
    n_cats = sum(1 for h in head if h.endswith("(%)"))
    assert head[1 : 2 + n_cats][-1] == "Total (count)"
    body, totals = rows[:-1], rows[-1]
    counts = [[None if c == "?" else int(c) for c in r[1 : 1 + n_cats]] for r in body]
    row_totals = [int(r[1 + n_cats]) for r in body]
    col_totals = [int(c) for c in totals[1 : 1 + n_cats]]
    grand = int(totals[1 + n_cats])
    assert sum(row_totals) == grand == sum(col_totals)
    # recover missing counts from a row total (checking against the column total)
    for i, row in enumerate(counts):
        for j, v in enumerate(row):
            if v is None:
                assert row.count(None) == 1, "more than one missing count in a row"
                row[j] = row_totals[i] - sum(x for x in row if x is not None)
    for j in range(n_cats):
        assert sum(counts[i][j] for i in range(len(counts))) == col_totals[j]
    for i in range(len(counts)):
        assert sum(counts[i]) == row_totals[i]
    base = (lambda i, j: row_totals[i]) if group == row_var else (lambda i, j: col_totals[j])
    return head, rows, counts, n_cats, base


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_every_cell_and_the_grader(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        head, rows, counts, n_cats, base = _solve(q.text)
        parts = []
        for i, cells in enumerate(rows[:-1]):
            for j in range(n_cats):
                for col, cell, expected in (
                    (head[1 + j], cells[1 + j], str(counts[i][j])),
                    (head[2 + n_cats + j], cells[2 + n_cats + j], _round1(Fraction(counts[i][j] * 100, base(i, j)))),
                ):
                    if cell == "?":
                        parts.append(f"{cells[0]} {col} = {expected}")
                    else:
                        assert cell == expected, (tier, seed, cells[0], col, cell, expected)
        # the answer text lists the blanks in reading order (row by row, column by column)
        ordered = sorted(
            parts,
            key=lambda p: next(
                (i, c)
                for i, r in enumerate(rows[:-1])
                for c in range(1, len(head))
                if p.startswith(f"{r[0]} {head[c]} =")
            ),
        )
        assert len(parts) == len(q.answer["blanks"])
        assert q.answer_text == ", ".join(ordered)
        assert grade_table(q, ", ".join(ordered)).correct, (tier, seed)


def test_the_group_variable_is_named_once_and_in_every_tier_wording():
    for tier in TIERS:
        for seed in range(N_SEEDS):
            q = make(TOPIC, seed=seed, difficulty=tier)
            _solve(q.text)


def test_groups_percentages_differ_so_there_is_something_to_compare():
    """For the right percentaging, each response category's share differs between groups by at least 4 points."""
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty="hard")
        head, rows, counts, n_cats, base = _solve(q.text)
        group_is_row = _group_variable(q.text.split("Table (")[0]) == head[0].split(" / ")[0]
        n_rows = len(counts)
        if group_is_row:
            shares = [[Fraction(counts[i][j], base(i, j)) for i in range(n_rows)] for j in range(n_cats)]
        else:
            shares = [[Fraction(counts[i][j], base(i, j)) for j in range(n_cats)] for i in range(n_rows)]
        assert all(max(s) - min(s) >= Fraction(4, 100) for s in shares)


def test_wrong_total_distractor_really_uses_the_other_total():
    """Recompute the 'wrong total' table from the counts and find it among the distractors."""
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        head, rows, counts, n_cats, base = _solve(q.text)
        group_is_row = _group_variable(q.text.split("Table (")[0]) == head[0].split(" / ")[0]
        row_totals = [sum(r) for r in counts]
        col_totals = [sum(counts[i][j] for i in range(len(counts))) for j in range(n_cats)]
        parts = []
        for i, cells in enumerate(rows[:-1]):
            for j in range(n_cats):
                if cells[2 + n_cats + j] == "?":
                    other = col_totals[j] if group_is_row else row_totals[i]
                    parts.append(f"{cells[0]} {head[2 + n_cats + j]} = {_round1(Fraction(counts[i][j] * 100, other))}")
        wrong = ", ".join(parts)
        assert wrong in q.distractors
        assert not grade_table(q, wrong).correct
