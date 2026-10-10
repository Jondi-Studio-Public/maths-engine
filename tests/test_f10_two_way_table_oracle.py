"""Independent oracle for `f10-two-way-table-relationship`.

Reads only `question.text`: it parses the one-line table and the list of paired records (reversed pairs and
"k x" repeats included), tabulates the records itself, and checks every given cell, every blank, the canonical
`answer_text` and the real `table` grader. It does not import the generator module.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make
from mathgen.grading.typed import grade_table

TOPIC = "f10_two_way_table"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("F10_TWO_WAY_ORACLE_SEEDS", "150"))

_TABLE_RE = re.compile(r"Table \((?P<head>.+?)\): (?P<body>.+)\.$")
_ENTRY_RE = re.compile(r"(?:(\d+) x )?\(([^(),]+), ([^(),]+)\)")


def _listing(text):
    return re.search(r"\): (.*?)\.(?: Some pairs| Complete the two-way)", text).group(1)


def _solve(text):
    m = _TABLE_RE.search(text)
    assert m, text
    head = m.group("head").split(" — ")
    rows = [r.split(" — ") for r in m.group("body").split("; ")]
    cols = head[1:-1]
    row_names = [r[0] for r in rows[:-1]]
    listing = _listing(text)
    counts = {}
    total = 0
    for k, a, b in _ENTRY_RE.findall(listing):
        k = int(k) if k else 1
        if a in row_names and b in cols:
            key = (a, b)
        else:
            assert b in row_names and a in cols, (a, b)
            key = (b, a)
        counts[key] = counts.get(key, 0) + k
        total += k
    full = {}
    for rn in row_names:
        for cn in cols:
            full[(rn, cn)] = counts.get((rn, cn), 0)
        full[(rn, "Total")] = sum(counts.get((rn, cn), 0) for cn in cols)
    for cn in cols:
        full[("Total", cn)] = sum(counts.get((rn, cn), 0) for rn in row_names)
    full[("Total", "Total")] = total
    return head, rows, full


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_tabulation_matches_every_cell_and_the_grader(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        head, rows, full = _solve(q.text)
        parts = []
        for row in rows:
            for j, cell in enumerate(row[1:]):
                value = full[(row[0], head[1 + j])]
                if cell == "?":
                    parts.append(f"{row[0]} {head[1 + j]} = {value}")
                else:
                    assert int(cell) == value, (tier, seed, row[0], head[1 + j])
        assert parts and len(parts) == len(q.answer["blanks"])
        assert grade_table(q, ", ".join(parts)).correct, (tier, seed)
        assert set(parts) == set(q.answer_text.split(", "))
        assert q.answer_text == ", ".join(parts)


def test_hard_repeat_entries_change_the_count_of_entries():
    """Counting entries instead of records must give a different table on hard questions that have a repeat."""
    seen = 0
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty="hard")
        listing = _listing(q.text)
        if re.search(r"\d+ x \(", listing):
            seen += 1
            n_entries = len(_ENTRY_RE.findall(listing))
            assert n_entries < _solve(q.text)[2][("Total", "Total")]
    assert seen >= N_SEEDS // 3
