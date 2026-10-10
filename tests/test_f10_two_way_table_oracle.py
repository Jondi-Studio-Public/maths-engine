"""Independent oracle for `f10-two-way-table-relationship`.

Reads only `question.text`. It parses the one-line table and any listed records or stated condition, then
solves the table with SymPy's linear solver (not the generator's solver) and requires a UNIQUE solution that
matches `question.answer_text` and the real `table` grader. It does not import the generator module.
"""

from __future__ import annotations

import os
import re

import pytest
import sympy

from mathgen import make
from mathgen.grading.typed import grade_table

TOPIC = "f10_two_way_table"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("F10_TWO_WAY_ORACLE_SEEDS", "120"))

_TABLE_RE = re.compile(r"Table \((?P<head>.+?)\): (?P<body>.+)\.$")
_REL_RE = re.compile(
    r"the total for (?P<a>.+?) is (?P<how>twice|three times|\d+ more than) ?(?:the total for )?(?P<b>.+?)\. Use"
)
_REL_RE = re.compile(
    r"the total for (?P<a>.+?) is (?:(?P<k>twice|three times)|(?P<m>\d+) more than) the total for (?P<b>.+?)\. Use this"
)


def _parse(text):
    m = _TABLE_RE.search(text)
    assert m, text
    head = m.group("head").split(" — ")
    rows = [r.split(" — ") for r in m.group("body").split("; ")]
    return head, rows


def _solve(text, tier):
    head, rows = _parse(text)
    cols = head[1:]
    n_rows, n_cols = len(rows) - 1, len(cols) - 1
    sym = {}
    grid = []
    for i, row in enumerate(rows):
        line = []
        for j, cell in enumerate(row[1:]):
            if cell == "?":
                sym[(i, j)] = sympy.Symbol(f"x{i}_{j}")
                line.append(sym[(i, j)])
            else:
                line.append(sympy.Integer(int(cell)))
        grid.append(line)
    eqs = []
    if tier == "easy":
        pairs = re.findall(r"\(([^(),]+), ([^(),]+)\)", text.split("Table (")[0])
        pairs = [p for p in pairs if p[0] in [r[0] for r in rows]]
        counts = {}
        for a, b in pairs:
            counts[(a, b)] = counts.get((a, b), 0) + 1
        for i in range(n_rows):
            for j in range(n_cols):
                eqs.append(grid[i][j] - counts.get((rows[i][0], cols[j]), 0))
        for i in range(n_rows):
            eqs.append(sum(grid[i][:n_cols]) - grid[i][n_cols])
        for j in range(n_cols):
            eqs.append(sum(grid[i][j] for i in range(n_rows)) - grid[n_rows][j])
        eqs.append(sum(grid[n_rows][:n_cols]) - grid[n_rows][n_cols])
        eqs.append(grid[n_rows][n_cols] - len(pairs))
    else:
        for i in range(n_rows + 1):
            eqs.append(sum(grid[i][:n_cols]) - grid[i][n_cols])
        for j in range(n_cols + 1):
            eqs.append(sum(grid[i][j] for i in range(n_rows)) - grid[n_rows][j])
        if tier == "hard":
            m = _REL_RE.search(text)
            assert m, text
            names_rows = [r[0] for r in rows[:-1]]
            if m.group("a") in cols:
                ta, tb = grid[n_rows][cols.index(m.group("a"))], grid[n_rows][cols.index(m.group("b"))]
            else:
                ta, tb = grid[names_rows.index(m.group("a"))][n_cols], grid[names_rows.index(m.group("b"))][n_cols]
            if m.group("k"):
                eqs.append(ta - (2 if m.group("k") == "twice" else 3) * tb)
            else:
                eqs.append(ta - tb - int(m.group("m")))
    unknowns = list(sym.values())
    solution = sympy.solve(eqs, unknowns, dict=True)
    return head, rows, sym, solution, eqs


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_unique_solution_matches_answer_and_grader(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        head, rows, sym, solution, _eqs = _solve(q.text, tier)
        assert len(solution) == 1, (tier, seed)
        sol = solution[0]
        assert set(sol) == set(sym.values()), (tier, seed, "not uniquely determined")
        parts = []
        for (i, j), s in sorted(sym.items()):
            value = sol[s]
            assert value == int(value) and value >= 0
            parts.append(f"{rows[i][0]} {head[1 + j]} = {int(value)}")
        assert grade_table(q, ", ".join(parts)).correct, (tier, seed, parts, q.answer_text)
        assert set(parts) == set(q.answer_text.split(", "))


def test_hard_condition_is_load_bearing():
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty="hard")
        _head, _rows, sym, _solution, eqs = _solve(q.text, "hard")
        without = sympy.solve(eqs[:-1], list(sym.values()), dict=True)
        assert len(without) == 1 and set(without[0]) != set(sym.values()), (seed, "condition is not needed")


def test_easy_records_are_all_in_the_table_and_counted():
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty="easy")
        grand = int(q.answer_text.split(", ")[-1].split("= ")[-1]) if "Total Total" in q.answer_text else None
        n = int(re.search(r"(\d+) ", q.text).group(1))
        assert 10 <= n <= 18
        if grand is not None:
            assert grand == n
