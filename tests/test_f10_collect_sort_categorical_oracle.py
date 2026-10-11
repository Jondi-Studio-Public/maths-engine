"""Independent oracle for `f10-collect-sort-compare-categorical-data`.

Reads only `question.text` (the data and the one-line table) and the public grader. It recounts the data from
the text, fills each blank by its own rules and checks the result against `question.answer_text` and the real
`table` grader. It does not import the generator module.
"""

from __future__ import annotations

import os
import re
from collections import Counter

import pytest

from mathgen import make
from mathgen.grading.typed import grade_table

TOPIC = "f10_collect_sort_categorical"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("F10_TABLE_ORACLE_SEEDS", "150"))

_TABLE_RE = re.compile(r"Table \((?P<head>.+?)\): (?P<body>.+)\.$")


def _table(text):
    m = _TABLE_RE.search(text)
    assert m, text
    head = m.group("head").split(" — ")
    rows = [r.split(" — ") for r in m.group("body").split("; ")]
    return head, rows


def _tally_value(s):
    return 5 * s.count("/") + s.count("|") - 4 * s.count("/")


def _solve(text, tier):
    """{(row label, column header): value} for every cell that is a '?', from the text alone."""
    head, rows = _table(text)
    out = {}
    if tier == "easy":
        if head[1] == "Tally":
            counts = {r[0]: _tally_value(r[1]) for r in rows}
            col = 2
        else:
            listed = re.search(r": ([a-z, ]+)\. Sort", text).group(1).split(", ")
            c = Counter(listed)
            counts = {r[0]: c[r[0].lower()] for r in rows}
            col = 1
        for r in rows:
            if r[col] == "?":
                out[(r[0], head[col])] = counts[r[0]]
        return out
    if tier == "medium":
        named = re.findall(r"chose (\w+)[;.]", text)
        if named:
            votes = named
        else:
            votes = re.search(r"were: ([a-z, ]+)\. Complete", text).group(1).split(", ")
        c = Counter(votes)
        top_name = head[2].removeprefix("Fewer votes than ")
        top = c[top_name.lower()]
        for r in rows:
            if r[1] == "?":
                out[(r[0], "Votes")] = c[r[0].lower()]
            if r[2] == "?":
                out[(r[0], head[2])] = top - c[r[0].lower()]
        return out
    data = [int(x) for x in re.search(r": ([\d, ]+)\. Complete", text).group(1).split(", ")]
    c = Counter(data)
    total = len(data)
    for r in rows:
        if r[1] != "?":
            continue
        label = r[0]
        if label.isdigit():
            v = c[int(label)]
        elif label == "Total":
            v = total
        else:
            m = re.fullmatch(r"(?:(\d+) (or more|or fewer)|(More than|Fewer than) (\d+))", label)
            assert m, label
            if m.group(1):
                t, rel = int(m.group(1)), m.group(2)
            else:
                t, rel = int(m.group(4)), m.group(3)
            test = {
                "or more": lambda x: x >= t,
                "or fewer": lambda x: x <= t,
                "More than": lambda x: x > t,
                "Fewer than": lambda x: x < t,
            }[rel]
            v = sum(n for x, n in c.items() if test(x))
        out[(label, head[1])] = v
    return out


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_canonical_answer_and_grader(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        solved = _solve(q.text, tier)
        assert solved, (tier, seed)
        given = ", ".join(f"{label} {col} = {v}" for (label, col), v in solved.items())
        assert grade_table(q, given).correct, (tier, seed, given, q.answer_text)
        assert {f"{label} {col} = {v}" for (label, col), v in solved.items()} == set(q.answer_text.split(", ")), (
            tier,
            seed,
        )
        assert len(solved) == len(q.answer["blanks"])


@pytest.mark.parametrize("tier", TIERS)
def test_no_single_answer_dominates_and_no_blank_is_trivial(tier):
    pool = Counter()
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        pool.update(q.answer["blanks"].values())
    assert max(pool.values()) <= 0.3 * sum(pool.values())
