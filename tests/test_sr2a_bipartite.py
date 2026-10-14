"""Tests for `bipartite-matching-matrix` (ACMGM110): the 0/1 matrix of an allocation problem and what it shows."""

from __future__ import annotations

import re
from itertools import product

import pytest

from mathgen import make
from tests.sr2a_support import (
    TIERS,
    check_follow_through,
    check_question_grading,
    check_tagged,
    distinct_texts,
    questions,
    top_share,
)

TOPIC = "sr2a_bipartite"
CHAIN = "sr2a_bipartite_chain"


def _split_names(text):
    return [x for x in re.split(r", | and ", text) if x]


def _order(text):
    rows = _split_names(re.search(r"a row for each of (.+?), in that order", text).group(1))
    cols = _split_names(re.search(r"a column for each of (.+?), in that order", text).group(1))
    return rows, cols


def _facts(text, rows, cols):
    """{worker: set of jobs} from the sentences that start with a worker's name."""
    out = {}
    for who in rows:
        m = re.search(rf"(?<![A-Za-z]){who} (can|cannot) ([^.]*)\.", text)
        if m:
            out[who] = (m.group(1), {c for c in cols if re.search(rf"\b{re.escape(c)}\b", m.group(2))})
    return out


def _matrix_from_workers(text, rows, cols):
    facts = _facts(text, rows, cols)
    m = []
    for who in rows:
        kind, jobs = facts[who]
        m.append([int((c in jobs) == (kind == "can")) for c in cols])
    return m


def _matrix_from_jobs(text, rows, cols):
    m = [[0] * len(cols) for _ in rows]
    for j, c in enumerate(cols):
        names = re.search(rf"{c.capitalize()}: ([^.]*)\.", text).group(1)
        for who in _split_names(names):
            m[rows.index(who)][j] = 1
    return m


def _best(m):
    """(largest matching, number of largest matchings) by trying every partial assignment of jobs to workers."""
    n, k = len(m), len(m[0])
    tally = {}
    for pick in product([None, *range(k)], repeat=n):
        used = [j for j in pick if j is not None]
        if len(set(used)) != len(used) or any(j is not None and not m[i][j] for i, j in enumerate(pick)):
            continue
        tally[len(used)] = tally.get(len(used), 0) + 1
    top = max(tally)
    return top, tally[top]


@pytest.mark.parametrize("tier", TIERS)
def test_answers_grade_correct_and_distractors_wrong(tier):
    for q in questions(TOPIC, tier):
        assert q.answer_kind == "matrix"
        check_question_grading(q)
        check_tagged(q)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_varied(tier):
    assert make(TOPIC, seed=3, difficulty=tier).to_json() == make(TOPIC, seed=3, difficulty=tier).to_json()
    assert distinct_texts(TOPIC, tier) >= 40
    assert top_share(TOPIC, tier) <= 0.1


def test_easy_matrix_matches_the_sentences():
    for q in questions(TOPIC, "easy", range(1, 61)):
        rows, cols = _order(q.text)
        assert q.answer == _matrix_from_workers(q.text, rows, cols)
        assert all(sum(r) for r in q.answer) and all(sum(c) for c in zip(*q.answer))


def test_medium_both_descriptions():
    seen = set()
    for q in questions(TOPIC, "medium", range(1, 61)):
        rows, cols = _order(q.text)
        if "can be given it are listed" in q.text:
            seen.add("by-job")
            assert q.answer == _matrix_from_jobs(q.text, rows, cols)
            assert q.answer != [list(r) for r in zip(*q.answer)]
        else:
            seen.add("cannot")
            assert q.answer == _matrix_from_workers(q.text, rows, cols)
    assert seen == {"by-job", "cannot"}


def test_hard_reduction_is_recomputed_from_the_lists():
    for q in questions(TOPIC, "hard", range(1, 61)):
        rows, cols = _order(q.text)
        m = _matrix_from_workers(q.text, rows, cols)
        forced = [(next(i for i in range(5) if m[i][j]), j) for j in range(5) if sum(r[j] for r in m) == 1]
        assert 1 <= len(forced) <= 2 and len({i for i, _ in forced}) == len(forced)
        left_rows = [i for i in range(5) if i not in {i for i, _ in forced}]
        left_cols = [j for j in range(5) if j not in {j for _, j in forced}]
        assert q.answer == [[m[i][j] for j in left_cols] for i in left_rows]
        assert len(q.answer) >= 3


def test_exam_chain_matching_numbers_are_recomputed():
    honoured = 0
    for q in questions(CHAIN, "exam", range(1, 41)):
        assert [p.answer_kind for p in q.parts] == ["matrix", "number", "number", "number"]
        check_question_grading(q)
        honoured += check_follow_through(q)
        rows, cols = _order(q.text)
        m = _matrix_from_workers(q.text, rows, cols)
        assert q.parts[0].answer == m
        best, ways = _best(m)
        n = len(rows)
        assert [int(p.answer_text) for p in q.parts[1:]] == [best, n - best, ways]
        assert n - best >= 1 and ways >= 2
    assert honoured > 20
