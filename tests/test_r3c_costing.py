"""Tests for `matrix-costing-model`: costs from a price row, quantity matrices and recipe matrices."""

from __future__ import annotations

import ast
import re
from fractions import Fraction

import pytest

from mathgen import make
from tests.r3c_support import (
    TIERS,
    as_rows,
    check_follow_through,
    check_question_grading,
    check_tagged,
    distinct_texts,
    questions,
)

TOPIC = "r3c_costing"
CHAIN = "r3c_costing_chain"


def _row(text, label):
    return [Fraction(x) for x in re.search(rf"{label} = \[([\d., ]+)\]", text).group(1).split(",")]


def _grid(text, label):
    return [
        [Fraction(str(v)) for v in row]
        for row in ast.literal_eval(re.search(rf"{label} = (\[\[.*?\]\])", text).group(1))
    ]


def _times(row, grid):
    return [sum(row[i] * grid[i][j] for i in range(len(grid))) for j in range(len(grid[0]))]


@pytest.mark.parametrize("tier", TIERS)
def test_answers_grade_correct_and_distractors_wrong(tier):
    for q in questions(TOPIC, tier):
        check_question_grading(q)
        check_tagged(q)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_varied(tier):
    assert make(TOPIC, seed=3, difficulty=tier).to_json() == make(TOPIC, seed=3, difficulty=tier).to_json()
    assert distinct_texts(TOPIC, tier) >= 40


def test_easy_cost_of_one_branch():
    for q in questions(TOPIC, "easy", range(1, 41)):
        assert q.answer_kind == "number"
        P, Q = _row(q.text, "P"), _grid(q.text, "Q")
        names = re.search(r"a column for each of (.+?) \(in that order\)", q.text).group(1)
        branches = [x for x in re.split(r", | and ", names) if x]
        asked = re.search(r"How much does the (.+?) branch pay", q.text).group(1)
        assert Fraction(q.answer_text) == _times(P, Q)[branches.index(asked)]


def test_medium_total_of_all_branches():
    for q in questions(TOPIC, "medium", range(1, 41)):
        assert q.answer_kind == "number"
        P, Q = _row(q.text, "P"), _grid(q.text, "Q")
        assert Fraction(q.answer_text) == sum(_times(P, Q))
        # the same total from the quantities of each item over all branches
        assert Fraction(q.answer_text) == sum(P[i] * sum(Q[i]) for i in range(len(P)))
        assert len(P) == len(Q) and len(Q[0]) >= 3


def test_hard_two_stage_product_for_one_outlet():
    for q in questions(TOPIC, "hard", range(1, 41)):
        assert q.answer_kind == "number"
        P, R, Q = _row(q.text, "P"), _grid(q.text, "R"), _grid(q.text, "Q")
        per_batch = _times(P, R)
        outlets = re.search(r"delivered to (.+?) \(columns", q.text).group(1)
        names = [x for x in re.split(r", | and ", outlets) if x]
        asked = re.search(r"ingredient cost for (.+?), in dollars", q.text).group(1)
        expected = _times(per_batch, Q)[names.index(asked)]
        assert Fraction(q.answer_text) == expected
        # associativity: the same answer by the other grouping
        RQ = [[sum(R[i][p] * Q[p][o] for p in range(len(Q))) for o in range(len(Q[0]))] for i in range(len(R))]
        assert _times(P, RQ)[names.index(asked)] == expected


def test_exam_chain_recomputed_and_followed_through():
    honoured = 0
    for q in questions(CHAIN, "exam", range(1, 41)):
        assert [p.answer_kind for p in q.parts] == ["number"] * 4
        check_question_grading(q)
        check_tagged(q)
        honoured += check_follow_through(q)
        P, R = _row(q.text, "P"), _grid(q.text, "R")
        batch = _times(P, R)
        assert [Fraction(q.parts[0].answer_text), Fraction(q.parts[1].answer_text)] == batch
        x, y = [int(v) for v in re.search(r"is (\d+) batches of .*? and (\d+) batches", q.parts[2].prompt).groups()]
        cost = x * batch[0] + y * batch[1]
        assert Fraction(q.parts[2].answer_text) == cost
        sells = [Fraction(v) for v in re.findall(r"\$([\d.]+) a batch", q.parts[3].prompt)]
        assert Fraction(q.parts[3].answer_text) == x * sells[0] + y * sells[1] - cost
        assert Fraction(q.parts[3].answer_text) > 0
    assert honoured > 20
