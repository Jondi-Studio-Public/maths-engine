"""Tests for `permutation-binary-matrices`: scrambling, unscrambling, finding P from a block, and relation matrices."""

from __future__ import annotations

import ast
import re

import pytest

from mathgen import make
from tests.r3c_support import (
    TIERS,
    check_follow_through,
    check_question_grading,
    check_tagged,
    distinct_texts,
    matrix_after,
    questions,
)

TOPIC = "r3c_permutation"
CHAIN = "r3c_permutation_chain"


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def _T(a):
    return [list(r) for r in zip(*a)]


def _is_permutation(P):
    return all(sum(r) == 1 for r in P) and all(sum(c) == 1 for c in zip(*P)) and all(v in (0, 1) for r in P for v in r)


def _pair(text):
    m = re.search(r"the block \(([\d, ]+)\) came out as \(([\d, ]+)\)", text)
    return [int(x) for x in m.group(1).split(",")], [int(x) for x in m.group(2).split(",")]


def _order(P):
    n, cur, k = len(P), P, 1
    ident = [[int(i == j) for j in range(n)] for i in range(n)]
    while cur != ident:
        cur, k = _mul(P, cur), k + 1
    return k


@pytest.mark.parametrize("tier", TIERS)
def test_answers_grade_correct_and_distractors_wrong(tier):
    for q in questions(TOPIC, tier):
        check_question_grading(q)
        check_tagged(q)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_varied(tier):
    assert make(TOPIC, seed=3, difficulty=tier).to_json() == make(TOPIC, seed=3, difficulty=tier).to_json()
    assert distinct_texts(TOPIC, tier) >= 40


def test_easy_scramble_and_relation_variants():
    kinds = set()
    for q in questions(TOPIC, "easy", range(1, 61)):
        if "relation" in q.text:
            kinds.add("relation")
            elements = [
                int(x)
                for x in re.findall(r"\d+", re.search(r"on the numbers ([\d, and]+) is defined", q.text).group(1))
            ]
            rules = {
                "a divides b": lambda a, b: b % a == 0,
                "b is more than 2 greater than a": lambda a, b: b - a > 2,
                "b is exactly 3 greater than a, or exactly 4 greater": lambda a, b: b - a in (3, 4),
            }
            rule = next(r for label, r in rules.items() if f"when {label}." in q.text)
            assert q.answer == [[int(rule(a, b)) for b in elements] for a in elements]
        else:
            kinds.add("scramble")
            P, M = matrix_after(q.text, "P"), matrix_after(q.text, "M")
            assert _is_permutation(P)
            assert q.answer == _mul(P, M)
    assert kinds == {"relation", "scramble"}


def test_medium_unscramble_uses_the_transpose():
    for q in questions(TOPIC, "medium", range(1, 41)):
        P, E = matrix_after(q.text, "P"), matrix_after(q.text, "E")
        assert _is_permutation(P) and P != _T(P)
        assert q.answer == _mul(_T(P), E)
        assert _mul(P, q.answer) == E


def test_hard_finds_p_then_unscrambles():
    for q in questions(TOPIC, "hard", range(1, 41)):
        before, after = _pair(q.text)
        assert len(set(before)) == len(before)
        n = len(before)
        # the only permutation matrix taking the block to its scrambled form
        P = [[int(after[i] == before[j]) for j in range(n)] for i in range(n)]
        assert _is_permutation(P) and P != _T(P)
        E = matrix_after(q.text, "E")
        assert q.answer == _mul(_T(P), E)
        assert n in (4, 5)


def test_exam_chain_recomputed_and_followed_through():
    honoured = 0
    for q in questions(CHAIN, "exam", range(1, 41)):
        assert [p.answer_kind for p in q.parts] == ["matrix", "matrix", "number", "number"]
        check_question_grading(q)
        check_tagged(q)
        honoured += check_follow_through(q)
        before, after = _pair(q.text)
        n = len(before)
        P = [[int(after[i] == before[j]) for j in range(n)] for i in range(n)]
        assert q.parts[0].answer == P
        M = ast.literal_eval(re.search(r"M = (\[\[.*?\]\])", q.parts[1].prompt).group(1))
        assert q.parts[1].answer == _mul(P, M)
        order = _order(P)
        assert int(q.parts[2].answer_text) == order
        runs = int(re.search(r"scrambled (\d+) times", q.parts[3].prompt).group(1))
        assert int(q.parts[3].answer_text) == runs % order != 0
    assert honoured > 20
