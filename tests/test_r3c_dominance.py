"""Tests for `dominance-matrix-ranking`: tournaments read back from the wording and ranked independently."""

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
    questions,
)

TOPIC = "r3c_dominance"
CHAIN = "r3c_dominance_chain"


def _teams(text):
    m = re.search(r"(?:row and a column for each of|a row for each of) (.+?), in that order", text)
    return [x for x in re.split(r", | and ", m.group(1)) if x]


def _results(text, teams):
    """Dominance matrix from "X beat Y." / "X lost to Y." sentences."""
    n = len(teams)
    D = [[0] * n for _ in range(n)]
    alt = "|".join(map(re.escape, teams))
    games = 0
    for a, verb, b in re.findall(rf"({alt}) (beat|lost to) ({alt})\.", text):
        w, l = (a, b) if verb == "beat" else (b, a)
        D[teams.index(w)][teams.index(l)] = 1
        games += 1
    assert games == n * (n - 1) // 2
    return D


def _mat(D):
    n = len(D)
    return [[sum(D[i][k] * D[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def _rank(teams, D):
    D2 = _mat(D)
    score = [sum(D[i]) + sum(D2[i]) for i in range(len(teams))]
    assert len(set(score)) == len(score)
    return [teams[i] for i in sorted(range(len(teams)), key=lambda i: -score[i])]


@pytest.mark.parametrize("tier", TIERS)
def test_answers_grade_correct_and_distractors_wrong(tier):
    for q in questions(TOPIC, tier):
        check_question_grading(q)
        check_tagged(q)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_varied(tier):
    assert make(TOPIC, seed=3, difficulty=tier).to_json() == make(TOPIC, seed=3, difficulty=tier).to_json()
    assert distinct_texts(TOPIC, tier) >= 40


def test_easy_ranks_by_wins_with_ties_in_listed_order():
    ties = 0
    for q in questions(TOPIC, "easy", range(1, 61)):
        assert q.answer_kind == "ordered_list"
        teams = _teams(q.text)
        D = ast.literal_eval(re.search(r"D = (\[\[.*?\]\])\.", q.text).group(1))
        wins = [sum(r) for r in D]
        assert q.answer == [teams[i] for i in sorted(range(len(teams)), key=lambda i: (-wins[i], i))]
        ties += len(set(wins)) < len(wins)
    assert ties == 60


def test_hard_and_exam_use_both_phrasings():
    joined = " ".join(q.text for q in questions(TOPIC, "hard", range(1, 41)))
    assert " beat " in joined and " lost to " in joined


def test_medium_ranks_from_the_given_matrix():
    for q in questions(TOPIC, "medium", range(1, 61)):
        assert q.answer_kind == "ordered_list"
        teams = _teams(q.text)
        D = ast.literal_eval(re.search(r"D = (\[\[.*?\]\])\.", q.text).group(1))
        assert q.answer == _rank(teams, D)
        wins_order = [teams[i] for i in sorted(range(len(teams)), key=lambda i: (-sum(D[i]), i))]
        assert q.answer != wins_order


def test_hard_has_a_tie_at_the_top_that_the_second_step_breaks():
    for q in questions(TOPIC, "hard", range(1, 41)):
        teams = sorted(
            set(
                re.findall(
                    r"\b(?:Kookaburras|Wombats|Emus|Dingoes|Cockatoos|Echidnas|Magpies|Brolgas|Wallabies)\b", q.text
                )
            )
        )
        D = _results(q.text, teams)
        wins = sorted((sum(r) for r in D), reverse=True)
        assert wins[0] == wins[1] != wins[2]
        assert q.answer == _rank(teams, D)
        assert len(teams) in (5, 6)


def test_exam_chain_recomputed_and_followed_through():
    honoured = 0
    for q in questions(CHAIN, "exam", range(1, 41)):
        assert [p.answer_kind for p in q.parts] == ["matrix", "number", "number", "number"]
        check_question_grading(q)
        check_tagged(q)
        honoured += check_follow_through(q)
        teams = _teams(q.parts[0].prompt)
        D = _results(q.text, teams)
        assert q.parts[0].answer == D
        team = next(t for t in teams if f"the {t}" in q.parts[1].prompt)
        k = teams.index(team)
        D2 = _mat(D)
        assert [int(p.answer_text) for p in q.parts[1:]] == [sum(D[k]), sum(D2[k]), sum(D[k]) + sum(D2[k])]
    assert honoured > 20
