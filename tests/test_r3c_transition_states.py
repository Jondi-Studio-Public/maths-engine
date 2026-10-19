"""Tests for `transition-matrix-states`: a population moved on by a transition matrix, read back from the wording."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from tests.r3c_support import (
    TIERS,
    check_follow_through,
    check_question_grading,
    check_tagged,
    column_sums_are_one,
    distinct_texts,
    names_in,
    parse_transition,
    questions,
    step,
    top_share,
)

TOPIC = "r3c_transition_states"
CHAIN = "r3c_transition_states_chain"


def _start(text, states):
    """The starting numbers by state from "there are 1500 tennis, 2500 netball and 500 swimming members"."""
    seg = re.search(
        r"(?:there are|At the start there are) ([^.]*?) (?:members|customers|shoppers|commuters)\.?", text
    ).group(1)
    out = {}
    for n, name in re.findall(r"(\d+) ([A-Za-z\- ]+?)(?:,| and |$)", seg):
        out[name.strip()] = Fraction(int(n))
    assert set(out) == set(states), (out, states)
    return out


def _order(text):
    return [x.strip() for x in re.search(r"in the order (.+?), for example", text).group(1).split(", ")]


@pytest.mark.parametrize("tier", TIERS)
def test_answers_grade_correct_and_distractors_wrong(tier):
    for q in questions(TOPIC, tier):
        check_question_grading(q)
        check_tagged(q)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_varied(tier):
    assert make(TOPIC, seed=3, difficulty=tier).to_json() == make(TOPIC, seed=3, difficulty=tier).to_json()
    assert distinct_texts(TOPIC, tier) >= 40
    assert top_share(TOPIC, tier) <= 0.1


@pytest.mark.parametrize("tier,steps", [("easy", 1), ("medium", 2)])
def test_vector_matches_the_movements_in_the_wording(tier, steps):
    for q in questions(TOPIC, tier, range(1, 41)):
        assert q.answer_kind == "matrix"
        states, T = parse_transition(q.text)
        assert column_sums_are_one(T, states)
        order = _order(q.text)
        assert set(order) == set(states)
        S = step(T, order, _start(q.text, order), steps)
        assert [row[0] for row in q.answer] == [int(S[n]) for n in order]
        assert all(S[n].denominator == 1 for n in order)


def test_medium_leaves_one_share_as_the_rest():
    for q in questions(TOPIC, "medium", range(1, 21)):
        assert "; the rest" in q.text


def test_hard_vector_after_three_periods():
    for q in questions(TOPIC, "hard", range(1, 41)):
        assert q.answer_kind == "matrix"
        states, T = parse_transition(q.text)
        assert len(states) == 4 and column_sums_are_one(T, states)
        total = int(re.search(r"There are (\d+)", q.text).group(1))
        shares = {
            name: Fraction(int(p), 100) * total
            for p, name in re.findall(r"(\d+)% ([A-Za-z\- ]+?)(?:,| and |\. )", q.text.split("shared")[1])
        }
        assert set(shares) == set(states)
        order = _order(q.text)
        S = step(T, order, shares, 3)
        assert [row[0] for row in q.answer] == [int(S[n]) for n in order]
        assert all(S[n].denominator == 1 for n in order)


def test_hard_needs_more_than_easy():
    easy = {len(names_in(q.text)) for q in questions(TOPIC, "easy", range(1, 21))}
    hard = {len(names_in(q.text)) for q in questions(TOPIC, "hard", range(1, 21))}
    assert easy == {3} and hard == {4}


def test_exam_chain_recomputed_and_followed_through():
    honoured = 0
    for q in questions(CHAIN, "exam", range(1, 41)):
        assert [p.answer_kind for p in q.parts] == ["number"] * 4
        check_question_grading(q)
        check_tagged(q)
        honoured += check_follow_through(q)
        states, T = parse_transition(q.text)
        assert column_sums_are_one(T, states)
        S0 = _start(q.text, states)
        S1, S2 = step(T, states, S0, 1), step(T, states, S0, 2)
        name = next(n for n in names_in(q.parts[1].prompt) if n in states)
        stay = T[name][name]
        assert int(q.parts[0].answer_text) == int(stay * 100)
        assert int(q.parts[1].answer_text) == int(S1[name])
        assert int(q.parts[2].answer_text) == int(S2[name])
        assert int(q.parts[3].answer_text) == abs(int(S2[name] - S0[name]))
    assert honoured > 20
