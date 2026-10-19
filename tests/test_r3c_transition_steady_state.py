"""Tests for `transition-matrix-steady-state`: the long-run shares of a three-state model, solved independently."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

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
    top_share,
)

TOPIC = "r3c_transition_steady_state"
CHAIN = "r3c_steady_state_chain"


def _long_run(T, states, unknown=None):
    """The solution of T s = s with the shares adding to 1, by SymPy's null space."""
    n = len(states)
    M = sympy.Matrix(
        n, n, lambda i, j: sympy.Rational(T[states[i]][states[j]].numerator, T[states[i]][states[j]].denominator)
    )
    (vector,) = (M - sympy.eye(n)).nullspace()
    vector = vector / sum(vector)
    return {s: Fraction(int(v.p), int(v.q)) for s, v in zip(states, vector)}


@pytest.mark.parametrize("tier", TIERS)
def test_answers_grade_correct_and_distractors_wrong(tier):
    for q in questions(TOPIC, tier):
        check_question_grading(q)
        check_tagged(q)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_varied(tier):
    assert make(TOPIC, seed=3, difficulty=tier).to_json() == make(TOPIC, seed=3, difficulty=tier).to_json()
    assert distinct_texts(TOPIC, tier) >= 40


def test_easy_share_matches_the_nullspace():
    for q in questions(TOPIC, "easy", range(1, 41)):
        states, T = parse_transition(q.text)
        assert column_sums_are_one(T, states)
        s = _long_run(T, states)
        question = q.text.split("What fraction")[1]
        (target,) = [n for n in names_in(question) if n in states]
        assert q.answer_kind == "fraction"
        assert Fraction(q.answer_text) == s[target]


def test_medium_two_groups_together():
    for q in questions(TOPIC, "medium", range(1, 41)):
        assert q.answer_kind == "fraction"
        states, T = parse_transition(q.text)
        s = _long_run(T, states)
        question = q.text.split("What fraction")[1]
        asked = [n for n in names_in(question) if n in states]
        assert len(asked) == 2
        assert Fraction(q.answer_text) == sum(s[n] for n in asked)
        assert "; the rest" in q.text


def test_hard_unknown_share_is_found_then_used():
    x = sympy.Symbol("x")
    for q in questions(TOPIC, "hard", range(1, 31)):
        assert q.answer_kind == "fraction"
        assert "x%" in q.text
        sentence = next(s for s in re.findall(r"Of the \w+ [^.]*\.", q.text) if "x%" in s)
        states, T0 = parse_transition(q.text, unknown=Fraction(1, 10))
        source = next(n for n in states if n in sentence.split(",")[0])
        dest = next(n for n in states if n in sentence.split("x%")[1] and n != source)
        stated = sympy.Rational(re.search(r"In the long run (\d+/\d+) of", q.text).group(1))
        stated_state = [
            n for n in names_in(q.text.split("In the long run")[1].split("What fraction")[0]) if n in states
        ][0]
        asked_state = [n for n in names_in(q.text.split("What fraction")[1]) if n in states][0]
        T = {d: {s: sympy.Rational(T0[d][s].numerator, T0[d][s].denominator) for s in states} for d in states}
        old = T[dest][source]
        T[dest][source] = x
        T[source][source] = T[source][source] + old - x
        M = sympy.Matrix(3, 3, lambda i, j: T[states[i]][states[j]]) - sympy.eye(3)
        M[2, :] = sympy.Matrix([[1, 1, 1]])
        sol = M.LUsolve(sympy.Matrix([0, 0, 1]))
        roots = [r for r in sympy.solve(sympy.Eq(sol[states.index(stated_state)], stated), x) if 0 < r < 1]
        assert len(roots) == 1
        assert sympy.Rational(q.answer_text) == sol[states.index(asked_state)].subs(x, roots[0])


def test_exam_chain_recomputed_and_followed_through():
    honoured = 0
    for q in questions(CHAIN, "exam", range(1, 41)):
        assert [p.answer_kind for p in q.parts] == ["fraction", "fraction", "fraction", "number"]
        check_question_grading(q)
        check_tagged(q)
        honoured += check_follow_through(q)
        states, T = parse_transition(q.text)
        s = _long_run(T, states)
        order = [next(n for n in names_in(p.prompt) if n in states) for p in q.parts[:3]]
        assert [Fraction(p.answer_text) for p in q.parts[:3]] == [s[n] for n in order]
        assert sum(Fraction(p.answer_text) for p in q.parts[:3]) == 1
        total = int(re.search(r"There are (\d+)", q.text).group(1))
        assert int(q.parts[3].answer_text) == s[order[2]] * total
    assert honoured > 10
