"""Exam source of `ucv1-markov-hitting-probability`: a four-part chain by first-step analysis on four states."""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2f_markov_absorption"
SKILL_ID = "ucv1-markov-hitting-probability"
SEEDS = range(300)
H1, H2 = sympy.symbols("h1 h2")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_STEM = re.compile(
    r"from state 1 it moves to state 2 with probability (\S+), to state 0 with probability (\S+), "
    r"and otherwise stays in state 1\. In one step from state 2 it moves to state 3 with probability (\S+), "
    r"to state 1 with probability (\S+), and otherwise stays in state 2\."
)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _frac(value) -> str:
    value = sympy.Rational(value)
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def _probabilities(q):
    return [sympy.Rational(t.rstrip(",")) for t in _STEM.search(q.stem).groups()]


def oracle(q):
    """Write the two first-step equations as stated (stay probability = 1 - the other two) and solve them together."""
    p1, q1, p2, q2 = _probabilities(q)
    s1, s2 = 1 - p1 - q1, 1 - p2 - q2
    sol = sympy.solve([sympy.Eq(H1, p1 * H2 + q1 * 0 + s1 * H1), sympy.Eq(H2, p2 * 1 + q2 * H1 + s2 * H2)], [H1, H2])
    return p1 / (p1 + q1), sol[H2], sol[H1], (sol[H1] + sol[H2]) / 2


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 3, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)
        assert all(len(p.distractors) >= 2 and p.option_notes for p in q.parts)


def test_oracle_agrees(questions):
    for q in questions:
        r1, h2, h1, mean = oracle(q)
        assert _canonical(q) == {"a": _frac(r1), "b": _frac(h2), "c": _frac(h1), "d": _frac(mean)}, q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 8


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_first_move_probability(questions):
    """Using p1 (the stay probability ignored) for (a), the learner's own values carry through (b), (c) and (d)."""
    checked = 0
    for q in questions[:150]:
        p1, q1, p2, q2 = _probabilities(q)
        a_wrong = p1
        r2 = p2 / (p2 + q2)
        b_wrong = r2 / (1 - (1 - r2) * a_wrong)
        c_wrong = a_wrong * b_wrong
        d_wrong = (b_wrong + c_wrong) / 2
        values = [a_wrong, b_wrong, c_wrong, d_wrong]
        if len(set(values)) < 4:  # follow-through is withheld from a repeated reply
            continue
        reply = dict(zip("abcd", map(_frac, values)))
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label, marks in (("b", 3), ("c", 1), ("d", 2)):
            assert result.parts[label].marks == marks, (q.text, label, reply)
            assert result.parts[label].follow_through is True
        assert result.marks == 6 and not result.correct
        checked += 1
    assert checked >= 100


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "97/101", "b": "89/103", "c": "83/107", "d": "79/109"})
        assert result.marks == 0 and not result.correct
