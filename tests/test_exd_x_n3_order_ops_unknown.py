"""Exam source of `order-of-operations`: evaluate E, then solve for n so a second expression equals E."""

from __future__ import annotations

import re

import sympy

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n3_order_ops_unknown"
SKILL_ID = "order-of-operations"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _eval(text):
    clean = text.replace("×", "*").replace("÷", "/").replace("−", "-").replace("²", "**2")
    return sympy.sympify(clean)


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages) for s in exam] == [(TOPIC, ["exam"])]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("order_of_operations", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        expr = re.search(r"E = (.*)\.$", q.stem).group(1)
        e_value = _eval(expr)
        p, qq, r = map(int, re.search(r"(\d+) × \(n \+ (\d+)\) − (\d+) has", q.text).groups())
        n = sympy.Symbol("n")
        sol = sympy.solve(sympy.Eq(p * (n + qq) - r, e_value), n)
        assert [q.parts[0].answer_text, q.parts[1].answer_text] == [str(e_value), str(sol[0])], (seed, q.text)


def test_grader_accepts_key_and_rejects_distractors():
    for seed in range(120):
        q = _exam(seed)
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert p.distractors
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (seed, p.label, d)


def test_follow_through_from_wrong_part_a():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        wrong = int(q.parts[0].distractors[0])
        p, qq, r = map(int, re.search(r"(\d+) × \(n \+ (\d+)\) − (\d+) has", q.text).groups())
        if (wrong + r) % p:
            continue
        n_wrong = (wrong + r) // p - qq
        if n_wrong < 1 or n_wrong == wrong:
            continue
        result = grade(_SKILL, q, {"a": str(wrong), "b": str(n_wrong)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks and result.parts["b"].follow_through
        checked += 1
    assert checked >= 10


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
