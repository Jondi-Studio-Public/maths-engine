"""Exam source of `f10-exponent-laws-positive-zero`: powers of different bases over a common base."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n3_common_base_powers"
SKILL_ID = "f10-exponent-laws-positive-zero"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _terms(q):
    text = re.search(r"N = (.*)\.$", q.stem).group(1)
    value = Fraction(1)
    tokens = re.split(r" ([×÷]) ", text)
    value = Fraction(1)
    op = "×"
    for tok in tokens:
        if tok in "×÷":
            op = tok
            continue
        if "^" in tok:
            base, exp = map(int, tok.split("^"))
            factor = Fraction(base) ** exp
        else:
            factor = Fraction(int(tok))
        value = value * factor if op == "×" else value / factor
    return value


def _form(q):
    c, b = map(int, re.search(r"form (\d+) × (\d+)\^k", q.parts[0].prompt).groups())
    return c, b


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages) for s in exam] == [(TOPIC, ["exam"])]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("exponent_laws_positive_zero", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        value = _terms(q)
        c, b = _form(q)
        k = 0
        rest = value / c
        while rest > 1:
            rest /= b
            k += 1
        assert rest == 1, (seed, q.stem)
        assert [int(p.answer_text) for p in q.parts] == [k, int(value)], (seed, q.text)


def test_grader_accepts_key_and_rejects_distractors():
    for seed in range(120):
        q = _exam(seed)
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 4
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert p.distractors
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (seed, p.label, d)


def test_follow_through_from_wrong_index():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        c, b = _form(q)
        wrong = int(q.parts[0].distractors[0])
        b_wrong = c * b**wrong
        result = grade(_SKILL, q, {"a": str(wrong), "b": str(b_wrong)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks and result.parts["b"].follow_through
        checked += 1
    assert checked == 120


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
