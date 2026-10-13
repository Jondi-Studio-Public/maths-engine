"""Tests for the multi-part exam source `exd_x_sc7_quantifier_threshold`
(senior-use-the-quantifiers-for-all-and-there-exists).

The oracle finds the vertex, the least value and the integer k from the rendered stem alone, by scanning integers
with exact arithmetic, and never imports the generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc7_quantifier_threshold"
SKILL_ID = "senior-use-the-quantifiers-for-all-and-there-exists"
HARD = ("quantifier_forall_exists_evaluate", {"variant": "select_quantifier"})
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _g(q):
    m = re.search(r"g\(x\) = (\d*)x\^2 ([+-]) (\d+)x,", q.stem)
    a = int(m.group(1) or 1)
    b = int(m.group(3)) * (1 if m.group(2) == "+" else -1)
    return a, b


def oracle(q) -> dict[str, str]:
    a, b = _g(q)
    # Vertex by completing the square, minimum value exactly.
    x0 = Fraction(-b, 2 * a)
    least = a * x0 * x0 + b * x0
    stem = q.stem
    assert least.denominator == 1 and x0.denominator == 1
    least, x0 = int(least), int(x0)
    if "For every real number x, g(x) + k >" in stem:
        # S true  <=>  least + k > 0
        k = next(k for k in range(-5000, 5000) if least + k > 0)
    elif "For every real number x, g(x) + k ≥" in stem:
        k = next(k for k in range(-5000, 5000) if least + k >= 0)
    else:
        assert "There is a real number x for which g(x) + k < 0" in stem
        k = next(k for k in range(5000, -5000, -1) if least + k < 0)
    return {"a": str(x0), "b": str(least), "c": str(k)}


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_prompt_variety_and_new_shapes_over_hard_tier():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[0], seed=s, difficulty="hard", **HARD[1])) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, "the exam source adds no shape beyond the hard tier"


def test_no_text_defects():
    for s in range(60):
        assert find_text_defects(_exam(s)) == [], s


def test_deterministic():
    for s in (0, 1, 7, 99):
        assert _exam(s).to_json() == _exam(s).to_json()


def test_source_is_wired_into_the_skill():
    last = _SKILL.sources[-1]
    assert last.topic == TOPIC and last.difficulty == "exam" and list(last.stages) == ["exam"]
    assert last.params == {"tier": "exam"}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_oracle_agrees_300_seeds(questions):
    bad = [(s, q.text) for s, q in zip(SEEDS, questions) if oracle(q) != _canonical(q)]
    assert bad == []


def test_all_three_statements_occur(questions):
    assert any("> 0" in q.stem for q in questions)
    assert any("≥ 0" in q.stem for q in questions)
    assert any("< 0" in q.stem for q in questions)


def test_answer_space():
    answers = [_exam(s).answer_text for s in SEEDS]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 0.05 * len(answers)


def test_key_accepted_and_distractors_rejected_100_seeds(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) == 3
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def _k_from_least(q, least: int) -> int:
    stem = q.stem
    if "g(x) + k >" in stem:
        return 1 - least
    if "≥" in stem:
        return -least
    return -least - 1


def test_follow_through_from_a_wrong_vertex(questions):
    for q in questions:
        a, b = _g(q)
        x0 = int(_canonical(q)["a"])
        wrong = -x0
        least_wrong = a * wrong * wrong + b * wrong
        k_wrong = _k_from_least(q, least_wrong)
        result = grade(_SKILL, q, {"a": str(wrong), "b": str(least_wrong), "c": str(k_wrong)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks, (q.text, wrong, least_wrong)
        assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, least_wrong, k_wrong)
        assert result.parts["b"].follow_through and result.parts["c"].follow_through
        assert not result.correct


def test_follow_through_from_a_wrong_least_value(questions):
    for q in questions[:100]:
        canon = {k: int(v) for k, v in _canonical(q).items()}
        least_wrong = canon["b"] - 7
        k_wrong = _k_from_least(q, least_wrong)
        result = grade(_SKILL, q, {"a": str(canon["a"]), "b": str(least_wrong), "c": str(k_wrong)})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks and result.parts["c"].follow_through


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": "9999", "b": "7777", "c": "-3333"})
        assert result.marks == 0 and not result.correct
