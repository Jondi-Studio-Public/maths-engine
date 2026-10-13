"""Tests for the multi-part exam source `exd_x_sc7_implication_counts`
(senior-use-implication-converse-equivalence-negation-contrapositive).

The oracle counts the integers by brute force from the rendered stem alone and never imports the generator module.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc7_implication_counts"
SKILL_ID = "senior-use-implication-converse-equivalence-negation-contrapositive"
HARD = ("implication_transform_identify", {"variant": "negation_truth"})
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _predicate(text: str):
    if m := re.fullmatch(r"n is divisible by (\d+)", text):
        k = int(m.group(1))
        return lambda n: n % k == 0
    if m := re.fullmatch(r"n is a multiple of (\d+)", text):
        k = int(m.group(1))
        return lambda n: n % k == 0
    if m := re.fullmatch(r"n is greater than (\d+)", text):
        k = int(m.group(1))
        return lambda n: n > k
    if m := re.fullmatch(r"n is less than (\d+)", text):
        k = int(m.group(1))
        return lambda n: n < k
    if text == "n is a perfect square":
        return lambda n: round(n**0.5) ** 2 == n
    raise AssertionError(text)


def _statements(q):
    top = int(re.search(r"from 1 to (\d+)", q.stem).group(1))
    p_text, q_text = re.findall("\u201c([^\u201d]+)\u201d", q.stem)
    return top, _predicate(p_text), _predicate(q_text)


def oracle(q) -> dict[str, str]:
    top, p, r = _statements(q)
    nums = range(1, top + 1)
    a = sum(1 for n in nums if p(n) and not r(n))
    b = sum(1 for n in nums if r(n) and not p(n))
    c = sum(1 for n in nums if p(n) != r(n))
    return {"a": str(a), "b": str(b), "c": str(c)}


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
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_oracle_agrees_300_seeds(questions):
    bad = [(s, q.text) for s, q in zip(SEEDS, questions) if oracle(q) != _canonical(q)]
    assert bad == []


def test_every_statement_kind_occurs(questions):
    for kind in ("divisible by", "a multiple of", "greater than", "less than", "a perfect square"):
        assert any(kind in q.stem for q in questions), kind


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


def test_follow_through_from_a_wrong_part_a(questions):
    checked = 0
    for q in questions:
        canon = {k: int(v) for k, v in _canonical(q).items()}
        wrong_a = canon["a"] + 3
        c_wrong = wrong_a + canon["b"]
        result = grade(_SKILL, q, {"a": str(wrong_a), "b": str(canon["b"]), "c": str(c_wrong)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks
        assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, wrong_a, c_wrong)
        assert result.parts["c"].follow_through is True
        assert not result.correct
        checked += 1
    assert checked == len(SEEDS)


def test_follow_through_from_a_wrong_part_b(questions):
    for q in questions[:100]:
        canon = {k: int(v) for k, v in _canonical(q).items()}
        wrong_b = canon["b"] + 2
        result = grade(_SKILL, q, {"a": str(canon["a"]), "b": str(wrong_b), "c": str(canon["a"] + wrong_b)})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks and result.parts["c"].follow_through


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": "9999", "b": "-7", "c": "-3"})
        assert result.marks == 0 and not result.correct
