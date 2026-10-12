"""Build tests for `f10-partition-one-two-digit-numbers` (`answer: set`): the skill's shape, the shared skill
contract families, the real `set` grader on every canonical answer and every distractor, and the tier ladder."""

from __future__ import annotations

import pytest

from mathgen import make
from mathgen.grading.typed import grade_set
from mathgen.skills import load_skills
from scripts.measure_tiers import classify_skill
from scripts.skill_contract import FAMILIES

SKILL_ID = "f10-partition-one-two-digit-numbers"
TOPIC = "f10_partition_tens_ones"
TIERS = ("easy", "medium", "hard")
SEEDS = range(80)

_SKILLS = {s.id: s for s in load_skills()[1]}
SKILL = _SKILLS[SKILL_ID]


def _questions(tier):
    return [make(TOPIC, seed=s, difficulty=tier) for s in SEEDS]


def test_skill_shape():
    assert SKILL.status == "active"
    assert SKILL.answer == "set"
    assert SKILL.domain == "early-foundations"
    assert [src.difficulty for src in SKILL.sources] == list(TIERS)
    assert all(src.topic == TOPIC for src in SKILL.sources)
    assert all(p in _SKILLS for p in SKILL.prerequisites)
    # Year 1 content (AC9M1N02): the exam tier is not required for a skill mapped at Year 6 or below.
    assert "exam" not in {src.difficulty for src in SKILL.sources}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_shared_skill_contract(family):
    FAMILIES[family](SKILL_ID)


@pytest.mark.parametrize("tier", TIERS)
def test_the_set_grader_accepts_the_answer_in_any_order(tier):
    for q in _questions(tier):
        assert q.answer_kind == "set"
        assert grade_set(q, q.answer_text).correct, q.text
        reply = "; ".join("(" + ", ".join(str(x) for x in item) + ")" for item in reversed(q.answer))
        assert grade_set(q, reply).correct, (q.text, reply)


@pytest.mark.parametrize("tier", TIERS)
def test_every_distractor_is_rejected_and_tagged(tier):
    declared = {m.id for m in SKILL.misconceptions}
    for q in _questions(tier):
        assert len(q.distractors) >= 2, q.text
        for text in q.distractors:
            assert not grade_set(q, text).correct, (q.text, text)
            assert q.option_notes[text] in declared, (q.text, text)


@pytest.mark.parametrize("tier", TIERS)
def test_a_missing_or_extra_partition_is_rejected(tier):
    for q in _questions(tier):
        items = ["(" + ", ".join(str(x) for x in item) + ")" for item in q.answer]
        assert not grade_set(q, ", ".join(items[:-1])).correct, q.text
        assert not grade_set(q, ", ".join(items + ["(99, 99)"])).correct, q.text


@pytest.mark.parametrize("tier", TIERS)
def test_more_than_one_valid_partition_and_no_repeats(tier):
    for q in _questions(tier):
        assert len(q.answer) >= 2, q.text
        pairs = [tuple(item) for item in q.answer]
        assert len(set(pairs)) == len(pairs), q.text


def test_the_tiers_escalate():
    verdict, adds = classify_skill(SKILL, 60)
    assert verdict == "escalates", adds
    assert adds == [False, True, True]


def test_questions_are_deterministic():
    for tier in TIERS:
        assert make(TOPIC, seed=7, difficulty=tier).to_json() == make(TOPIC, seed=7, difficulty=tier).to_json()
