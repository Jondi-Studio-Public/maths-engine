"""Build tests for `f10-linear-conjecture-testing` (`answer: mcq`): the skill's shape, the shared skill contract
families, the real `mcq` grader, a key that cannot be read off the wording, and the tier ladder."""

from __future__ import annotations

import pytest

from mathgen import make
from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from scripts.measure_tiers import classify_skill
from scripts.skill_contract import FAMILIES

SKILL_ID = "f10-linear-conjecture-testing"
TOPIC = "f10_linear_conjecture"
TIERS = ("easy", "medium", "hard", "exam")
SEEDS = range(150)

_SKILLS = {s.id: s for s in load_skills()[1]}
SKILL = _SKILLS[SKILL_ID]


def _questions(tier):
    return [make(TOPIC, seed=s, difficulty=tier) for s in SEEDS]


def test_skill_shape():
    assert SKILL.status == "active"
    assert SKILL.answer == "mcq"
    assert SKILL.domain == "linear-relations"
    assert [src.difficulty for src in SKILL.sources] == list(TIERS)
    assert SKILL.sources[-1].stages == ["exam"]
    assert all(src.topic == TOPIC for src in SKILL.sources)
    assert all(p in _SKILLS for p in SKILL.prerequisites)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_shared_skill_contract(family):
    FAMILIES[family](SKILL_ID)


@pytest.mark.parametrize("tier", TIERS)
def test_the_real_grader_accepts_only_the_key(tier):
    for q in _questions(tier):
        options = mcq_options(q)
        assert len(options) == 4 and len(set(options)) == 4, q.text
        for index, option in enumerate(options):
            assert grade_mcq(q, str(index)).correct == (option == q.answer_text), (q.text, option)
        assert not grade_mcq(q, "9").correct


@pytest.mark.parametrize("tier", TIERS)
def test_every_distractor_is_tagged_with_a_declared_note(tier):
    declared = {m.id for m in SKILL.misconceptions}
    for q in _questions(tier):
        assert set(q.option_notes) == set(q.distractors), q.text
        assert set(q.option_notes.values()) <= declared, q.text


@pytest.mark.parametrize("tier", TIERS)
def test_the_key_position_varies_and_the_wording_does_not_give_the_key_away(tier):
    positions, longest, shortest, supports = set(), 0, 0, 0
    questions = _questions(tier)
    for q in questions:
        options = mcq_options(q)
        positions.add(options.index(q.answer_text))
        lengths = [len(o) for o in options]
        longest += len(q.answer_text) == max(lengths) and lengths.count(max(lengths)) == 1
        shortest += len(q.answer_text) == min(lengths) and lengths.count(min(lengths)) == 1
        supports += q.answer_text.startswith("No ")
    assert positions == {0, 1, 2, 3}
    n = len(questions)
    assert longest < 0.5 * n, "the key is too often the unique longest option"
    assert shortest < 0.5 * n, "the key is too often the unique shortest option"
    assert 0.2 * n < supports < 0.65 * n, "both kinds of key (supported, counterexample) should be common"


@pytest.mark.parametrize("tier", TIERS)
def test_the_prompt_is_text_only_with_the_table_in_the_wording(tier):
    for q in _questions(tier)[:40]:
        assert q.figure is None
        assert "x: " in q.text and "y: " in q.text
        assert "\n" not in q.text


def test_the_tiers_escalate():
    verdict, adds = classify_skill(SKILL, 60)
    assert verdict == "escalates", adds
    assert adds == [False, True, True, True]


def test_questions_are_deterministic():
    for tier in TIERS:
        assert make(TOPIC, seed=5, difficulty=tier).to_json() == make(TOPIC, seed=5, difficulty=tier).to_json()
