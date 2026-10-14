"""Build tests for `ucv1-weak-solution-regularity` (`answer: mcq`): the skill's shape, the shared skill contract
families, the real `mcq` grader, a key that cannot be read off the wording, and the tier ladder."""

from __future__ import annotations

import pytest

from mathgen import make
from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from scripts.measure_tiers import classify_skill
from scripts.skill_contract import FAMILIES

SKILL_ID = "ucv1-weak-solution-regularity"
TOPIC = "ucv1_weak_solution_regularity"
TIERS = ("easy", "medium", "hard", "exam")
SEEDS = range(150)

_SKILLS = {s.id: s for s in load_skills()[1]}
SKILL = _SKILLS[SKILL_ID]


def _questions(tier):
    return [make(TOPIC, seed=s, difficulty=tier) for s in SEEDS]


def test_skill_shape():
    assert SKILL.status == "active"
    assert SKILL.answer == "mcq"
    assert SKILL.domain == "advanced-pdes-and-dynamics"
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
            assert grade_mcq(q, index).correct == (option == q.answer_text), (q.text, option)


@pytest.mark.parametrize("tier", TIERS)
def test_every_distractor_is_tagged_with_a_declared_note(tier):
    declared = {m.id for m in SKILL.misconceptions}
    for q in _questions(tier):
        assert set(q.option_notes) == set(q.distractors), q.text
        assert set(q.option_notes.values()) <= declared, q.text


@pytest.mark.parametrize("tier", TIERS)
def test_the_key_position_varies_and_is_not_the_middle_or_the_longest_by_design(tier):
    positions, longest, orders = set(), 0, []
    for q in _questions(tier):
        options = mcq_options(q)
        positions.add(options.index(q.answer_text))
        lengths = [len(o) for o in options]
        longest += len(q.answer_text) == max(lengths) and lengths.count(max(lengths)) == 1
        orders.append(sorted(options).index(q.answer_text))
    assert positions == {0, 1, 2, 3}
    assert longest < 0.6 * len(orders)
    # The wrong orders are drawn on both sides of the right one, so the key is sometimes the lowest statement and
    # sometimes the highest.
    assert set(orders) == {0, 1, 2, 3}


def test_the_fourth_order_equation_appears_only_in_the_exam_tier():
    for tier in TIERS:
        has_beam = any("clamped beam" in q.text for q in _questions(tier))
        assert has_beam == (tier == "exam")


def test_the_tiers_escalate():
    verdict, adds = classify_skill(SKILL, 60)
    assert verdict == "escalates", adds
    assert adds == [False, True, True, True]


def test_questions_are_deterministic():
    for tier in TIERS:
        assert make(TOPIC, seed=3, difficulty=tier).to_json() == make(TOPIC, seed=3, difficulty=tier).to_json()


def test_the_latex_prompt_carries_the_sobolev_spaces():
    for tier in TIERS:
        for q in _questions(tier)[:30]:
            assert "H^{" in q.latex and r"\int" in q.latex, q.latex
            assert "₀" not in q.latex and "¹" not in q.latex, q.latex
