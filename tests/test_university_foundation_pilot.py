"""Focused contract tests for the university-foundation generator pilot."""

from __future__ import annotations

import json

import pytest

from mathgen.grading import typed as grading
from mathgen import skills as skills_mod
from mathgen import make
from mathgen.core import PRESETS, REGISTRY


PILOT = (
    "propositional_truth_value",
    "finite_set_operations",
    "modular_arithmetic",
)


@pytest.mark.parametrize("topic", PILOT)
def test_pilot_topics_are_registered_and_deterministic(topic):
    assert topic in REGISTRY
    for seed in range(30):
        first = make(topic, seed=seed)
        second = make(topic, seed=seed)
        assert first.to_json() == second.to_json()
        payload = json.loads(first.to_json())
        assert payload["topic"] == topic
        assert first.text and first.answer_text and len(first.distractors) >= 2
        assert first.answer_text not in first.distractors
        assert len(first.distractors) == len(set(first.distractors))


@pytest.mark.parametrize("topic", PILOT)
@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard"))
def test_pilot_presets_generate(topic, difficulty):
    assert difficulty in PRESETS[topic]
    for seed in range(10):
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.difficulty == difficulty
        assert question.answer is not None


def test_pilot_answers_are_accepted_by_existing_graders():
    for seed in range(40):
        proposition = make("propositional_truth_value", seed=seed)
        options = grading.mcq_options(proposition)
        correct_index = options.index(proposition.answer_text)
        assert grading.grade("mcq", proposition, str(correct_index)).correct

        sets = make("finite_set_operations", seed=seed)
        assert grading.grade("list", sets, sets.answer_text).correct

        modular = make("modular_arithmetic", seed=seed)
        assert grading.grade("number", modular, modular.answer_text).correct


def test_pilot_skills_are_active_and_have_provenance_metadata():
    _domains, loaded = skills_mod.load_skills()
    by_id = {skill.id: skill for skill in loaded}
    expected = {
        "propositional-truth-evaluation": "propositional_truth_value",
        "finite-set-operations": "finite_set_operations",
        "modular-arithmetic-congruence": "modular_arithmetic",
    }
    for skill_id, topic in expected.items():
        skill = by_id[skill_id]
        assert skill.status == "active"
        assert skill.sources and all(source.topic == topic for source in skill.sources if source.difficulty != "exam")
        assert skill.framework_areas
        assert len(skill.alignments) >= 2
        assert all(alignment.status == "candidate" for alignment in skill.alignments)
        assert all(alignment.framework == "university-core-v1" for alignment in skill.alignments)
        assert all(alignment.area in skill.framework_areas for alignment in skill.alignments)


def test_pilot_skills_pass_live_loader_validation():
    domains, loaded = skills_mod.load_skills()
    report = skills_mod.validate(domains, loaded)
    assert report.ok, report.errors
