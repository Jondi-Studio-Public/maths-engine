"""Focused contract tests for the five-topic statistics/learning batch."""

from __future__ import annotations

import pytest

from mathgen.grading import typed as grading
from mathgen import skills as skills_mod
from mathgen import make
from mathgen.core import PRESETS, REGISTRY


PILOT = {
    "hypergeometric_probability": "fraction",
    "finite_distribution_variance": "fraction",
    "bernoulli_mle": "fraction",
    "least_squares_line": "pair",
    "gradient_descent_quadratic": "fraction",
}


@pytest.mark.parametrize("topic,answer_kind", PILOT.items())
def test_stat_learning_topics_are_deterministic_and_gradable(topic, answer_kind):
    assert topic in REGISTRY
    assert set(PRESETS[topic]) == {"easy", "medium", "hard"}
    for seed in range(70):
        first = make(topic, seed=seed)
        second = make(topic, seed=seed)
        assert first.to_json() == second.to_json()
        assert first.answer_text and first.answer_text not in first.distractors
        assert len(first.distractors) >= 2
        assert len(first.distractors) == len(set(first.distractors))
        assert grading.grade(answer_kind, first, first.answer_text).correct


@pytest.mark.parametrize("topic", PILOT)
@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard"))
def test_stat_learning_presets_generate(topic, difficulty):
    for seed in range(20):
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.difficulty == difficulty
        assert question.answer is not None


def test_stat_learning_skills_are_active_and_reachable():
    _domains, loaded = skills_mod.load_skills()
    by_id = {skill.id: skill for skill in loaded}
    expected = {
        "hypergeometric-probability": ("hypergeometric_probability", {"probability", "statistics"}),
        "finite-distribution-variance": ("finite_distribution_variance", {"probability", "statistics"}),
        "bernoulli-mle": ("bernoulli_mle", {"statistics"}),
        "least-squares-line": ("least_squares_line", {"statistics", "optimisation"}),
        "gradient-descent-quadratic": ("gradient_descent_quadratic", {"optimisation", "statistical-learning"}),
    }
    for skill_id, (topic, areas) in expected.items():
        skill = by_id[skill_id]
        assert skill.status == "active"
        assert set(skill.framework_areas) == areas
        assert all(source.topic == topic for source in skill.sources if source.difficulty != "exam")
        assert all(alignment.framework == "university-core-v1" for alignment in skill.alignments)
        assert {alignment.area for alignment in skill.alignments} == areas
        assert all(
            prerequisite in by_id and by_id[prerequisite].status == "active" for prerequisite in skill.prerequisites
        )


def test_stat_learning_skills_pass_structural_validation():
    domains, loaded = skills_mod.load_skills()
    report = skills_mod.validate(domains, loaded)
    assert report.ok, report.errors


@pytest.mark.parametrize("topic", PILOT)
def test_stat_learning_questions_round_trip_to_json(topic):
    import json

    for seed in range(12):
        payload = json.loads(make(topic, seed=seed).to_json())
        assert payload["topic"] == topic
        assert payload["answer"] is not None
