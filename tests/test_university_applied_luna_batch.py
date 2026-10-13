"""Focused contract tests for the bounded university applied batch."""

from __future__ import annotations

import pytest

from mathgen.grading import typed as grading
from mathgen import skills as skills_mod
from mathgen import make
from mathgen.core import PRESETS, REGISTRY


PILOT = {
    "convex_quadratic_minimum": "pair",
    "lagrange_constrained_point": "pair",
    "markov_two_state_step": "fraction",
    "markov_two_state_stationary": "fraction",
    "finite_joint_covariance": "fraction",
}


@pytest.mark.parametrize("topic,answer_kind", PILOT.items())
def test_luna_batch_is_deterministic_and_gradable(topic, answer_kind):
    assert topic in REGISTRY
    assert set(PRESETS[topic]) == {"easy", "medium", "hard"}
    for seed in range(50):
        first = make(topic, seed=seed)
        second = make(topic, seed=seed)
        assert first.to_json() == second.to_json()
        assert first.answer_text and first.answer_text not in first.distractors
        assert len(first.distractors) >= 2
        assert len(first.distractors) == len(set(first.distractors))
        assert grading.grade(answer_kind, first, first.answer_text).correct


@pytest.mark.parametrize("topic", PILOT)
@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard"))
def test_luna_batch_presets_generate(topic, difficulty):
    for seed in range(12):
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.difficulty == difficulty
        assert question.answer is not None


def test_luna_batch_skill_manifests_are_active_and_reachable():
    _domains, loaded = skills_mod.load_skills()
    by_id = {skill.id: skill for skill in loaded}
    expected = {
        "convex-quadratic-minimum": ("convex_quadratic_minimum", {"optimisation"}),
        "lagrange-constrained-point": (
            "lagrange_constrained_point",
            {"multivariable-and-vector-calculus", "optimisation"},
        ),
        "markov-two-state-step": ("markov_two_state_step", {"probability", "stochastic-processes"}),
        "markov-two-state-stationary": ("markov_two_state_stationary", {"probability", "stochastic-processes"}),
        "finite-joint-covariance": ("finite_joint_covariance", {"probability", "statistics"}),
    }
    for skill_id, (topic, areas) in expected.items():
        skill = by_id[skill_id]
        assert skill.status == "active"
        assert set(skill.framework_areas) == areas
        assert all(source.topic == topic for source in skill.sources if source.difficulty != "exam")
        assert all(alignment.framework == "university-core-v1" for alignment in skill.alignments)
        assert {alignment.area for alignment in skill.alignments} == areas or {
            alignment.area for alignment in skill.alignments
        } <= areas
        assert all(prerequisite in by_id for prerequisite in skill.prerequisites)


def test_luna_batch_passes_structural_skill_validation():
    domains, loaded = skills_mod.load_skills()
    report = skills_mod.validate(domains, loaded)
    assert report.ok, report.errors
