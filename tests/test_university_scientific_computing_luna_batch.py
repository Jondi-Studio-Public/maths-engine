"""Focused gate for the Luna scientific-computing generator batch."""

from __future__ import annotations

import re

import pytest

from mathgen.grading import typed as grading
from mathgen import skills as skills_mod
from mathgen import make
from mathgen.core import PRESETS, REGISTRY


BATCH = {
    "condition_number_diagonal": "fraction",
    "jacobi_iteration_2x2": "pair",
    "explicit_euler_linear_ode": "fraction",
    "centered_difference_polynomial": "fraction",
    "bisection_interval_update": "pair",
}


@pytest.mark.parametrize("topic,answer_kind", BATCH.items())
def test_topics_are_registered_deterministic_and_gradable(topic, answer_kind):
    assert topic in REGISTRY
    assert set(PRESETS[topic]) == {"easy", "medium", "hard"}
    for seed in range(40):
        first = make(topic, seed=seed)
        second = make(topic, seed=seed)
        assert first.to_json() == second.to_json()
        assert first.answer_text
        assert len(first.distractors) >= 2
        assert len(first.distractors) == len(set(first.distractors))
        assert first.answer_text not in first.distractors
        assert grading.grade(answer_kind, first, first.answer_text).correct


@pytest.mark.parametrize("topic", BATCH)
@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard"))
def test_all_presets_generate(topic, difficulty):
    for seed in range(30):
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.difficulty == difficulty
        assert question.answer is not None


def test_distribution_reaches_distinct_numeric_cases():
    condition_numbers = {make("condition_number_diagonal", seed=s, difficulty="hard").answer_text for s in range(150)}
    euler_values = {make("explicit_euler_linear_ode", seed=s, difficulty="hard").answer_text for s in range(150)}
    derivative_values = {
        make("centered_difference_polynomial", seed=s, difficulty="hard").answer_text for s in range(150)
    }
    assert len(condition_numbers) >= 6
    assert len(euler_values) >= 10
    assert len(derivative_values) >= 10


def test_jacobi_has_nontrivial_pair_components_and_bisection_halves_bracket():
    jacobi = {make("jacobi_iteration_2x2", seed=s, difficulty="hard").answer_text for s in range(120)}
    intervals = {make("bisection_interval_update", seed=s, difficulty="hard").answer_text for s in range(120)}
    assert len(jacobi) >= 20
    assert len(intervals) >= 2
    assert any("/" in item for item in jacobi | intervals)


@pytest.mark.parametrize("seed", (1, 15, 38, 59))
def test_centered_difference_uses_unit_coefficient_sign_format(seed):
    question = make("centered_difference_polynomial", seed=seed)
    for field in (question.text, question.latex, *question.working):
        assert not re.search(r"(?<![\d.])-?1x\b", field)
    if "x^3" in question.latex:
        assert "1x^3" not in question.latex


def test_new_skills_have_reachable_prerequisites_and_candidate_alignments():
    _domains, loaded = skills_mod.load_skills()
    by_id = {skill.id: skill for skill in loaded}
    expected = {
        "condition-number-diagonal": "condition_number_diagonal",
        "jacobi-iteration-2x2": "jacobi_iteration_2x2",
        "explicit-euler-linear-ode": "explicit_euler_linear_ode",
        "centered-difference-polynomial": "centered_difference_polynomial",
        "bisection-interval-update": "bisection_interval_update",
    }
    for skill_id, topic in expected.items():
        skill = by_id[skill_id]
        assert skill.status == "active"
        assert skill.framework_areas == ["numerical-and-computational-mathematics"]
        assert all(alignment.status == "candidate" for alignment in skill.alignments)
        assert all(pre in by_id and by_id[pre].status == "active" for pre in skill.prerequisites)
        assert all(source.topic == topic for source in skill.sources if source.difficulty != "exam")
