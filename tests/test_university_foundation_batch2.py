"""Focused tests for the second university-foundation generator batch."""

from __future__ import annotations

import pytest

from mathgen.grading import typed as grading
from mathgen import skills as skills_mod
from mathgen import make
from mathgen.core import PRESETS, REGISTRY


PILOT = {
    "vector_projection": "pair",
    "two_by_two_eigenvalues": "list",
    "discrete_expected_value": "fraction",
    "conditional_probability_counts": "fraction",
    "euclidean_algorithm_gcd": "number",
}


@pytest.mark.parametrize("topic,answer_kind", PILOT.items())
def test_batch2_topics_are_registered_deterministic_and_gradable(topic, answer_kind):
    assert topic in REGISTRY
    assert set(PRESETS[topic]) == {"easy", "medium", "hard"}
    for seed in range(35):
        first = make(topic, seed=seed)
        second = make(topic, seed=seed)
        assert first.to_json() == second.to_json()
        assert first.answer_text and len(first.distractors) >= 2
        assert first.answer_text not in first.distractors
        assert len(first.distractors) == len(set(first.distractors))
        assert grading.grade(answer_kind, first, first.answer_text).correct


@pytest.mark.parametrize("topic", PILOT)
@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard"))
def test_batch2_presets_generate(topic, difficulty):
    for seed in range(10):
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.difficulty == difficulty
        assert question.answer is not None


def test_batch2_skills_have_narrow_direct_framework_alignment():
    _domains, loaded = skills_mod.load_skills()
    by_id = {skill.id: skill for skill in loaded}
    expected = {
        "vector-projection": ("vector_projection", {"linear-algebra"}),
        "two-by-two-eigenvalues": ("two_by_two_eigenvalues", {"linear-algebra"}),
        "discrete-expected-value": ("discrete_expected_value", {"probability"}),
        "conditional-probability-counts": ("conditional_probability_counts", {"probability"}),
        "euclidean-algorithm-gcd": ("euclidean_algorithm_gcd", {"number-theory"}),
    }
    for skill_id, (topic, areas) in expected.items():
        skill = by_id[skill_id]
        assert skill.status == "active"
        assert skill.framework_areas == sorted(areas) or set(skill.framework_areas) == areas
        assert all(alignment.framework == "university-core-v1" for alignment in skill.alignments)
        assert {alignment.area for alignment in skill.alignments} == areas
        assert all(source.topic == topic for source in skill.sources if source.difficulty != "exam")
