"""Focused tests for the third university-foundation generator batch."""

from __future__ import annotations

import pytest

from mathgen.grading import typed as grading
from mathgen import skills as skills_mod
from mathgen import make
from mathgen.core import PRESETS, REGISTRY


PILOT = {
    "matrix_rank_small": "number",
    "directional_derivative": "number",
    "jacobian_two_variable": "mcq",
    "newton_method_iteration": "fraction",
    "composite_trapezoidal_polynomial": "fraction",
}


@pytest.mark.parametrize("topic,answer_kind", PILOT.items())
def test_batch3_topics_are_registered_deterministic_and_gradable(topic, answer_kind):
    assert topic in REGISTRY
    assert set(PRESETS[topic]) == {"easy", "medium", "hard"}
    for seed in range(35):
        first = make(topic, seed=seed)
        second = make(topic, seed=seed)
        assert first.to_json() == second.to_json()
        assert first.answer_text and len(first.distractors) >= 2
        assert first.answer_text not in first.distractors
        assert len(first.distractors) == len(set(first.distractors))
        if answer_kind == "mcq":
            options = grading.mcq_options(first)
            given = options.index(first.answer_text)
        else:
            given = first.answer_text
        assert grading.grade(answer_kind, first, given).correct


@pytest.mark.parametrize("topic", PILOT)
@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard"))
def test_batch3_presets_generate(topic, difficulty):
    for seed in range(10):
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.difficulty == difficulty
        assert question.answer is not None


def test_batch3_skills_have_direct_framework_areas_and_reachable_prerequisites():
    _domains, loaded = skills_mod.load_skills()
    by_id = {skill.id: skill for skill in loaded}
    expected = {
        "matrix-rank-small": ("matrix_rank_small", {"linear-algebra"}),
        "directional-derivative": ("directional_derivative", {"multivariable-and-vector-calculus"}),
        "jacobian-two-variable": ("jacobian_two_variable", {"multivariable-and-vector-calculus"}),
        "newton-method-iteration": ("newton_method_iteration", {"numerical-and-computational-mathematics"}),
        "composite-trapezoidal-polynomial": (
            "composite_trapezoidal_polynomial",
            {"numerical-and-computational-mathematics"},
        ),
    }
    for skill_id, (topic, areas) in expected.items():
        skill = by_id[skill_id]
        assert skill.status == "active"
        assert set(skill.framework_areas) == areas
        assert all(alignment.framework == "university-core-v1" for alignment in skill.alignments)
        assert {alignment.area for alignment in skill.alignments} == areas
        assert all(source.topic == topic for source in skill.sources if source.difficulty != "exam")
        assert all(prerequisite in by_id for prerequisite in skill.prerequisites)


@pytest.mark.parametrize("difficulty,max_rank", (("easy", 2), ("medium", 2), ("hard", 3)))
def test_matrix_rank_sampling_covers_full_and_deficient_cases(difficulty, max_rank):
    ranks = {make("matrix_rank_small", seed=seed, difficulty=difficulty).answer for seed in range(120)}
    assert ranks == set(range(1, max_rank + 1))
    assert max_rank in ranks
    assert any(rank < max_rank for rank in ranks)
