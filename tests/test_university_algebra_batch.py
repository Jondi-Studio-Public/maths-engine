"""Focused contract tests for the Luna university-algebra batch.

The tests deliberately exercise independent generation, deterministic replay,
all difficulty presets, existing answer graders, and the proposed mapping
fragment without silently merging that proposal into the canonical map.
"""

from __future__ import annotations

import yaml
import pytest

from mathgen.grading import typed as grading
from mathgen import skills as skills_mod
from mathgen import make
from mathgen.core import PRESETS, REGISTRY


BATCH = {
    "permutation_composition": "list",
    "modular_multiplicative_inverse": "number",
    "cyclic_group_element_order": "number",
    "integer_polynomial_quotient_remainder": "mcq",
    "propositional_formula_equivalence": "mcq",
}


@pytest.mark.parametrize("topic,answer_kind", BATCH.items())
def test_topics_are_registered_deterministic_and_gradable(topic, answer_kind):
    assert topic in REGISTRY
    assert set(PRESETS[topic]) == {"easy", "medium", "hard"}
    for seed in range(45):
        first = make(topic, seed=seed)
        second = make(topic, seed=seed)
        assert first.to_json() == second.to_json()
        assert first.answer_text
        assert len(first.distractors) >= 2
        assert first.answer_text not in first.distractors
        assert len(first.distractors) == len(set(first.distractors))
        if answer_kind == "mcq":
            given = str(grading.mcq_options(first).index(first.answer_text))
        else:
            given = first.answer_text
        assert grading.grade(answer_kind, first, given).correct


@pytest.mark.parametrize("topic", BATCH)
@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard"))
def test_each_preset_generates(topic, difficulty):
    for seed in range(12):
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.difficulty == difficulty
        assert question.answer is not None


def test_equivalence_batch_contains_both_truth_outcomes():
    # The "Not equivalent" answers now say why (one implies the other, or neither), so compare the verdict.
    answers = {make("propositional_formula_equivalence", seed=seed).answer_text for seed in range(80)}
    assert {answer.split(":")[0] for answer in answers} == {"Equivalent", "Not equivalent"}


def test_skills_are_active_with_direct_framework_alignment_and_reachable_prereqs():
    _domains, loaded = skills_mod.load_skills()
    by_id = {skill.id: skill for skill in loaded}
    expected = {
        "permutation-composition": ("permutation_composition", {"abstract-algebra"}),
        "modular-multiplicative-inverse": ("modular_multiplicative_inverse", {"number-theory"}),
        "cyclic-group-element-order": ("cyclic_group_element_order", {"abstract-algebra"}),
        "integer-polynomial-quotient-remainder": ("integer_polynomial_quotient_remainder", {"abstract-algebra"}),
        "propositional-formula-equivalence": (
            "propositional_formula_equivalence",
            {"logic-computability-and-information"},
        ),
    }
    for skill_id, (topic, areas) in expected.items():
        skill = by_id[skill_id]
        assert skill.status == "active"
        assert all(prerequisite in by_id for prerequisite in skill.prerequisites)
        assert skill.sources and all(source.topic == topic for source in skill.sources if source.difficulty != "exam")
        assert set(skill.framework_areas) == areas
        assert {alignment.area for alignment in skill.alignments} == areas
        assert all(alignment.framework == "university-core-v1" for alignment in skill.alignments)
        assert all(alignment.status == "candidate" for alignment in skill.alignments)
