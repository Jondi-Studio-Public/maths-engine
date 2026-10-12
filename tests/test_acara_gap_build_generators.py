"""Focused contract tests for the first accepted ACARA gap-build items."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_mcq, grade_number, mcq_options
from mathgen.skills import load_skills
from mathgen import make


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("topic", ["triangle_area_base_height", "prime_factorisation_exponent"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_gap_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in range(40):
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


def test_triangle_area_uses_number_grading_and_canonical_answer():
    for difficulty in ("easy", "medium", "hard"):
        for seed in range(20):
            question = make("triangle_area_base_height", seed=seed, difficulty=difficulty)
            assert grade_number(question, question.answer_text).correct
            assert all(not grade_number(question, distractor).correct for distractor in question.distractors)


def test_prime_factorisation_exponent_uses_mcq_canonical_answer():
    for difficulty in ("easy", "medium", "hard"):
        for seed in range(20):
            question = make("prime_factorisation_exponent", seed=seed, difficulty=difficulty)
            options = mcq_options(question)
            assert len(options) == len(set(options))
            correct_index = options.index(question.answer_text)
            assert grade_mcq(question, str(correct_index)).correct
            for index, option in enumerate(options):
                if option != question.answer_text:
                    assert not grade_mcq(question, str(index)).correct


def test_gap_skill_contracts_have_exact_prerequisites_answer_kinds_and_stage_sources():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    triangle = by_id["triangle-area-base-height"]
    prime = by_id["prime-factorisation-exponent"]
    assert triangle.prerequisites == ["rectangle-area", "fraction-of-quantity"]
    assert triangle.answer == "number"
    assert prime.prerequisites == ["prime-factorisation", "evaluate-powers"]
    assert prime.answer == "mcq"
    assert [source.stages for source in triangle.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert [source.stages for source in prime.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
