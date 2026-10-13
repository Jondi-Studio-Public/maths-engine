"""The floating-point error-budget skill (`ucv1-floating-point-error-budget`): loader, validator and grader.

It is a typed number graded to its stated precision, with no rubric. The other skills built on the `crit`
generators are rubric-marked written answers, and those skill files are not part of this package;
`tests/test_crit_build.py` and `tests/test_crit_oracle.py` test their generators directly.
"""

from __future__ import annotations

import pytest

import mathgen
from mathgen import skills as skills_mod
from mathgen.grading import GRADERS

TIERS = ["easy", "medium", "hard", "exam"]
FLOAT_ID = "ucv1-floating-point-error-budget"
FLOAT_TOPIC = "float_error_budget"


@pytest.fixture(scope="module")
def loaded():
    domains, skill_list = skills_mod.load_skills()
    return domains, skill_list, {s.id: s for s in skill_list}


def test_the_floating_point_skill_is_a_normal_number_skill(loaded):
    skill = loaded[2][FLOAT_ID]
    assert skill.status == "active" and skill.answer == "number" and skill.rubric_id is None
    assert skill.capability_gates == [] and skill.placement_eligible is False
    assert skill.domain == "advanced-scientific-computing"
    assert [src.difficulty for src in skill.sources] == TIERS
    assert {src.topic for src in skill.sources} == {FLOAT_TOPIC}
    assert [src.stages for src in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert len(skill.misconceptions) >= 4


def test_the_library_validates_with_no_error_for_the_floating_point_skill(loaded):
    domains, skill_list, _ = loaded
    report = skills_mod.validate(domains, skill_list)
    assert not [e for e in report.errors if e.startswith(f"{FLOAT_ID}:")], report.errors


def test_the_floating_point_questions_grade_as_numbers(loaded):
    skill = loaded[2][FLOAT_ID]
    grade = GRADERS["number"]
    for src in skill.sources:
        for seed in range(25):
            q = mathgen.make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))
            assert grade(q, q.answer_text).correct, (q.text, q.answer_text)
            for wrong in q.distractors:
                assert not grade(q, wrong).correct, (q.text, wrong, q.answer_text)
