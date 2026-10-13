"""Production checks for curriculum expansion Batch 1."""

from pathlib import Path

from mathgen.grading import typed as grading
from mathgen import skills as skills_mod
from mathgen import make


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SKILLS = REPO_ROOT / "skills"

PRIMARY_IDS = {
    "fraction-of-quantity",
    "compare-common-fractions",
    "decimal-place-value",
    "round-decimals",
    "fraction-decimal-percent",
    "percentage-of-quantity",
    "percentage-discount",
}

MEASUREMENT_IDS = {
    "convert-metric-length",
    "convert-metric-mass",
    "convert-metric-capacity",
    "rectangle-perimeter",
    "rectangle-area",
}


def test_batch_one_skills_are_active_and_ordered_before_readiness():
    domains, skills = skills_mod.load_skills(PRODUCTION_SKILLS)
    by_id = {skill.id: skill for skill in skills}
    domain_ids = [domain.id for domain in domains]

    assert domain_ids[:3] == ["early-foundations", "primary-number", "measurement"]
    assert domain_ids.index("readiness") > domain_ids.index("measurement")
    assert PRIMARY_IDS | MEASUREMENT_IDS <= set(by_id)
    for skill_id in PRIMARY_IDS | MEASUREMENT_IDS:
        skill = by_id[skill_id]
        assert skill.status == "active"
        assert skill.learner_band == "primary"
        assert (skill.placement_questions, skill.placement_pass) == (3, 2)


def test_batch_one_prerequisites_branch_instead_of_forming_one_chain():
    _domains, skills = skills_mod.load_skills(PRODUCTION_SKILLS)
    by_id = {skill.id: skill for skill in skills}

    assert by_id["fraction-of-quantity"].prerequisites == ["whole-number-division-facts"]
    assert by_id["compare-common-fractions"].prerequisites == ["whole-number-division-facts"]
    assert set(by_id["fraction-decimal-percent"].prerequisites) == {"fraction-of-quantity", "decimal-place-value"}
    assert by_id["rectangle-perimeter"].prerequisites == ["add-within-100"]
    assert by_id["rectangle-area"].prerequisites == ["multiplication-facts"]


def test_batch_one_sources_generate_reproducibly_with_answers_and_distractors():
    _domains, skills = skills_mod.load_skills(PRODUCTION_SKILLS)
    batch = [skill for skill in skills if skill.id in PRIMARY_IDS | MEASUREMENT_IDS]

    for skill in batch:
        for source in skill.sources:
            if source.difficulty == "exam":
                continue
            params = {key: tuple(value) if isinstance(value, list) else value for key, value in source.params.items()}
            for seed in range(1, 11):
                first = make(source.topic, seed=seed, difficulty=source.difficulty, **params)
                second = make(source.topic, seed=seed, difficulty=source.difficulty, **params)
                assert first.to_json() == second.to_json()
                assert first.answer_text
                assert len(first.distractors) >= 2
                if skill.answer == "mcq":
                    correct_index = grading.mcq_options(first).index(first.answer_text)
                    given = str(correct_index)
                else:
                    given = first.answer_text
                assert grading.grade(skill.answer, first, given).correct
