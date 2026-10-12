"""Cross-batch production checks for curriculum expansion batches 2–6."""

from pathlib import Path

from mathgen.grading import typed as grading
from mathgen import skills as skills_mod
from mathgen import make


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SKILLS = REPO_ROOT / "skills"
EXPECTED_COUNTS = {
    "expansion-batch-2": 8,
    "expansion-batch-3": 8,
    "expansion-batch-4": 8,
    "expansion-batch-5": 8,
    "expansion-batch-6": 12,
}


def _expanded_skills():
    return skills_mod.load_skills(PRODUCTION_SKILLS)[1]


def test_each_remaining_batch_has_the_planned_active_skills():
    skills = _expanded_skills()
    for tag, expected in EXPECTED_COUNTS.items():
        batch = [skill for skill in skills if tag in skill.tags]
        assert len(batch) == expected
        assert all(skill.status == "active" for skill in batch)
        assert all(skill.sources for skill in batch)
        assert all(skill.misconceptions for skill in batch)


def test_every_remaining_batch_source_reproduces_and_self_grades():
    batch_tags = set(EXPECTED_COUNTS)
    skills = [skill for skill in _expanded_skills() if batch_tags.intersection(skill.tags)]
    for skill in skills:
        for source in skill.sources:
            params = {key: tuple(value) if isinstance(value, list) else value for key, value in source.params.items()}
            for seed in range(1, 11):
                question = make(source.topic, seed=seed, difficulty=source.difficulty, **params)
                assert (
                    question.to_json()
                    == make(source.topic, seed=seed, difficulty=source.difficulty, **params).to_json()
                )
                if skill.answer == "mcq":
                    given = str(grading.mcq_options(question).index(question.answer_text))
                else:
                    given = question.answer_text
                assert grading.grade(skill.answer, question, given).correct, (skill.id, seed, question.answer_text)


def test_expansion_graph_contains_real_branches_and_cross_domain_links():
    by_id = {skill.id: skill for skill in _expanded_skills()}
    assert by_id["money-change"].prerequisites == ["australian-coin-values", "subtract-within-100"]
    assert set(by_id["binomial-point-probability"].prerequisites) == {
        "combinations",
        "simple-event-probability",
        "senior-identify-contexts-suitable-for-modelling-by-binomial-random",
    }
    assert set(by_id["complex-modulus"].prerequisites) == {"complex-imaginary-part", "pythagorean-hypotenuse"}
    assert by_id["moving-average"].prerequisites == ["calculate-mean"]
