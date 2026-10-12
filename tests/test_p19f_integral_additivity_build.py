"""Tests `p19f_integral_additivity_linearity` (ACMMM128): additivity and linearity of the integral, `answer:
number`."""

from collections import Counter

import pytest

from mathgen.grading.typed import GRADERS, grade_number
from mathgen.skills import load_skills
from mathgen import make

SEEDS = range(60)
CONCENTRATION_SEEDS = range(200)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p19f_integral_additivity_linearity"
SKILL = "senior-recognise-and-use-the-additivity-and-linearity-of"


# --- Determinism -------------------------------------------------------


# --- Distractor hygiene --------------------------------------------------


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_and_rejects(tier):
    assert GRADERS["number"] is grade_number
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"seed={seed}: canonical {q.answer_text!r} not accepted"
        for d in q.distractors:
            result = grade_number(q, d)
            assert not result.correct, f"seed={seed}: distractor {d!r} accepted"
        wrong = str(q.answer + 1)
        result = grade_number(q, wrong)
        assert not result.correct, f"seed={seed}: deliberately wrong {wrong!r} accepted"


# --- Answer kind: exact integer, no decimal point --------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_bare_integer(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, int)
        assert "." not in q.answer_text, f"seed={seed}: answer_text {q.answer_text!r} has a decimal point"
        for d in q.distractors:
            assert "." not in d, f"seed={seed}: distractor {d!r} has a decimal point"
            assert d.lstrip("-").isdigit(), f"seed={seed}: distractor {d!r} not integer-shaped"


# --- Distinct rendered questions -------------------------------------------


# --- Answer concentration: top key under 15% of 200 draws per tier ---------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_concentration_under_15_percent(tier):
    counts = Counter(make(TOPIC, seed=seed, difficulty=tier).answer_text for seed in CONCENTRATION_SEEDS)
    top_key, top_count = counts.most_common(1)[0]
    frac = top_count / len(CONCENTRATION_SEEDS)
    assert frac < 0.15, f"{tier}: top answer {top_key!r} is {frac:.1%} of {len(CONCENTRATION_SEEDS)} draws (>= 15%)"


# --- Independent-check tripwire ---------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    from mathgen.core import VerificationError, check

    with pytest.raises(VerificationError):
        check(5 == 6, "should fail")


# --- Skill loading -----------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL), None)
    assert skill is not None, f"{SKILL} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["definite-integral"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert set(skill.sources[3].stages) == {"exam"}
