"""The difference-quotient co-located pair.

Row 1: senior-examine-the-behaviour-of-the-difference-quotient-frac (ACMMM081,
topic p9c_difference_quotient_behaviour, answer: number). Row 2:
senior-use-the-notation-frac-delta-y-delta-x (ACMMM079, topic
p9c_delta_y_delta_x_notation, answer: fraction). See the generator module
docstring for the one-sentence boundary test and the ACMMM082 boundary.
"""

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_fraction, grade_number
from mathgen.skills import load_skills
from mathgen import make

SEEDS = range(60)
TIERS = ("easy", "medium", "hard")

TOPIC_1 = "p9c_difference_quotient_behaviour"
SKILL_1 = "senior-examine-the-behaviour-of-the-difference-quotient-frac"
TOPIC_2 = "p9c_delta_y_delta_x_notation"
SKILL_2 = "senior-use-the-notation-frac-delta-y-delta-x"


# --- Determinism -------------------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_generator_is_deterministic(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        repeat = make(topic, seed=seed, difficulty=tier)
        assert q.to_json() == repeat.to_json()


# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors, f"seed={seed}: answer in distractors"
        assert len(q.distractors) == len(set(q.distractors)), f"seed={seed}: dup distractors"
        assert len(q.distractors) >= 1


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("tier", TIERS)
def test_row1_real_grader_accepts_and_rejects(tier):
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grade_number(q, d)
            assert not result.correct, f"seed={seed}: distractor {d!r} accepted"
        wrong = str(q.answer + 1)
        result = grade_number(q, wrong)
        assert not result.correct, f"seed={seed}: deliberately wrong {wrong!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_row2_real_grader_accepts_and_rejects(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        result = grade_fraction(q, q.answer_text)
        assert result.correct, f"seed={seed}: canonical not accepted"
        for d in q.distractors:
            result = grade_fraction(q, d)
            assert not result.correct, f"seed={seed}: distractor {d!r} accepted"
        answer_value = Fraction(q.answer["p"], q.answer["q"])
        wrong = str(answer_value + 1)
        result = grade_fraction(q, wrong)
        assert not result.correct, f"seed={seed}: deliberately wrong {wrong!r} accepted"


# --- Answer kind -----------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_row1_answer_is_bare_integer(tier):
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        assert isinstance(q.answer, int)
        assert "." not in q.answer_text, f"seed={seed}: answer_text {q.answer_text!r} has a decimal point"


@pytest.mark.parametrize("tier", TIERS)
def test_row2_answer_is_exact_reduced_fraction(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        p, den = q.answer["p"], q.answer["q"]
        assert den > 0
        frac = Fraction(p, den)
        assert frac.numerator == p and frac.denominator == den, (
            f"seed={seed}: answer {{'p': {p}, 'q': {den}}} is not already in lowest terms"
        )


# --- Formatted alike --------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_row1_non_decimal_distractors_share_answer_shape(tier):
    """The three int-valued distractors (sign-error, forgot-power-multiply / dropped-linear-term) must look like
    the bare-integer answer; only the structurally-required reads-off-least-converged-value distractor is a
    decimal."""
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        non_decimal = [d for d in q.distractors if "." not in d]
        for d in non_decimal:
            assert d.lstrip("-").isdigit(), f"seed={seed}: {d!r} not shaped like the integer answer"


@pytest.mark.parametrize("tier", TIERS)
def test_row2_distractors_are_all_fraction_shaped(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        for d in q.distractors:
            # every candidate is produced by rational_text: an optional
            # leading '-', digits, and an optional '/digits'
            body = d.lstrip("-")
            parts = body.split("/")
            assert 1 <= len(parts) <= 2
            assert all(part.isdigit() for part in parts), f"seed={seed}: {d!r} not fraction-shaped"


# --- Distinct rendered questions -------------------------------------------


@pytest.mark.parametrize("topic", [TOPIC_1, TOPIC_2])
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(topic, tier):
    texts = {make(topic, seed=seed, difficulty=tier).text for seed in SEEDS}
    assert len(texts) >= 20, f"{topic}/{tier}: only {len(texts)} distinct questions over 60 seeds"


# --- Independent-check tripwire ---------------------------------------------


def test_row1_independent_check_fires_on_corrupted_answer():
    from mathgen.core import VerificationError, check_equivalent

    with pytest.raises(VerificationError):
        check_equivalent(5, 6, "should fail")


def test_row2_independent_check_fires_on_corrupted_answer():
    from mathgen.core import VerificationError, check_equivalent

    with pytest.raises(VerificationError):
        check_equivalent(Fraction(1, 2), Fraction(1, 3), "should fail")


# --- Skill loading -----------------------------------------------------


def test_skill_1_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_1), None)
    assert skill is not None, f"{SKILL_1} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == []
    assert len(skill.sources) == 3
    for source in skill.sources:
        assert source.topic == TOPIC_1
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}


def test_skill_2_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_2), None)
    assert skill is not None, f"{SKILL_2} not found"
    assert skill.answer == "fraction"
    assert skill.placement_eligible is False
    assert skill.prerequisites == []
    assert len(skill.sources) == 3
    for source in skill.sources:
        assert source.topic == TOPIC_2
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
