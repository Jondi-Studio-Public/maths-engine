"""Tests the skill defining the imaginary number i as a root (`senior-define-the-imaginary-number-i-as-a-root`,
ACMSM067)."""

import re

import pytest

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p7c_imaginary_root_build import _verify_power_cycle, _verify_sqrt

TOPIC = "imaginary_number_i_root"
SEEDS = range(40)
SWEEP_SEEDS = range(60)


# --- Determinism & distractor hygiene ----------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert len(question.distractors) == 3, question.distractors


# --- Real grader (learner-typed strings, not the generator's own check) -----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_rejects_distractors(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        result = grade_expression(question, question.answer_text)
        assert result.correct, f"seed {seed}/{difficulty}: canonical not accepted"
        for distractor in question.distractors:
            result = grade_expression(question, distractor)
            assert not result.correct, f"seed {seed}/{difficulty}: distractor {distractor!r} was accepted"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        wrong = "42"
        assert not grade_expression(question, wrong).correct, (
            f"seed {seed}/{difficulty}: deliberate wrong {wrong!r} accepted"
        )


def test_explicit_multiplication_grades_correct():
    """The specific case that proved the i-notation design: an explicit '*'
    learner input grades correct against an implicit-form canonical, both for
    the medium/hard "-i" family and the easy "-ki" family."""
    found_medium = found_easy = False
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        if q.answer_text == "-i" and not found_medium:
            assert grade_expression(q, "-1*i").correct
            found_medium = True
        q = make(TOPIC, seed=seed, difficulty="easy")
        if q.answer_text not in ("i", "-i") and q.answer_text.endswith("i") and not found_easy:
            k_text = q.answer_text[:-1]
            assert grade_expression(q, f"{k_text}*i").correct, (
                f"seed {seed}: explicit '*' rejected for canonical {q.answer_text!r}"
            )
            found_easy = True
        if found_medium and found_easy:
            break
    assert found_medium, "no medium-tier seed within 60 produced canonical '-i'"
    assert found_easy, "no easy-tier seed within 60 produced a k>1 canonical"


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard"] for s in range(10)])
def test_capital_i_is_rejected_known_limitation(difficulty, seed):
    """Known, accepted limitation: capital 'I' parses to sympy's true
    imaginary unit, never the Symbol('i') the canonical answer is built from,
    so it is always rejected. Asserted so a future change that "fixes" this
    accidentally is caught as a behaviour change, not silently regressed."""
    question = make(TOPIC, seed=seed, difficulty=difficulty)
    capitalised = question.answer_text.replace("i", "I")
    if capitalised == question.answer_text:
        return  # answer had no 'i' to capitalise (should not happen here)
    assert not grade_expression(question, capitalised).correct, (
        f"seed {seed}/{difficulty}: capital-I form {capitalised!r} was wrongly accepted"
    )


# --- Independent-check firing on corrupted answers ---------------------------


def test_power_cycle_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        _verify_power_cycle(7, "1")  # true answer for n=7 is "-i"
    with pytest.raises(VerificationError):
        _verify_power_cycle(-13, "1")  # true answer for n=-13 is "i"
    _verify_power_cycle(7, "-i")  # sanity: the correct answer does not raise


def test_sqrt_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        _verify_sqrt(5, complex(0, 6))  # wrong imaginary coefficient
    with pytest.raises(VerificationError):
        _verify_sqrt(5, complex(5, 0))  # missing the imaginary unit entirely
    _verify_sqrt(5, complex(0, 5))  # sanity: the correct claim does not raise


# --- Pool size and hard-tier sub-range coverage -------------------------------


def test_easy_tier_pool_is_exactly_20_by_construction():
    """k = 1..20 gives exactly 20 distinct radicands/answers -- the
    stated pool size, which clears the 20-distinct-over-60-seeds floor on its
    own. A uniform draw over only 20 slots does not reliably fill all 20 in
    just 60 seeds (the classic coupon-collector gap), so this samples 400
    seeds to confirm the pool is exactly 20 and never more (never an
    off-by-one k range) rather than resampling the 60-seed floor, which the
    other tiers' test already covers for medium/hard."""
    texts = {make(TOPIC, seed=s, difficulty="easy").text for s in range(400)}
    assert len(texts) == 20, texts


def test_hard_tier_covers_both_sub_ranges_within_60_seeds():
    saw_negative = False
    saw_large_positive = False
    for seed in SWEEP_SEEDS:
        question = make(TOPIC, seed=seed, difficulty="hard")
        match = re.search(r"i\^(\(-?\d+\)|-?\d+)", question.text)
        assert match, question.text
        n = int(match.group(1).strip("()"))
        if n < 0:
            saw_negative = True
        elif n >= 25:
            saw_large_positive = True
    assert saw_negative, "no negative-exponent seed seen in 60 hard-tier seeds"
    assert saw_large_positive, "no large-positive-exponent seed seen in 60 hard-tier seeds"


# --- Answer kind and shape -----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_and_distractors_formatted_alike(difficulty):
    """No option should be identifiable purely by shape (e.g. only one option
    carrying an 'i')."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        options = [question.answer_text] + list(question.distractors)
        has_i = [("i" in opt) for opt in options]
        # At least one distractor must share the answer's "has i" shape so the
        # answer is not the unique option of its kind.
        assert has_i.count(has_i[0]) >= 2, options


# --- Skill loading -----


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-define-the-imaginary-number-i-as-a-root"), None)
    assert skill is not None
    assert skill.answer == "expression"
    assert skill.placement_eligible is False
    assert skill.prerequisites == []


def test_skill_sources_cover_correct_stages():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == "senior-define-the-imaginary-number-i-as-a-root"), None)
    assert skill is not None
    assert len(skill.sources) == 4
    for source in skill.sources[:3]:
        assert source.topic == TOPIC
    assert set(skill.sources[0].stages) == {0, 1, 2}
    assert set(skill.sources[1].stages) == {3, 4}
    assert set(skill.sources[2].stages) == {5, 6, "maintenance"}
    assert skill.sources[3].difficulty == "exam"
    assert set(skill.sources[3].stages) == {"exam"}
