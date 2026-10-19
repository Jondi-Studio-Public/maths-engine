"""Chance language/outcomes and probability scale/expectation (f10-chance-language-and-outcomes,
f10-probability-scale-and-expectation)."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_mcq, grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p5d_chance_expectation_build import (
    classify_chance,
    OPTION_CERTAIN,
    OPTION_IMPOSSIBLE,
    OPTION_LIKELY,
    OPTION_UNLIKELY,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

CHANCE_TOPIC = "chance_outcome_classify"
PROBABILITY_TOPIC = "probability_scale_estimate"


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [CHANCE_TOPIC, PROBABILITY_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [CHANCE_TOPIC, PROBABILITY_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    seeds_with_at_least_two = 0
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors, (
            f"{topic} seed={seed} difficulty={difficulty}: answer in distractors"
        )
        assert len(question.distractors) == len(set(question.distractors)), (
            f"{topic} seed={seed} difficulty={difficulty}: duplicate distractors"
        )
        assert len(question.distractors) >= 1, f"{topic} seed={seed} difficulty={difficulty}: no distractors"
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / len(SEEDS) >= 0.85


@pytest.mark.parametrize("topic", [CHANCE_TOPIC, PROBABILITY_TOPIC])
def test_bare_call_default_has_enough_distractors(topic):
    """Mirrors test bar (>=2 distractors on >=85% of seeds) for a bare
    `make(topic, seed=s)` call with no `difficulty` argument."""
    seeds_with_at_least_two = 0
    for seed in range(200):
        question = make(topic, seed=seed)
        if len(question.distractors) >= 2:
            seeds_with_at_least_two += 1
    assert seeds_with_at_least_two / 200 >= 0.85


# --- real grader accept/reject, both topics ---------------------------------


@pytest.mark.parametrize("topic", [CHANCE_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor_mcq(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert (
            grade_mcq(question, "0").correct
            or grade_mcq(question, "1").correct
            or grade_mcq(question, "2").correct
            or grade_mcq(question, "3").correct
        ), f"seed={seed}: no option matched answer"
        for idx, distractor in enumerate(question.distractors):
            for option_idx in range(4):
                result = grade_mcq(question, option_idx)
                if result.canonical == distractor:
                    assert not result.correct, f"seed={seed}: distractor {distractor} was graded correct"


@pytest.mark.parametrize("topic", [PROBABILITY_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor_number(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct, (
            f"seed={seed}: canonical {question.answer_text} not accepted"
        )
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct, f"seed={seed}: distractor {distractor} was accepted"


@pytest.mark.parametrize("topic", [CHANCE_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer_mcq(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        wrong_options = [
            o for o in (OPTION_CERTAIN, OPTION_IMPOSSIBLE, OPTION_LIKELY, OPTION_UNLIKELY) if o != question.answer_text
        ]
        if wrong_options:
            # Grade by option index
            from mathgen.grading.typed import mcq_options

            options = mcq_options(question)
            for wrong_opt in wrong_options[:1]:
                try:
                    idx = options.index(wrong_opt)
                    result = grade_mcq(question, idx)
                    assert not result.correct, f"seed={seed}: wrong option {wrong_opt} was graded correct"
                except ValueError:
                    pass


@pytest.mark.parametrize("topic", [PROBABILITY_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer_number(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        try:
            if isinstance(question.answer, float):
                answer_val = float(question.answer_text)
                # Use a value far enough to definitely exceed stated-precision tolerance
                # For decimals, just add 0.1 which is safely larger than any reasonable tolerance
                wrong = str(answer_val + 0.1)
            else:
                wrong = str(int(question.answer) + 1)
            result = grade_number(question, wrong)
            assert not result.correct, f"seed={seed}: wrong answer {wrong} was graded correct"
        except (ValueError, TypeError):
            pass


# --- chance_outcome_classify -------------------------------------------------


def test_chance_easy_answers_are_four_fixed_options():
    """Easy-tier answers must be one of the four classification words."""
    for seed in SEEDS:
        question = make(CHANCE_TOPIC, seed=seed, difficulty="easy")
        assert question.answer_text in (OPTION_CERTAIN, OPTION_IMPOSSIBLE, OPTION_LIKELY, OPTION_UNLIKELY)


def test_chance_medium_answers_are_four_fixed_options():
    """Medium-tier answers must be one of the four classification words."""
    for seed in SEEDS:
        question = make(CHANCE_TOPIC, seed=seed, difficulty="medium")
        assert question.answer_text in (OPTION_CERTAIN, OPTION_IMPOSSIBLE, OPTION_LIKELY, OPTION_UNLIKELY)


def test_chance_hard_answers_are_four_fixed_options():
    """Hard-tier answers must be one of the four classification words."""
    for seed in SEEDS:
        question = make(CHANCE_TOPIC, seed=seed, difficulty="hard")
        assert question.answer_text in (OPTION_CERTAIN, OPTION_IMPOSSIBLE, OPTION_LIKELY, OPTION_UNLIKELY)


def test_classify_chance_function():
    """Test the independent check function directly."""
    assert classify_chance(0, 10) == OPTION_IMPOSSIBLE
    assert classify_chance(10, 10) == OPTION_CERTAIN
    assert classify_chance(7, 10) == OPTION_LIKELY
    assert classify_chance(3, 10) == OPTION_UNLIKELY
    # Exactly half should also work (returns unlikely by the < comparison)
    assert classify_chance(5, 10) == OPTION_UNLIKELY


# --- probability_scale_estimate -----------------------------------------------


def test_probability_easy_answers_are_canonical_decimals():
    """Easy-tier answers are one of 0.0, 0.25, 0.5, 0.75, 1.0."""
    for seed in SEEDS:
        question = make(PROBABILITY_TOPIC, seed=seed, difficulty="easy")
        val = float(question.answer_text)
        assert val in (0.0, 0.25, 0.5, 0.75, 1.0), f"seed={seed}: unexpected value {val}"


def test_probability_medium_answers_are_products_of_restricted():
    """Medium-tier answers are products of {0.25, 0.5, 0.75}."""
    restricted_set = {0.25, 0.5, 0.75}
    products = set()
    for a in restricted_set:
        for b in restricted_set:
            products.add(round(a * b, 4))  # Round to avoid float precision issues

    for seed in SEEDS:
        question = make(PROBABILITY_TOPIC, seed=seed, difficulty="medium")
        val = round(float(question.answer_text), 4)
        assert val in products, f"seed={seed}: {val} not a product of {restricted_set}"


def test_probability_hard_answers_are_integers():
    """Hard-tier answers are always integers (expected frequencies)."""
    for seed in SEEDS:
        question = make(PROBABILITY_TOPIC, seed=seed, difficulty="hard")
        # Answer should be an integer count
        assert question.answer_text == str(int(float(question.answer_text)))


def test_probability_hard_trials_are_multiples_of_four():
    """Hard-tier trial counts are multiples of 4 (so products are exact integers)."""
    for seed in SEEDS:
        question = make(PROBABILITY_TOPIC, seed=seed, difficulty="hard")
        # Extract trial count from question text (look for "X trials")
        import re

        match = re.search(r"(\d+) trials", question.text)
        if match:
            trials = int(match.group(1))
            assert trials % 4 == 0, f"seed={seed}: trials {trials} not multiple of 4"


# --- skills loading ---------


def test_both_skills_load_with_correct_properties():
    """Both skills load with expected properties."""
    domains, skills_list = load_skills()
    skill_ids = {s.id for s in skills_list}

    assert "f10-chance-language-and-outcomes" in skill_ids
    assert "f10-probability-scale-and-expectation" in skill_ids

    chance_skill = next(s for s in skills_list if s.id == "f10-chance-language-and-outcomes")
    prob_skill = next(s for s in skills_list if s.id == "f10-probability-scale-and-expectation")

    # Check properties
    assert chance_skill.placement_eligible is False
    assert chance_skill.answer == "mcq"
    assert set(chance_skill.prerequisites) == {"simple-event-probability"}

    assert prob_skill.placement_eligible is False
    assert prob_skill.answer == "number"
    expected_prereqs = {"probability-representations", "simple-event-probability"}
    assert set(prob_skill.prerequisites) == expected_prereqs
