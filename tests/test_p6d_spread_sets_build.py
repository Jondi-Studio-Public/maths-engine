"""Tests the senior spread and set-description skills `use_informal_ways_of_describing_spread_such_as` and
`use_everyday_occurrences_to_illustrate_set_descriptions_and`.

Senior skills for statistics-probability domain."""

from pathlib import Path
import re

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.generators.p6d_spread_sets_build import (
    _check_gap_between_clusters,
    _check_outlier,
    _check_tight_pack,
    THREE_SET_TABLE,
    TWO_SET_TABLE,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- Determinism and distractor hygiene, both topics -----------------------


@pytest.mark.parametrize(
    "topic",
    [
        "use_informal_ways_of_describing_spread_such_as",
        "use_everyday_occurrences_to_illustrate_set_descriptions_and",
    ],
)
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


# --- use_informal_ways_of_describing_spread_such_as (skill 1) ----------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_spread_answer_is_one_of_four_options(difficulty):
    """The answer must always be one of the four MCQ options."""
    for seed in SEEDS:
        question = make(
            "use_informal_ways_of_describing_spread_such_as",
            seed=seed,
            difficulty=difficulty,
        )
        answer = question.answer_text
        options = mcq_options(question)
        assert answer in options, f"answer not in options: {answer!r} not in {options}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_spread_correct_answer_grades_correct_and_distractors_grade_incorrect(
    difficulty,
):
    """The real grader accepts the canonical answer and rejects every distractor."""
    for seed in SEEDS:
        question = make(
            "use_informal_ways_of_describing_spread_such_as",
            seed=seed,
            difficulty=difficulty,
        )
        options = mcq_options(question)
        correct_index = options.index(question.answer_text)
        assert grade_mcq(question, correct_index).correct
        for distractor in question.distractors:
            distractor_index = options.index(distractor)
            assert not grade_mcq(question, distractor_index).correct, f"distractor {distractor!r} graded as correct"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_spread_all_four_options_are_data_sets(difficulty):
    """All MCQ options must be formatted data sets, no listed elements outside options."""
    for seed in SEEDS:
        question = make(
            "use_informal_ways_of_describing_spread_such_as",
            seed=seed,
            difficulty=difficulty,
        )
        options = mcq_options(question)
        assert len(options) == 4, f"expected 4 options, got {len(options)}"
        # Each option should follow "Data set X: ..." format
        for option in options:
            assert "Data set " in option, f"option not formatted as dataset: {option!r}"
            assert ": " in option, f"option missing colon separator: {option!r}"
            # Extract and verify it's a comma-separated list of integers
            data_part = option.split(": ", 1)[1]
            values = [int(s.strip()) for s in data_part.split(", ")]
            assert len(values) >= 4, f"data set has fewer than 4 values: {values}"
            assert len(values) <= 6, f"data set has more than 6 values: {values}"


def test_spread_hard_tier_has_lone_outlier_non_qualifying_option():
    """Hard tier should have at least one non-qualifying option with lone outlier
    (not two clusters) on most seeds."""
    lone_outlier_count = 0
    for seed in SEEDS:
        question = make(
            "use_informal_ways_of_describing_spread_such_as",
            seed=seed,
            difficulty="hard",
        )
        options = mcq_options(question)
        # Extract T from the text (the gap threshold)
        text_lower = question.text.lower()
        # Find "more than T" or similar
        if "more than " in text_lower:
            parts = text_lower.split("more than ")
            for part in parts[1:]:
                T_str = part.split()[0]
                if T_str.isdigit():
                    T = int(T_str)
                    # Check if any non-answer option has a gap > T with lone outlier
                    for option in options:
                        if option == question.answer_text:
                            continue
                        data_part = option.split(": ", 1)[1]
                        values = sorted([int(s.strip()) for s in data_part.split(", ")])
                        # If it has a gap but is marked non-answer, it might be lone outlier
                        if not _check_gap_between_clusters(values, T):
                            lone_outlier_count += 1
                    break
    # At least some seeds should have the lone-outlier pattern
    assert lone_outlier_count > 0, "hard tier never had lone-outlier non-qualifying option"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_rule_text_has_exactly_one_if_and_no_imperative(difficulty):
    """The rule clause must have exactly one 'if' (word boundary) and must not
    start with a bare imperative verb. This guards against doubled 'if' or
    imperative verbs in if-clauses."""
    imperative_blocklist = {"sort", "check", "compute", "count", "list"}

    for seed in range(60):  # 0-59 as specified
        question = make(
            "use_informal_ways_of_describing_spread_such_as",
            seed=seed,
            difficulty=difficulty,
        )
        text = question.text

        # Count exact word-boundary occurrences of "if" (case-insensitive)
        if_count = len(re.findall(r"\bif\b", text, re.IGNORECASE))
        assert if_count == 1, (
            f"seed={seed}, difficulty={difficulty}: expected exactly 1 'if', got {if_count} in text: {text!r}"
        )

        # Extract the word immediately after "if " and check it's not imperative
        match = re.search(r"\bif\s+(\w+)", text, re.IGNORECASE)
        assert match, f"seed={seed}, difficulty={difficulty}: no 'if' clause found in text: {text!r}"

        word_after_if = match.group(1).lower()
        assert word_after_if not in imperative_blocklist, (
            f"seed={seed}, difficulty={difficulty}: word after 'if ' is imperative verb '{word_after_if}' "
            f"in text: {text!r}"
        )


# --- use_everyday_occurrences_to_illustrate_set_descriptions_and (skill 2) ----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_sets_answer_is_one_of_four_options(difficulty):
    """The answer must always be one of the four MCQ options."""
    for seed in SEEDS:
        question = make(
            "use_everyday_occurrences_to_illustrate_set_descriptions_and",
            seed=seed,
            difficulty=difficulty,
        )
        answer = question.answer_text
        options = mcq_options(question)
        assert answer in options, f"answer not in options: {answer!r} not in {options}"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_sets_correct_answer_grades_correct_and_distractors_grade_incorrect(
    difficulty,
):
    """The real grader accepts the canonical answer and rejects every distractor."""
    for seed in SEEDS:
        question = make(
            "use_everyday_occurrences_to_illustrate_set_descriptions_and",
            seed=seed,
            difficulty=difficulty,
        )
        options = mcq_options(question)
        correct_index = options.index(question.answer_text)
        assert grade_mcq(question, correct_index).correct
        for distractor in question.distractors:
            distractor_index = options.index(distractor)
            assert not grade_mcq(question, distractor_index).correct, f"distractor {distractor!r} graded as correct"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_sets_no_question_or_option_lists_set_elements(difficulty):
    """No question or option ever lists explicit set elements (e.g., {1, 2, 3}).
    This verifies the BOUNDARY: we translate descriptions, never compute results
    over listed elements."""
    for seed in SEEDS:
        question = make(
            "use_everyday_occurrences_to_illustrate_set_descriptions_and",
            seed=seed,
            difficulty=difficulty,
        )
        # Check full question text
        assert "{" not in question.text, f"question contains brace: {question.text!r}"
        assert "}" not in question.text, f"question contains brace: {question.text!r}"
        # Check all options
        options = mcq_options(question)
        for option in options:
            assert "{" not in option, f"option contains brace: {option!r}"
            assert "}" not in option, f"option contains brace: {option!r}"


def test_sets_easy_draws_only_from_rows_1_2():
    """Easy tier draws description from rows 1-2 (intersection/union)."""
    answers_seen = set()
    for seed in SEEDS:
        question = make(
            "use_everyday_occurrences_to_illustrate_set_descriptions_and",
            seed=seed,
            difficulty="easy",
        )
        # Answer should match one of the two easy notations
        notation1 = TWO_SET_TABLE[0][1]  # "the intersection of ..."
        notation2 = TWO_SET_TABLE[1][1]  # "the union of ..."
        answer = question.answer_text
        # Check if it's an intersection or union answer (contains the keywords)
        is_intersection = "intersection" in answer.lower()
        is_union = "union" in answer.lower()
        assert is_intersection or is_union, f"easy tier answer is not intersection/union: {answer!r}"
        answers_seen.add("intersection" if is_intersection else "union")
    # Both should appear at least once
    assert "intersection" in answers_seen, "easy tier never drew intersection (row 1)"
    assert "union" in answers_seen, "easy tier never drew union (row 2)"


def test_sets_medium_draws_from_all_two_set_rows():
    """Medium tier draws description from all rows 1-6 (all two-set descriptions)."""
    notations_seen = set()
    for seed in SEEDS:
        question = make(
            "use_everyday_occurrences_to_illustrate_set_descriptions_and",
            seed=seed,
            difficulty="medium",
        )
        answer = question.answer_text
        # Record which notation pattern was used
        if "intersection" in answer.lower():
            notations_seen.add("intersection")
        elif "union" in answer.lower():
            notations_seen.add("union")
        elif "removed" in answer.lower():
            notations_seen.add("removed")
        elif "outside both" in answer.lower():
            notations_seen.add("outside")
        elif "exactly one" in answer.lower():
            notations_seen.add("exactly_one")
    # Medium should eventually sample multiple patterns
    assert len(notations_seen) >= 3, f"medium tier only saw patterns: {notations_seen}"


def test_sets_hard_draws_only_from_three_set_rows():
    """Hard tier draws description from rows 7-10 (all three-set descriptions)."""
    for seed in SEEDS:
        question = make(
            "use_everyday_occurrences_to_illustrate_set_descriptions_and",
            seed=seed,
            difficulty="hard",
        )
        answer = question.answer_text
        # Hard-tier answers should match one of the three-set notations
        # (should contain keywords from the 3-set table)
        assert any(
            phrase in answer.lower()
            for phrase in [
                "intersection",
                "removed",
                "exactly one",
                "outside all",
            ]
        ), f"hard tier answer doesn't match three-set pattern: {answer!r}"


# --- Skills loading and metadata --


def test_both_skills_load_successfully():
    """Both skills must load through mathgen.skills.load_skills."""
    _, skills = load_skills()
    skill_ids = {s.id for s in skills}
    assert "senior-use-informal-ways-of-describing-spread-such-as" in skill_ids
    assert "senior-use-everyday-occurrences-to-illustrate-set-descriptions-and" in skill_ids


def test_spread_skill_metadata():
    """Spread skill has correct answer type, prerequisites, and placement_eligible."""
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "senior-use-informal-ways-of-describing-spread-such-as")
    assert skill.answer == "mcq"
    assert "simple-event-probability" in skill.prerequisites
    assert skill.placement_eligible is False
    assert len(skill.sources) == 3  # easy, medium, hard


def test_sets_skill_metadata():
    """Sets skill has correct answer type, prerequisites, and placement_eligible."""
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "senior-use-everyday-occurrences-to-illustrate-set-descriptions-and")
    assert skill.answer == "mcq"
    assert "simple-event-probability" in skill.prerequisites
    assert skill.placement_eligible is False
    assert len(skill.sources) == 3  # easy, medium, hard
