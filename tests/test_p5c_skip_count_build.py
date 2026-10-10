"""Focused contract tests for f10-skip-count-pattern-sequences."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_ordered_list
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p5c_skip_count_build import _verify_terms

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


# --- determinism / distractor hygiene -------------------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_deterministic_across_seeds_and_unique_distractors(difficulty):
    """Each seed produces the same question on replay, and distractors are
    distinct, non-empty, and never equal to the answer."""
    for seed in SEEDS:
        question = make("skip_count_pattern_sequences", seed=seed, difficulty=difficulty)
        repeat = make("skip_count_pattern_sequences", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert isinstance(question.answer, list)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_correct_answer_grades_correct_and_every_distractor_grades_incorrect(difficulty):
    """The real grader accepts the canonical answer and rejects every distractor."""
    for seed in SEEDS:
        question = make("skip_count_pattern_sequences", seed=seed, difficulty=difficulty)
        assert grade_ordered_list(question, question.answer_text).correct, (
            f"seed {seed}: canonical answer {question.answer_text} did not grade correct"
        )
        for i, distractor in enumerate(question.distractors):
            result = grade_ordered_list(question, distractor)
            assert not result.correct, (
                f"seed {seed}: distractor {i} ({distractor}) graded incorrect when it should reject"
            )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_transposed_answer_grades_incorrect(difficulty):
    """Reordering the same terms must grade wrong (order-sensitive grader)."""
    checked = 0
    for seed in SEEDS:
        question = make("skip_count_pattern_sequences", seed=seed, difficulty=difficulty)
        items = question.answer
        if len(items) < 2 or items[0] == items[1]:
            continue
        transposed = [items[1], items[0]] + list(items[2:])
        given = ", ".join(str(item) for item in transposed)
        assert not grade_ordered_list(question, given).correct
        checked += 1
    assert checked > 0


# --- difficulty-specific constraints -----------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_easy_and_medium_ask_for_2_terms(difficulty):
    """Easy and medium difficulty ask for exactly 2 next terms."""
    for seed in SEEDS:
        question = make("skip_count_pattern_sequences", seed=seed, difficulty=difficulty)
        assert len(question.answer) == 2, (
            f"seed {seed}: expected 2 terms, got {len(question.answer)}: {question.answer}"
        )


def test_hard_asks_for_3_terms():
    """Hard difficulty asks for exactly 3 next terms."""
    for seed in SEEDS:
        question = make("skip_count_pattern_sequences", seed=seed, difficulty="hard")
        assert len(question.answer) == 3, (
            f"seed {seed}: expected 3 terms, got {len(question.answer)}: {question.answer}"
        )


def test_easy_uses_friendly_starts():
    """Easy tier: start value is always a multiple of the step."""
    for seed in SEEDS:
        question = make("skip_count_pattern_sequences", seed=seed, difficulty="easy")
        # The first term is shown, so it's in the text
        first_term = question.answer[0] - 3  # Reverse: answer[0] = a + 3*step
        # Actually, let me extract the first term directly from the question text
        # Extract the displayed text (first 3 terms) from the text
        import re

        match = re.search(r"(\d+),\s*(\d+),\s*(\d+)", question.text)
        assert match is not None
        shown_terms = [int(match.group(i)) for i in range(1, 4)]
        a = shown_terms[0]
        step = shown_terms[1] - shown_terms[0]
        assert a % step == 0, f"seed {seed}: easy tier start {a} is not a multiple of step {step}"


def test_medium_uses_unfriendly_starts():
    """Medium tier: start value is NOT a multiple of the step."""
    for seed in SEEDS:
        question = make("skip_count_pattern_sequences", seed=seed, difficulty="medium")
        import re

        match = re.search(r"(\d+),\s*(\d+),\s*(\d+)", question.text)
        assert match is not None
        shown_terms = [int(match.group(i)) for i in range(1, 4)]
        a = shown_terms[0]
        step = shown_terms[1] - shown_terms[0]
        assert a % step != 0, f"seed {seed}: medium tier start {a} is a multiple of step {step}"


def test_hard_has_crossing_boundary():
    """Hard tier: extension crosses a place-value boundary (multiple of 100)."""
    for seed in SEEDS:
        question = make("skip_count_pattern_sequences", seed=seed, difficulty="hard")
        import re

        match = re.search(r"(\d+),\s*(\d+),\s*(\d+)", question.text)
        assert match is not None
        shown_terms = [int(match.group(i)) for i in range(1, 4)]
        a = shown_terms[0]
        step = shown_terms[1] - shown_terms[0]

        # Hard tier constraint: extension crosses 100 (or 200, etc.)
        # The third required term is a + 5*step
        last_required_term = a + 5 * step

        # Find the multiple of 100 crossed
        last_shown = a + 2 * step
        M = ((last_shown // 100) + 1) * 100

        # Verify crossing: last_shown < M <= last_required_term
        assert last_shown < M, f"seed {seed}: last_shown {last_shown} >= M {M}"
        assert M <= last_required_term, f"seed {seed}: M {M} > last_required_term {last_required_term}"


# --- independent check -------------------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    """The independent check recomputes terms via closed-form and fires
    on a corrupted answer."""
    a, step, k = 10, 5, 3
    correct_terms = [a + (3 + j) * step for j in range(k)]
    _verify_terms(a, step, k, correct_terms)  # must not raise

    corrupted = list(reversed(correct_terms))
    with pytest.raises(VerificationError):
        _verify_terms(a, step, k, corrupted)


def test_independent_check_detects_mutation():
    """Mutating a single element in the answer list triggers the check."""
    a, step, k = 23, 2, 2
    correct_terms = [a + (3 + j) * step for j in range(k)]

    mutated = list(correct_terms)
    mutated[0] = mutated[0] + 1
    with pytest.raises(VerificationError):
        _verify_terms(a, step, k, mutated)


# --- misconceptions -------------------------------------------------------


def test_wrong_step_distractor_is_distinct_from_answer():
    """The wrong-step distractor uses a different step value."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make("skip_count_pattern_sequences", seed=seed, difficulty=difficulty)
            # Extract step from the displayed terms
            import re

            match = re.search(r"(\d+),\s*(\d+),\s*(\d+)", question.text)
            assert match is not None
            shown_terms = [int(match.group(i)) for i in range(1, 4)]
            true_step = shown_terms[1] - shown_terms[0]

            # At least one distractor should use a wrong step
            # (the wrong-step distractor)
            found_wrong_step = False
            for distractor in question.distractors:
                distractor_nums = [int(x.strip()) for x in distractor.split(",")]
                if len(distractor_nums) >= 2:
                    distractor_step = distractor_nums[1] - distractor_nums[0]
                    if distractor_step != true_step:
                        found_wrong_step = True
                        break
            assert found_wrong_step, f"seed {seed} ({difficulty}): no wrong-step distractor found"


def test_stopped_incrementing_distractor_exists():
    """At least one distractor should repeat the last shown term."""
    for seed in SEEDS:
        for difficulty in ["easy", "medium", "hard"]:
            question = make("skip_count_pattern_sequences", seed=seed, difficulty=difficulty)
            import re

            match = re.search(r"(\d+),\s*(\d+),\s*(\d+)", question.text)
            assert match is not None
            shown_terms = [int(match.group(i)) for i in range(1, 4)]
            last_shown = shown_terms[-1]

            found_stopped = False
            for distractor in question.distractors:
                distractor_nums = [int(x.strip()) for x in distractor.split(",")]
                if all(n == last_shown for n in distractor_nums):
                    found_stopped = True
                    break
            assert found_stopped, f"seed {seed} ({difficulty}): no stopped-incrementing distractor found"


# --- skill loading -------------------------------------------------------


def test_skill_loads_with_correct_metadata():
    """The skill loads with correct prerequisites, answer kind, and stages."""
    _, all_skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in all_skills}
    skill = by_id.get("f10-skip-count-pattern-sequences")
    assert skill is not None, "skill f10-skip-count-pattern-sequences not found"

    assert skill.prerequisites == ["f10-pattern-copy-repeat-unit", "skip-count-2-5-10"]
    assert skill.answer == "ordered_list"
    assert skill.placement_eligible is False


def test_skill_sources_cover_difficulties():
    """The skill's sources list covers all three difficulties."""
    _, all_skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in all_skills}
    skill = by_id.get("f10-skip-count-pattern-sequences")
    assert skill is not None

    difficulties_covered = {s.difficulty for s in skill.sources}
    assert difficulties_covered == {"easy", "medium", "hard"}, (
        f"expected all three difficulties, got {difficulties_covered}"
    )
