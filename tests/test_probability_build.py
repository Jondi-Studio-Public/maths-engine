"""Focused contract tests for the probability pilot
(finite-sample-space-probability -- ACMEM155)."""

from pathlib import Path

import pytest
from fractions import Fraction

from mathgen.grading.typed import grade_fraction
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.probability_build import _verify_probability

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(difficulty):
    for seed in SEEDS:
        question = make("finite_sample_space_probability", seed=seed, difficulty=difficulty)
        repeat = make("finite_sample_space_probability", seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


def test_probability_grades_correct_answer_and_rejects_distractors():
    for difficulty in ("easy", "medium", "hard"):
        for seed in SEEDS:
            question = make("finite_sample_space_probability", seed=seed, difficulty=difficulty)
            assert grade_fraction(question, question.answer_text).correct
            for distractor in question.distractors:
                assert not grade_fraction(question, distractor).correct


def test_probability_in_lowest_terms():
    """Probabilities must always be in lowest terms. Verify this by parsing
    the answer_text and checking the numerator and denominator are coprime."""
    from math import gcd

    for difficulty in ("easy", "medium", "hard"):
        for seed in range(100):
            question = make("finite_sample_space_probability", seed=seed, difficulty=difficulty)
            # Parse the answer text (e.g., "3/4" or "1")
            if "/" in question.answer_text:
                parts = question.answer_text.split("/")
                p, q = int(parts[0]), int(parts[1])
                assert gcd(p, q) == 1, f"Answer {p}/{q} is not in lowest terms at seed {seed}"
            # For integers (e.g., "1"), q is implicitly 1, which is always coprime


def test_probability_is_in_valid_range():
    """Probabilities must be strictly between 0 and 1 (not certain or
    impossible)."""
    for difficulty in ("easy", "medium", "hard"):
        for seed in range(100):
            question = make("finite_sample_space_probability", seed=seed, difficulty=difficulty)
            p = Fraction(question.answer["p"], question.answer["q"])
            assert 0 < p < 1, f"Probability {p} is not in (0, 1) at seed {seed}"


def test_probability_reducible_cases_occur():
    """Verify that at least one answer's unreduced form differs from the
    reduced form. For example, "2/4 = 1/2" shows reducibility."""
    import re

    found_reducible = False
    for difficulty in ("easy", "medium", "hard"):
        for seed in range(200):
            question = make("finite_sample_space_probability", seed=seed, difficulty=difficulty)
            # Parse the third working line, which should be:
            # "P(...) = {count}/{total} = {rational_text}."
            if len(question.working) >= 3:
                line = question.working[2]
                # Look for pattern: "P(...) = {a}/{b} = {answer}."
                match = re.search(r"= (\d+)/(\d+) = (\S+)\.$", line)
                if match:
                    a, b, reduced_text = int(match.group(1)), int(match.group(2)), match.group(3)
                    answer_text = question.answer_text
                    if f"{a}/{b}" != answer_text and Fraction(a, b) == Fraction(answer_text):
                        found_reducible = True
                        break
        if found_reducible:
            break
    assert found_reducible, "No reducible fraction found in sampled seeds"


def test_probability_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        _verify_probability([1, 2, 3, 4], [1, 2], Fraction(1, 3))  # Should be 1/2
    # A genuinely correct probability must still pass.
    _verify_probability([1, 2, 3, 4], [1, 2], Fraction(1, 2))


def test_probability_answer_format():
    """Verify that the answer dict has the correct structure for
    grade_fraction."""
    question = make("finite_sample_space_probability", seed=1, difficulty="easy")
    assert isinstance(question.answer, dict)
    assert "p" in question.answer
    assert "q" in question.answer
    assert isinstance(question.answer["p"], int)
    assert isinstance(question.answer["q"], int)


def test_probability_skill_contracts_load_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    finite_sample = by_id["finite-sample-space-probability"]

    assert finite_sample.prerequisites == ["simple-event-probability"]
    assert finite_sample.answer == "fraction"
    assert finite_sample.placement_eligible is False
    assert finite_sample.status == "active"
    assert [source.difficulty for source in finite_sample.sources] == ["easy", "medium", "hard"]
    assert {source.topic for source in finite_sample.sources} == {"finite_sample_space_probability"}
