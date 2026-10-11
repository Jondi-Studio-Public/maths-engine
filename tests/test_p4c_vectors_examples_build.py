"""Examples of vectors including displacement and velocity
(senior-examine-examples-of-vectors-including-displacement-and-velocity)."""

from pathlib import Path
import re

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p4c_vectors_examples_build import (
    TRIPLES,
    _verify_magnitude_from_legs,
    _verify_triple,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(500)

TOPIC = "examine_examples_of_vectors_including_displacement_and_velocity"


# --- determinism / distractor hygiene ----------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generator_is_deterministic_across_500_seeds(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        repeat = make(TOPIC, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(difficulty):
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1


# --- answer format and type --------------------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_is_always_integer_string(difficulty):
    """All tiers return plain integer answer_text, no decimal point."""
    for seed in range(50):
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert "." not in question.answer_text
        assert isinstance(question.answer, int)


# --- grader acceptance/rejection -----------------------------------------


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_rejects_distractors(difficulty):
    for seed in range(50):
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_deliberately_wrong_answer(difficulty):
    for seed in range(50):
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        wrong = question.answer + 1
        assert not grade_number(question, str(wrong)).correct


# --- easy tier (single_leg) --------------------------------------------------


def test_easy_tier_always_asks_for_magnitude():
    """Easy tier: questions ask for magnitude of a single displacement."""
    for seed in range(50):
        question = make(TOPIC, seed=seed, difficulty="easy")
        assert "magnitude" in question.text or "displacement" in question.text


def test_easy_tier_perpendicularity_preserved():
    """Easy tier: directions are perpendicular (east/west with north/south)."""
    for seed in range(100):
        question = make(TOPIC, seed=seed, difficulty="easy")
        # Check that e.g. "east then north" or "west then south" appear
        has_valid_pair = (
            ("east" in question.text and "north" in question.text)
            or ("west" in question.text and "north" in question.text)
            or ("east" in question.text and "south" in question.text)
            or ("west" in question.text and "south" in question.text)
        )
        assert has_valid_pair, f"Seed {seed}: not perpendicular: {question.text}"


def test_easy_tier_added_instead_of_pythagoras_strictly_exceeds_magnitude():
    """Easy: leg_a + leg_b > m always (algebraic proof)."""
    for leg_a, leg_b, m in TRIPLES:
        # Algebraically: (leg_a + leg_b)^2 = leg_a^2 + 2*leg_a*leg_b + leg_b^2 > leg_a^2 + leg_b^2 = m^2
        # So leg_a + leg_b > m always (both sides positive)
        assert leg_a + leg_b > m, f"Triple {(leg_a, leg_b, m)}: leg sum {leg_a + leg_b} not > m {m}"


def test_easy_tier_forgot_one_leg_strictly_less_than_magnitude():
    """Easy: max(leg_a, leg_b) < m always (hypotenuse strictly > leg)."""
    for leg_a, leg_b, m in TRIPLES:
        assert max(leg_a, leg_b) < m, f"Triple {(leg_a, leg_b, m)}: leg {max(leg_a, leg_b)} not < m {m}"


def test_easy_tier_all_9_triples_sampled_over_sweep():
    """All 9 Pythagorean triples should be sampled over a large seed sweep."""
    sampled_triples = set()
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="easy")
        # Extract legs from text like "travels 3 km east then 4 km north"
        match = re.search(r"travels (\d+) km \w+ then (\d+) km", question.text)
        if match:
            leg_a, leg_b = int(match.group(1)), int(match.group(2))
            # Find which triple this is
            for t_a, t_b, t_m in TRIPLES:
                if {leg_a, leg_b} == {t_a, t_b}:
                    sampled_triples.add((t_a, t_b, t_m))
    assert len(sampled_triples) >= 8, f"Only {len(sampled_triples)} distinct triples sampled: {sampled_triples}"


def test_easy_tier_independent_check_fires_on_corrupted_answer():
    _verify_magnitude_from_legs(3, 4, 5)  # 5^2 == 3^2 + 4^2
    with pytest.raises(VerificationError):
        _verify_magnitude_from_legs(3, 4, 6)  # 6^2 != 3^2 + 4^2


# --- medium tier (velocity_with_red_herring) --------------------------------


def test_medium_tier_red_herring_never_equals_1():
    """Medium: time t is sampled from 2..9, never 1."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="medium")
        # Extract time from text like "Over 3 seconds"
        match = re.search(r"Over (\d+) seconds", question.text)
        assert match, f"Seed {seed}: could not extract time: {question.text}"
        t = int(match.group(1))
        assert 2 <= t <= 9, f"Seed {seed}: t={t} not in range [2, 9]"


def test_medium_tier_multiplies_by_irrelevant_number_always_offered():
    """Medium: the multiplies-by-irrelevant-number distractor is always offered."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="medium")
        # Extract time and magnitude
        match_t = re.search(r"Over (\d+) seconds", question.text)
        assert match_t
        t = int(match_t.group(1))
        answer = int(question.answer_text)
        # The distractor should be m * t
        expected_distractor = str(answer * t)
        assert expected_distractor in question.distractors, (
            f"Seed {seed}: multiplies-by-irrelevant {expected_distractor} not in {question.distractors}"
        )


def test_medium_tier_red_herring_not_used_in_computation():
    """Medium: time t is not used in the Pythagorean computation, verified via independent check."""
    for seed in range(50):
        question = make(TOPIC, seed=seed, difficulty="medium")
        # The independent check should verify leg_a^2 + leg_b^2 = m^2, ignoring t
        # Verify the answer is correct for the legs only
        match = re.search(r"component of (\d+) m/s.*northward component of (\d+) m/s", question.text)
        if match:
            leg_a, leg_b = int(match.group(1)), int(match.group(2))
            answer = int(question.answer_text)
            assert answer * answer == leg_a * leg_a + leg_b * leg_b, f"Seed {seed}: {answer}^2 != {leg_a}^2 + {leg_b}^2"


def test_medium_tier_added_instead_of_pythagoras_strictly_exceeds_magnitude():
    """Medium: leg_a + leg_b > m always."""
    for leg_a, leg_b, m in TRIPLES:
        assert leg_a + leg_b > m


def test_medium_tier_forgot_one_leg_strictly_less_than_magnitude():
    """Medium: max(leg_a, leg_b) < m always."""
    for leg_a, leg_b, m in TRIPLES:
        assert max(leg_a, leg_b) < m


# --- hard tier (two_leg_journey) ----------------------------------------


def test_hard_tier_summed_legs_in_answer():
    """Hard tier: the answer should be the magnitude of the summed resultant."""
    for seed in range(50):
        question = make(TOPIC, seed=seed, difficulty="hard")
        # Extract all four leg values from text like "walks 1 km east then 2 km north, then walks a further 3 km east and 4 km north"
        match = re.search(
            r"walks (\d+) km east then (\d+) km north, then walks a further (\d+) km east and (\d+) km north",
            question.text,
        )
        assert match, f"Seed {seed}: could not parse hard-tier text: {question.text}"
        leg_a1, leg_b1, leg_a2, leg_b2 = map(int, match.groups())
        total_a = leg_a1 + leg_a2
        total_b = leg_b1 + leg_b2
        answer = int(question.answer_text)
        # Verify: answer^2 == total_a^2 + total_b^2
        assert answer * answer == total_a * total_a + total_b * total_b, (
            f"Seed {seed}: {answer}^2 != {total_a}^2 + {total_b}^2"
        )


def test_hard_tier_dropped_second_leg_formatted_to_1_decimal():
    """Hard: dropped-the-second-leg distractor must have .1f formatting (1 decimal place).
    It must never be a string-identical integer, so it can never equal canonical answer_text."""
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty="hard")
        answer_text = question.answer_text
        # Answer is always a plain integer (no decimal point)
        assert "." not in answer_text, f"Seed {seed}: answer_text {answer_text} has decimal"
        # Every distractor should differ from answer_text
        for distractor in question.distractors:
            assert distractor != answer_text, f"Seed {seed}: distractor {distractor} equals answer_text {answer_text}"
            # If distractor has a decimal point, it's the dropped-second-leg one
            # (the only one that might be formatted differently)


def test_hard_tier_dropped_second_leg_has_decimal_point():
    """Hard: the dropped-the-second-leg distractor should have a decimal point (formatted .1f)."""
    count_with_decimal = 0
    for seed in range(100):
        question = make(TOPIC, seed=seed, difficulty="hard")
        for distractor in question.distractors:
            if "." in distractor:
                count_with_decimal += 1
                # Verify it's formatted with 1 decimal place
                parts = distractor.split(".")
                assert len(parts) == 2, f"Seed {seed}: distractor {distractor} has wrong format"
                assert len(parts[1]) == 1, f"Seed {seed}: distractor {distractor} not 1 decimal place"
    assert count_with_decimal >= 50, f"Only {count_with_decimal} distractors have decimal point"


def test_hard_tier_added_instead_of_pythagoras_strictly_exceeds_magnitude():
    """Hard: (leg_a1+leg_a2) + (leg_b1+leg_b2) > m always (since (P,Q,m) is a Pythagorean triple)."""
    for P, Q, m in TRIPLES:
        assert P + Q > m, f"Triple {(P, Q, m)}: sum {P + Q} not > m {m}"


def test_hard_tier_forgot_one_leg_strictly_less_than_magnitude():
    """Hard: P < m and Q < m always (since (P,Q,m) is a Pythagorean triple)."""
    for P, Q, m in TRIPLES:
        assert P < m, f"Triple {(P, Q, m)}: P {P} not < m {m}"
        assert Q < m, f"Triple {(P, Q, m)}: Q {Q} not < m {m}"


def test_hard_tier_independent_check_fires_on_corrupted_answer():
    """Hard: independent check verifies Pythagorean identity on summed legs."""
    _verify_magnitude_from_legs(8, 15, 17)  # 17^2 == 8^2 + 15^2
    with pytest.raises(VerificationError):
        _verify_magnitude_from_legs(8, 15, 18)  # 18^2 != 8^2 + 15^2


# --- skill contract ---------------------------------------------------------


def test_skill_contract_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}

    skill = by_id["senior-examine-examples-of-vectors-including-displacement-and-velocity"]

    assert skill.prerequisites == ["vector-addition"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert {source.topic for source in skill.sources} == {TOPIC, "exd_x_g2_velocity_composition"}
    # Confirm exact stages: [6], [6], [6, "maintenance"], and the exam source's ["exam"]
    stages_list = [source.stages for source in skill.sources]
    assert stages_list == [[6], [6], [6, "maintenance"], ["exam"]], (
        f"Expected [[6], [6], [6, 'maintenance'], ['exam']], got {stages_list}"
    )
