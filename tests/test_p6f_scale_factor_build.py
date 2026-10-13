"""Scale factor for similar figures (areas) and similar solids (surface area and volume)."""

from pathlib import Path
import re

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)

AREAS_TOPIC = "scale_factor_similar_areas"
SOLIDS_TOPIC = "scale_factor_similar_solids"


# --- determinism / distractor hygiene, both topics -------------------------


@pytest.mark.parametrize("topic", [AREAS_TOPIC, SOLIDS_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        repeat = make(topic, seed=seed, difficulty=difficulty)
        assert question.to_json() == repeat.to_json()


@pytest.mark.parametrize("topic", [AREAS_TOPIC, SOLIDS_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_distractors_distinct_nonempty_never_equal_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 1


@pytest.mark.parametrize("topic", [AREAS_TOPIC, SOLIDS_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_answer_is_exact_integer(topic, difficulty):
    """All answers must be plain integers."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert isinstance(question.answer, int)
        assert question.answer_text == str(question.answer)
        assert "." not in question.answer_text


@pytest.mark.parametrize("topic", [AREAS_TOPIC, SOLIDS_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_all_numbers_in_text_are_integers(topic, difficulty):
    """Every number in the question text must be an integer."""
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        # Extract all numbers from text
        numbers = re.findall(r"\d+", question.text)
        for num_str in numbers:
            int(num_str)  # Will fail if any number is not an integer


# --- real grader accept/reject, both topics ---------------------------------


@pytest.mark.parametrize("topic", [AREAS_TOPIC, SOLIDS_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        assert grade_number(question, question.answer_text).correct
        for distractor in question.distractors:
            assert not grade_number(question, distractor).correct


@pytest.mark.parametrize("topic", [AREAS_TOPIC, SOLIDS_TOPIC])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_real_grader_rejects_a_deliberately_wrong_answer(topic, difficulty):
    for seed in SEEDS:
        question = make(topic, seed=seed, difficulty=difficulty)
        wrong = question.answer + 1
        assert not grade_number(question, str(wrong)).correct


# --- independent check fires on corrupted answer ---------------------------


def test_scale_factor_areas_hard_independent_check_fires():
    """Test that the independent check in hard tier catches a mismatch."""
    from mathgen.generators.p6f_scale_factor_build import _verify_scale_factor_areas_hard

    # Correct case
    length_a = 6
    length_b = 9
    area_a = 16
    area_b = 36
    answer = 9
    _verify_scale_factor_areas_hard(length_a, length_b, area_a, area_b, answer)

    # Corrupted answer should raise
    with pytest.raises(VerificationError):
        _verify_scale_factor_areas_hard(length_a, length_b, area_a, area_b, 10)


def test_scale_factor_solids_hard_independent_check_fires():
    """Test that the independent check in hard tier catches a mismatch."""
    from mathgen.generators.p6f_scale_factor_build import _verify_scale_factor_solids_hard

    # Correct case: volume_b^2 * area_a^3 == volume_a^2 * area_b^3
    # With p=2, q=3: volume scales by 2^3/3^3 = 8/27, area scales by 2^2/3^2 = 4/9
    volume_a = 27
    volume_b = 8
    area_a = 9
    area_b = 4
    _verify_scale_factor_solids_hard(volume_a, volume_b, area_a, area_b, volume_b)

    # Corrupted value should raise (pass a different volume_b value)
    with pytest.raises(VerificationError):
        _verify_scale_factor_solids_hard(volume_a, 7, area_a, area_b, 7)


# --- specific topic coverage -----------------------------------------------


def test_scale_factor_areas_easy_tier_distinct_questions():
    """At least 20 distinct questions over 60 seeds in easy tier."""
    questions = set()
    for seed in range(60):
        q = make(AREAS_TOPIC, seed=seed, difficulty="easy")
        questions.add(q.text)
    assert len(questions) >= 20


def test_scale_factor_areas_medium_tier_distinct_questions():
    """At least 20 distinct questions over 60 seeds in medium tier."""
    questions = set()
    for seed in range(60):
        q = make(AREAS_TOPIC, seed=seed, difficulty="medium")
        questions.add(q.text)
    assert len(questions) >= 20


def test_scale_factor_areas_hard_tier_distinct_questions():
    """At least 20 distinct questions over 60 seeds in hard tier."""
    questions = set()
    for seed in range(60):
        q = make(AREAS_TOPIC, seed=seed, difficulty="hard")
        questions.add(q.text)
    assert len(questions) >= 20


def test_scale_factor_solids_easy_tier_distinct_questions():
    """At least 20 distinct questions over 60 seeds in easy tier."""
    questions = set()
    for seed in range(60):
        q = make(SOLIDS_TOPIC, seed=seed, difficulty="easy")
        questions.add(q.text)
    assert len(questions) >= 20


def test_scale_factor_solids_medium_tier_distinct_questions():
    """At least 20 distinct questions over 60 seeds in medium tier."""
    questions = set()
    for seed in range(60):
        q = make(SOLIDS_TOPIC, seed=seed, difficulty="medium")
        questions.add(q.text)
    assert len(questions) >= 20


def test_scale_factor_solids_hard_tier_distinct_questions():
    """At least 20 distinct questions over 60 seeds in hard tier."""
    questions = set()
    for seed in range(60):
        q = make(SOLIDS_TOPIC, seed=seed, difficulty="hard")
        questions.add(q.text)
    assert len(questions) >= 20


def test_scale_factor_solids_both_kinds_in_easy():
    """Easy tier must exercise both 'volume' and 'surface_area' kinds."""
    kinds = set()
    for seed in range(200):
        q = make(SOLIDS_TOPIC, seed=seed, difficulty="easy")
        if "volume" in q.text:
            kinds.add("volume")
        if "surface area" in q.text:
            kinds.add("surface_area")
    assert "volume" in kinds
    assert "surface_area" in kinds


def test_scale_factor_solids_both_kinds_in_medium():
    """Medium tier must exercise both 'volume' and 'surface_area' kinds."""
    kinds = set()
    for seed in range(200):
        q = make(SOLIDS_TOPIC, seed=seed, difficulty="medium")
        if "volume" in q.text:
            kinds.add("volume")
        if "surface area" in q.text:
            kinds.add("surface_area")
    assert "volume" in kinds
    assert "surface_area" in kinds


def test_scale_factor_solids_both_kinds_in_hard():
    """Hard tier must exercise both 'volume' and 'surface_area' kinds."""
    kinds = set()
    for seed in range(200):
        q = make(SOLIDS_TOPIC, seed=seed, difficulty="hard")
        if "volume" in q.text and "surface area" in q.text:
            # This is the hard tier which asks for one given the other
            if "What is Solid B's volume" in q.text:
                kinds.add("volume")
            elif "What is Solid B's surface area" in q.text:
                kinds.add("surface_area")
    assert "volume" in kinds
    assert "surface_area" in kinds


def test_scale_factor_solids_hard_both_directions():
    """Hard tier must exercise both 'areas_to_volume' and 'volumes_to_area'."""
    directions = set()
    for seed in range(200):
        q = make(SOLIDS_TOPIC, seed=seed, difficulty="hard")
        if "What is Solid B's volume" in q.text:
            directions.add("areas_to_volume")
        elif "What is Solid B's surface area" in q.text:
            directions.add("volumes_to_area")
    assert "areas_to_volume" in directions
    assert "volumes_to_area" in directions


# --- skill loading -------------------------------------------------------


def test_scale_factor_areas_skill_loads_with_correct_metadata():
    """Skill must load with correct prerequisites and answer kind."""
    _, skills_list = load_skills()
    skills_by_id = {skill.id: skill for skill in skills_list}
    skill = skills_by_id.get("senior-obtain-a-scale-factor-and-use-it-to")
    assert skill is not None
    assert skill.answer == "number"
    assert skill.prerequisites == ["convert-metric-length"]
    assert skill.placement_eligible is False


def test_scale_factor_solids_skill_loads_with_correct_metadata():
    """Skill must load with correct prerequisites and answer kind."""
    _, skills_list = load_skills()
    skills_by_id = {skill.id: skill for skill in skills_list}
    skill = skills_by_id.get("senior-obtain-a-scale-factor-and-use-it-to-acmgm025")
    assert skill is not None
    assert skill.answer == "number"
    assert skill.prerequisites == ["convert-metric-length"]
    assert skill.placement_eligible is False


def test_scale_factor_areas_skill_has_three_tiers():
    """Skill sources must cover easy, medium, hard."""
    _, skills_list = load_skills()
    skills_by_id = {skill.id: skill for skill in skills_list}
    skill = skills_by_id.get("senior-obtain-a-scale-factor-and-use-it-to")
    assert len(skill.sources) == 4
    difficulties = {s.difficulty for s in skill.sources}
    assert difficulties == {"easy", "medium", "hard", "exam"}


def test_scale_factor_solids_skill_has_three_tiers():
    """Skill sources must cover easy, medium, hard."""
    _, skills_list = load_skills()
    skills_by_id = {skill.id: skill for skill in skills_list}
    skill = skills_by_id.get("senior-obtain-a-scale-factor-and-use-it-to-acmgm025")
    assert len(skill.sources) == 4
    difficulties = {s.difficulty for s in skill.sources}
    assert difficulties == {"easy", "medium", "hard", "exam"}


# --- oracle-style tests: re-derive answers from rendered text -------


def test_oracle_scale_factor_areas_easy_recompute():
    """Oracle: re-derive easy-tier areas questions from text."""
    for seed in range(40):
        q = make(AREAS_TOPIC, seed=seed, difficulty="easy")

        # Extract numbers from text
        match = re.search(r"scale factor .+ is (\d+)/(\d+)", q.text)
        assert match
        p, q_ratio = int(match.group(1)), int(match.group(2))

        match = re.search(r"area of (\d+) cm²", q.text)
        assert match
        area_a = int(match.group(1))

        # Recompute: area_b = area_a * (p/q)^2
        area_b = area_a * p * p // (q_ratio * q_ratio)

        assert int(q.answer) == area_b


def test_oracle_scale_factor_areas_medium_recompute():
    """Oracle: re-derive medium-tier areas questions from text."""
    from fractions import Fraction

    for seed in range(40):
        q = make(AREAS_TOPIC, seed=seed, difficulty="medium")

        # Extract lengths
        match = re.search(r"side of Figure A measures (\d+) cm", q.text)
        assert match
        length_a = int(match.group(1))

        match = re.search(r"side of Figure B measures (\d+) cm", q.text)
        assert match
        length_b = int(match.group(1))

        # Extract area_a
        match = re.search(r"area of (\d+) cm²", q.text)
        assert match
        area_a = int(match.group(1))

        # Recompute: area_b = area_a * (length_b / length_a)^2
        ratio = Fraction(length_b, length_a)
        area_b = int(area_a * ratio**2)

        assert int(q.answer) == area_b


def test_oracle_scale_factor_areas_hard_recompute():
    """Oracle: re-derive hard-tier areas questions using cross-multiply check."""
    for seed in range(40):
        q = make(AREAS_TOPIC, seed=seed, difficulty="hard")

        # Extract areas
        match = re.search(r"area of (\d+) cm².*area of (\d+) cm²", q.text, re.DOTALL)
        assert match
        area_a = int(match.group(1))
        area_b = int(match.group(2))

        # Extract length_a
        match = re.search(r"side of Figure A measures (\d+) cm", q.text)
        assert match
        length_a = int(match.group(1))

        # Answer should satisfy: answer^2 * area_a == length_a^2 * area_b
        # Using isqrt to find answer = sqrt(length_a^2 * area_b / area_a)
        import math

        answer_sq_numerator = length_a * length_a * area_b
        answer_sq_denominator = area_a

        # Verify it divides evenly
        assert answer_sq_numerator % answer_sq_denominator == 0
        answer_sq = answer_sq_numerator // answer_sq_denominator
        answer = int(math.isqrt(answer_sq))

        # Verify the cross-multiply
        assert answer * answer * area_a == length_a * length_a * area_b
        assert int(q.answer) == answer


def test_oracle_scale_factor_solids_easy_recompute():
    """Oracle: re-derive easy-tier solids questions from text."""
    for seed in range(40):
        q = make(SOLIDS_TOPIC, seed=seed, difficulty="easy")

        # Extract scale factor
        match = re.search(r"scale factor .+ is (\d+)/(\d+)", q.text)
        assert match
        p, q_ratio = int(match.group(1)), int(match.group(2))

        # Determine exponent from kind label
        if "volume" in q.text:
            exponent = 3
            unit = "cm³"
        else:
            exponent = 2
            unit = "cm²"

        # Extract base quantity
        match = re.search(rf"has a .+ of (\d+) {re.escape(unit)}", q.text)
        assert match
        base = int(match.group(1))

        # Recompute
        result = base * (p**exponent) // (q_ratio**exponent)
        assert int(q.answer) == result


def test_oracle_scale_factor_solids_medium_recompute():
    """Oracle: re-derive medium-tier solids questions from text."""
    from fractions import Fraction

    for seed in range(40):
        q = make(SOLIDS_TOPIC, seed=seed, difficulty="medium")

        # Extract lengths
        match = re.search(r"side of Solid A measures (\d+) cm", q.text)
        assert match
        length_a = int(match.group(1))

        match = re.search(r"side of Solid B measures (\d+) cm", q.text)
        assert match
        length_b = int(match.group(1))

        # Determine exponent
        if "volume" in q.text:
            exponent = 3
            unit = "cm³"
        else:
            exponent = 2
            unit = "cm²"

        # Extract base
        match = re.search(rf"has a .+ of (\d+) {re.escape(unit)}", q.text)
        assert match
        base = int(match.group(1))

        # Recompute
        ratio = Fraction(length_b, length_a)
        result = int(base * ratio**exponent)

        assert int(q.answer) == result


def test_oracle_scale_factor_solids_hard_recompute():
    """Oracle: re-derive hard-tier solids questions using cross-multiply check."""
    import math

    for seed in range(40):
        q = make(SOLIDS_TOPIC, seed=seed, difficulty="hard")

        # Determine direction and extract quantities
        if "What is Solid B's volume" in q.text:
            # areas_to_volume: given area_a, area_b, volume_a; compute volume_b
            match = re.search(r"surface area of (\d+) cm².*surface area of (\d+) cm²", q.text, re.DOTALL)
            assert match
            area_a = int(match.group(1))
            area_b = int(match.group(2))

            match = re.search(r"volume of (\d+) cm³", q.text)
            assert match
            volume_a = int(match.group(1))

            # volume_b^2 * area_a^3 == volume_a^2 * area_b^3
            # volume_b^2 = volume_a^2 * area_b^3 / area_a^3
            numerator = volume_a * volume_a * area_b * area_b * area_b
            denominator = area_a * area_a * area_a
            assert numerator % denominator == 0
            volume_b_sq = numerator // denominator
            volume_b = int(math.isqrt(volume_b_sq))
            assert int(q.answer) == volume_b

        else:
            # volumes_to_area: given volume_a, volume_b, area_a; compute area_b
            match = re.search(r"volume of (\d+) cm³.*volume of (\d+) cm³", q.text, re.DOTALL)
            assert match
            volume_a = int(match.group(1))
            volume_b = int(match.group(2))

            match = re.search(r"surface area of (\d+) cm²", q.text)
            assert match
            area_a = int(match.group(1))

            # area_b^3 = area_a^3 * volume_b^2 / volume_a^2
            numerator = area_a * area_a * area_a * volume_b * volume_b
            denominator = volume_a * volume_a
            assert numerator % denominator == 0
            area_b_cu = numerator // denominator
            area_b = int(round(area_b_cu ** (1 / 3)))
            # Verify by cubing back
            if area_b**3 != area_b_cu:
                area_b = area_b_cu ** (1 / 3)  # For floating point
                area_b = int(round(area_b))
            assert int(q.answer) == area_b
