"""Tests `p11d_parallel_line_angle_theorems` and `p11d_compare_angle_names`: determinism, tier floors,
independent checks and skill loading.

Geometry angles build."""

import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p11d_geometry_angles_build import (
    _check_easy_medium,
    _check_hard,
    _check_order,
)

TOPIC1 = "p11d_parallel_line_angle_theorems"
TOPIC2 = "p11d_compare_angle_names"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)


# -- Skill 1: p11d_parallel_line_angle_theorems ---------------------------------


class TestSkill1DeterminismAndShape:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism_across_seeds(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC1, seed=seed, difficulty=difficulty)
            repeat = make(TOPIC1, seed=seed, difficulty=difficulty)
            assert q.to_json() == repeat.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_working_text_nonempty(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC1, seed=seed, difficulty=difficulty)
            assert isinstance(q.working, list) and len(q.working) >= 2
            for step in q.working:
                assert isinstance(step, str) and step

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC1, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 2, f"seed {seed} {difficulty}: fewer than 2 distractors survived"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_and_distractors_formatted_alike(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC1, seed=seed, difficulty=difficulty)
            int(q.answer_text)  # raises if not a plain integer string
            for dist in q.distractors:
                int(dist)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(TOPIC1, seed=seed, difficulty=difficulty)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected answer {q.answer_text}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_distractors(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(TOPIC1, seed=seed, difficulty=difficulty)
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed}: grader accepted distractor {dist}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_wrong_answer(self, difficulty):
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(TOPIC1, seed=seed, difficulty=difficulty)
            wrong = str(int(q.answer_text) + 137)
            result = grader(q, wrong)
            assert not result.correct, f"seed {seed}: grader accepted wrong answer {wrong}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions_over_60_seeds(self, difficulty):
        texts = {make(TOPIC1, seed=seed, difficulty=difficulty).text for seed in range(60)}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"

    def test_easy_never_names_co_interior(self):
        for seed in range(60):
            q = make(TOPIC1, seed=seed, difficulty="easy")
            assert "co-interior" not in q.text
            assert re.search(r"it is (corresponding|alternate) to another angle", q.text)

    def test_medium_and_hard_draw_all_three_relationships(self):
        for difficulty, pattern in (
            ("medium", re.compile(r"it is (corresponding|alternate|co-interior) (?:to|with) another angle")),
            ("hard", re.compile(r"forming two (corresponding|alternate|co-interior) angles")),
        ):
            seen = set()
            for seed in range(60):
                q = make(TOPIC1, seed=seed, difficulty=difficulty)
                seen.add(pattern.search(q.text).group(1))
            assert seen == {"corresponding", "alternate", "co-interior"}, f"{difficulty}: only saw {seen} over 60 seeds"

    def test_never_states_angles_at_a_point_or_vertically_opposite(self):
        """Boundary against the prerequisite f10-angle-relationships: never
        a 360-degree-at-a-point or vertically-opposite construction."""
        for difficulty in TIERS:
            for seed in range(60):
                q = make(TOPIC1, seed=seed, difficulty=difficulty)
                assert "360" not in q.text
                assert "vertically opposite" not in q.text.lower()
                assert "point" not in q.text.lower()
                assert "parallel lines" in q.text
                assert "transversal" in q.text

    def test_never_chains_two_relationships(self):
        for difficulty in TIERS:
            for seed in range(60):
                q = make(TOPIC1, seed=seed, difficulty=difficulty)
                # exactly one relationship keyword family appears, not two distinct ones chained
                names = [n for n in ("corresponding", "alternate", "co-interior") if n in q.text]
                assert len(names) == 1, f"{difficulty} seed {seed}: text names {names}"


class TestSkill1Floors:
    """Figure plausibility floor: every angle value lands in
    [20, 160] degrees, swept over 400 seeds per tier, not spot-checked."""

    def test_easy_medium_floor_holds_400_seeds(self):
        for difficulty in ("easy", "medium"):
            for seed in range(400):
                q = make(TOPIC1, seed=seed, difficulty=difficulty)
                x = int(re.search(r"measures (\d+) degrees", q.text).group(1))
                answer = int(q.answer_text)
                assert 20 <= x <= 160, f"{difficulty} seed {seed}: stated angle {x} outside [20, 160]"
                assert 20 <= answer <= 160, f"{difficulty} seed {seed}: answer {answer} outside [20, 160]"
                assert x != 90, f"{difficulty} seed {seed}: midpoint 90 must be excluded"

    def test_hard_floor_holds_400_seeds(self):
        for seed in range(400):
            q = make(TOPIC1, seed=seed, difficulty="hard")
            angle = int(q.answer_text)
            assert 20 <= angle <= 160, f"seed {seed}: solved angle {angle} outside [20, 160]"

    def test_hard_x_always_positive_integer(self):
        for seed in range(400):
            q = make(TOPIC1, seed=seed, difficulty="hard")
            m = re.search(r"x = (\d+)", q.working[2])
            assert m is not None
            assert int(m.group(1)) > 0

    def test_class_balance_400_seeds_medium_and_hard(self):
        """Class balance: medium and hard tiers each
        draw all three relationship types close to uniformly at 400 seeds."""
        for difficulty, pattern in (
            ("medium", re.compile(r"it is (corresponding|alternate|co-interior) (?:to|with) another angle")),
            ("hard", re.compile(r"forming two (corresponding|alternate|co-interior) angles")),
        ):
            counts = {"corresponding": 0, "alternate": 0, "co-interior": 0}
            for seed in range(400):
                q = make(TOPIC1, seed=seed, difficulty=difficulty)
                counts[pattern.search(q.text).group(1)] += 1
            for name, count in counts.items():
                assert 100 <= count <= 175, f"{difficulty}: {name} count {count} not close to uniform over 400"


class TestSkill1IndependentCheck:
    def test_check_easy_medium_fires_on_corrupted_answer(self):
        text = (
            "Two parallel lines are cut by a transversal. One angle measures 60 degrees, "
            "and it is corresponding to another angle formed by the same transversal. "
            "Find the size of that other angle, in degrees."
        )
        _check_easy_medium(text, "60")  # correct, does not raise
        with pytest.raises(VerificationError):
            _check_easy_medium(text, "61")

    def test_check_easy_medium_fires_on_corrupted_co_interior_answer(self):
        text = (
            "Two parallel lines are cut by a transversal. One angle measures 60 degrees, "
            "and it is co-interior with another angle formed by the same transversal. "
            "Find the size of that other angle, in degrees."
        )
        _check_easy_medium(text, "120")  # correct: 180 - 60
        with pytest.raises(VerificationError):
            _check_easy_medium(text, "60")  # equal-rule misapplied

    def test_check_hard_fires_on_corrupted_answer(self):
        text = (
            "Two parallel lines are cut by a transversal, forming two corresponding angles. "
            "One angle measures (2x + 4) degrees and the other measures (3x - 16) degrees. "
            "Corresponding angles are always equal. Find the value of the first angle, in degrees."
        )
        _check_hard(text, "44")  # x = 20, angle = 44
        with pytest.raises(VerificationError):
            _check_hard(text, "45")

    def test_check_hard_fires_on_co_interior(self):
        text = (
            "Two parallel lines are cut by a transversal, forming two co-interior angles. "
            "One angle measures (2x + 20) degrees and the other measures (3x + 10) degrees. "
            "Co-interior angles always sum to 180 degrees. Find the value of the first angle, in degrees."
        )
        # 2x+20 + 3x+10 = 180 -> 5x + 30 = 180 -> x = 30 -> first angle = 2*30+20 = 80
        _check_hard(text, "80")
        with pytest.raises(VerificationError):
            _check_hard(text, "30")  # reports x instead of the angle


# -- Skill 2: p11d_compare_angle_names -------------------------------------------

_ALL_NAMES = {"acute", "right", "obtuse", "straight", "reflex", "revolution"}
_RANK = {n: i for i, n in enumerate(["acute", "right", "obtuse", "straight", "reflex", "revolution"])}


def _drawn_names(text):
    m = re.search(r": (.+)\.$", text)
    return [n.strip() for n in m.group(1).split(",")]


class TestSkill2DeterminismAndShape:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism_across_seeds(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC2, seed=seed, difficulty=difficulty)
            repeat = make(TOPIC2, seed=seed, difficulty=difficulty)
            assert q.to_json() == repeat.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_working_text_nonempty(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC2, seed=seed, difficulty=difficulty)
            assert isinstance(q.working, list) and len(q.working) >= 2

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC2, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 2

    @pytest.mark.parametrize("difficulty,expected_n", [("easy", 3), ("medium", 4), ("hard", 5)])
    def test_names_are_distinct_and_from_the_fixed_pool(self, difficulty, expected_n):
        for seed in SEEDS:
            q = make(TOPIC2, seed=seed, difficulty=difficulty)
            names = _drawn_names(q.text)
            assert len(names) == expected_n
            assert len(set(names)) == expected_n
            assert set(names) <= _ALL_NAMES

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer(self, difficulty):
        grader = GRADERS["ordered_list"]
        for seed in SEEDS:
            q = make(TOPIC2, seed=seed, difficulty=difficulty)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected answer {q.answer_text}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_distractors(self, difficulty):
        grader = GRADERS["ordered_list"]
        for seed in SEEDS:
            q = make(TOPIC2, seed=seed, difficulty=difficulty)
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed}: grader accepted distractor {dist}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_wrong_answer(self, difficulty):
        grader = GRADERS["ordered_list"]
        for seed in range(10):
            q = make(TOPIC2, seed=seed, difficulty=difficulty)
            names = _drawn_names(q.text)
            wrong = ", ".join(reversed(sorted(names, key=lambda n: _RANK[n])))
            if wrong == q.answer_text:
                continue
            result = grader(q, wrong)
            assert not result.correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions_over_60_seeds(self, difficulty):
        texts = {make(TOPIC2, seed=seed, difficulty=difficulty).text for seed in range(60)}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"

    def test_answer_is_correctly_ranked(self):
        for difficulty in TIERS:
            for seed in range(60):
                q = make(TOPIC2, seed=seed, difficulty=difficulty)
                ranks = [_RANK[n] for n in q.answer]
                assert ranks == sorted(ranks), f"{difficulty} seed {seed}: {q.answer} not correctly ranked"

    def test_never_states_a_degree_value(self):
        """Boundary against angle-types: never receives a stated degree value."""
        for difficulty in TIERS:
            for seed in range(60):
                q = make(TOPIC2, seed=seed, difficulty=difficulty)
                assert "degree" not in q.text.lower()
                assert not re.search(r"\d", q.text)

    def test_never_uses_turn_fraction_vocabulary(self):
        """Boundary against f10-angle-turn-right-reference: never mixes in
        quarter/half/three-quarter/full-turn vocabulary."""
        for difficulty in TIERS:
            for seed in range(60):
                q = make(TOPIC2, seed=seed, difficulty=difficulty)
                assert "turn" not in q.text.lower()
                assert "right angle" not in q.text.lower()

    def test_first_last_swap_only_at_n_ge_4(self):
        """At N=3 first-last-swap is IDENTICAL to full reversal, so at most 2
        distractor candidates are offered (reversal, adjacent-swap); at N>=4
        first-last-swap is a genuinely distinct third candidate."""
        for seed in range(60):
            q = make(TOPIC2, seed=seed, difficulty="easy")
            order_asc = q.answer
            first_last = list(order_asc)
            first_last[0], first_last[-1] = first_last[-1], first_last[0]
            assert ", ".join(first_last) == ", ".join(reversed(order_asc)), (
                "N=3 first-last-swap must equal full reversal"
            )
            assert len(q.distractors) <= 2

        for difficulty in ("medium", "hard"):
            for seed in range(60):
                q = make(TOPIC2, seed=seed, difficulty=difficulty)
                order_asc = q.answer
                first_last = list(order_asc)
                first_last[0], first_last[-1] = first_last[-1], first_last[0]
                first_last_text = ", ".join(first_last)
                reversed_text = ", ".join(reversed(order_asc))
                assert first_last_text != reversed_text, (
                    f"{difficulty}: first-last-swap collapsed into full reversal at N>=4"
                )


class TestSkill2IndependentCheck:
    def test_check_order_fires_on_corrupted_answer(self):
        text = "Arrange these angle types in order from smallest to largest: reflex, right, acute."
        _check_order(text, "acute, right, reflex")
        with pytest.raises(VerificationError):
            _check_order(text, "acute, reflex, right")


class TestSkillsLoad:
    def test_skill1_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "f10-parallel-line-angle-theorems"), None)
        assert skill is not None
        assert skill.domain == "geometry"
        assert skill.answer == "number"
        assert skill.prerequisites == ["f10-angle-relationships"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 3

    def test_skill2_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "f10-estimate-compare-angle-names"), None)
        assert skill is not None
        assert skill.domain == "primary-geometry-time"
        assert skill.answer == "ordered_list"
        assert set(skill.prerequisites) == {"angle-types", "f10-angle-turn-right-reference"}
        assert skill.placement_eligible is False
        assert len(skill.sources) == 3
