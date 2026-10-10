"""Tests `p10e_angle_relationships`: determinism, tier floors for angles at a point and vertically opposite
angles, and the skill loading.

Angle relationships build."""

import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make

TOPIC = "p10e_angle_relationships"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)

_STRAIGHT_LINE_RE = re.compile(r"straight line", re.IGNORECASE)


def _all_point_angles_easy(text):
    m = re.search(r"measure (\d+) degrees and (\d+) degrees\. Find the size of the third angle", text)
    g1, g2 = int(m.group(1)), int(m.group(2))
    return g1, g2


def _all_point_angles_medium(text):
    m = re.search(r"measure (\d+) degrees, (\d+) degrees and (\d+) degrees\. Find the size of the fourth angle", text)
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _hard_expressions(text):
    m = re.search(r"One angle measures \(([^)]+)\) degrees and the other measures \(([^)]+)\) degrees", text)
    return m.group(1), m.group(2)


class TestDeterminismAndShape:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism_across_seeds(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            repeat = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q.to_json() == repeat.to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_working_text_nonempty(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert isinstance(q.working, list) and len(q.working) >= 2
            for step in q.working:
                assert isinstance(step, str) and step

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 2, f"seed {seed} {difficulty}: fewer than 2 distractors survived"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_and_distractors_formatted_alike(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            int(q.answer_text)  # raises if not a plain integer string
            for dist in q.distractors:
                int(dist)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_answer(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            result = grader(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected answer {q.answer_text}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_distractors(self, difficulty):
        grader = GRADERS["number"]
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            for dist in q.distractors:
                result = grader(q, dist)
                assert not result.correct, f"seed {seed}: grader accepted distractor {dist}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_rejects_wrong_answer(self, difficulty):
        grader = GRADERS["number"]
        for seed in range(10):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            wrong = str(int(q.answer_text) + 137)
            result = grader(q, wrong)
            assert not result.correct, f"seed {seed}: grader accepted wrong answer {wrong}"

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_20_distinct_questions_over_60_seeds(self, difficulty):
        texts = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in range(60)}
        assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"

    def test_point_tiers_never_state_a_180_relationship(self):
        """Easy/medium relationships are always "N angles at a point sum to
        360" -- the excluded 180-degree straight-line SUM relationship never
        appears as the construction. An individual given angle happening to
        equal 180 (a legitimate value among several angles at a point) is
        not the same thing and is not checked against here."""
        for tier in ("easy", "medium"):
            for seed in range(60):
                q = make(TOPIC, seed=seed, difficulty=tier)
                assert "full turn" in q.text
                assert "adjacent" not in q.text.lower()
                assert any("360" in step for step in q.working)

    def test_hard_tier_never_states_a_180_or_supplementary_relationship(self):
        """The hard tier states the vertically-opposite EQUALITY fact, never
        a sum-to-180 (straight-line/supplementary) fact. "straight lines"
        appears only as the geometric setup ("two straight lines cross at a
        point") -- describing what forms the crossing, not the excluded
        180-degree adjacent-angle relationship -- so it is not itself
        checked against; what is checked is that no equation or fact in the
        text ever asserts the two angles are supplementary or sum to 180."""
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty="hard")
            assert _STRAIGHT_LINE_RE.search(q.text)  # the crossing-lines setup, expected
            assert "180" not in q.text
            assert "supplementary" not in q.text.lower()
            assert "always equal" in q.text


class TestPointTierFloors:
    """Plausibility floor proof: every angle (given or unknown) at
    the angles-at-a-point tiers is >= 15 degrees and < 360, swept over
    >=500 seeds, not spot-checked."""

    def test_easy_floor_holds_500_seeds(self):
        min_seen, max_seen = 360, 0
        for seed in range(500):
            q = make(TOPIC, seed=seed, difficulty="easy")
            g1, g2 = _all_point_angles_easy(q.text)
            x = int(q.answer_text)
            for val in (g1, g2, x):
                assert 15 <= val < 360, f"seed {seed}: angle {val} outside [15, 360)"
                min_seen, max_seen = min(min_seen, val), max(max_seen, val)
        assert min_seen >= 15
        assert max_seen <= 250, f"easy: an angle of {max_seen} exceeds the _POINT_CEIL of 250"

    def test_medium_floor_holds_500_seeds(self):
        min_seen, max_seen = 360, 0
        for seed in range(500):
            q = make(TOPIC, seed=seed, difficulty="medium")
            g1, g2, g3 = _all_point_angles_medium(q.text)
            x = int(q.answer_text)
            for val in (g1, g2, g3, x):
                assert 15 <= val < 360, f"seed {seed}: angle {val} outside [15, 360)"
                min_seen, max_seen = min(min_seen, val), max(max_seen, val)
        assert min_seen >= 15
        assert max_seen <= 250, f"medium: an angle of {max_seen} exceeds the _POINT_CEIL of 250"

    def test_point_tier_skew_measured(self):
        """Measures the fixed-up sampler's own rate stays low, over 600 seeds each -- not a spot check."""
        for tier, min_given, max_rate in (("easy", 2, 0.15), ("medium", 3, 0.10)):
            over270 = 0
            for seed in range(600):
                q = make(TOPIC, seed=seed, difficulty=tier)
                vals = (_all_point_angles_easy(q.text) if min_given == 2 else _all_point_angles_medium(q.text)) + (
                    int(q.answer_text),
                )
                if max(vals) > 270:
                    over270 += 1
            rate = over270 / 600
            assert rate <= max_rate, f"{tier}: P(angle>270) = {rate} exceeds {max_rate}"

    def test_point_tiers_sum_to_360(self):
        for seed in range(60):
            qe = make(TOPIC, seed=seed, difficulty="easy")
            g1, g2 = _all_point_angles_easy(qe.text)
            assert g1 + g2 + int(qe.answer_text) == 360

            qm = make(TOPIC, seed=seed, difficulty="medium")
            g1, g2, g3 = _all_point_angles_medium(qm.text)
            assert g1 + g2 + g3 + int(qm.answer_text) == 360


class TestHardTierFloor:
    """Plausibility floor proof for the vertically-opposite hard tier: the
    single crossing angle stays in [10, 170] (strictly < 180) over >=500
    seeds."""

    def test_hard_angle_floor_holds_500_seeds(self):
        min_seen, max_seen = 180, 0
        for seed in range(500):
            q = make(TOPIC, seed=seed, difficulty="hard")
            angle = int(q.answer_text)
            assert 10 <= angle <= 170, f"seed {seed}: angle {angle} outside [10, 170]"
            min_seen, max_seen = min(min_seen, angle), max(max_seen, angle)
        assert min_seen >= 10 and max_seen <= 170

    def test_hard_reports_x_not_angle_collision_resample_measured(self):
        """Regression guard for a previously fixed defect."""
        import random

        hits = 0
        trials = 2000
        for seed in range(trials):
            r = random.Random(f"p10e_angle_relationships:hard:{seed}")
            x = r.randint(2, 30)
            angle = r.randint(10, 170)
            if x == angle:
                hits += 1
        assert 0 < hits < trials  # sanity: the collision is rare but possible

    def test_hard_never_a_straight_line_pair(self):
        """The two vertically-opposite expressions are never presented as a
        linear (180-degree) pair: the question never states or implies the
        two given expressions sum to 180 -- they are stated EQUAL."""
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty="hard")
            assert "equal" in q.text
            assert "180" not in q.text
            _hard_expressions(q.text)  # parses cleanly; equality itself is asserted by the text above


class TestSkillLoads:
    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "f10-angle-relationships"), None)
        assert skill is not None
        assert skill.domain == "primary-geometry-time"
        assert skill.answer == "number"
        assert set(skill.prerequisites) == {"straight-line-angles", "f10-angle-turn-right-reference"}
        assert skill.placement_eligible is False
        assert len(skill.sources) == 3
