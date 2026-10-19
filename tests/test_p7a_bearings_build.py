"""Tests the bearings skill (`senior-solve-problems-involving-bearings`)."""

import math
import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p7a_bearings_build import (
    bearings_problem,
    BEARING_PAIRS,
    _check_back_bearing,
    _check_leg_distance,
    _check_leg_bearing,
)

TOPIC = "bearings_problem"
SKILL_ID = "senior-solve-problems-involving-bearings"
SEEDS = range(40)
GRADER = GRADERS["number"]


def _mk(difficulty, seed):
    """Generate through the skill's own sources: difficulty alone selects the
    preset, which supplies the variant param -- matching what the skill
    actually drives."""
    return make(TOPIC, seed=seed, difficulty=difficulty)


# ---------------------------------------------------------------------------
# Easy: back bearing


class TestBackBearing:
    def test_determinism(self):
        for seed in SEEDS:
            q1 = _mk("easy", seed)
            q2 = _mk("easy", seed)
            assert q1.to_json() == q2.to_json()

    def test_working_nonempty(self):
        for seed in SEEDS:
            q = _mk("easy", seed)
            assert isinstance(q.working, list) and len(q.working) >= 1
            assert all(isinstance(s, str) and s for s in q.working)

    def test_distractors_distinct_nonempty(self):
        for seed in SEEDS:
            q = _mk("easy", seed)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_grader_accepts_answer_rejects_distractors(self):
        for seed in SEEDS:
            q = _mk("easy", seed)
            result = GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected {q.answer_text!r}"
            for d in q.distractors:
                result = GRADER(q, d)
                assert not result.correct, f"seed {seed}: grader accepted distractor {d!r}"

    def test_grader_rejects_a_deliberately_wrong_answer(self):
        for seed in range(10):
            q = _mk("easy", seed)
            wrong = (int(q.answer_text) + 47) % 360
            result = GRADER(q, str(wrong))
            assert not result.correct, f"seed {seed}: grader accepted deliberately-wrong {wrong}"

    def test_answer_is_an_exact_three_digit_bearing(self):
        for seed in SEEDS:
            q = _mk("easy", seed)
            assert re.fullmatch(r"\d{3}", q.answer_text), q.answer_text
            assert 0 <= int(q.answer_text) < 360

    def test_full_draw_space_of_bearings(self):
        """>=20 distinct rendered questions over 60 seeds (bearing_ab is
        drawn uniformly over 0..359, so this is a low bar)."""
        texts = {_mk("easy", seed).text for seed in range(60)}
        assert len(texts) >= 20

    def test_independent_check_fires_on_corrupted_answer(self):
        with pytest.raises(VerificationError):
            _check_back_bearing(bearing_ab=10, claimed=999)

    def test_independent_check_agrees_by_hand_all_four_quadrant_starts(self):
        """Direct hand-worked spot check, independent of the generator's own
        construction: reversing a bearing in each compass quadrant."""
        for bearing_ab, expected in [(0, 180), (90, 270), (180, 0), (270, 90), (45, 225), (359, 179)]:
            _check_back_bearing(bearing_ab, expected)
            with pytest.raises(VerificationError):
                _check_back_bearing(bearing_ab, (expected + 1) % 360)


# ---------------------------------------------------------------------------
# Medium: straight-line distance via Pythagoras


class TestLegDistance:
    def test_determinism(self):
        for seed in SEEDS:
            q1 = _mk("medium", seed)
            q2 = _mk("medium", seed)
            assert q1.to_json() == q2.to_json()

    def test_working_nonempty(self):
        for seed in SEEDS:
            q = _mk("medium", seed)
            assert isinstance(q.working, list) and len(q.working) >= 1
            assert all(isinstance(s, str) and s for s in q.working)

    def test_distractors_distinct_nonempty(self):
        for seed in SEEDS:
            q = _mk("medium", seed)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_grader_accepts_answer_rejects_distractors(self):
        for seed in SEEDS:
            q = _mk("medium", seed)
            result = GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected {q.answer_text!r}"
            for d in q.distractors:
                result = GRADER(q, d)
                assert not result.correct, f"seed {seed}: grader accepted distractor {d!r}"

    def test_grader_rejects_a_deliberately_wrong_answer(self):
        for seed in range(10):
            q = _mk("medium", seed)
            wrong = f"{float(q.answer_text) + 3.7:.1f}"
            result = GRADER(q, wrong)
            assert not result.correct, f"seed {seed}: grader accepted deliberately-wrong {wrong}"

    def test_answer_never_a_trailing_zero(self):
        """The listed defect class: 'to 1 decimal place' rendering a
        Pythagorean-triple's exact integer hypotenuse as e.g. '5.0'."""
        for seed in range(200):
            q = _mk("medium", seed)
            assert not q.answer_text.endswith(".0"), f"seed {seed}: trailing-zero answer {q.answer_text!r}"

    def test_answer_states_exactly_one_decimal_place(self):
        for seed in SEEDS:
            q = _mk("medium", seed)
            assert re.fullmatch(r"\d+\.\d", q.answer_text), q.answer_text

    def test_full_draw_space(self):
        texts = {_mk("medium", seed).text for seed in range(60)}
        assert len(texts) >= 20

    def test_all_four_bearing_pairs_occur(self):
        seen = set()
        for seed in range(60):
            q = _mk("medium", seed)
            m = re.search(
                r"bearing of (\d{3})\S* for \d+ km, arriving at point M, "
                r"then turns and walks on a bearing of (\d{3})",
                q.text,
            )
            assert m, q.text
            seen.add((int(m.group(1)), int(m.group(2))))
        assert seen == set(BEARING_PAIRS), seen

    def test_independent_check_fires_on_corrupted_answer(self):
        with pytest.raises(VerificationError):
            _check_leg_distance(L1=3, L2=4, hyp_full=999.0)

    def test_hypotenuse_never_sits_on_the_rounding_boundary(self):
        """The generator resamples any leg pair whose hypotenuse is within
        0.05 of the tenths boundary; recompute the true hypotenuse from the
        stated legs (not the already-rounded answer_text) and confirm."""
        pattern = re.compile(
            r"bearing of \d{3}\S* for (\d+) km, arriving at point M, "
            r"then turns and walks on a bearing of \d{3}\S* for (\d+) km"
        )
        for seed in range(200):
            q = _mk("medium", seed)
            m = pattern.search(q.text)
            assert m, q.text
            L1, L2 = int(m.group(1)), int(m.group(2))
            hyp = math.hypot(L1, L2)
            distance_from_boundary = abs(hyp * 10 - round(hyp * 10))
            assert distance_from_boundary <= 0.46, (
                f"seed {seed}: hypotenuse {hyp} sits within {0.5 - distance_from_boundary:.4f} of a rounding boundary"
            )


# ---------------------------------------------------------------------------
# Hard: resultant bearing


HARD_TEXT_RE = re.compile(
    r"bearing of (\d{3})\S* for (\d+) m, arriving at point M, then turns and walks "
    r"on a bearing of (\d{3})\S* for (\d+) m, arriving at point B\. "
    r"Find the bearing (from A to B|from B to A)"
)


def _hard_hand_worked(b1, L1, b2, L2, direction):
    """A from-scratch, textbook-style hand derivation, independent of both the
    generator's construction (2D vector sum + atan2) and its own in-generator
    check (b1 + arctan(L2/L1))."""
    alpha = math.degrees(math.atan(L2 / L1))
    forward = round(b1 + alpha) % 360
    if direction == "from A to B":
        return forward
    return (forward + 180) % 360


class TestLegBearing:
    def test_determinism(self):
        for seed in SEEDS:
            q1 = _mk("hard", seed)
            q2 = _mk("hard", seed)
            assert q1.to_json() == q2.to_json()

    def test_working_nonempty(self):
        for seed in SEEDS:
            q = _mk("hard", seed)
            assert isinstance(q.working, list) and len(q.working) >= 2
            assert all(isinstance(s, str) and s for s in q.working)

    def test_distractors_distinct_nonempty(self):
        for seed in SEEDS:
            q = _mk("hard", seed)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_grader_accepts_answer_rejects_distractors(self):
        for seed in SEEDS:
            q = _mk("hard", seed)
            result = GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected {q.answer_text!r}"
            for d in q.distractors:
                result = GRADER(q, d)
                assert not result.correct, f"seed {seed}: grader accepted distractor {d!r}"

    def test_grader_rejects_a_deliberately_wrong_answer(self):
        for seed in range(10):
            q = _mk("hard", seed)
            wrong = (int(q.answer_text) + 61) % 360
            result = GRADER(q, str(wrong))
            assert not result.correct, f"seed {seed}: grader accepted deliberately-wrong {wrong}"

    def test_answer_is_an_exact_three_digit_bearing(self):
        for seed in SEEDS:
            q = _mk("hard", seed)
            assert re.fullmatch(r"\d{3}", q.answer_text), q.answer_text
            assert 0 <= int(q.answer_text) < 360

    def test_full_draw_space(self):
        texts = {_mk("hard", seed).text for seed in range(60)}
        assert len(texts) >= 20

    def test_independent_check_fires_on_corrupted_answer(self):
        with pytest.raises(VerificationError):
            _check_leg_bearing(b1=0, L1=3, L2=4, claimed_ab=999, claimed_ba=999)

    def test_resultant_bearing_never_sits_on_the_rounding_boundary(self):
        """Mirrors the generator's own resample guard (>0.48 of a degree from
        the nearest integer is resampled): recompute the true angle from the
        stated legs and confirm no shipped question is within 0.02 of it."""
        for seed in range(200):
            q = _mk("hard", seed)
            m = HARD_TEXT_RE.search(q.text)
            assert m, q.text
            b1, L1, b2, L2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            theta1, theta2 = math.radians(b1), math.radians(b2)
            dx = L1 * math.sin(theta1) + L2 * math.sin(theta2)
            dy = L1 * math.cos(theta1) + L2 * math.cos(theta2)
            raw = math.degrees(math.atan2(dx, dy)) % 360
            distance_from_boundary = abs(raw - round(raw))
            assert distance_from_boundary <= 0.48, (
                f"seed {seed}: resultant bearing {raw} sits within "
                f"{0.5 - distance_from_boundary:.4f} of a rounding boundary"
            )

    def test_all_four_bearing_pair_combinations_occur_and_hand_verify(self):
        """A required extra check: sweep the hard tier's 60 seeds,
        assert all four (000+090, 090+180, 180+270, 270+000) combinations are
        drawn, and hand-verify the reported bearing for EACH occurrence
        against a from-scratch textbook derivation (not the generator's own
        in-generator check)."""
        seen_combos = set()
        checked = 0
        for seed in range(60):
            q = _mk("hard", seed)
            m = HARD_TEXT_RE.search(q.text)
            assert m, q.text
            b1, L1, b2, L2, direction = (
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
                int(m.group(4)),
                m.group(5),
            )
            seen_combos.add((b1, b2))
            expected = _hard_hand_worked(b1, L1, b2, L2, direction)
            assert expected == int(q.answer_text), (
                f"seed {seed}: hand-worked {expected} != reported {q.answer_text} "
                f"(b1={b1}, L1={L1}, b2={b2}, L2={L2}, direction={direction})"
            )
            checked += 1
        assert seen_combos == set(BEARING_PAIRS), (
            f"not all four bearing-pair combinations were drawn over 60 seeds: {seen_combos}"
        )
        assert checked == 60


# ---------------------------------------------------------------------------
# Skill loading


class TestSkillLoads:
    def test_skill_loads_with_correct_metadata(self):
        domains, skills = load_skills()
        skill = next((s for s in skills if s.id == SKILL_ID), None)
        assert skill is not None, f"skill {SKILL_ID} not found"
        assert skill.answer == "number"
        assert skill.prerequisites == ["senior-apply-the-cosine-and-sine-ratios-to-find", "f10-angle-relationships"]
        assert skill.placement_eligible is False
        assert len(skill.sources) == 6  # w3 added medium and hard sources with non-cardinal legs

    def test_skill_sources_cover_all_difficulties(self):
        domains, skills = load_skills()
        skill = next(s for s in skills if s.id == SKILL_ID)
        difficulties = {source.difficulty for source in skill.sources}
        assert difficulties == {"easy", "medium", "hard", "exam"}

    def test_skill_sources_cover_expected_stages(self):
        domains, skills = load_skills()
        skill = next(s for s in skills if s.id == SKILL_ID)
        by_difficulty = {s.difficulty: s.stages for s in skill.sources if s.topic == "bearings_problem"}
        assert by_difficulty["easy"] == [6]
        assert by_difficulty["medium"] == [6]
        assert by_difficulty["hard"] == [6, "maintenance"]

    def test_misconceptions_present(self):
        domains, skills = load_skills()
        skill = next(s for s in skills if s.id == SKILL_ID)
        assert len(skill.misconceptions) >= 1


# ---------------------------------------------------------------------------
# Unknown variant


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        bearings_problem(__import__("random").Random(0), variant="not_a_variant")
