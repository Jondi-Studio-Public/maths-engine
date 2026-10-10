"""Tests for: practical right/non-right triangle trigonometry
(``senior-solve-practical-problems-involving-the-trigonometry-of-right``, ACMGM037)."""

from __future__ import annotations

import math
import random
import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.core import REGISTRY
from mathgen.text_quality import prompt_shape
from mathgen.generators.p14a_practical_triangle_build import (
    PHI_POOL,
    PSI_POOL,
    _draw_easy,
    _draw_medium,
    _draw_hard,
    _draw_exam,
    _interior_from_bearings,
    _verify_pt_via_qt_cosine,
    _verify_qt_via_pt_cosine,
    _verify_cosine_sas,
    _verify_bearing_interior,
    _verify_elevation,
)

TOPIC = "p14a_practical_triangle"
GRADER = GRADERS["number"]
SEEDS = range(60)

TIERS = ["easy", "medium", "hard", "exam"]


def _num(text: str) -> float:
    return float(text)


class _CommonBehaviour:
    TIER: str

    def test_determinism(self):
        for seed in SEEDS:
            q1 = make(TOPIC, seed=seed, difficulty=self.TIER)
            q2 = make(TOPIC, seed=seed, difficulty=self.TIER)
            assert q1.to_json() == q2.to_json()

    def test_working_nonempty(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            assert isinstance(q.working, list) and len(q.working) >= 2
            for step in q.working:
                assert isinstance(step, str) and step

    def test_distractors_distinct_nonempty_never_equal_answer(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_distractors_never_within_grader_tolerance(self):
        # answer_text/distractors all state exactly 1 decimal place, so the
        # real grade_number tolerance is 0.05; every offered distractor must
        # differ by more than 0.051 (the same margin pick_distractors/
        # stated_precision_tolerance(1) uses).
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            answer_value = _num(q.answer_text)
            for dist in q.distractors:
                assert abs(_num(dist) - answer_value) > 0.051

    def test_grader_accepts_answer_rejects_distractors_and_wrong(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            result = GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: rejected canonical answer {q.answer_text!r}"
            for dist in q.distractors:
                result = GRADER(q, dist)
                assert not result.correct, f"seed {seed}: accepted distractor {dist!r}"
            wrong = _num(q.answer_text) + 7.3
            result = GRADER(q, str(round(wrong, 1)))
            assert not result.correct, f"seed {seed}: accepted deliberately wrong {wrong}"

    def test_answer_states_exactly_one_decimal_place(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            assert isinstance(q.answer, float)
            literal = re.search(r"-?\d+\.\d+", q.answer_text)
            assert literal, f"seed {seed}: answer_text {q.answer_text!r} has no decimal literal"
            assert len(literal.group().split(".")[1]) == 1
            for dist in q.distractors:
                d_literal = re.search(r"-?\d+\.\d+", dist)
                assert d_literal, f"seed {seed}: distractor {dist!r} has no decimal literal"
                assert len(d_literal.group().split(".")[1]) == 1

    def test_at_least_20_distinct_questions_over_60_seeds(self):
        texts = {make(TOPIC, seed=seed, difficulty=self.TIER).text for seed in SEEDS}
        assert len(texts) >= 20, f"only {len(texts)} distinct questions over {len(SEEDS)} seeds"

    def test_plausible_height(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            assert 5.0 <= _num(q.answer_text) <= 300.0

    def test_prompt_variety_at_least_3_shapes(self):
        shapes = {prompt_shape(make(TOPIC, seed=seed, difficulty=self.TIER)) for seed in SEEDS}
        assert len(shapes) >= 3


class TestEasy(_CommonBehaviour):
    TIER = "easy"

    def test_red_first_ground_triangle_filter(self):
        """Prove the [20, 140] ground-triangle-angle filter is load-bearing:
        without it, a meaningful fraction of raw (phi, psi) draws land
        outside the plausible range (proven red first)."""
        failures = 0
        for seed in range(200):
            rng = random.Random(f"p14a-red-first:{seed}")
            phi = rng.choice(PHI_POOL)
            psi = rng.choice(PSI_POOL)
            angle_t = 180 - phi - psi
            if not (20 <= phi <= 140 and 20 <= psi <= 140 and 20 <= angle_t <= 140):
                failures += 1
        assert failures > 0, "expected some raw (phi, psi) draws to fall outside [20, 140] without the filter"

    def test_ground_triangle_angles_plausible(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            phi = int(re.search(r"[Aa]ngle QPT = (\d+) degrees", q.text).group(1))
            psi = int(re.search(r"[Aa]ngle PQT = (\d+) degrees", q.text).group(1))
            angle_t = 180 - phi - psi
            assert 20 <= phi <= 140
            assert 20 <= psi <= 140
            assert 20 <= angle_t <= 140

    def test_elevation_angle_excludes_45_and_bounded(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            theta = int(re.search(r"elevation from P to the top of the tower is (\d+) degrees", q.text).group(1))
            assert theta != 45
            assert 15 <= theta <= 75

    def test_verify_pt_fires_on_corrupted_pt(self):
        rng = random.Random("p14a-verify-easy")
        draw = _draw_easy(rng)
        _verify_pt_via_qt_cosine(draw["d"], draw["phi"], draw["psi"], draw["angle_t"], draw["pt"])
        with pytest.raises(VerificationError):
            _verify_pt_via_qt_cosine(draw["d"], draw["phi"], draw["psi"], draw["angle_t"], draw["pt"] + 5.0)

    def test_verify_elevation_fires_on_corrupted_height(self):
        rng = random.Random("p14a-verify-easy-elevation")
        draw = _draw_easy(rng)
        _verify_elevation(draw["pt"], draw["theta"], draw["height"])
        with pytest.raises(VerificationError):
            _verify_elevation(draw["pt"], draw["theta"], draw["height"] + 5.0)

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "senior-solve-practical-problems-involving-the-trigonometry-of-right")
        assert skill.answer == "number"
        assert skill.placement_eligible is False
        assert any(src.topic == TOPIC and src.difficulty == "easy" for src in skill.sources)


class TestMedium(_CommonBehaviour):
    TIER = "medium"

    def test_ground_triangle_angles_plausible(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            phi = int(re.search(r"[Aa]ngle QPT = (\d+) degrees", q.text).group(1))
            psi = int(re.search(r"[Aa]ngle PQT = (\d+) degrees", q.text).group(1))
            angle_t = 180 - phi - psi
            assert 20 <= phi <= 140
            assert 20 <= psi <= 140
            assert 20 <= angle_t <= 140

    def test_elevation_is_from_q_not_p(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            assert "elevation from Q to the top of the tower" in q.text
            assert "elevation from P to the top of the tower" not in q.text

    def test_verify_qt_fires_on_corrupted_qt(self):
        rng = random.Random("p14a-verify-medium")
        draw = _draw_medium(rng)
        _verify_qt_via_pt_cosine(draw["d"], draw["phi"], draw["psi"], draw["angle_t"], draw["qt"])
        with pytest.raises(VerificationError):
            _verify_qt_via_pt_cosine(draw["d"], draw["phi"], draw["psi"], draw["angle_t"], draw["qt"] + 5.0)

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "senior-solve-practical-problems-involving-the-trigonometry-of-right")
        assert any(src.topic == TOPIC and src.difficulty == "medium" for src in skill.sources)


class TestHard(_CommonBehaviour):
    TIER = "hard"

    def test_red_first_triangle_inequality_and_angle_filter(self):
        """Prove the derived-angle sanity filter is load-bearing for the
        SAS (cosine-rule) tier: without checking the OTHER two angles,
        some raw (d, e, psi) draws would produce a sliver triangle even
        though psi itself is in-range."""
        failures = 0
        for seed in range(200):
            rng = random.Random(f"p14a-red-first-hard:{seed}")
            d = rng.randint(20, 150)
            e = rng.randint(20, 150)
            psi = rng.choice(PSI_POOL)
            pt_sq = d * d + e * e - 2 * d * e * math.cos(math.radians(psi))
            if pt_sq <= 0:
                failures += 1
                continue
            pt = math.sqrt(pt_sq)
            angle_t = math.degrees(math.acos((pt * pt + e * e - d * d) / (2 * pt * e)))
            angle_p = 180 - psi - angle_t
            if not (20 <= angle_t <= 140 and 20 <= angle_p <= 140):
                failures += 1
        assert failures > 0, "expected some raw (d, e, psi) draws to fall outside [20, 140] without the filter"

    def test_sas_sides_and_included_angle_plausible(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            d = int(re.search(r"PQ = (\d+) m", q.text).group(1))
            e = int(re.search(r"QT = (\d+) m", q.text).group(1))
            psi = int(re.search(r"[Aa]ngle PQT = (\d+) degrees", q.text).group(1))
            assert 20 <= d <= 150
            assert 20 <= e <= 150
            assert 20 <= psi <= 140

    def test_verify_cosine_sas_fires_on_corrupted_pt(self):
        rng = random.Random("p14a-verify-hard")
        draw = _draw_hard(rng)
        _verify_cosine_sas(draw["d"], draw["e"], draw["psi"], draw["pt"])
        with pytest.raises(VerificationError):
            _verify_cosine_sas(draw["d"], draw["e"], draw["psi"], draw["pt"] + 5.0)

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "senior-solve-practical-problems-involving-the-trigonometry-of-right")
        assert any(src.topic == TOPIC and src.difficulty == "hard" for src in skill.sources)


class TestExam(_CommonBehaviour):
    TIER = "exam"

    def test_wraparound_direction_both_occur_over_60_seeds(self):
        """Prove the bearing-difference wraparound is genuinely exercised:
        both the direction that needs no supplement (raw diff <= 180) and
        the direction that does (raw diff > 180) occur across seeds."""
        directions = set()
        for seed in SEEDS:
            rng = random.Random(f"{TOPIC}:{self.TIER}:{seed}")
            draw = _draw_exam(rng)
            directions.add(draw["direction"])
        assert directions == {1, -1}, f"expected both bearing directions, got {directions}"

    def test_interior_from_bearings_handles_wraparound(self):
        # No wraparound needed: bearing_t is simply bearing_q + phi.
        assert _interior_from_bearings(10, 70) == 60
        # Wraparound needed: raw difference is 300, supplement is 60.
        assert _interior_from_bearings(350, 290) == 60
        # Exact boundary sanity: a straight difference of 180 needs no supplement.
        assert _interior_from_bearings(0, 180) == 180

    def test_verify_bearing_interior_fires_on_corrupted_phi(self):
        rng = random.Random("p14a-verify-exam-bearing")
        draw = _draw_exam(rng)
        _verify_bearing_interior(draw["bearing_q"], draw["bearing_t"], draw["phi"])
        with pytest.raises(VerificationError):
            _verify_bearing_interior(draw["bearing_q"], draw["bearing_t"], draw["phi"] + 10)

    def test_bearings_and_ground_triangle_plausible(self):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=self.TIER)
            bq = int(re.search(r"bearing of Q from P is (\d+) degrees", q.text).group(1))
            bt = int(re.search(r"bearing of T from P is (\d+) degrees", q.text).group(1))
            psi = int(re.search(r"[Aa]ngle PQT = (\d+) degrees", q.text).group(1))
            assert 0 <= bq <= 359
            assert 0 <= bt <= 359
            phi = _interior_from_bearings(bq, bt)
            angle_t = 180 - phi - psi
            assert 20 <= phi <= 140
            assert 20 <= psi <= 140
            assert 20 <= angle_t <= 140

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "senior-solve-practical-problems-involving-the-trigonometry-of-right")
        assert any(src.topic == TOPIC and src.difficulty == "exam" for src in skill.sources)
        skill_src = next(src for src in skill.sources if src.difficulty == "exam")
        assert skill_src.stages == ["exam"]


def test_topic_registered_once_in_this_module():
    _, fn = REGISTRY[TOPIC]
    assert fn.__module__.endswith("p14a_practical_triangle_build")


def test_module_name_ends_in_build():
    from mathgen.generators import p14a_practical_triangle_build

    assert p14a_practical_triangle_build.__name__.endswith("_build")
