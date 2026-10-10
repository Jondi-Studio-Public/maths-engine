"""Tests the sine rule (angle, side and ambiguous case) and cosine rule generators.

Trig rules build."""

import random
import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p8b_trig_rules_build import (
    _draw_once_sine_rule_angle,
    _draw_once_sine_rule_ambiguous,
    _draw_once_sine_rule_solve,
    _draw_once_cosine_rule_angle,
    _verify_sine_rule_angle,
    _verify_sine_rule_ambiguous,
    _verify_sine_rule_solve,
    _verify_cosine_rule_angle,
)

SEEDS = range(60)
GRADER = GRADERS["number"]

TOPIC_DIFFICULTY = {
    "sine_rule_angle": "medium",
    "sine_rule_ambiguous": "hard",
    "sine_rule_solve": "hard",
    "cosine_rule_angle": "hard",
}


def _angle_value(text: str) -> float:
    m = re.search(r"([A-Za-z])\s*=\s*(-?\d+(?:\.\d+)?)", text)
    return float(m.group(2))


class _CommonBehaviour:
    """Shared assertions parametrised per topic; subclasses set TOPIC."""

    TOPIC: str

    def _difficulty(self):
        return TOPIC_DIFFICULTY[self.TOPIC]

    def test_determinism(self):
        for seed in SEEDS:
            q1 = make(self.TOPIC, seed=seed, difficulty=self._difficulty())
            q2 = make(self.TOPIC, seed=seed, difficulty=self._difficulty())
            assert q1.to_json() == q2.to_json()

    def test_working_nonempty(self):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=self._difficulty())
            assert isinstance(q.working, list) and len(q.working) >= 2
            for step in q.working:
                assert isinstance(step, str) and step

    def test_distractors_distinct_nonempty_never_equal_answer(self):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=self._difficulty())
            assert q.answer_text not in q.distractors
            assert len(q.distractors) == len(set(q.distractors))
            assert len(q.distractors) >= 1

    def test_distractors_never_within_grader_tolerance(self):
        # Every distractor and the answer state exactly 1 decimal place, so
        # the real grade_number tolerance is 0.05 -- assert every offered
        # distractor differs by more than that (0.051, the same margin
        # `pick_distractors`/`GRADER_TOLERANCE` uses).
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=self._difficulty())
            answer_value = _angle_value(q.answer_text)
            for dist in q.distractors:
                assert abs(_angle_value(dist) - answer_value) > 0.051

    def test_grader_accepts_answer_rejects_distractors_and_wrong(self):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=self._difficulty())
            result = GRADER(q, q.answer_text)
            assert result.correct, f"seed {seed}: rejected canonical answer {q.answer_text!r}"
            for dist in q.distractors:
                result = GRADER(q, dist)
                assert not result.correct, f"seed {seed}: accepted distractor {dist!r}"
            wrong = _angle_value(q.answer_text) + 5.3
            result = GRADER(q, str(round(wrong, 1)))
            assert not result.correct, f"seed {seed}: accepted deliberately wrong {wrong}"

    def test_answer_states_exactly_one_decimal_place(self):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty=self._difficulty())
            assert isinstance(q.answer, float)
            literal = re.search(r"-?\d+\.\d+", q.answer_text)
            assert literal, f"seed {seed}: answer_text {q.answer_text!r} has no decimal literal"
            assert len(literal.group().split(".")[1]) == 1
            for dist in q.distractors:
                d_literal = re.search(r"-?\d+\.\d+", dist)
                assert d_literal, f"seed {seed}: distractor {dist!r} has no decimal literal"
                assert len(d_literal.group().split(".")[1]) == 1

    def test_at_least_20_distinct_questions_over_60_seeds(self):
        texts = {make(self.TOPIC, seed=seed, difficulty=self._difficulty()).text for seed in SEEDS}
        assert len(texts) >= 20, f"only {len(texts)} distinct questions over {len(SEEDS)} seeds"


class TestSineRuleAngle(_CommonBehaviour):
    TOPIC = "sine_rule_angle"

    def test_red_first_without_rejection_loop(self):
        # Prove the [30, 150] filter is load-bearing: without it, a
        # meaningful fraction of raw draws land outside the plausible range.
        failures = 0
        for seed in range(60):
            rng = random.Random(f"sine_rule_angle:red-first:{seed}")
            A = rng.choice([30, 35, 40, 45, 50, 55, 60, 65, 70])
            a = rng.randint(8, 25)
            b = rng.randint(8, 25)
            if not a > b:
                failures += 1
                continue
            import math

            sinA = math.sin(math.radians(A))
            ratio = b * sinA / a
            if not (0 < ratio <= 0.95):
                failures += 1
                continue
            B = round(math.degrees(math.asin(ratio)), 1)
            C = round(180 - A - B, 1)
            if not (30 <= A <= 150 and 30 <= B <= 150 and 30 <= C <= 150):
                failures += 1
        assert failures > 0, "expected some raw draws to fall outside [30, 150] without the filter"

    def test_plausible_range(self):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty="medium")
            A = float(re.search(r"angle A = (-?\d+(?:\.\d+)?)", q.text).group(1))
            B = _angle_value(q.answer_text)
            C = 180 - A - B
            assert 30 <= A <= 150
            assert 30 <= B <= 150
            assert 30 <= C <= 150

    def test_a_strictly_greater_than_b(self):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty="medium")
            a = int(re.search(r"side a = (\d+)", q.text).group(1))
            b = int(re.search(r"side b = (\d+)", q.text).group(1))
            assert a > b

    def test_verify_fires_on_corrupted_answer(self):
        draw = None
        rng = random.Random("sine_rule_angle:verify-test")
        while draw is None:
            draw = _draw_once_sine_rule_angle(rng)
        _verify_sine_rule_angle(draw)  # correct: must not raise
        corrupted = draw.__class__(**{**draw.__dict__, "B_exact": draw.B_exact + 5.0})
        with pytest.raises(VerificationError):
            _verify_sine_rule_angle(corrupted)

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "sine-rule")
        assert skill.answer == "number"
        assert any(src.topic == "sine_rule_angle" and src.difficulty == "medium" for src in skill.sources)


class TestSineRuleAmbiguous(_CommonBehaviour):
    TOPIC = "sine_rule_ambiguous"

    def test_red_first_without_rejection_loop(self):
        import math

        failures = 0
        for seed in range(60):
            rng = random.Random(f"sine_rule_ambiguous:red-first:{seed}")
            A = rng.choice(list(range(30, 46)))
            b = rng.randint(15, 60)
            a = rng.randint(15, 60)
            sinA = math.sin(math.radians(A))
            if not (b * sinA < a < b):
                failures += 1
                continue
            ratio = b * sinA / a
            if not (0 < ratio < 1):
                failures += 1
                continue
            B1 = math.degrees(math.asin(ratio))
            B2 = 180 - B1
            C2 = 180 - A - B2
            if not (30 <= A <= 150 and 30 <= round(B2, 1) <= 150 and 30 <= round(C2, 1) <= 150):
                failures += 1
        assert failures > 0, "expected some raw draws to fall outside [30, 150] without the filter"

    def test_plausible_range_of_chosen_branch(self):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty="hard")
            A = float(re.search(r"angle A = (-?\d+(?:\.\d+)?)", q.text).group(1))
            B2 = _angle_value(q.answer_text)
            C2 = 180 - A - B2
            assert 30 <= A <= 150
            assert 30 <= B2 <= 150
            assert 30 <= C2 <= 150

    def test_rejected_branch_is_genuinely_distinct(self):
        for seed in SEEDS:
            draw = None
            rng = random.Random(f"sine_rule_ambiguous:branch-check:{seed}")
            while draw is None:
                draw = _draw_once_sine_rule_ambiguous(rng)
            assert draw.B1 != draw.B2
            assert draw.C1 != draw.C2
            assert draw.C1 > 0

    def test_question_names_obtuse_triangle(self):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty="hard")
            assert "OBTUSE" in q.text
            assert _angle_value(q.answer_text) > 90

    def test_verify_fires_on_corrupted_answer(self):
        draw = None
        rng = random.Random("sine_rule_ambiguous:verify-test")
        while draw is None:
            draw = _draw_once_sine_rule_ambiguous(rng)
        _verify_sine_rule_ambiguous(draw)
        corrupted = draw.__class__(**{**draw.__dict__, "B2_exact": draw.B2_exact - 4.0})
        with pytest.raises(VerificationError):
            _verify_sine_rule_ambiguous(corrupted)

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "sine-rule")
        assert any(src.topic == "sine_rule_ambiguous" and src.difficulty == "hard" for src in skill.sources)


class TestSineRuleSolve(_CommonBehaviour):
    TOPIC = "sine_rule_solve"

    def test_red_first_without_rejection_loop(self):
        failures = 0
        for seed in range(60):
            rng = random.Random(f"sine_rule_solve:red-first:{seed}")
            A = rng.choice(list(range(30, 121)))
            B = rng.choice(list(range(30, 121)))
            C = 180 - A - B
            if not (30 <= A <= 150 and 30 <= B <= 150 and 30 <= C <= 150):
                failures += 1
        assert failures > 0, "expected some raw (A, B) draws to fall outside [30, 150] without the filter"

    def test_plausible_range(self):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty="hard")
            A = float(re.search(r"angle A = (-?\d+(?:\.\d+)?)", q.text).group(1))
            B = float(re.search(r"angle B = (-?\d+(?:\.\d+)?)", q.text).group(1))
            C = 180 - A - B
            assert 30 <= A <= 150
            assert 30 <= B <= 150
            assert 30 <= C <= 150

    def test_verify_fires_on_corrupted_answer(self):
        draw = None
        rng = random.Random("sine_rule_solve:verify-test")
        while draw is None:
            draw = _draw_once_sine_rule_solve(rng)
        _verify_sine_rule_solve(draw)
        corrupted = draw.__class__(**{**draw.__dict__, "c": draw.c + 3.0})
        with pytest.raises(VerificationError):
            _verify_sine_rule_solve(corrupted)

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "sine-rule")
        assert any(src.topic == "sine_rule_solve" and src.difficulty == "hard" for src in skill.sources)


class TestCosineRuleAngle(_CommonBehaviour):
    TOPIC = "cosine_rule_angle"

    def test_red_first_without_rejection_loop(self):
        import math

        failures = 0
        for seed in range(60):
            rng = random.Random(f"cosine_rule_angle:red-first:{seed}")
            a = rng.randint(5, 20)
            b = rng.randint(5, 20)
            c = rng.randint(5, 20)
            if not (abs(a - b) < c < a + b):
                failures += 1
                continue
            cosC = (a * a + b * b - c * c) / (2 * a * b)
            if not (-1 < cosC < 1):
                failures += 1
                continue
            C = math.degrees(math.acos(cosC))
            cosA = (b * b + c * c - a * a) / (2 * b * c)
            cosB = (a * a + c * c - b * b) / (2 * a * c)
            if not (-1 < cosA < 1 and -1 < cosB < 1):
                failures += 1
                continue
            A = math.degrees(math.acos(cosA))
            B = math.degrees(math.acos(cosB))
            if not (30 <= A <= 150 and 30 <= B <= 150 and 30 <= C <= 150):
                failures += 1
        assert failures > 0, "expected some raw (a, b, c) draws to fall outside [30, 150] without the filter"

    def test_plausible_range_and_triangle_inequality(self):
        for seed in SEEDS:
            q = make(self.TOPIC, seed=seed, difficulty="hard")
            a = int(re.search(r"side a = (\d+)", q.text).group(1))
            b = int(re.search(r"side b = (\d+)", q.text).group(1))
            c = int(re.search(r"side c = (\d+)", q.text).group(1))
            assert abs(a - b) < c < a + b
            C = _angle_value(q.answer_text)
            assert 30 <= C <= 150

    def test_verify_fires_on_corrupted_answer(self):
        draw = None
        rng = random.Random("cosine_rule_angle:verify-test")
        while draw is None:
            draw = _draw_once_cosine_rule_angle(rng)
        _verify_cosine_rule_angle(draw)
        corrupted = draw.__class__(**{**draw.__dict__, "C": draw.C + 10.0})
        with pytest.raises(VerificationError):
            _verify_cosine_rule_angle(corrupted)

    def test_skill_loads(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "cosine-rule")
        assert any(src.topic == "cosine_rule_angle" and src.difficulty == "hard" for src in skill.sources)


def test_geometry_module_untouched_topics_unaffected():
    """Sanity: this module's topics are new names, not reused ones -- the
    legacy `trigonometry` topic (geometry.py) is unaffected by construction
    since this module never imports it and never re-registers its name."""
    from mathgen.core import REGISTRY

    for topic in ("sine_rule_angle", "sine_rule_ambiguous", "sine_rule_solve", "cosine_rule_angle"):
        _, fn = REGISTRY[topic]
        assert fn.__module__.endswith("p8b_trig_rules_build")
    assert REGISTRY["trigonometry"][1].__module__.endswith("geometry")
