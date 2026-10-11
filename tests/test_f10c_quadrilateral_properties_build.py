"""Tests `f10c_quadrilateral_properties` (`f10-quadrilateral-properties`): determinism, number grading to the
stated precision, tagged distractors that clear the grader, answer variety and the generator's own checks."""

import collections

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators import f10c_quadrilateral_properties_build as gen
from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills

TOPIC = "f10c_quadrilateral_properties"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(80)


def _skill():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}["f10-quadrilateral-properties"]


class TestShape:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            assert (
                make(TOPIC, seed=seed, difficulty=difficulty).to_json()
                == make(TOPIC, seed=seed, difficulty=difficulty).to_json()
            )

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_the_question_states_its_precision_and_unit_and_is_self_contained(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert q.text.endswith("to 1 decimal place.")
            assert ("in cm," in q.text) or ("in degrees," in q.text)
            assert q.answer_text == f"{q.answer:.1f}"
            assert isinstance(q.answer, float) and q.answer > 0
            assert q.working and all(isinstance(w, str) and w for w in q.working)
            assert "*" not in q.text and "**" not in q.text

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_a_property_is_stated_in_the_question(self, difficulty):
        keys = ("bisect", "equal", "add to 180", "right angles", "perpendicular bisector", "isosceles")
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert any(k in q.text for k in keys), q.text

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_two_distractors_all_distinct_and_in_range(self, difficulty):
        for seed in range(200):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert 2 <= len(q.distractors) <= 3
            assert len(set(q.distractors)) == len(q.distractors) and q.answer_text not in q.distractors
            for d in q.distractors:
                v = float(d)
                assert d == f"{v:.1f}" and v > 0 and abs(v * 2 - round(v * 2)) < 1e-9
                if "degrees" in q.text:
                    assert v < 360 and (v < 180 or "sum of angles" in q.text)


class TestGrading:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_accepted_and_every_distractor_rejected_by_the_real_grader(self, difficulty):
        for seed in range(150):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert grade_number(q, q.answer_text).correct
            if q.answer == int(q.answer):
                assert grade_number(q, str(int(q.answer))).correct  # a whole number is the same value
            for d in q.distractors:
                assert not grade_number(q, d).correct, (q.text, d)
                assert not grade_number(q, str(int(float(d)))).correct if float(d) == int(float(d)) else True

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_values_off_by_a_tenth_are_rejected(self, difficulty):
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            for off in (-0.5, -0.1, 0.1, 0.5):
                assert not grade_number(q, f"{q.answer + off:.1f}").correct


class TestVariety:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answers_are_varied_and_questions_distinct(self, difficulty):
        answers = collections.Counter(make(TOPIC, seed=s, difficulty=difficulty).answer_text for s in range(400))
        assert max(answers.values()) / 400 <= 0.1
        assert len(answers) >= 30
        assert len({make(TOPIC, seed=s, difficulty=difficulty).text for s in range(100)}) >= 90

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_every_case_at_the_tier_is_reachable(self, difficulty):
        stems = set()
        for s in range(400):
            q = make(TOPIC, seed=s, difficulty=difficulty)
            stems.add(q.working[0].split(" ")[0] + "|" + q.option_notes[q.distractors[0]])
        assert len(stems) >= 5

    def test_hard_tier_asks_for_more_steps_than_easy(self):
        easy = [len(make(TOPIC, seed=s, difficulty="easy").working) for s in range(100)]
        hard = [len(make(TOPIC, seed=s, difficulty="hard").working) for s in range(100)]
        assert sum(hard) / 100 >= sum(easy) / 100


class TestMisconceptionTags:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_every_distractor_is_tagged_with_a_declared_misconception(self, difficulty):
        ids = {m.id for m in _skill().misconceptions}
        for seed in range(200):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert set(q.option_notes) == set(q.distractors)
            assert set(q.option_notes.values()) <= ids

    def test_every_declared_misconception_is_exercised(self):
        used = set()
        for tier in TIERS:
            for seed in range(600):
                used |= set(make(TOPIC, seed=seed, difficulty=tier).option_notes.values())
        assert used == {m.id for m in _skill().misconceptions}


class TestOwnVerification:
    def test_generator_raises_if_the_independent_route_disagrees(self, monkeypatch):
        real = gen._dist
        monkeypatch.setattr(gen, "_dist", lambda p, q: real(p, q) + 1.0)
        with pytest.raises(VerificationError):
            for seed in range(30):
                make(TOPIC, seed=seed, difficulty="easy")

    def test_helpers_on_known_figures(self):
        assert abs(gen._angle_at((1, 0), (0, 0), (0, 1)) - 90) < 1e-9
        assert gen._cross((0, 0), (2, 2), (0, 2), (2, 0)) == (1.0, 1.0)

    def test_skill_shape(self):
        skill = _skill()
        assert skill.answer == "number" and skill.status == "active"
        assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard"}
        assert "ac9m8sp02" in skill.tags
