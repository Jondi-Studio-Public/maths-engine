"""Tests `f10_geometric_construction_algorithm`: determinism, tier shapes, distractors, grading, skill loading."""

import re
from collections import Counter

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.f10_geometric_construction_build import _check_easy, _check_hard, _check_medium
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills

TOPIC = "f10_geometric_construction_algorithm"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)


class TestShape:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            a = make(TOPIC, seed=seed, difficulty=difficulty)
            assert a.to_json() == make(TOPIC, seed=seed, difficulty=difficulty).to_json()

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_working_and_distractors(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert len(q.working) >= 3 and all(isinstance(w, str) and w for w in q.working)
            assert q.answer_text not in q.distractors
            assert len(set(q.distractors)) == len(q.distractors) == 3
            assert set(q.option_notes) == set(q.distractors)
            int(q.answer_text)
            for d in q.distractors:
                int(d)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_at_least_30_distinct_questions(self, difficulty):
        assert len({make(TOPIC, seed=s, difficulty=difficulty).text for s in range(60)}) >= 30

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_no_answer_dominates(self, difficulty):
        counts = Counter(make(TOPIC, seed=s, difficulty=difficulty).answer_text for s in range(300))
        assert max(counts.values()) <= 0.25 * 300

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_text_asks_for_a_number_not_an_explanation(self, difficulty):
        for seed in range(40):
            text = make(TOPIC, seed=seed, difficulty=difficulty).text.lower()
            assert "explain" not in text and "prove" not in text and "design" not in text
            assert "refinement" in text


class TestGrading:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_answer_accepted_distractors_and_nearby_values_rejected(self, difficulty):
        grade = GRADERS["number"]
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert grade(q, q.answer_text).correct
            for d in q.distractors:
                assert not grade(q, d).correct, f"seed {seed}: accepted distractor {d}"
            assert not grade(q, str(int(q.answer_text) + 1)).correct
            assert not grade(q, str(int(q.answer_text) - 1)).correct


class TestTierShapes:
    def test_easy_uses_both_constructions(self):
        seen = set()
        for seed in range(40):
            text = make(TOPIC, seed=seed, difficulty="easy").text
            seen.add("perpendicular bisector" if "perpendicular bisector" in text else "semicircle")
        assert seen == {"perpendicular bisector", "semicircle"}

    def test_medium_replaces_a_step_and_has_four_steps(self):
        for seed in range(40):
            text = make(TOPIC, seed=seed, difficulty="medium").text
            assert len(re.findall(r"Step \d:", text)) == 4
            assert re.search(r"A refinement replaces Step [234] with:", text)

    def test_medium_never_uses_pythagoras(self):
        for seed in range(40):
            assert " cm" not in make(TOPIC, seed=seed, difficulty="medium").text

    def test_hard_covers_all_three_refinement_targets_including_the_decoy(self):
        seen = set()
        for seed in range(80):
            q = make(TOPIC, seed=seed, difficulty="hard")
            m = re.search(r"A refinement changes (Step \d)", q.text)
            seen.add(m.group(1))
            if m.group(1) == "Step 2":
                # the compass setting is a decoy: the answer is the unrefined AC
                assert "unchanged by the refinement" in q.working[-1]
        assert seen == {"Step 1", "Step 2", "Step 4"}

    def test_every_misconception_note_is_declared_in_the_skill(self):
        _, skills = load_skills()
        declared = {
            m.id for m in next(s for s in skills if s.id == "f10-geometric-construction-algorithm").misconceptions
        }
        used = set()
        for difficulty in TIERS:
            for seed in range(120):
                used |= set(make(TOPIC, seed=seed, difficulty=difficulty).option_notes.values())
        assert used <= declared
        assert used == declared


class TestIndependentCheck:
    def test_easy_check_fires_on_corrupted_answer(self):
        q = make(TOPIC, seed=3, difficulty="easy")
        with pytest.raises(VerificationError):
            _check_easy(q.text, int(q.answer_text) + 1)

    def test_medium_check_fires_on_corrupted_answer(self):
        q = make(TOPIC, seed=3, difficulty="medium")
        with pytest.raises(VerificationError):
            _check_medium(q.text, int(q.answer_text) + 15)

    def test_hard_check_fires_on_corrupted_answer(self):
        q = make(TOPIC, seed=3, difficulty="hard")
        with pytest.raises(VerificationError):
            _check_hard(q.text, int(q.answer_text) + 1)


class TestSkillLoads:
    def test_loads(self):
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "f10-geometric-construction-algorithm"), None)
        assert skill is not None
        assert skill.domain == "geometry" and skill.answer == "number" and skill.status == "active"
        assert [s.difficulty for s in skill.sources[:3]] == TIERS
        assert skill.placement_eligible is False
        assert {"ac9m9sp03", "ac9m10sp03"} <= set(skill.tags)
