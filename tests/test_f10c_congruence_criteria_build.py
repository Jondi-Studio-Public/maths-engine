"""Tests `f10c_congruence_criteria` (`f10-congruence-similarity-criteria`): determinism, the closed answer
space, one correct option per question, tagged distractors and the generator's own verification."""

import collections

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators import f10c_congruence_criteria_build as gen
from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills

TOPIC = "f10c_congruence_criteria"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)


def _skill():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}["f10-congruence-similarity-criteria"]


class TestShape:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_determinism(self, difficulty):
        for seed in SEEDS:
            assert (
                make(TOPIC, seed=seed, difficulty=difficulty).to_json()
                == make(TOPIC, seed=seed, difficulty=difficulty).to_json()
            )

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_four_distinct_labelled_options_from_the_closed_pool(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            options = [q.answer_text] + q.distractors
            assert len(options) == 4 and len(set(options)) == 4
            assert set(options) <= set(gen.ALL_LABELS)
            assert q.answer_text not in gen.FALSE_TESTS  # a false test is only ever a distractor
            assert q.working and all(isinstance(w, str) and w for w in q.working)

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_grader_accepts_the_answer_and_rejects_every_distractor(self, difficulty):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            options = mcq_options(q)
            assert grade_mcq(q, str(options.index(q.answer_text))).correct
            for d in q.distractors:
                assert not grade_mcq(q, str(options.index(d))).correct
            assert not grade_mcq(q, "99").correct

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_enough_distinct_questions(self, difficulty):
        assert len({make(TOPIC, seed=s, difficulty=difficulty).text for s in SEEDS}) >= 55

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_no_answer_dominates_and_the_correct_option_moves(self, difficulty):
        answers = collections.Counter()
        positions = collections.Counter()
        for seed in range(300):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            answers[q.answer_text] += 1
            positions[mcq_options(q).index(q.answer_text)] += 1
        assert max(answers.values()) / 300 <= 0.35
        assert len(answers) >= 5
        assert max(positions.values()) / 300 <= 0.4 and len(positions) == 4


class TestAnswerSpace:
    def test_easy_is_congruence_only_and_names_all_five_tests(self):
        seen = {make(TOPIC, seed=s, difficulty="easy").answer_text for s in range(200)}
        assert seen == set(gen.CONGRUENT_VALID)

    def test_medium_reaches_congruence_similarity_and_neither(self):
        seen = {make(TOPIC, seed=s, difficulty="medium").answer_text for s in range(400)}
        assert seen == set(gen.CONGRUENT_VALID) | set(gen.SIMILAR) | {gen.NEITHER}

    def test_hard_reaches_neither_and_both_families(self):
        seen = {make(TOPIC, seed=s, difficulty="hard").answer_text for s in range(500)}
        assert gen.NEITHER in seen and {gen.S_AA, gen.S_SAS, gen.C_SAS, gen.C_RHS} <= seen

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_a_similarity_label_is_never_offered_beside_a_congruence_answer(self, difficulty):
        for seed in range(300):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            if q.answer_text in gen.CONGRUENT_VALID:
                assert not (set(q.distractors) & set(gen.SIMILAR)), q.text

    @pytest.mark.parametrize("difficulty", TIERS)
    def test_asa_and_aas_are_not_offered_against_each_other(self, difficulty):
        """Two angles and a side meet one of them whichever way the side sits, once the third angle is found, so
        the sibling is never a wrong option beside the keyed one."""
        for seed in range(300):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            if q.answer_text in (gen.C_ASA, gen.C_AAS):
                assert not ({gen.C_ASA, gen.C_AAS} & set(q.distractors))


class TestMisconceptionTags:
    @pytest.mark.parametrize("difficulty", TIERS)
    def test_every_distractor_is_tagged_with_a_skill_misconception(self, difficulty):
        ids = {m.id for m in _skill().misconceptions}
        for seed in range(150):
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert set(q.option_notes) == set(q.distractors)
            assert set(q.option_notes.values()) <= ids

    def test_every_skill_misconception_is_exercised_and_nothing_untagged_is_declared(self):
        used = set()
        for tier in TIERS:
            for seed in range(400):
                used |= set(make(TOPIC, seed=seed, difficulty=tier).option_notes.values())
        declared = {m.id for m in _skill().misconceptions}
        assert used == declared

    def test_false_tests_carry_their_named_misconceptions(self):
        for tier in TIERS:
            for seed in range(200):
                q = make(TOPIC, seed=seed, difficulty=tier)
                if gen.C_SSA in q.distractors:
                    assert q.option_notes[gen.C_SSA] == gen.M_SSA
                if gen.C_AAA in q.distractors:
                    assert q.option_notes[gen.C_AAA] == gen.M_AAA


class TestHardTraps:
    def test_ssa_stems_have_neither_keyed_with_the_angle_not_between_the_sides(self):
        hits = 0
        for seed in range(500):
            q = make(TOPIC, seed=seed, difficulty="hard")
            if q.answer_text == gen.NEITHER and gen.M_ANY_ANGLE in q.option_notes.values():
                hits += 1
                assert gen.C_SSA in q.distractors
        assert hits >= 10

    def test_right_angle_between_legs_is_sas_and_with_the_hypotenuse_is_rhs(self):
        saw_sas = saw_rhs = False
        for seed in range(500):
            q = make(TOPIC, seed=seed, difficulty="hard")
            if q.option_notes.get(gen.C_RHS) == gen.M_RHS_SAS:
                assert q.answer_text == gen.C_SAS
                saw_sas = True
            if q.option_notes.get(gen.C_SAS) == gen.M_RHS_SAS:
                assert q.answer_text == gen.C_RHS
                saw_rhs = True
        assert saw_sas and saw_rhs


class TestOwnVerification:
    def test_generator_raises_if_the_definitions_disagree_with_the_key(self, monkeypatch):
        monkeypatch.setattr(gen, "_valid_labels", lambda t1, t2: set())
        with pytest.raises(VerificationError):
            make(TOPIC, seed=0, difficulty="easy")

    def test_valid_labels_on_hand_built_cases(self):
        T = gen.Tri
        # SAS: sides AB, BC with angle B between them (vertices 0, 1, 2).
        sas = (T({(0, 1): 5, (1, 2): 7}, {1: 40}), T({(0, 1): 5, (1, 2): 7}, {1: 40}))
        assert gen._valid_labels(*sas) == {gen.C_SAS, gen.S_SAS}
        # SSA: the angle at A is not between AB and BC.
        ssa = (T({(0, 1): 8, (1, 2): 6}, {0: 40}), T({(0, 1): 8, (1, 2): 6}, {0: 40}))
        assert gen._valid_labels(*ssa) == set()
        # RHS: right angle at B, hypotenuse AC and leg AB.
        rhs = (T({(0, 2): 13, (0, 1): 5}, {1: 90}), T({(0, 2): 13, (0, 1): 5}, {1: 90}))
        assert gen._valid_labels(*rhs) == {gen.C_RHS}
        # AA with the third angle found by the angle sum.
        aa = (T(angles={0: 40, 1: 65}), T(angles={0: 40, 2: 75}))
        assert gen._valid_labels(*aa) == {gen.S_AA}
        aa_bad = (T(angles={0: 40, 1: 65}), T(angles={0: 40, 2: 70}))
        assert gen._valid_labels(*aa_bad) == set()
        # One ratio breaks.
        broken = (T({(0, 1): 4, (1, 2): 5, (0, 2): 6}, {}), T({(0, 1): 8, (1, 2): 10, (0, 2): 13}, {}))
        assert gen._valid_labels(*broken) == set()


class TestLoadSkills:
    def test_skill_shape(self):
        skill = _skill()
        assert skill.answer == "mcq" and skill.status == "active"
        assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}
        assert "ac9m8sp01" in skill.tags and "ac9m8sp04" in skill.tags
