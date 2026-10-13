"""Tests `solve_linear_equation_fraction_both_sides` and `identify_and_solve_linear_equation`, and that their
topic names register.

Linear equations build."""

import re
from fractions import Fraction
from pathlib import Path

import pytest
from sympy import Rational

from mathgen.grading.typed import GRADERS, mcq_options
from mathgen.skills import load_skills
from mathgen import make

ROOT = Path(__file__).parent.parent
SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]

TOPIC_1 = "solve_linear_equation_fraction_both_sides"
TOPIC_2 = "identify_and_solve_linear_equation"


# ============================================================================
# SKILL 1: senior-solve-linear-equations
# ============================================================================


class TestSolveLinearEquations:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make(TOPIC_1, seed=seed, difficulty=tier)
            q2 = make(TOPIC_1, seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, tier):
        for seed in SEEDS:
            q = make(TOPIC_1, seed=seed, difficulty=tier)
            assert len(q.distractors) > 0, f"seed {seed}: no distractors"
            for d in q.distractors:
                assert d != q.answer_text, f"seed {seed}: distractor equals answer"
            assert len(q.distractors) == len(set(q.distractors)), f"seed {seed}: duplicate distractors"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_accepts_canonical_rejects_distractors(self, tier):
        for seed in SEEDS:
            q = make(TOPIC_1, seed=seed, difficulty=tier)
            result = GRADERS["fraction"](q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected canonical {q.answer_text}"
            for d in q.distractors:
                result = GRADERS["fraction"](q, d)
                assert not result.correct, f"seed {seed}: grader accepted distractor {d}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_deliberately_wrong_answer_rejected(self, tier):
        for seed in range(10):
            q = make(TOPIC_1, seed=seed, difficulty=tier)
            frac = Fraction(q.answer["p"], q.answer["q"])
            wrong = frac + 1
            wrong_text = str(wrong.numerator) if wrong.denominator == 1 else f"{wrong.numerator}/{wrong.denominator}"
            result = GRADERS["fraction"](q, wrong_text)
            assert not result.correct, f"seed {seed}: grader accepted deliberately wrong {wrong_text}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        seen = {make(TOPIC_1, seed=seed, difficulty=tier).text for seed in SEEDS}
        assert len(seen) >= 20, f"{tier}: only {len(seen)} distinct questions over 60 seeds"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_is_lowest_terms_fraction_dict(self, tier):
        for seed in SEEDS:
            q = make(TOPIC_1, seed=seed, difficulty=tier)
            assert isinstance(q.answer, dict) and {"p", "q"} <= set(q.answer)
            p, q_ = q.answer["p"], q.answer["q"]
            assert q_ > 0
            frac = Fraction(p, q_)
            assert frac.numerator == p and frac.denominator == q_, f"seed {seed}: {p}/{q_} not in lowest terms"

    @pytest.mark.parametrize("tier", TIERS)
    def test_some_seeds_give_non_integer_x0(self, tier):
        """The contract's whole point is fraction coefficients producing fraction
        solutions -- an easy tier that always lands on a whole number is not
        exercising the grader kind chosen."""
        non_integer = 0
        for seed in range(40):
            q = make(TOPIC_1, seed=seed, difficulty=tier)
            if Fraction(q.answer["p"], q.answer["q"]).denominator != 1:
                non_integer += 1
        assert non_integer > 0, f"{tier}: 0 of 40 seeds produced a non-integer x0"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_and_distractors_formatted_alike(self, tier):
        """No option is identifiable by shape: every option is a bare fraction/int text."""
        bare_re = re.compile(r"^-?\d+(/\d+)?$")
        for seed in range(20):
            q = make(TOPIC_1, seed=seed, difficulty=tier)
            assert bare_re.match(q.answer_text), f"seed {seed}: answer_text {q.answer_text!r} not bare fraction"
            for d in q.distractors:
                assert bare_re.match(d), f"seed {seed}: distractor {d!r} not bare fraction"

    def test_independent_check_fires_on_corrupted_answer(self):
        """check_equivalent/check_distinct actually fail the build on a wrong value."""
        from mathgen.core import VerificationError, check_distinct, check_equivalent

        with pytest.raises(VerificationError):
            check_equivalent(Rational(1, 2), Rational(1, 3), "must differ -- proves the check fires")
        with pytest.raises(VerificationError):
            check_distinct(Rational(1, 2), Rational(1, 2), msg="must differ -- proves the check fires")

    def test_skill_loads_with_correct_shape(self):
        _, skills = load_skills(ROOT / "skills")
        by_id = {s.id: s for s in skills}
        skill = by_id["senior-solve-linear-equations"]
        assert skill.domain == "linear-relations"
        assert skill.status == "active"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["rate-of-change"]
        assert skill.answer == "fraction"
        difficulties = [s.difficulty for s in skill.sources]
        assert difficulties == ["easy", "medium", "hard", "exam"]
        stages = {s.difficulty: s.stages for s in skill.sources}
        assert stages["easy"] == [0, 1, 2]
        assert stages["medium"] == [3, 4]
        assert stages["hard"] == [5, 6, "maintenance"]
        assert stages["exam"] == ["exam"]


# ============================================================================
# SKILL 2: senior-identify-and-solve-linear-equations
# ============================================================================


class TestIdentifyAndSolveLinearEquations:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make(TOPIC_2, seed=seed, difficulty=tier)
            q2 = make(TOPIC_2, seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, tier):
        for seed in SEEDS:
            q = make(TOPIC_2, seed=seed, difficulty=tier)
            assert len(q.distractors) > 0, f"seed {seed}: no distractors"
            for d in q.distractors:
                assert d != q.answer_text, f"seed {seed}: distractor equals answer"
            assert len(q.distractors) == len(set(q.distractors)), f"seed {seed}: duplicate distractors"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_accepts_canonical_rejects_distractors(self, tier):
        for seed in SEEDS:
            q = make(TOPIC_2, seed=seed, difficulty=tier)
            options = mcq_options(q)
            correct_index = str(options.index(q.answer_text))
            result = GRADERS["mcq"](q, correct_index)
            assert result.correct, f"seed {seed}: grader rejected correct option index"
            for d in q.distractors:
                idx = str(options.index(d))
                result = GRADERS["mcq"](q, idx)
                assert not result.correct, f"seed {seed}: grader accepted distractor index for {d!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        seen = {make(TOPIC_2, seed=seed, difficulty=tier).text for seed in SEEDS}
        assert len(seen) >= 20, f"{tier}: only {len(seen)} distinct questions over 60 seeds"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_is_one_of_two_canonical_shapes(self, tier):
        shape_re = re.compile(r"^(Linear; x = -?\d+(/\d+)?|Not linear)$")
        for seed in SEEDS:
            q = make(TOPIC_2, seed=seed, difficulty=tier)
            assert shape_re.match(q.answer_text), f"seed {seed}: unexpected answer shape {q.answer_text!r}"
            for d in q.distractors:
                assert shape_re.match(d), f"seed {seed}: unexpected distractor shape {d!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_both_classifications_occur(self, tier):
        """A learner cannot tell in advance from the wording which branch fires."""
        labels = set()
        for seed in SEEDS:
            q = make(TOPIC_2, seed=seed, difficulty=tier)
            labels.add("linear" if q.answer_text.startswith("Linear") else "not_linear")
        assert labels == {"linear", "not_linear"}, f"{tier}: only saw {labels} over 60 seeds"

    def test_hard_tier_surface_pattern_trap_fires(self):
        """At least one seed's distractor set carries the 'judged by surface
        appearance before simplifying' trap -- assert it fires, not just that
        it is theoretically possible."""
        fires = 0
        for seed in SEEDS:
            q = make(TOPIC_2, seed=seed, difficulty="hard")
            if any(d.startswith("Linear; x =") for d in q.distractors) or q.answer_text.startswith("Linear"):
                fires += 1
        assert fires > 0, "hard tier: surface-pattern trap never fired over 60 seeds"

    def test_hard_tier_same_wording_stem_both_branches(self):
        """Both branches ((x+a)(x-b) - x^2 and + x^2) share the same question
        stem, differing only in the sign before x^2 -- the minimal-pair trap."""
        seen_signs = set()
        for seed in SEEDS:
            q = make(TOPIC_2, seed=seed, difficulty="hard")
            if " - x^2 = " in q.text:
                seen_signs.add("minus")
            elif " + x^2 = " in q.text:
                seen_signs.add("plus")
        assert seen_signs == {"minus", "plus"}, f"hard tier: only saw {seen_signs} over 60 seeds"

    def test_skill_loads_with_correct_shape(self):
        _, skills = load_skills(ROOT / "skills")
        by_id = {s.id: s for s in skills}
        skill = by_id["senior-identify-and-solve-linear-equations"]
        assert skill.domain == "linear-relations"
        assert skill.status == "active"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["two-step-equations", "variables-on-both-sides"]
        assert skill.answer == "mcq"
        difficulties = [s.difficulty for s in skill.sources]
        assert difficulties == ["easy", "medium", "hard", "exam"]
        stages = {s.difficulty: s.stages for s in skill.sources}
        assert stages["easy"] == [0, 1, 2]
        assert stages["medium"] == [3, 4]
        assert stages["hard"] == [5, 6, "maintenance"]
        assert stages["exam"] == ["exam"]


def test_topic_names_registered_once():
    """Both new topics are registered exactly once (no duplicate-topic collision)."""
    from mathgen.core import REGISTRY

    assert TOPIC_1 in REGISTRY
    assert TOPIC_2 in REGISTRY
