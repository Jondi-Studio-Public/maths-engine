"""Tests for rationalise-binomial-denominator."""

import re

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)
PROOF_SEEDS = range(60)  # 60 seeds, matching the module's own sweep width.

_QUESTION_RE = re.compile(
    r"^Rationalise the denominator of (\d+)/\((.+?)\)\. Give your answer as a single "
    r"simplified expression with a rational denominator\.$"
)
# The denominator group captured above carries a unicode '√' (never ASCII
# "sqrt(" -- see the generator's `_denom_text` docstring); the real grader
# normalises '√' back to "sqrt(" before comparing, so building the
# "own wrong form" string straight from this capture is still a fair proof.


def _all_questions(tier, seeds=SEEDS):
    return [make("t1_rationalise_binomial", seed=s, difficulty=tier) for s in seeds]


class TestRationaliseBinomialDenominator:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make("t1_rationalise_binomial", seed=seed, difficulty=tier)
            q2 = make("t1_rationalise_binomial", seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors
            assert q1.answer_form == q2.answer_form

    @pytest.mark.parametrize("tier", TIERS)
    def test_question_text_is_recognised_shape(self, tier):
        for q in _all_questions(tier):
            assert _QUESTION_RE.match(q.text), f"{tier}: unrecognised prompt shape {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_form_is_rationalised(self, tier):
        for q in _all_questions(tier):
            assert q.answer_form == "rationalised", f"{tier}: {q.text!r} should be answer_form=rationalised"

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, tier):
        for q in _all_questions(tier):
            assert len(q.distractors) > 0, f"{tier}: no distractors for {q.text!r}"
            for d in q.distractors:
                assert d != q.answer_text, f"{tier}: distractor equals answer in {q.text!r}"
            assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors in {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_three_misconceptions_present_every_question(self, tier):
        # All three misconceptions are constructed unconditionally on every
        # question (unlike a form-dependent skill) -- confirm pick_distractors
        # never had to drop one because it accidentally collided with another
        # candidate or the answer, over a wide seed sweep.
        for q in _all_questions(tier, seeds=range(200)):
            assert len(q.distractors) == 3, f"{tier}: expected 3 distractors, got {q.distractors} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_accepts_canonical_answer(self, tier):
        for q in _all_questions(tier):
            result = GRADERS["expression"](q, q.answer_text)
            assert result.correct, f"{tier}: grader rejected canonical {q.answer_text!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_rejects_every_distractor(self, tier):
        for q in _all_questions(tier):
            for d in q.distractors:
                result = GRADERS["expression"](q, d)
                assert not result.correct, f"{tier}: grader accepted distractor {d!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_rejects_deliberately_wrong_answer(self, tier):
        for q in _all_questions(tier, seeds=range(10)):
            wrong = "999 - sqrt(999)" if "(" in q.answer_text else "999"
            result = GRADERS["expression"](q, wrong)
            assert not result.correct, f"{tier}: grader accepted deliberately wrong {wrong!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        texts = {q.text for q in _all_questions(tier)}
        assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions (need >= 20)"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_and_distractors_with_radical_formatted_alike(self, tier):
        # Every option carrying a radical is shown as "(numerator)/denominator"
        # -- the same shape as the answer -- never bracket-free while the
        # answer is bracketed or vice versa. `forgot-numerator` is the one
        # misconception explicitly defined as "a plain rational number, no
        # radical at all" (skill YAML) and is exempt from this rule by its
        # own definition; it is checked instead by test_forgot_numerator_
        # distractor_has_no_radical below.
        for q in _all_questions(tier):
            answer_has_paren = q.answer_text.startswith("(")
            for d in q.distractors:
                if "sqrt(" not in d:
                    continue
                assert d.startswith("(") == answer_has_paren, (
                    f"{tier}: radical-bearing option {d!r} does not match answer shape {q.answer_text!r} for {q.text!r}"
                )

    @pytest.mark.parametrize("tier", TIERS)
    def test_forgot_numerator_distractor_has_no_radical(self, tier):
        for q in _all_questions(tier):
            no_radical = [d for d in q.distractors if "sqrt(" not in d]
            assert len(no_radical) == 1, f"{tier}: expected exactly one no-radical distractor in {q.distractors}"
            assert re.match(r"^\d+/\d+$", no_radical[0]), f"{tier}: forgot-numerator distractor malformed: {no_radical}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_denominator_has_no_surviving_radical(self, tier):
        # The declared answer_form ("rationalised") already makes the real
        # grader enforce this; re-assert it directly against the generator's
        # own answer_text as a construction sanity check.
        for q in _all_questions(tier):
            if "/" not in q.answer_text:
                continue  # a bare two-term answer (D' == 1) carries no denominator at all
            denom = q.answer_text.rsplit("/", 1)[1]
            assert "sqrt(" not in denom, f"{tier}: answer denominator still carries a radical: {q.answer_text!r}"


class TestOwnWrongFormRejectedProof:
    """The mandatory proof: the question's own un-rationalised fraction, built from `question.text` alone
    (never the generator's internal a/b/c/s variables), is REJECTED by the real grader against this question's
    own answer_text -- for every question, over >= 40 seeds at every tier."""

    @pytest.mark.parametrize("tier", TIERS)
    def test_own_unrationalised_form_rejected(self, tier):
        checked = 0
        for q in _all_questions(tier, seeds=PROOF_SEEDS):
            m = _QUESTION_RE.match(q.text)
            assert m, f"{tier}: unrecognised prompt shape {q.text!r}"
            c_text, denom_text = m.group(1), m.group(2)
            wrong = f"{c_text}/({denom_text})"
            result = GRADERS["expression"](q, wrong)
            assert not result.correct, (
                f"{tier}: question {q.text!r} (answer {q.answer_text!r}) accepted its own "
                f"un-rationalised form {wrong!r}"
            )
            checked += 1
        assert checked >= 40, f"{tier}: only {checked} questions sampled (need >= 40)"
        print(f"{tier}: own-wrong-form-rejected proof: {checked}/{checked} passed, 0 failed")


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "rationalise-binomial-denominator")
        assert skill.domain == "secondary-number"
        assert skill.answer == "expression"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {"simplify-surds", "difference-of-squares"}
        stages_covered = set()
        for source in skill.sources:
            assert source.topic in ("t1_rationalise_binomial", "t1_rationalise_binomial_sum")
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
