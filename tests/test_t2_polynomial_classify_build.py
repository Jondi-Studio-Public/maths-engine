"""Tests `classify-polynomials`: answers and skill loading.

Tests for classify-polynomials."""

import re

import pytest

from mathgen.grading.typed import GRADERS, mcq_options
from mathgen.skills import load_skills
from mathgen import make

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)

_MODE_A_PREFIX = "Classify this polynomial by degree and number of terms: "
_MODE_B_PREFIX = "Is this a polynomial? If not, why not: "
_ANSWER_RE = re.compile(r"^Degree (-?\d+), (monomial|binomial|trinomial)$")
_CATEGORIES = {
    "Yes, it is a polynomial",
    "No - it has a negative exponent",
    "No - it has a fractional exponent",
    "No - the variable is in the denominator",
}


def _all_questions(tier, seeds=SEEDS):
    return [make("polynomial_classify", seed=s, difficulty=tier) for s in seeds]


def _mode(q):
    if q.text.startswith(_MODE_A_PREFIX):
        return "A"
    assert q.text.startswith(_MODE_B_PREFIX), f"unrecognised prompt shape: {q.text!r}"
    return "B"


class TestClassifyPolynomials:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(40):
            q1 = make("polynomial_classify", seed=seed, difficulty=tier)
            q2 = make("polynomial_classify", seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, tier):
        for q in _all_questions(tier):
            assert len(q.distractors) > 0, f"{tier}: no distractors for {q.text!r}"
            for d in q.distractors:
                assert d != q.answer_text, f"{tier}: distractor equals answer in {q.text!r}"
            assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors in {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_three_distractors_except_the_deliberate_equivalent_reading_pair(self, tier):
        # pick_distractors drops a collision silently, so a count below 3 is normally the signal
        # that a candidate collided with the answer or another candidate
        # -- EXCEPT for a deliberate exclusion:
        # a "negative exponent"/"variable in the denominator" question always drops the other,
        # mathematically-equivalent label from the pool, so those two categories legitimately carry
        # exactly 2.
        equivalent_pair = {"No - it has a negative exponent", "No - the variable is in the denominator"}
        for q in _all_questions(tier):
            if q.answer_text in equivalent_pair:
                assert len(q.distractors) == 2, (
                    f"{tier}: expected exactly 2 distractors (equivalent-reading label excluded) for "
                    f"{q.text!r} -> {q.answer_text!r} vs {q.distractors!r}"
                )
            else:
                assert len(q.distractors) == 3, (
                    f"{tier}: only {len(q.distractors)} distractors (a candidate collided) for {q.text!r} "
                    f"-> {q.answer_text!r} vs {q.distractors!r}"
                )

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_accepts_canonical_answer(self, tier):
        for q in _all_questions(tier):
            options = mcq_options(q)
            idx = str(options.index(q.answer_text))
            result = GRADERS["mcq"](q, idx)
            assert result.correct, f"{tier}: grader rejected canonical index {idx!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_rejects_every_distractor(self, tier):
        for q in _all_questions(tier):
            options = mcq_options(q)
            for d in q.distractors:
                idx = str(options.index(d))
                result = GRADERS["mcq"](q, idx)
                assert not result.correct, f"{tier}: grader accepted distractor index {idx!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_rejects_deliberately_wrong_answer(self, tier):
        for q in _all_questions(tier, seeds=range(10)):
            options = mcq_options(q)
            wrong_idx = str(len(options) + 5)  # out of range -> never a valid option
            result = GRADERS["mcq"](q, wrong_idx)
            assert not result.correct, f"{tier}: grader accepted out-of-range index for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        texts = {q.text for q in _all_questions(tier)}
        assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions (need >= 20)"

    @pytest.mark.parametrize("tier", TIERS)
    def test_options_formatted_alike_per_mode(self, tier):
        for q in _all_questions(tier):
            options = [q.answer_text] + list(q.distractors)
            if _mode(q) == "A":
                for o in options:
                    assert _ANSWER_RE.match(o), f"{tier}: option {o!r} not 'Degree N, type' shape in {q.text!r}"
            else:
                for o in options:
                    assert o in _CATEGORIES, f"{tier}: option {o!r} not a fixed category string in {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_both_modes_present_across_tier(self, tier):
        questions = _all_questions(tier, seeds=range(200))
        mode_a = sum(1 for q in questions if _mode(q) == "A")
        mode_b = sum(1 for q in questions if _mode(q) == "B")
        assert mode_a > 0, f"{tier}: no Mode A questions over 200 seeds"
        assert mode_b > 0, f"{tier}: no Mode B questions over 200 seeds"

    @pytest.mark.parametrize("tier", TIERS)
    def test_mode_a_distractor_shapes_measured_over_200_seeds(self, tier):
        wrong_degree_right_type = 0
        right_degree_wrong_type = 0
        x_exponent_only = 0
        for q in _all_questions(tier, seeds=range(200)):
            if _mode(q) != "A":
                continue
            answer_m = _ANSWER_RE.match(q.answer_text)
            true_degree, true_type = int(answer_m.group(1)), answer_m.group(2)
            for d in q.distractors:
                dm = _ANSWER_RE.match(d)
                d_degree, d_type = int(dm.group(1)), dm.group(2)
                if d_type == true_type and d_degree != true_degree:
                    wrong_degree_right_type += 1
                    if tier == "hard":
                        # The x-exponent-only misconception is always < true
                        # degree and always keeps the true type label -- the
                        # SAME shape as "wrong-degree-right-type" from the
                        # oracle's point of view (both are just "some other
                        # degree, correct type"). Identify it from the
                        # rendered TEXT's own top-degree term, never from
                        # generator state: parse the top-degree term's x
                        # exponent, and this distractor's degree equals it.
                        expr = q.text[len(_MODE_A_PREFIX) :]
                        x_matches = [
                            (m.start(), int(m.group(1)) if m.group(1) else 1)
                            for m in re.finditer(r"x(?:\^(\d+))?", expr)
                        ]
                        if any(x_exp == d_degree for _pos, x_exp in x_matches) and d_degree < true_degree:
                            x_exponent_only += 1
                elif d_degree == true_degree and d_type != true_type:
                    right_degree_wrong_type += 1
        assert wrong_degree_right_type > 0, f"{tier}: wrong-degree-right-type never exercised over 200 seeds"
        assert right_degree_wrong_type > 0, f"{tier}: right-degree-wrong-type never exercised over 200 seeds"
        if tier == "hard":
            assert x_exponent_only > 0, f"{tier}: x-exponent-only distractor never exercised over 200 seeds"

    def test_hard_tier_x_over_2_trap_and_variable_denominator_both_appear(self):
        # "pair this with the x/2 trap ... so both appear across the tier's
        # own 60-seed sweep".
        questions = _all_questions("hard", seeds=range(60))
        x_over_2_trap = sum(
            1 for q in questions if _mode(q) == "B" and q.answer_text == "Yes, it is a polynomial" and "/" in q.text
        )
        variable_denominator = sum(1 for q in questions if q.answer_text == "No - the variable is in the denominator")
        assert x_over_2_trap > 0, "hard: x/2 constant-denominator trap never exercised over 60 seeds"
        assert variable_denominator > 0, "hard: variable-denominator violation never exercised over 60 seeds"


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "classify-polynomials")
        assert skill.domain == "expressions-equivalence"
        assert skill.answer == "mcq"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["combine-like-terms"]
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == "polynomial_classify"
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
