"""Tests `expand-binomial-trinomial` (`skills/expressions-equivalence/expand-binomial-trinomial.yaml`): answers
and rejection of the unexpanded product typed back."""

import pytest

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.text_quality import prompt_shape

TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)
PROOF_SEEDS = range(60)  # 60 seeds, matching the sweep width.
TOPIC = "exd_e2_expand_binomial_trinomial"


def _all_questions(tier, seeds=SEEDS):
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


def _own_unexpanded_text(question) -> str:
    """The question's own literal unexpanded product, parsed straight out of
    its own displayed text -- every question here is worded exactly
    "Expand and simplify (BINOMIAL)(SECOND FACTOR)."."""
    prefix, suffix = "Expand and simplify ", "."
    assert question.text.startswith(prefix) and question.text.endswith(suffix), (
        f"unrecognised question shape: {question.text!r}"
    )
    return question.text[len(prefix) : -len(suffix)]


class TestSharedAcrossTiers:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make(TOPIC, seed=seed, difficulty=tier)
            q2 = make(TOPIC, seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors
            assert q1.answer_form == q2.answer_form

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_form_is_expanded_on_every_question(self, tier):
        for q in _all_questions(tier):
            assert q.answer_form == "expanded", f"{tier}: {q.text!r} missing answer_form=expanded"

    @pytest.mark.parametrize("tier", TIERS)
    def test_canonical_never_carries_a_bracket(self, tier):
        for q in _all_questions(tier):
            assert "(" not in q.answer_text, f"{tier}: canonical {q.answer_text!r} carries a bracket"

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_nonempty_distinct_never_equal_answer(self, tier):
        for q in _all_questions(tier):
            assert len(q.distractors) >= 2, f"{tier}: too few distractors for {q.text!r}: {q.distractors}"
            for d in q.distractors:
                assert d != q.answer_text, f"{tier}: distractor equals answer in {q.text!r}"
                assert "(" not in d, f"{tier}: distractor {d!r} carries a bracket (must match answer's own shape)"
            assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors"

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
            wrong = "999x^2 + 999x - 999"
            result = GRADERS["expression"](q, wrong)
            assert not result.correct, f"{tier}: grader accepted deliberately wrong {wrong!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        texts = {q.text for q in _all_questions(tier)}
        assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions (need >= 20)"

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_3_prompt_shapes(self, tier):
        shapes = {prompt_shape(q) for q in _all_questions(tier)}
        assert len(shapes) >= 3, (
            f"{tier}: only {len(shapes)} distinct prompt shapes in 60 seeds (prompt variety needs >= 3)"
        )


class TestOwnWrongFormRejectedProof:
    """Every question, not a sample: the question's own unexpanded product,
    parsed straight out of its own text, must be rejected, and the canonical
    accepted -- the mandatory hazard proof."""

    @pytest.mark.parametrize("tier", TIERS)
    def test_own_unexpanded_product_rejected_every_seed(self, tier):
        checked = 0
        for q in _all_questions(tier, seeds=PROOF_SEEDS):
            own_form = _own_unexpanded_text(q)
            own_result = GRADERS["expression"](q, own_form)
            canon_result = GRADERS["expression"](q, q.answer_text)
            assert not own_result.correct, f"{tier}: accepted own unexpanded form {own_form!r} for {q.text!r}"
            assert canon_result.correct, f"{tier}: rejected canonical {q.answer_text!r} for {q.text!r}"
            checked += 1
        assert checked >= 40, f"{tier}: only {checked} questions checked (need >= 40)"
        print(f"expand-binomial-trinomial {tier}: own-wrong-form-rejected proof: {checked}/{checked} passed")


class TestVariableContentPerTier:
    """Confirms the stated escalation is real in the generated text/answer,
    not just claimed in the docstring."""

    def test_easy_has_no_y_and_no_symbolic_parameter(self):
        for q in _all_questions("easy"):
            # NOT a check on the raw q.text -- "simplify" itself contains a
            # "y", so the check is on the parsed product text (just the two
            # bracketed factors) and the answer, never the instruction prose.
            product_text = _own_unexpanded_text(q)
            assert "y" not in product_text and "y" not in q.answer_text, f"easy: unexpected y in {q.text!r}"
            assert "a" not in q.answer_text and "b" not in q.answer_text, (
                f"easy: unexpected symbolic parameter in {q.answer_text!r}"
            )

    def test_medium_has_y_but_no_symbolic_parameter(self):
        for q in _all_questions("medium"):
            assert "y" in q.answer_text, f"medium: expected a y term in {q.answer_text!r}"
            assert "a" not in q.answer_text and "b" not in q.answer_text, (
                f"medium: unexpected symbolic parameter in {q.answer_text!r}"
            )

    def test_hard_carries_symbol_a_but_not_b(self):
        for q in _all_questions("hard"):
            assert "x + a" in q.text, f"hard: expected the (x + a) binomial in {q.text!r}"
            assert "a" in q.answer_text, f"hard: expected a to survive into {q.answer_text!r}"
            assert "b" not in q.answer_text, f"hard: unexpected b in {q.answer_text!r}"

    def test_exam_carries_both_symbols_and_varies_which_slot_b_replaces(self):
        slots_seen = set()
        for q in _all_questions("exam"):
            assert "x + a" in q.text, f"exam: expected the (x + a) binomial in {q.text!r}"
            assert "a" in q.answer_text and "b" in q.answer_text, (
                f"exam: expected both a and b to survive into {q.answer_text!r}"
            )
            if "bx" in q.text and "by" not in q.text:
                slots_seen.add("x")
            elif "by" in q.text:
                slots_seen.add("y")
            else:
                slots_seen.add("const")
        assert len(slots_seen) == 3, f"exam: expected b to replace all 3 slots across seeds, saw {slots_seen}"


class TestDroppedCrossTermMisconception:
    """dropped-cross-term is medium tier and up only -- see the generator's
    own module docstring for why easy (a binomial x binomial) has no xy term
    to drop in the first place."""

    def test_easy_has_exactly_two_misconception_distractors(self):
        for q in _all_questions("easy", seeds=range(30)):
            assert len(q.distractors) <= 2, f"easy: expected at most 2 distractors, got {q.distractors}"

    @pytest.mark.parametrize("tier", ["medium", "hard", "exam"])
    def test_has_y_tiers_have_up_to_three_misconception_distractors(self, tier):
        for q in _all_questions(tier, seeds=range(30)):
            assert len(q.distractors) <= 3, f"{tier}: expected at most 3 distractors, got {q.distractors}"


class TestSkillLoads:
    def test_expand_binomial_trinomial_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "expand-binomial-trinomial")
        assert skill.domain == "expressions-equivalence"
        assert skill.answer == "expression"
        assert skill.placement_eligible is False
        stages_covered = set()
        difficulties = set()
        for source in skill.sources:
            assert source.topic == TOPIC
            stages_covered.update(source.stages)
            difficulties.add(source.difficulty)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
        assert difficulties == {"easy", "medium", "hard", "exam"}
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].stages == ["exam"]
