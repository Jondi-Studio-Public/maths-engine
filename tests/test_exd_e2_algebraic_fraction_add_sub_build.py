"""Tests for algebraic-fraction-add-subtract."""

import re

import pytest
from sympy import Symbol
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import symbolically_equal

TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)
PROOF_SEEDS = range(60)  # 60 seeds, matching the module's own sweep width.
_x = Symbol("x")
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

_SENTENCE_SUFFIXES = (
    ", giving your answer as a single fraction.",
    " to a single fraction.",
    ".",
)

# Marks a connector between two consecutive bracketed fraction terms in the
# question's own rendered text, e.g. ")/2 + (" or ")/9 - (" -- used only to
# recover each connector's sign (+1/-1) from `q.text`, never to rebuild the
# expression itself (see `_strip_sentence_frame`'s own docstring for why no
# rebuild is needed here, unlike the mul_div test file).
_CONNECTOR_RE = re.compile(r"\)/\d+ ([+-]) \(")


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMATIONS, local_dict={"x": _x})


def _all_questions(tier, seeds=SEEDS):
    return [make("exd_e2_algebraic_fraction_add_sub", seed=s, difficulty=tier) for s in seeds]


def _strip_sentence_frame(text: str) -> str:
    """The bare, literal unmerged-sum expression a question's text wraps --
    strips the "Simplify " prefix and whichever of the generator's three
    sentence frames was chosen for this seed. Unlike the mul_div test
    file's `_own_unreduced_text` (which has to REBUILD a cancelled product's
    uncancelled form from regex-captured numbers), this skill's prompt IS
    ALREADY the unmerged-sum text a learner might copy straight back -- the
    generator's own `_expr_text` produces exactly this string, so stripping
    the sentence frame is the whole job; no reconstruction, no risk of a
    hand-typed-once-and-reused text."""
    prefix = "Simplify "
    assert text.startswith(prefix), f"unrecognised sentence prefix: {text!r}"
    body = text[len(prefix) :]
    for suffix in _SENTENCE_SUFFIXES:
        if body.endswith(suffix):
            return body[: -len(suffix)]
    raise AssertionError(f"unrecognised sentence frame: {text!r}")


def _connector_signs(expr_text: str) -> list[str]:
    """The list of top-level connector signs ('+'/'-') between consecutive
    fraction terms, in order -- length 1 on easy/medium/hard, 2 on exam."""
    return _CONNECTOR_RE.findall(expr_text)


class TestAlgebraicFractionAddSub:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make("exd_e2_algebraic_fraction_add_sub", seed=seed, difficulty=tier)
            q2 = make("exd_e2_algebraic_fraction_add_sub", seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors
            assert q1.answer_form == q2.answer_form

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty_never_equal_answer(self, tier):
        for q in _all_questions(tier):
            assert len(q.distractors) > 0, f"{tier}: no distractors for {q.text!r}"
            for d in q.distractors:
                assert d != q.answer_text, f"{tier}: distractor equals answer in {q.text!r}"
            assert len(q.distractors) == len(set(q.distractors)), f"{tier}: duplicate distractors in {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_form_is_single_fraction(self, tier):
        for q in _all_questions(tier):
            assert q.answer_form == "single_fraction", f"{tier}: {q.text!r} should be answer_form=single_fraction"

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
    def test_distractor_own_raw_text_not_flagged_as_unmerged_sum(self, tier):
        """Every distractor is itself a single combined fraction, formatted
        the same shape as the answer -- confirm the real grader rejects a
        distractor for being the WRONG VALUE, not because its own raw text
        happens to trip the single_fraction unmerged-sum check: grade each
        distractor's text against ITSELF as if it were the canonical answer
        (a fresh question whose own answer_text is the distractor) and
        confirm THAT grades correct -- proving the distractor's shape alone
        clears single_fraction, so its rejection above is for value."""
        from dataclasses import replace

        for q in _all_questions(tier, seeds=range(20)):
            for d in q.distractors:
                self_check_question = replace(q, answer_text=d, answer=None)
                result = GRADERS["expression"](self_check_question, d)
                assert result.correct, (
                    f"{tier}: distractor {d!r} trips single_fraction against itself -- "
                    f"rejection above would be for shape, not value"
                )

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_rejects_deliberately_wrong_answer(self, tier):
        for q in _all_questions(tier, seeds=range(10)):
            wrong = "x + 999"
            result = GRADERS["expression"](q, wrong)
            assert not result.correct, f"{tier}: grader accepted deliberately wrong {wrong!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        texts = {q.text for q in _all_questions(tier)}
        assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions (need >= 20)"

    def test_easy_never_exercises_common_denominator_error_or_forgot_sign(self):
        # Easy is addition-only and coprime-denominator: only
        # added-denominators is ever meaningfully distinct there (an honest caveat) -- exactly one distractor, every seed.
        for q in _all_questions("easy"):
            assert len(q.distractors) == 1, f"easy: expected exactly 1 distractor, got {q.distractors!r}"

    @pytest.mark.parametrize("tier", ["medium", "hard"])
    def test_distractor_count_matches_sign_and_tier(self, tier):
        # Deterministic, every seed: added-denominators always present (+1);
        # common-denominator-error present iff tier is hard (+1);
        # forgot-to-distribute-the-sign present iff a '-' connector occurs.
        base = 2 if tier == "hard" else 1
        for q in _all_questions(tier):
            expr = _strip_sentence_frame(q.text)
            signs = _connector_signs(expr)
            expected = base + (1 if "-" in signs else 0)
            assert len(q.distractors) == expected, (
                f"{tier}: expected {expected} distractors (signs={signs}), got {q.distractors!r} for {q.text!r}"
            )

    def test_exam_distractor_count_matches_signs(self):
        # added-denominators + common-denominator-error always present (+2);
        # forgot-to-distribute-the-sign present iff any connector is '-'.
        for q in _all_questions("exam"):
            expr = _strip_sentence_frame(q.text)
            signs = _connector_signs(expr)
            assert len(signs) == 2, f"exam: expected 2 connectors, got {signs!r} for {q.text!r}"
            expected = 2 + (1 if "-" in signs else 0)
            assert len(q.distractors) == expected, (
                f"exam: expected {expected} distractors (signs={signs}), got {q.distractors!r} for {q.text!r}"
            )

    @pytest.mark.parametrize("tier", ["medium", "hard", "exam"])
    def test_forgot_to_distribute_the_sign_exercisable_over_200_seeds(self, tier):
        # Not exercisable on easy (addition-only) -- an honest
        # caveat. Measured, not assumed, on every other tier.
        seen = False
        for q in _all_questions(tier, seeds=range(200)):
            expr = _strip_sentence_frame(q.text)
            if "-" in _connector_signs(expr):
                seen = True
                break
        assert seen, f"{tier}: no subtraction observed over 200 seeds"

    @pytest.mark.parametrize("tier", ["hard", "exam"])
    def test_common_denominator_error_exercisable_over_200_seeds(self, tier):
        # A distractor differs from the answer only in denominator (same
        # numerator text) and that denominator is larger than the answer's
        # own -- the common-denominator-error shape.
        found = False
        for q in _all_questions(tier, seeds=range(200)):
            ans_num, ans_den = q.answer_text.rsplit("/", 1)
            for d in q.distractors:
                num, den = d.rsplit("/", 1)
                if num == ans_num and int(den) > int(ans_den):
                    found = True
                    break
            if found:
                break
        assert found, f"{tier}: common-denominator-error distractor never observed over 200 seeds"

    def test_exam_clears_three_distinct_shapes(self):
        from mathgen.text_quality import prompt_shape

        shapes = {prompt_shape(q) for q in _all_questions("exam")}
        assert len(shapes) >= 3, f"exam: only {len(shapes)} distinct prompt shapes over 60 seeds (need >= 3): {shapes}"


class TestOwnFormRejectedProof:
    """Also confirms the canonical, graded against itself, is ACCEPTED."""

    @pytest.mark.parametrize("tier", TIERS)
    def test_own_unmerged_sum_is_value_equal_but_rejected(self, tier):
        checked = 0
        for q in _all_questions(tier, seeds=PROOF_SEEDS):
            unmerged_text = _strip_sentence_frame(q.text)
            unmerged_expr = _parse(unmerged_text)
            checked += 1
            assert symbolically_equal(unmerged_expr, q.answer), (
                f"{tier}: {unmerged_text!r} is not value-equal to {q.answer!r} for {q.text!r} -- "
                "construction bug, the rejection proof below would be meaningless"
            )
            result = GRADERS["expression"](q, unmerged_text)
            assert not result.correct, (
                f"{tier}: grader accepted the question's own unmerged-sum form "
                f"{unmerged_text!r} for {q.text!r} (answer {q.answer_text!r})"
            )
            canonical_result = GRADERS["expression"](q, q.answer_text)
            assert canonical_result.correct, (
                f"{tier}: grader rejected the canonical {q.answer_text!r} against itself for {q.text!r}"
            )
        assert checked >= 40, f"{tier}: only {checked} questions checked (need >= 40)"
        print(f"{tier}: own-unmerged-sum value-equal-but-rejected proof: {checked}/{checked} passed, 0 failed")


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "algebraic-fraction-add-subtract")
        assert skill.domain == "expressions-equivalence"
        assert skill.answer == "expression"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {"fraction-arithmetic", "simplify-algebraic-fractions"}
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == "exd_e2_algebraic_fraction_add_sub"
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance", "exam"}
        exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
        assert len(exam_sources) == 1
        assert exam_sources[0].stages == ["exam"]
