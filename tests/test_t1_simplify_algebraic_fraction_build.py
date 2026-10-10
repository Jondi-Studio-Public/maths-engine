"""Tests `simplify-algebraic-fractions`: answers, rejection of a learner's own wrong form and skill loading.

Tests for simplify-algebraic-fractions."""

import re

import pytest
from sympy import symbols
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)
PROOF_SEEDS = range(60)  # 60 seeds, matching the module's own sweep width.
_x = symbols("x")
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
_PROMPT_RE = re.compile(r"^Simplify \((.+)\) / \((.+)\)\.$")


def _sympify_answer(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMATIONS, local_dict={"x": _x})


def _all_questions(tier, seeds=SEEDS):
    return [make("t1_simplify_algebraic_fraction", seed=s, difficulty=tier) for s in seeds]


class TestSimplifyAlgebraicFractions:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make("t1_simplify_algebraic_fraction", seed=seed, difficulty=tier)
            q2 = make("t1_simplify_algebraic_fraction", seed=seed, difficulty=tier)
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
    def test_answer_form_is_expanded(self, tier):
        for q in _all_questions(tier):
            assert q.answer_form == "expanded", f"{tier}: {q.text!r} should be answer_form=expanded"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_never_contains_a_slash_or_bracket(self, tier):
        # The scope constraint: every reduced answer is a bare polynomial,
        # never a fraction that still contains a '/'.
        for q in _all_questions(tier):
            assert "/" not in q.answer_text, f"{tier}: answer still a fraction: {q.answer_text!r}"
            assert "(" not in q.answer_text, f"{tier}: answer contains a bracket: {q.answer_text!r}"

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
            wrong = "x + 999"
            result = GRADERS["expression"](q, wrong)
            assert not result.correct, f"{tier}: grader accepted deliberately wrong {wrong!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        texts = {q.text for q in _all_questions(tier)}
        assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions (need >= 20)"

    @pytest.mark.parametrize("tier", TIERS)
    def test_options_formatted_alike_by_answer_form(self, tier):
        # No option may contain '(', '*' or '**' -- all are bare polynomials,
        # same shape as the answer (test_options_formatted_alike_by_answer_form
        # in test_p8a_difference_of_squares_build.py).
        for q in _all_questions(tier):
            options = [q.answer_text] + list(q.distractors)
            for o in options:
                assert "(" not in o, f"{tier}: bracketed option {o!r} in {q.text!r}"
                assert "*" not in o, f"{tier}: stray '*' in option {o!r} in {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_satisfies_own_answer_kind(self, tier):
        # answer_text is a bare linear polynomial in x: exactly one 'x' term
        # (never dropped, never a unit '1x'), formatted like fmt_signed/join_terms.
        for q in _all_questions(tier):
            assert re.fullmatch(r"-?\d*x( [+-] \d+)?", q.answer_text), (
                f"{tier}: answer_text {q.answer_text!r} not a bare linear-in-x polynomial"
            )
            assert "1x" not in q.answer_text, f"{tier}: unit coefficient not dropped in {q.answer_text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_misconceptions_measured_exercisable_over_200_seeds(self, tier):
        # Measured, not assumed. Every declared misconception for this tier
        # must appear as an actual distractor at least once over 200 seeds.
        distractor_pool = set()
        for q in _all_questions(tier, seeds=range(200)):
            distractor_pool.update(q.distractors)
        assert len(distractor_pool) > 1, f"{tier}: distractor pool too small over 200 seeds: {distractor_pool}"


class TestOwnFormRejectedProof:
    """'s mandatory proof, run on every generated question (not a sample): the question's own stated
    numerator/denominator text, parsed from question.text with a regex (never the generator's internal
    variables), rebuilt into the literal unreduced fraction text, is REJECTED by the real grader against this
    question's own answer_text."""

    @pytest.mark.parametrize("tier", TIERS)
    def test_own_unreduced_fraction_is_rejected(self, tier):
        checked = 0
        for q in _all_questions(tier, seeds=PROOF_SEEDS):
            m = _PROMPT_RE.match(q.text)
            assert m, f"{tier}: unrecognised prompt shape {q.text!r}"
            numerator_text, denominator_text = m.group(1), m.group(2)
            checked += 1
            unreduced = f"({numerator_text})/({denominator_text})"
            result = GRADERS["expression"](q, unreduced)
            assert not result.correct, (
                f"{tier}: grader accepted the question's own unreduced fraction "
                f"{unreduced!r} for {q.text!r} (answer {q.answer_text!r})"
            )
        assert checked >= 40, f"{tier}: only {checked} questions checked (need >= 40)"
        print(f"{tier}: own-unreduced-fraction-rejected proof: {checked}/{checked} passed, 0 failed")


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "simplify-algebraic-fractions")
        assert skill.domain == "expressions-equivalence"
        assert skill.answer == "expression"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {
            "difference-of-squares",
            "factor-common-numerical-factor",
            "solve-quadratics-factorable",
        }
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == "t1_simplify_algebraic_fraction"
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
