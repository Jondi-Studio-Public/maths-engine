"""Tests for: zero-product-property + polynomial-long-division."""

import re
from pathlib import Path

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_distinct, check_equivalent

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def _parse(text):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMATIONS)


ROOT = Path(__file__).parent.parent
SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]


# ============================================================================
# SKILL 1: zero-product-property
# ============================================================================


class TestZeroProductProperty:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make("zero_product_property", seed=seed, difficulty=tier)
            q2 = make("zero_product_property", seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty_and_never_equal_answer(self, tier):
        for seed in SEEDS:
            q = make("zero_product_property", seed=seed, difficulty=tier)
            assert len(q.distractors) >= 2, f"seed {seed}: too few distractors: {q.distractors}"
            for d in q.distractors:
                assert d != q.answer_text
            assert len(q.distractors) == len(set(q.distractors))

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_accepts_canonical_answer(self, tier):
        for seed in SEEDS:
            q = make("zero_product_property", seed=seed, difficulty=tier)
            result = GRADERS["list"](q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected canonical: {q.answer_text}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_rejects_every_distractor(self, tier):
        for seed in SEEDS:
            q = make("zero_product_property", seed=seed, difficulty=tier)
            for d in q.distractors:
                result = GRADERS["list"](q, d)
                assert not result.correct, f"seed {seed}: grader accepted distractor {d}: {q.text}"

    def test_real_grader_rejects_deliberately_wrong_answer(self):
        q = make("zero_product_property", seed=0, difficulty="easy")
        result = GRADERS["list"](q, "x = 999 or x = 998")
        assert not result.correct

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        texts = {make("zero_product_property", seed=s, difficulty=tier).text for s in SEEDS}
        assert len(texts) >= 20, f"tier {tier}: only {len(texts)} distinct questions (need >= 20)"

    def test_easy_tier_is_all_monic_no_bracket_ever_shows_a_leading_coefficient(self):
        pattern = re.compile(r"\(\d+x")  # a leading coefficient > 1 before x
        for seed in SEEDS:
            q = make("zero_product_property", seed=seed, difficulty="easy")
            assert not pattern.search(q.text), f"seed {seed}: easy tier is not all-monic: {q.text}"

    @pytest.mark.parametrize("tier", ["medium", "hard"])
    def test_medium_and_hard_always_have_a_non_monic_factor(self, tier):
        pattern = re.compile(r"\([2-9]\d*x")
        for seed in SEEDS:
            q = make("zero_product_property", seed=seed, difficulty=tier)
            assert pattern.search(q.text), f"seed {seed}: no non-monic factor: {q.text}"

    def test_hard_tier_has_three_factors(self):
        for seed in SEEDS:
            q = make("zero_product_property", seed=seed, difficulty="hard")
            assert q.text.count("(") == 3, f"seed {seed}: not three factors: {q.text}"

    def test_forgot_to_divide_misconception_exercisable_at_medium_and_hard_not_easy(self):
        """Measured, not assumed, over every seed at every tier: the forgot-to-divide-by-coefficient distractor
        is a genuinely different value whenever a non-monic factor exists (medium/hard) and collapses to the
        correct answer (so pick_distractors drops it) at easy, where every factor is monic -- disclosed
        honestly."""
        counts = {}
        for tier in TIERS:
            n = 0
            for seed in SEEDS:
                q = make("zero_product_property", seed=seed, difficulty=tier)
                # 2 structural misconceptions always present (sign-flip-both,
                # sign-flip-one); a 3rd distractor slot means forgot-to-divide
                # survived pick_distractors as a genuinely different value.
                if len(q.distractors) >= 3:
                    n += 1
            counts[tier] = n
        assert counts["easy"] == 0, f"forgot-to-divide unexpectedly exercisable at easy: {counts}"
        assert counts["medium"] == len(SEEDS), f"forgot-to-divide not always exercisable at medium: {counts}"
        assert counts["hard"] == len(SEEDS), f"forgot-to-divide not always exercisable at hard: {counts}"

    def test_independent_check_fires_on_corrupted_answer(self):
        with pytest.raises(VerificationError):
            check_equivalent(sympy.sympify("3"), sympy.sympify("4"), "must fail")
        with pytest.raises(VerificationError):
            check_distinct(sympy.sympify("3"), sympy.sympify("3"), msg="must fail")

    def test_skill_loads_with_correct_shape(self):
        _, skills = load_skills(ROOT / "skills")
        by_id = {s.id: s for s in skills}
        skill = by_id["zero-product-property"]
        assert skill.domain == "quadratics"
        assert skill.status == "active"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["two-step-equations"]
        assert skill.answer == "list"
        assert [s.difficulty for s in skill.sources] == [*TIERS, "exam"]
        for s in skill.sources:
            if s.difficulty == "exam":
                continue  # the exam source is covered by this skill's tests/test_exd_x_* file
            assert s.topic == "zero_product_property"


# ============================================================================
# SKILL 2: polynomial-long-division
# ============================================================================


class TestPolynomialLongDivision:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make("polynomial_long_division", seed=seed, difficulty=tier)
            q2 = make("polynomial_long_division", seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors

    @pytest.mark.parametrize("tier", TIERS)
    def test_distractors_distinct_nonempty_and_never_equal_answer(self, tier):
        for seed in SEEDS:
            q = make("polynomial_long_division", seed=seed, difficulty=tier)
            assert len(q.distractors) >= 2, f"seed {seed}: too few distractors: {q.distractors}"
            for d in q.distractors:
                assert d != q.answer_text
            assert len(q.distractors) == len(set(q.distractors))

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_accepts_canonical_answer(self, tier):
        for seed in SEEDS:
            q = make("polynomial_long_division", seed=seed, difficulty=tier)
            result = GRADERS["expression"](q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected canonical: {q.answer_text}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_real_grader_rejects_every_distractor(self, tier):
        for seed in SEEDS:
            q = make("polynomial_long_division", seed=seed, difficulty=tier)
            for d in q.distractors:
                result = GRADERS["expression"](q, d)
                assert not result.correct, f"seed {seed}: grader accepted distractor {d}: {q.text}"

    def test_real_grader_rejects_deliberately_wrong_answer(self):
        q = make("polynomial_long_division", seed=0, difficulty="easy")
        result = GRADERS["expression"](q, "x + 999")
        assert not result.correct

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        texts = {make("polynomial_long_division", seed=s, difficulty=tier).text for s in SEEDS}
        assert len(texts) >= 20, f"tier {tier}: only {len(texts)} distinct questions (need >= 20)"

    @pytest.mark.parametrize("tier", TIERS)
    def test_answer_and_distractors_carry_no_python_power_or_star(self, tier):
        for seed in SEEDS:
            q = make("polynomial_long_division", seed=seed, difficulty=tier)
            for option in [q.answer_text, *q.distractors, q.text]:
                assert "**" not in option, f"seed {seed}: ** in {option!r}"
                assert "*" not in option, f"seed {seed}: bare * in {option!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_divisor_is_never_monic(self, tier):
        """Boundary vs the shipped integer-polynomial-quotient-remainder,
        whose divisor is always monic: every tier here must have a divisor
        leading coefficient > 1."""
        pattern = re.compile(r"g\(x\) = ([2-9]\d*)x")
        for seed in SEEDS:
            q = make("polynomial_long_division", seed=seed, difficulty=tier)
            match = pattern.search(q.text)
            assert match, f"seed {seed}: divisor is monic (or unparsed): {q.text}"

    def test_hard_tier_divisor_is_quadratic(self):
        for seed in SEEDS:
            q = make("polynomial_long_division", seed=seed, difficulty="hard")
            match = re.search(r"g\(x\) = (.+)\. Give", q.text)
            assert match, q.text
            divisor_expr = _parse(match.group(1))
            assert sympy.degree(divisor_expr, sympy.Symbol("x")) == 2, f"seed {seed}: {q.text}"

    def test_independent_check_fires_on_corrupted_answer(self):
        with pytest.raises(VerificationError):
            check_equivalent(sympy.sympify("x + 1"), sympy.sympify("x + 2"), "must fail")

    def test_skill_loads_with_correct_shape(self):
        _, skills = load_skills(ROOT / "skills")
        by_id = {s.id: s for s in skills}
        skill = by_id["polynomial-long-division"]
        assert skill.domain == "quadratics"
        assert skill.status == "active"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["expand-single-brackets", "combine-like-terms"]
        assert skill.answer == "expression"
        assert [s.difficulty for s in skill.sources] == [*TIERS, "exam"]
        for s in skill.sources:
            if s.difficulty == "exam":
                continue  # the exam source is covered by this skill's tests/test_exd_x_* file
            assert s.topic == "polynomial_long_division"

    # -- the disclosed, measured grading gap -----

    _DIVIDEND_DIVISOR_RE = re.compile(r"f\(x\) = (.+) by g\(x\) = (.+)\. Give")

    @pytest.mark.parametrize("tier", TIERS)
    def test_measure_the_raw_dividend_over_divisor_loophole(self, tier):
        """The single-rational-expression format is accepted by value
        equivalence, which also accepts the UNREDUCED dividend/divisor typed
        straight back from the question -- a learner who never divided
        anything is marked correct. Measured, not assumed: this is an exact
        100% accept rate by algebraic identity (Q + R/D IS dividend/divisor
        by construction, not an approximation of it), reported per tier over
        60 seeds so a future change to this generator that broke the
        identity would show up as a drop from 60/60."""
        accepted = 0
        for seed in SEEDS:
            q = make("polynomial_long_division", seed=seed, difficulty=tier)
            match = self._DIVIDEND_DIVISOR_RE.search(q.text)
            assert match, q.text
            dividend_text, divisor_text = match.group(1), match.group(2)
            raw = f"({dividend_text})/({divisor_text})"
            result = GRADERS["expression"](q, raw)
            if result.correct:
                accepted += 1
        assert accepted == len(SEEDS), (
            f"tier {tier}: raw dividend/divisor accepted {accepted}/{len(SEEDS)} times "
            "(expected all: the raw dividend and divisor must be accepted)"
        )
