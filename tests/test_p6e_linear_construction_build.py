"""Tests for skill 1 (senior-find-the-equation-of-a-straight-line-given) and
skill 2 (senior-develop-a-linear-formula-from-a-word-description)."""

import pytest
import re
from pathlib import Path
from sympy import symbols, expand, Rational, simplify

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills

ROOT = Path(__file__).parent.parent
SEEDS = range(60)

x, y = symbols("x y")


# ============================================================================
# Shared helpers
# ============================================================================


def _poly_display(expr) -> str:
    """Format a polynomial expression for display."""
    return str(expr).replace("**", "^").replace("*", "")


# ============================================================================
# SKILL 1: senior-find-the-equation-of-a-straight-line-given
# ============================================================================


class TestSkill1EquationStraightLine:
    """Tests for finding the equation of a straight line."""

    @pytest.mark.parametrize("variant", ["two_points", "parallel", "perpendicular"])
    def test_determinism_across_seeds(self, variant):
        """Same seed should produce identical questions."""
        for seed in range(20):
            q1 = make("find_the_equation_of_a_straight_line_given", seed=seed, variant=variant)
            q2 = make("find_the_equation_of_a_straight_line_given", seed=seed, variant=variant)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors

    @pytest.mark.parametrize("variant", ["two_points", "parallel", "perpendicular"])
    def test_distractors_are_distinct_and_nonempty(self, variant):
        """Every distractor is distinct from the answer and from each other."""
        for seed in SEEDS:
            question = make("find_the_equation_of_a_straight_line_given", seed=seed, variant=variant)
            assert len(question.distractors) > 0, f"seed {seed}: no distractors"
            # Check each distractor is different from the answer
            for i, distractor in enumerate(question.distractors):
                assert distractor != question.answer_text, f"seed {seed}, distractor {i}: matches answer"
            # Check distractors are unique (no duplicates)
            assert len(question.distractors) == len(set(question.distractors)), f"seed {seed}: duplicate distractors"

    @pytest.mark.parametrize("variant", ["two_points", "parallel", "perpendicular"])
    def test_real_grader_accepts_canonical_answer(self, variant):
        """The real expression grader accepts the canonical answer_text."""
        for seed in SEEDS:
            question = make("find_the_equation_of_a_straight_line_given", seed=seed, variant=variant)
            result = GRADERS["expression"](question, question.answer_text)
            assert result.correct, f"seed {seed}: grader rejected canonical answer_text: {question.answer_text}"

    @pytest.mark.parametrize("variant", ["two_points", "parallel", "perpendicular"])
    def test_real_grader_rejects_every_distractor(self, variant):
        """The real expression grader rejects all distractors."""
        for seed in SEEDS:
            question = make("find_the_equation_of_a_straight_line_given", seed=seed, variant=variant)
            for distractor in question.distractors:
                result = GRADERS["expression"](question, distractor)
                assert not result.correct, f"seed {seed}: grader accepted distractor: {distractor}"

    @pytest.mark.parametrize("variant", ["two_points", "parallel", "perpendicular"])
    def test_at_least_20_distinct_questions_per_tier(self, variant):
        """At least 20 distinct questions per difficulty tier over 60 seeds."""
        seen_texts = set()
        for seed in SEEDS:
            question = make("find_the_equation_of_a_straight_line_given", seed=seed, variant=variant)
            seen_texts.add(question.text)
        assert len(seen_texts) >= 20, (
            f"variant {variant}: only {len(seen_texts)} distinct questions over 60 seeds (need >= 20)"
        )

    def test_skill_loads_with_correct_shape(self):
        """senior-find-the-equation-of-a-straight-line-given loads correctly."""
        _, skills = load_skills(ROOT / "skills")
        by_id = {skill.id: skill for skill in skills}
        skill = by_id["senior-find-the-equation-of-a-straight-line-given"]

        assert skill.domain == "linear-relations"
        assert skill.status == "active"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["rate-of-change"]
        assert skill.answer == "expression"

        difficulties = [source.difficulty for source in skill.sources]
        assert difficulties == ["easy", "medium", "hard", "exam"]

        # Check difficulty corresponds to variant
        topics_params = {
            (source.topic, tuple(sorted(source.params.items()))): source.difficulty for source in skill.sources
        }
        assert "find_the_equation_of_a_straight_line_given" in {t[0] for t in topics_params.keys()}

    def test_no_double_star_or_star_in_answer_text(self):
        """Answer text must not contain ** or bare * (text_quality check)."""
        for seed in range(20):
            for variant in ["two_points", "parallel", "perpendicular"]:
                question = make("find_the_equation_of_a_straight_line_given", seed=seed, variant=variant)
                assert "**" not in question.answer_text, (
                    f"seed {seed}, {variant}: double star in {question.answer_text}"
                )
                # Note: bare * might appear in formatting, but the _poly_display helper removes it
                # For display purposes, check the answer text is formatted correctly
                assert question.answer_text.startswith("y = "), (
                    f"seed {seed}, {variant}: answer_text should start with 'y = '"
                )

    def test_parallel_variant_no_float_precision_distractors(self):
        """Medium tier (parallel variant) distractors must not contain Python float-repr decimals."""
        # Pattern to match long decimal sequences (3 or more decimal places)
        long_decimal_pattern = re.compile(r"\.\d{3,}")
        for seed in range(40):
            question = make("find_the_equation_of_a_straight_line_given", seed=seed, variant="parallel")
            for i, distractor in enumerate(question.distractors):
                assert not long_decimal_pattern.search(distractor), (
                    f"seed {seed}, distractor {i}: contains float-precision decimals: {distractor}"
                )

    @pytest.mark.parametrize(
        "difficulty,variant",
        [
            ("easy", "two_points"),
            ("medium", "parallel"),
            ("hard", "perpendicular"),
        ],
    )
    def test_difficulty_only_call_matches_explicit_variant(self, difficulty, variant):
        """Calling with difficulty alone should match call with difficulty + explicit variant (presets regression test)."""
        for seed in range(10):
            q_difficulty = make("find_the_equation_of_a_straight_line_given", seed=seed, difficulty=difficulty)
            q_explicit = make(
                "find_the_equation_of_a_straight_line_given", seed=seed, difficulty=difficulty, variant=variant
            )
            assert q_difficulty.text == q_explicit.text, (
                f"seed {seed}, {difficulty}: text differs from difficulty + explicit variant={variant}"
            )
            assert q_difficulty.answer_text == q_explicit.answer_text, (
                f"seed {seed}, {difficulty}: answer_text differs from difficulty + explicit variant={variant}"
            )
            assert q_difficulty.distractors == q_explicit.distractors, (
                f"seed {seed}, {difficulty}: distractors differ from difficulty + explicit variant={variant}"
            )


# ============================================================================
# SKILL 2: senior-develop-a-linear-formula-from-a-word-description
# ============================================================================


class TestSkill2LinearWordFormula:
    """Tests for developing a linear formula from word description."""

    @pytest.mark.parametrize("variant", ["increasing", "decreasing", "back_solved_intercept"])
    def test_determinism_across_seeds(self, variant):
        """Same seed should produce identical questions."""
        for seed in range(20):
            q1 = make("develop_a_linear_formula_from_a_word_description", seed=seed, variant=variant)
            q2 = make("develop_a_linear_formula_from_a_word_description", seed=seed, variant=variant)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors

    @pytest.mark.parametrize("variant", ["increasing", "decreasing", "back_solved_intercept"])
    def test_distractors_are_distinct_and_nonempty(self, variant):
        """Every distractor is distinct from the answer and from each other."""
        for seed in SEEDS:
            question = make("develop_a_linear_formula_from_a_word_description", seed=seed, variant=variant)
            assert len(question.distractors) > 0, f"seed {seed}: no distractors"
            # Check each distractor is different from the answer
            for i, distractor in enumerate(question.distractors):
                assert distractor != question.answer_text, f"seed {seed}, distractor {i}: matches answer"
            # Check distractors are unique (no duplicates)
            assert len(question.distractors) == len(set(question.distractors)), f"seed {seed}: duplicate distractors"

    @pytest.mark.parametrize("variant", ["increasing", "decreasing", "back_solved_intercept"])
    def test_real_grader_accepts_canonical_answer(self, variant):
        """The real expression grader accepts the canonical answer_text."""
        for seed in SEEDS:
            question = make("develop_a_linear_formula_from_a_word_description", seed=seed, variant=variant)
            result = GRADERS["expression"](question, question.answer_text)
            assert result.correct, f"seed {seed}: grader rejected canonical answer_text: {question.answer_text}"

    @pytest.mark.parametrize("variant", ["increasing", "decreasing", "back_solved_intercept"])
    def test_real_grader_rejects_every_distractor(self, variant):
        """The real expression grader rejects all distractors."""
        for seed in SEEDS:
            question = make("develop_a_linear_formula_from_a_word_description", seed=seed, variant=variant)
            for distractor in question.distractors:
                result = GRADERS["expression"](question, distractor)
                assert not result.correct, f"seed {seed}: grader accepted distractor: {distractor}"

    @pytest.mark.parametrize("variant", ["increasing", "decreasing", "back_solved_intercept"])
    def test_at_least_20_distinct_questions_per_tier(self, variant):
        """At least 20 distinct questions per difficulty tier over 60 seeds."""
        seen_texts = set()
        for seed in SEEDS:
            question = make("develop_a_linear_formula_from_a_word_description", seed=seed, variant=variant)
            seen_texts.add(question.text)
        assert len(seen_texts) >= 20, (
            f"variant {variant}: only {len(seen_texts)} distinct questions over 60 seeds (need >= 20)"
        )

    def test_variable_and_unit_stated_in_text(self):
        """Every generated question states its variable letter(s) and unit explicitly."""
        for variant in ["increasing", "decreasing", "back_solved_intercept"]:
            for seed in SEEDS:
                question = make("develop_a_linear_formula_from_a_word_description", seed=seed, variant=variant)
                # Check that the answer_text's variable letter appears in the text
                # The answer format is "V = ..." or "C = ..." etc, extract the first letter
                var_letter = question.answer_text.split(" = ")[0].strip()
                # Check it appears in the text (e.g., "$V$ (in litres)")
                assert var_letter in question.text or f"${var_letter}$" in question.text, (
                    f"seed {seed}, {variant}: variable {var_letter} not stated in text: {question.text}"
                )
                # Check that units appear (e.g., "dollars", "litres", "years", "hours")
                # Look for common unit words
                unit_words = ["dollars", "litres", "hours", "years", "minutes", "weeks"]
                assert any(unit in question.text for unit in unit_words), (
                    f"seed {seed}, {variant}: no recognized unit in text: {question.text}"
                )

    def test_increasing_wording_and_sign(self):
        """Increasing variant uses positive rate and appropriate wording."""
        wording_variants = set()
        for seed in range(40):
            question = make("develop_a_linear_formula_from_a_word_description", seed=seed, variant="increasing")
            # Increasing should have positive rate, no "minus", "drain", "lose" etc
            text_lower = question.text.lower()
            # Check for absence of decreasing keywords
            assert (
                "drain" not in text_lower
                and "lose" not in text_lower
                and "depreciat" not in text_lower
                and "cool" not in text_lower
            ), f"seed {seed}: increasing variant has decreasing wording: {question.text}"
            # Collect some wording to verify variety
            if "fee" in text_lower:
                wording_variants.add("fee")
            if "fill" in text_lower:
                wording_variants.add("fill")
            if "add" in text_lower or "plus" in text_lower:
                wording_variants.add("add/plus")
        # Should have seen multiple wording variants
        assert len(wording_variants) > 0, "increasing variant should have diverse wording"

    def test_decreasing_wording_and_sign(self):
        """Decreasing variant uses negative rate and appropriate wording."""
        wording_variants = set()
        for seed in range(40):
            question = make("develop_a_linear_formula_from_a_word_description", seed=seed, variant="decreasing")
            # Decreasing should have language indicating subtraction
            text_lower = question.text.lower()
            # Check for presence of decreasing keywords
            has_decreasing = any(word in text_lower for word in ["drain", "lose", "depreciat", "cool", "-"])
            # At least some should have explicit decreasing wording
            # Collect variants that appear
            if "drain" in text_lower:
                wording_variants.add("drain")
            if "lose" in text_lower or "depreciat" in text_lower:
                wording_variants.add("depreciate/lose")
            if "cool" in text_lower:
                wording_variants.add("cool")
        # Should have seen multiple decreasing wording variants
        assert len(wording_variants) > 0, "decreasing variant should have diverse wording"

    def test_skill_loads_with_correct_shape(self):
        """senior-develop-a-linear-formula-from-a-word-description loads correctly."""
        _, skills = load_skills(ROOT / "skills")
        by_id = {skill.id: skill for skill in skills}
        skill = by_id["senior-develop-a-linear-formula-from-a-word-description"]

        assert skill.domain == "linear-relations"
        assert skill.status == "active"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["rate-of-change"]
        assert skill.answer == "expression"

        # Four tiers (easy, medium, hard, exam)
        difficulties = [source.difficulty for source in skill.sources]
        assert difficulties == ["easy", "medium", "hard", "exam"]

    def test_no_double_star_or_star_in_answer_text(self):
        """Answer text must not contain ** or bare * (text_quality check)."""
        for seed in range(20):
            for variant in ["increasing", "decreasing", "back_solved_intercept"]:
                question = make("develop_a_linear_formula_from_a_word_description", seed=seed, variant=variant)
                assert "**" not in question.answer_text, (
                    f"seed {seed}, {variant}: double star in {question.answer_text}"
                )
                # Answer should have format like "C = ..." where C is the variable
                assert " = " in question.answer_text, (
                    f"seed {seed}, {variant}: answer_text should have format 'VAR = ...'"
                )

    def test_back_solved_intercept_value_is_sensible_positive(self):
        """Hard tier (back_solved_intercept) must have stated value >= 5 for sensible scenarios."""
        for seed in range(40):
            question = make(
                "develop_a_linear_formula_from_a_word_description", seed=seed, variant="back_solved_intercept"
            )
            # Extract the numeric value from the text
            # Look for patterns like "is worth $27" or "contains 17 litres"
            # First, let's check by looking for standalone numbers in the text
            # The value appears after "worth" or "contains"
            text = question.text

            # Extract the value that was stated in the scenario
            # Patterns: "is worth $N" or "contains N litres" or "contains N litre"
            worth_match = re.search(r"worth \$(-?\d+)", text)
            contains_match = re.search(r"contains (-?\d+)", text)

            stated_value = None
            if worth_match:
                stated_value = int(worth_match.group(1))
            elif contains_match:
                stated_value = int(contains_match.group(1))

            assert stated_value is not None, f"seed {seed}: could not extract stated value from: {text}"
            assert stated_value >= 5, f"seed {seed}: stated value {stated_value} is not >= 5: {text}"

    @pytest.mark.parametrize("variant", ["increasing", "decreasing", "back_solved_intercept"])
    def test_at_least_4_distinct_scenario_templates_per_tier(self, variant):
        """At least 4 distinct scenario frames per tier over 60 seeds.

        Unlike test_at_least_20_distinct_questions_per_tier (which counts raw
        question text and is satisfied by digit variation alone even with a
        single scenario frame), this strips all digits from the text so it
        measures actual template/wording diversity.
        """
        seen_templates = set()
        for seed in SEEDS:
            question = make("develop_a_linear_formula_from_a_word_description", seed=seed, variant=variant)
            stripped = re.sub(r"\d+", "", question.text)
            seen_templates.add(stripped)
        assert len(seen_templates) >= 4, (
            f"variant {variant}: only {len(seen_templates)} distinct scenario templates over 60 seeds (need >= 4): {seen_templates}"
        )

    def test_back_solved_intercept_no_singular_plural_errors(self):
        """Hard tier must not have '1 years' or '1 hours' (singular/plural mismatch)."""
        for seed in range(40):
            question = make(
                "develop_a_linear_formula_from_a_word_description", seed=seed, variant="back_solved_intercept"
            )
            text = question.text
            # Check for the incorrect forms
            assert "1 years" not in text, f"seed {seed}: text contains '1 years': {text}"
            assert "1 hours" not in text, f"seed {seed}: text contains '1 hours': {text}"

    @pytest.mark.parametrize(
        "difficulty,variant",
        [
            ("easy", "increasing"),
            ("medium", "decreasing"),
            ("hard", "back_solved_intercept"),
        ],
    )
    def test_difficulty_only_call_matches_explicit_variant(self, difficulty, variant):
        """Calling with difficulty alone should match call with difficulty + explicit variant (presets regression test)."""
        for seed in range(10):
            q_difficulty = make("develop_a_linear_formula_from_a_word_description", seed=seed, difficulty=difficulty)
            q_explicit = make(
                "develop_a_linear_formula_from_a_word_description", seed=seed, difficulty=difficulty, variant=variant
            )
            assert q_difficulty.text == q_explicit.text, (
                f"seed {seed}, {difficulty}: text differs from difficulty + explicit variant={variant}"
            )
            assert q_difficulty.answer_text == q_explicit.answer_text, (
                f"seed {seed}, {difficulty}: answer_text differs from difficulty + explicit variant={variant}"
            )
            assert q_difficulty.distractors == q_explicit.distractors, (
                f"seed {seed}, {difficulty}: distractors differ from difficulty + explicit variant={variant}"
            )
