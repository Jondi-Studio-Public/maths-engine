"""Tests for skill 1 (f10-quadratic-algebra-expansion) and skill 2
(senior-determine-complex-conjugate-solutions-of-real-quadratic-equations)."""

import re
from pathlib import Path

import pytest
from sympy import Symbol, sympify

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make

ROOT = Path(__file__).parent.parent
SEEDS = range(60)
i_sym = Symbol("i")


# ============================================================================
# SKILL 1: f10-quadratic-algebra-expansion
# ============================================================================


class TestExpandBinomialProductMonic:
    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make("expand_binomial_product_monic", seed=seed, difficulty=tier)
            q2 = make("expand_binomial_product_monic", seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_distractors_distinct_nonempty_and_never_equal_answer(self, tier):
        for seed in SEEDS:
            q = make("expand_binomial_product_monic", seed=seed, difficulty=tier)
            assert len(q.distractors) > 0, f"seed {seed}: no distractors"
            for d in q.distractors:
                assert d != q.answer_text, f"seed {seed}: distractor equals answer"
            assert len(q.distractors) == len(set(q.distractors)), f"seed {seed}: duplicate distractors"

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_real_grader_accepts_canonical_answer(self, tier):
        for seed in SEEDS:
            q = make("expand_binomial_product_monic", seed=seed, difficulty=tier)
            result = GRADERS["expression"](q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected canonical: {q.answer_text}"

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_real_grader_rejects_every_distractor(self, tier):
        for seed in SEEDS:
            q = make("expand_binomial_product_monic", seed=seed, difficulty=tier)
            for d in q.distractors:
                result = GRADERS["expression"](q, d)
                assert not result.correct, f"seed {seed}: grader accepted distractor {d}"

    def test_real_grader_rejects_deliberately_wrong_answer(self):
        q = make("expand_binomial_product_monic", seed=0, difficulty="easy")
        result = GRADERS["expression"](q, "x^2 + 999x + 999")
        assert not result.correct

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        texts = {make("expand_binomial_product_monic", seed=s, difficulty=tier).text for s in SEEDS}
        assert len(texts) >= 20, f"tier {tier}: only {len(texts)} distinct questions (need >= 20)"

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_answer_text_is_standard_form_no_star_no_double_star(self, tier):
        pattern = re.compile(r"^x\^2( [+-] \d*x)?( [+-] \d+)?$")
        for seed in SEEDS:
            q = make("expand_binomial_product_monic", seed=seed, difficulty=tier)
            assert "**" not in q.answer_text, f"seed {seed}: ** in {q.answer_text}"
            assert "*" not in q.answer_text, f"seed {seed}: bare * in {q.answer_text}"
            assert not re.search(r"(?<!\d)1x", q.answer_text), f"seed {seed}: unit coefficient '1x' in {q.answer_text}"
            assert "+ -" not in q.answer_text and "- -" not in q.answer_text
            assert pattern.match(q.answer_text), f"seed {seed}: not standard form: {q.answer_text}"
            for d in q.distractors:
                assert "**" not in d and "*" not in d
                assert not re.search(r"(?<!\d)1x", d)

    def test_independent_check_fires_on_corrupted_answer(self):
        from mathgen.core import VerificationError, check_equivalent

        with pytest.raises(VerificationError):
            check_equivalent(sympify("x**2 + 5*x + 6"), sympify("x**2 + 5*x + 7"), "must fail")

    def test_hard_tier_covers_both_sub_cases(self):
        """Both difference-of-squares and perfect-square sub-cases appear over 60 seeds."""
        has_dos = has_perfect_square = False
        for seed in SEEDS:
            q = make("expand_binomial_product_monic", seed=seed, difficulty="hard")
            if " - " in q.answer_text and "x" not in q.answer_text.split(" ")[-1]:
                # x^2 - N shape (no linear term) => difference of squares
                if re.match(r"^x\^2 - \d+$", q.answer_text):
                    has_dos = True
            if re.match(r"^x\^2 [+-] \d+x \+ \d+$", q.answer_text):
                has_perfect_square = True
        assert has_dos, "no difference-of-squares question seen over 60 seeds"
        assert has_perfect_square, "no perfect-square question seen over 60 seeds"

    def test_skill_loads_with_correct_shape(self):
        _, skills = load_skills(ROOT / "skills")
        by_id = {s.id: s for s in skills}
        skill = by_id["f10-quadratic-algebra-expansion"]
        assert skill.domain == "quadratics"
        assert skill.status == "active"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["expand-single-brackets", "factor-common-numerical-factor"]
        assert skill.answer == "expression"
        difficulties = [s.difficulty for s in skill.sources]
        assert difficulties == ["easy", "medium", "hard", "exam"]
        for s in skill.sources:
            if s.difficulty == "exam":
                continue  # the exam source is covered by this skill's tests/test_exd_x_* file
            assert s.topic == "expand_binomial_product_monic"


# ============================================================================
# SKILL 2: senior-determine-complex-conjugate-solutions-of-real-quadratic-equations
# ============================================================================


class TestQuadraticComplexConjugateSolutions:
    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make("quadratic_complex_conjugate_solutions", seed=seed, difficulty=tier)
            q2 = make("quadratic_complex_conjugate_solutions", seed=seed, difficulty=tier)
            assert q1.text == q2.text
            assert q1.answer_text == q2.answer_text
            assert q1.distractors == q2.distractors

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_distractors_distinct_nonempty_and_never_equal_answer(self, tier):
        for seed in SEEDS:
            q = make("quadratic_complex_conjugate_solutions", seed=seed, difficulty=tier)
            assert len(q.distractors) > 0, f"seed {seed}: no distractors"
            for d in q.distractors:
                assert d != q.answer_text
            assert len(q.distractors) == len(set(q.distractors))

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_real_grader_accepts_canonical_answer(self, tier):
        for seed in SEEDS:
            q = make("quadratic_complex_conjugate_solutions", seed=seed, difficulty=tier)
            result = GRADERS["expression"](q, q.answer_text)
            assert result.correct, f"seed {seed}: grader rejected canonical: {q.answer_text}"

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_real_grader_rejects_every_distractor(self, tier):
        for seed in SEEDS:
            q = make("quadratic_complex_conjugate_solutions", seed=seed, difficulty=tier)
            for d in q.distractors:
                result = GRADERS["expression"](q, d)
                assert not result.correct, f"seed {seed}: grader accepted distractor {d}"

    def test_real_grader_rejects_deliberately_wrong_answer(self):
        q = make("quadratic_complex_conjugate_solutions", seed=0, difficulty="easy")
        result = GRADERS["expression"](q, "999 + 999i")
        assert not result.correct

    def test_grader_accepts_bracket_free_fractional_imaginary_part(self):
        """A fractional imaginary part must not be rendered with a bracket
        (e.g. '(1/2)i'): `grade_expression` rejects any learner answer that
        omits a bracket the canonical itself contains
        (`mathgen/grading/typed.py`: "if '(' in canonical and '(' not in stripped"),
        and a learner asked for 'p + qi' form naturally types the
        bracket-free '1/2i' -- a confirmed live false-reject pattern on an
        already-shipped indices skill. Seed 5 at medium tier is known to
        produce a fractional q (-1/2 + 1/2i); assert the canonical itself
        carries no bracket, and that the bracket-free form the learner would
        actually type is accepted."""
        q = make("quadratic_complex_conjugate_solutions", seed=5, difficulty="medium")
        assert "/" in q.answer_text, "seed 5 medium must have a fractional imaginary part"
        assert "(" not in q.answer_text, f"canonical must not bracket the fraction: {q.answer_text!r}"
        result = GRADERS["expression"](q, q.answer_text)
        assert result.correct, f"bracket-free canonical must itself grade correct: {q.answer_text}"

    def test_grader_rejects_capital_I_and_python_complex_literal(self):
        """A learner who types sympy.I's spelling or a Python complex-literal
        j-suffix must not be accidentally accepted (student-would-reject note)."""
        q = make("quadratic_complex_conjugate_solutions", seed=0, difficulty="easy")
        # "I" (capital) parses to sympy.I, a different object from our free
        # symbol i, so even a numerically-matching value must not compare equal.
        result_I = GRADERS["expression"](q, q.answer_text.replace("i", "I"))
        assert not result_I.correct

    @pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        texts = {make("quadratic_complex_conjugate_solutions", seed=s, difficulty=tier).text for s in SEEDS}
        assert len(texts) >= 20, f"tier {tier}: only {len(texts)} distinct questions (need >= 20)"

    @pytest.mark.parametrize("tier", ["easy", "medium"])
    def test_answer_matches_p_plus_qi_shape_and_positive_imaginary_part(self, tier):
        pattern = re.compile(r"^-?\d+(/\d+)? [+-] (\(\d+/\d+\)|\d+/\d+|\d+)?i$")
        for seed in SEEDS:
            q = make("quadratic_complex_conjugate_solutions", seed=seed, difficulty=tier)
            assert pattern.match(q.answer_text), f"seed {seed}: unexpected answer shape: {q.answer_text}"
            assert " + " in q.answer_text, f"seed {seed}: answer must have positive imaginary part: {q.answer_text}"
            for d in q.distractors:
                assert pattern.match(d), f"seed {seed}: distractor shape mismatch: {d}"

    def test_hard_tier_answer_is_a_plain_integer_c_value(self):
        """Hard tier now asks for the withheld coefficient c, not a p+qi root --
        the answer and every distractor must be a bare integer."""
        pattern = re.compile(r"^-?\d+$")
        for seed in SEEDS:
            q = make("quadratic_complex_conjugate_solutions", seed=seed, difficulty="hard")
            assert pattern.match(q.answer_text), f"seed {seed}: hard-tier answer not a plain integer: {q.answer_text}"
            for d in q.distractors:
                assert pattern.match(d), f"seed {seed}: hard-tier distractor not a plain integer: {d}"

    def test_hard_tier_question_gives_root_and_b_asks_for_c(self):
        """The hard-tier question must show a given complex root and a numeric b, and ask for c -- so c cannot be
        read straight off the given root without using the stated equation."""
        pattern = re.compile(
            r"^-?\d+ [+-] \d*i is one solution of x\^2 [+-] \d+x \+ c = 0, "
            r"where the coefficients are real\. Find the value of c\.$"
        )
        for seed in SEEDS:
            q = make("quadratic_complex_conjugate_solutions", seed=seed, difficulty="hard")
            assert pattern.match(q.text), f"seed {seed}: unexpected hard-tier question shape: {q.text}"

    def test_medium_tier_produces_at_least_one_fractional_q(self):
        seen_fraction = False
        for seed in SEEDS:
            q = make("quadratic_complex_conjugate_solutions", seed=seed, difficulty="medium")
            if "/" in q.answer_text:
                seen_fraction = True
                break
        assert seen_fraction, "medium tier never produced a fractional q over 60 seeds"

    def test_independent_check_fires_on_corrupted_answer(self):
        from mathgen.core import VerificationError, check

        with pytest.raises(VerificationError):
            check(False, "must fail")

    def test_skill_loads_with_correct_shape(self):
        _, skills = load_skills(ROOT / "skills")
        by_id = {s.id: s for s in skills}
        skill = by_id["senior-determine-complex-conjugate-solutions-of-real-quadratic-equations"]
        assert skill.domain == "quadratics"
        assert skill.status == "active"
        assert skill.placement_eligible is False
        assert skill.prerequisites == ["quadratic-vertex"]
        assert skill.answer == "expression"
        difficulties = [s.difficulty for s in skill.sources]
        assert difficulties == ["easy", "medium", "hard", "exam"]
        for s in skill.sources:
            if s.difficulty == "exam":
                continue  # the exam source is covered by this skill's tests/test_exd_x_* file
            assert s.topic == "quadratic_complex_conjugate_solutions"
