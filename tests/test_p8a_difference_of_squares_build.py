"""Tests `difference_of_squares`: expanding and factorising special products, and rejection of a learner's own
wrong form.

Tests for difference-of-squares."""

import re

import pytest
import sympy
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


def _sympify_answer(answer_text: str):
    """Parse a canonical answer_text (implicit multiplication, e.g. '4x^2 - 9'
    or '(2x + 3)(2x - 3)') the same way mathgen/grading/typed.py's own parser does --
    plain sympy.sympify chokes on '81x**2' with no explicit '*'."""
    return parse_expr(answer_text.replace("^", "**"), transformations=_TRANSFORMATIONS, local_dict={"x": _x})


def _all_questions(tier, seeds=SEEDS):
    return [make("difference_of_squares", seed=s, difficulty=tier) for s in seeds]


class TestDifferenceOfSquares:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(20):
            q1 = make("difference_of_squares", seed=seed, difficulty=tier)
            q2 = make("difference_of_squares", seed=seed, difficulty=tier)
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
    def test_answer_form_is_set_correctly(self, tier):
        for q in _all_questions(tier):
            if q.text.startswith("Expand"):
                assert q.answer_form == "expanded", f"{tier}: {q.text!r} should be answer_form=expanded"
            else:
                assert q.text.startswith("Factorise"), f"{tier}: unrecognised prompt shape {q.text!r}"
                assert q.answer_form == "factored", f"{tier}: {q.text!r} should be answer_form=factored"

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
            wrong = "x^2 + 999x + 999" if q.answer_form == "expanded" else "(x + 999)(x - 999)"
            result = GRADERS["expression"](q, wrong)
            assert not result.correct, f"{tier}: grader accepted deliberately wrong {wrong!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        texts = {q.text for q in _all_questions(tier)}
        assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions (need >= 20)"

    @pytest.mark.parametrize("tier", TIERS)
    def test_options_formatted_alike_by_answer_form(self, tier):
        # Every option in a question is either all-expanded (no bracket) or
        # all-factored (every option has a bracket), matching the question's
        # own answer_form -- no option identifiable by shape alone.
        for q in _all_questions(tier):
            options = [q.answer_text] + list(q.distractors)
            has_bracket = ["(" in o for o in options]
            if q.answer_form == "expanded":
                assert not any(has_bracket), f"{tier}: expanded question has a bracketed option: {q.text!r} {options}"
                assert all("**" not in o and "*" not in o for o in options), f"{tier}: stray '*' in {options}"
            else:
                assert all(has_bracket), f"{tier}: factored question has a bracket-free option: {q.text!r} {options}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_both_directions_present_across_tier(self, tier):
        questions = _all_questions(tier, seeds=range(200))
        expand_n = sum(1 for q in questions if q.answer_form == "expanded")
        factorise_n = sum(1 for q in questions if q.answer_form == "factored")
        assert expand_n > 0, f"{tier}: no 'expand' questions over 200 seeds"
        assert factorise_n > 0, f"{tier}: no 'factorise' questions over 200 seeds"

    @pytest.mark.parametrize("tier", TIERS)
    def test_misconceptions_measured_exercisability_over_200_seeds(self, tier):
        # Measured, not assumed. Classify every question
        # by (direction, form) via its own prompt shape, form identified from
        # the bracket signs the PROMPT itself states -- never re-deriving from
        # the generator's internal choice.
        counts = {
            ("expand", "dos"): 0,
            ("expand", "ps_plus"): 0,
            ("expand", "ps_minus"): 0,
            ("factorise", "dos"): 0,
            ("factorise", "ps_plus"): 0,
            ("factorise", "ps_minus"): 0,
        }
        expand_dos_re = re.compile(r"^Expand \([^)]*\+[^)]*\)\([^)]*-[^)]*\)\.$")
        expand_sq_re = re.compile(r"^Expand \((.*)\)\^2\.$")
        for q in _all_questions(tier, seeds=range(200)):
            if q.text.startswith("Expand"):
                m = expand_sq_re.match(q.text)
                if m:
                    inner = m.group(1)
                    counts[("expand", "ps_plus" if "+" in inner else "ps_minus")] += 1
                elif expand_dos_re.match(q.text):
                    counts[("expand", "dos")] += 1
                else:
                    raise AssertionError(f"unrecognised expand prompt shape: {q.text!r}")
            else:
                # Factorise prompt states the trinomial; classify by term
                # count and sign structure of the trinomial text itself: dos
                # has 2 terms (no middle term), ps_plus's 3 terms are all
                # joined by "+", ps_minus's middle term is joined by "-".
                trinomial = q.text[len("Factorise ") :].split(" using")[0]
                term_count = 1 + trinomial.count(" + ") + trinomial.count(" - ")
                if term_count == 2:
                    counts[("factorise", "dos")] += 1
                elif " - " in trinomial:
                    counts[("factorise", "ps_minus")] += 1
                else:
                    counts[("factorise", "ps_plus")] += 1
        for key, n in counts.items():
            assert n > 0, f"{tier}: combination {key} never exercised over 200 seeds: {counts}"


class TestOwnFormRejectedProof:
    """'s mandatory proof: grading the question's OWN form back at it is REJECTED, built with
    sympy.factor/sympy.expand of the CANONICAL answer itself -- never a regex over the prompt text (a regex
    over "Expand 4(5x - 7)" drops the leading coefficient)."""

    @pytest.mark.parametrize("tier", TIERS)
    def test_expand_questions_reject_a_factored_form_of_their_own_answer(self, tier):
        checked = 0
        for q in _all_questions(tier, seeds=PROOF_SEEDS):
            if q.answer_form != "expanded":
                continue
            checked += 1
            canonical = _sympify_answer(q.answer_text)
            factored = str(sympy.factor(canonical))
            result = GRADERS["expression"](q, factored)
            assert not result.correct, (
                f"{tier}: expand question {q.text!r} (answer {q.answer_text!r}) "
                f"accepted its own factored form {factored!r}"
            )
        assert checked >= 20, f"{tier}: only {checked} 'expand' questions sampled (need >= 20)"
        print(f"{tier}: own-form-rejected proof (expand): {checked}/{checked} passed, 0 failed")

    @pytest.mark.parametrize("tier", TIERS)
    def test_factorise_questions_reject_an_expanded_form_of_their_own_answer(self, tier):
        checked = 0
        for q in _all_questions(tier, seeds=PROOF_SEEDS):
            if q.answer_form != "factored":
                continue
            checked += 1
            canonical = _sympify_answer(q.answer_text)
            expanded = str(sympy.expand(canonical))
            result = GRADERS["expression"](q, expanded)
            assert not result.correct, (
                f"{tier}: factorise question {q.text!r} (answer {q.answer_text!r}) "
                f"accepted its own expanded form {expanded!r}"
            )
        assert checked >= 20, f"{tier}: only {checked} 'factorise' questions sampled (need >= 20)"
        print(f"{tier}: own-form-rejected proof (factorise): {checked}/{checked} passed, 0 failed")


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "difference-of-squares")
        assert skill.domain == "expressions-equivalence"
        assert skill.answer == "expression"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {"expand-single-brackets", "factor-common-numerical-factor"}
        stages_covered = set()
        for source in skill.sources:
            if source.difficulty == "exam":
                continue  # the exam source is covered by this skill's tests/test_exd_x_* file
            assert source.topic == "difference_of_squares"
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
