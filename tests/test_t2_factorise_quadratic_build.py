"""Tests `factorise-quadratic`: answers, difficulty progression and swapped-cross-term distractors.

Tests for factorise-quadratic."""

import re

import sympy
import pytest
from sympy import symbols
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import make

TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)
_x = symbols("x")
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def _sympify_answer(answer_text: str):
    return parse_expr(answer_text.replace("^", "**"), transformations=_TRANSFORMATIONS, local_dict={"x": _x})


def _all_questions(tier, seeds=SEEDS):
    return [make("factorise_quadratic", seed=s, difficulty=tier) for s in seeds]


def _leading_coef(text: str) -> int:
    """Parse a-coefficient out of a 'Factorise ...' question text (own-form-independent
    of the generator internals -- just enough to classify a=1 vs a!=1 for the tests below)."""
    trinomial = _sympify_answer(text[len("Factorise ") :].rstrip("."))
    return int(sympy.Poly(trinomial, _x).all_coeffs()[0])


class TestFactoriseQuadratic:
    @pytest.mark.parametrize("tier", TIERS)
    def test_determinism_across_seeds(self, tier):
        for seed in range(40):
            q1 = make("factorise_quadratic", seed=seed, difficulty=tier)
            q2 = make("factorise_quadratic", seed=seed, difficulty=tier)
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
    def test_answer_form_is_factored(self, tier):
        for q in _all_questions(tier):
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
            wrong = "(x + 999)(x - 999)"
            result = GRADERS["expression"](q, wrong)
            assert not result.correct, f"{tier}: grader accepted deliberately wrong {wrong!r} for {q.text!r}"

    @pytest.mark.parametrize("tier", TIERS)
    def test_at_least_20_distinct_questions_per_tier(self, tier):
        texts = {q.text for q in _all_questions(tier)}
        assert len(texts) >= 20, f"{tier}: only {len(texts)} distinct questions (need >= 20)"

    @pytest.mark.parametrize("tier", TIERS)
    def test_misconceptions_every_candidate_distinct_when_present(self, tier):
        # A candidate that is well-defined for a seed must be a genuinely
        # different value from the correct answer -- checked directly here
        # (not just trusted from the generator's own internal check_distinct).
        for q in _all_questions(tier, seeds=range(200)):
            answer_val = _sympify_answer(q.answer_text)
            for d in q.distractors:
                d_val = _sympify_answer(d)
                assert sympy.simplify(answer_val - d_val) != 0, (
                    f"{tier}: distractor {d!r} is the same value as {q.answer_text!r} for {q.text!r}"
                )


class TestDifficultyProgression:
    def test_easy_is_a_equals_1(self):
        for q in _all_questions("easy"):
            assert _leading_coef(q.text) == 1, f"easy: expected a == 1, got {q.text!r}"

    def test_medium_is_a_equals_1(self):
        for q in _all_questions("medium"):
            assert _leading_coef(q.text) == 1, f"medium: expected a == 1, got {q.text!r}"

    def test_hard_is_a_not_equal_1(self):
        for q in _all_questions("hard"):
            assert _leading_coef(q.text) != 1, f"hard: expected a != 1, got {q.text!r}"

    def test_hard_negative_leading_coefficient_fires_over_60_seeds(self):
        negatives = sum(1 for q in _all_questions("hard") if _leading_coef(q.text) < 0)
        assert negatives > 0, "hard: negative-leading-coefficient shape never fired over 60 seeds"


_BRACKET_RE = re.compile(r"^(-?\d*)x ([+-]) (\d+)$")
_PAIR_RE = re.compile(r"^\((.+?)\)\((.+?)\)$")


def _parse_bracket(inner: str):
    m = _BRACKET_RE.match(inner)
    assert m, f"unrecognised bracket shape: {inner!r}"
    coef_str, sign, const_str = m.groups()
    coef = -1 if coef_str == "-" else (1 if coef_str == "" else int(coef_str))
    const = int(const_str) if sign == "+" else -int(const_str)
    return coef, const


def _parse_pair(text: str):
    m = _PAIR_RE.match(text)
    assert m, f"unrecognised bracket-pair shape: {text!r}"
    return _parse_bracket(m.group(1)), _parse_bracket(m.group(2))


class TestSwappedCrossTerms:
    """Exercised over a 200-seed hard-tier sweep, and correctly dropped (not
    silently wrong) on any seed where the underlying displayed p == r (the
    swap is then a no-op, identical to the true answer)."""

    def test_swapped_cross_terms_fires_over_200_seeds(self):
        fired = 0
        for seed in range(200):
            q = make("factorise_quadratic", seed=seed, difficulty="hard")
            (p_disp, q_disp), (r_disp, s_disp) = _parse_pair(q.answer_text)
            if p_disp == r_disp:
                continue
            swapped_val = sympy.expand((r_disp * _x + q_disp) * (p_disp * _x + s_disp))
            for d in q.distractors:
                (dp, dq), (dr, ds) = _parse_pair(d)
                d_val = sympy.expand((dp * _x + dq) * (dr * _x + ds))
                if sympy.simplify(d_val - swapped_val) == 0:
                    fired += 1
                    break
        assert fired > 0, "swapped-cross-terms never observed among distractors over 200 hard-tier seeds"

    def test_swapped_cross_terms_dropped_when_p_disp_equals_r_disp(self):
        checked = 0
        for seed in range(200):
            q = make("factorise_quadratic", seed=seed, difficulty="hard")
            (p_disp, q_disp), (r_disp, s_disp) = _parse_pair(q.answer_text)
            if p_disp != r_disp:
                continue
            checked += 1
            true_val = sympy.expand((p_disp * _x + q_disp) * (r_disp * _x + s_disp))
            for d in q.distractors:
                (dp, dq), (dr, ds) = _parse_pair(d)
                d_val = sympy.expand((dp * _x + dq) * (dr * _x + ds))
                assert sympy.simplify(d_val - true_val) != 0, f"distractor {d!r} equals the true answer for {q.text!r}"
        print(f"p_disp == r_disp seen on {checked}/200 hard-tier seeds, all correctly dropped")


class TestOwnFormRejectedProof:
    """'s mandatory proof: grading the EXPANDED form of the canonical factored answer back is REJECTED by the
    real grader."""

    @pytest.mark.parametrize("tier", TIERS)
    def test_expanded_form_of_answer_is_rejected(self, tier):
        checked = 0
        for q in _all_questions(tier, seeds=range(40)):
            checked += 1
            canonical = _sympify_answer(q.answer_text)
            expanded = str(sympy.expand(canonical))
            result = GRADERS["expression"](q, expanded)
            assert not result.correct, (
                f"{tier}: {q.text!r} (answer {q.answer_text!r}) accepted its own expanded form {expanded!r}"
            )
        assert checked >= 20, f"{tier}: only {checked} questions sampled (need >= 20)"
        print(f"{tier}: own-form-rejected proof: {checked}/{checked} passed, 0 failed")


class TestSkillLoads:
    def test_skill_loads_with_expected_shape(self):
        _, skills = load_skills()
        skill = next(s for s in skills if s.id == "factorise-quadratic")
        assert skill.domain == "quadratics"
        assert skill.answer == "expression"
        assert skill.placement_eligible is False
        assert set(skill.prerequisites) == {"expand-single-brackets"}
        stages_covered = set()
        for source in skill.sources:
            assert source.topic == "factorise_quadratic"
            stages_covered.update(source.stages)
        assert stages_covered == {0, 1, 2, 3, 4, 5, 6, "maintenance"}
