"""Tests the total-change-by-integration skill (`senior-calculate-total-change-by-integrating-instantaneous-or-
marginal`, ACMMM133)."""

import re
from pathlib import Path

import pytest
from sympy import symbols, integrate, Rational

from mathgen.grading.typed import grade_expression
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(40)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generators_are_deterministic_across_seeds_and_have_unique_distractors(difficulty):
    """Determinism across >=40 seeds at every difficulty."""
    for seed in SEEDS:
        question = make(
            "calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty=difficulty
        )
        repeat = make(
            "calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty=difficulty
        )
        assert question.to_json() == repeat.to_json()
        assert question.answer_text not in question.distractors
        assert len(question.distractors) == len(set(question.distractors))
        assert len(question.distractors) >= 2


def test_easy_grades_correct_answer_and_rejects_distractors():
    """Real grader accepts canonical answer and rejects every distractor."""
    for seed in SEEDS:
        question = make("calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty="easy")
        assert grade_expression(question, question.answer_text).correct, f"easy seed {seed}: canonical answer rejected"
        for distractor in question.distractors:
            result = grade_expression(question, distractor)
            assert not result.correct, (
                f"easy seed {seed}: distractor '{distractor}' accepted (answer: {question.answer_text})"
            )


def test_medium_grades_correct_answer_and_rejects_distractors():
    """Real grader accepts canonical answer and rejects every distractor."""
    for seed in SEEDS:
        question = make(
            "calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty="medium"
        )
        assert grade_expression(question, question.answer_text).correct, (
            f"medium seed {seed}: canonical answer rejected"
        )
        for distractor in question.distractors:
            result = grade_expression(question, distractor)
            assert not result.correct, (
                f"medium seed {seed}: distractor '{distractor}' accepted (answer: {question.answer_text})"
            )


def test_hard_grades_correct_answer_and_rejects_distractors():
    """Real grader accepts canonical answer and rejects every distractor."""
    for seed in SEEDS:
        question = make("calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty="hard")
        assert grade_expression(question, question.answer_text).correct, f"hard seed {seed}: canonical answer rejected"
        for distractor in question.distractors:
            result = grade_expression(question, distractor)
            assert not result.correct, (
                f"hard seed {seed}: distractor '{distractor}' accepted (answer: {question.answer_text})"
            )


def test_deliberately_wrong_answer_is_rejected():
    """A deliberately corrupted answer is rejected by the real grader."""
    for difficulty in ["easy", "medium", "hard"]:
        for seed in range(20):
            question = make(
                "calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty=difficulty
            )
            # Make a deliberately wrong answer by negating or adding 1
            if question.answer_text.startswith("-"):
                wrong = question.answer_text[1:]
            else:
                wrong = f"-{question.answer_text}"
            assert not grade_expression(question, wrong).correct, (
                f"{difficulty} seed {seed}: wrong answer '{wrong}' was accepted"
            )


def test_answer_never_contains_letters():
    """Answer is a bare number/fraction, never with unit letters."""
    for difficulty in ["easy", "medium", "hard"]:
        for seed in range(60):
            question = make(
                "calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty=difficulty
            )
            assert not any(ch.isalpha() for ch in question.answer_text), (
                f"{difficulty} seed {seed}: answer '{question.answer_text}' contains letter"
            )


def test_hard_tier_prompt_contains_disambiguation_clause():
    """Hard tier prompt literally contains 'not the final volume' to name the reading."""
    for seed in range(60):
        question = make("calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty="hard")
        assert "not the final volume" in question.text, f"hard seed {seed}: prompt missing disambiguation clause"


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_at_least_20_distinct_renders_per_tier_over_60_seeds(difficulty):
    """At least 20 distinct rendered questions per tier over 60 seeds."""
    seen_texts = set()
    for seed in range(60):
        question = make(
            "calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty=difficulty
        )
        seen_texts.add(question.text)
    assert len(seen_texts) >= 20, f"{difficulty}: only {len(seen_texts)} distinct questions over 60 seeds"


# --- Skill contract loading ------------------------------------------------


def test_skill_loads_with_expected_shape():
    """Skill loads through load_skills with expected prerequisites and answer kind."""
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id["senior-calculate-total-change-by-integrating-instantaneous-or-marginal"]

    assert skill.id == "senior-calculate-total-change-by-integrating-instantaneous-or-marginal"
    assert skill.name == "Calculate total change by integrating instantaneous or marginal rate of change"
    assert skill.domain == "integration"
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert "integrate-polynomial" in skill.prerequisites
    assert skill.answer == "expression"


def test_skill_has_stage_covering_sources():
    """Skill has sources covering stages 0-6 and maintenance for all three difficulties."""
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id["senior-calculate-total-change-by-integrating-instantaneous-or-marginal"]

    # Collect all stages across all sources
    all_stages = set()
    for source in skill.sources:
        all_stages.update(source.stages)

    # Must cover 0-6 and maintenance
    expected_stages = {0, 1, 2, 3, 4, 5, 6, "maintenance"}
    assert expected_stages.issubset(all_stages), f"stages covered: {all_stages}, expected at least {expected_stages}"


# --- Oracle: independent verification -----------------------------------------------
# The oracle parses rendered questions and derives the answer from scratch,
# without importing from the generator.


class RateChangeOracle:
    """Parse a rendered question and independently compute the total change by integration."""

    @staticmethod
    def parse_linear_rate(text: str):
        """Parse a rate function r(t) = c*t or r(t) = c*t + d from the question text."""
        # Look for "r(t) = " pattern, ending before parenthetical unit or similar marker
        match = re.search(r"r\(t\)\s*=\s*(.+?)(?:\s*\(|\s+litres|\s+dollars)", text)
        if not match:
            return None

        rate_str = match.group(1).strip()
        # Handle the prose form: replace · with *
        rate_str = rate_str.replace("·", "*")

        try:
            from sympy import sympify, symbols as sym_symbols

            t_sym = sym_symbols("t")
            rate_expr = sympify(rate_str)
            return rate_expr
        except Exception:
            return None

    @staticmethod
    def parse_bounds(text: str):
        """Extract interval bounds [a, b] or [0, b] from the question text."""
        # Easy: "first N minutes" -> [0, N]
        # Medium/Hard: "from M to N units/minutes" -> [M, N]

        match = re.search(r"from\s+(\d+)\s+to\s+(\d+)", text)
        if match:
            return int(match.group(1)), int(match.group(2))

        match = re.search(r"first\s+(\d+)", text)
        if match:
            return 0, int(match.group(1))

        return None

    @staticmethod
    def compute_answer(rate_expr, bounds):
        """Compute the definite integral of rate_expr over bounds."""
        from sympy import symbols as sym_symbols, integrate as sym_integrate

        t_sym = sym_symbols("t")
        a, b = bounds
        result = sym_integrate(rate_expr, (t_sym, a, b))
        return result


def test_oracle_validates_easy_tier():
    """Oracle independently validates easy tier answers."""
    oracle = RateChangeOracle()
    for seed in range(20):
        question = make("calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty="easy")

        rate = oracle.parse_linear_rate(question.text)
        bounds = oracle.parse_bounds(question.text)

        assert rate is not None, f"easy seed {seed}: could not parse rate"
        assert bounds is not None, f"easy seed {seed}: could not parse bounds"

        oracle_ans = oracle.compute_answer(rate, bounds)
        canonical_ans = Rational(question.answer_text)

        assert oracle_ans == canonical_ans, f"easy seed {seed}: oracle {oracle_ans} != canonical {canonical_ans}"


def test_oracle_validates_medium_tier():
    """Oracle independently validates medium tier answers."""
    oracle = RateChangeOracle()
    for seed in range(20):
        question = make(
            "calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty="medium"
        )

        rate = oracle.parse_linear_rate(question.text)
        bounds = oracle.parse_bounds(question.text)

        assert rate is not None, f"medium seed {seed}: could not parse rate"
        assert bounds is not None, f"medium seed {seed}: could not parse bounds"

        oracle_ans = oracle.compute_answer(rate, bounds)
        canonical_ans = Rational(question.answer_text)

        assert oracle_ans == canonical_ans, f"medium seed {seed}: oracle {oracle_ans} != canonical {canonical_ans}"


def test_oracle_validates_hard_tier():
    """Oracle independently validates hard tier answers."""
    oracle = RateChangeOracle()
    for seed in range(20):
        question = make("calculate_total_change_by_integrating_instantaneous_or_marginal", seed=seed, difficulty="hard")

        rate = oracle.parse_linear_rate(question.text)
        bounds = oracle.parse_bounds(question.text)

        assert rate is not None, f"hard seed {seed}: could not parse rate"
        assert bounds is not None, f"hard seed {seed}: could not parse bounds"

        oracle_ans = oracle.compute_answer(rate, bounds)
        canonical_ans = Rational(question.answer_text)

        assert oracle_ans == canonical_ans, f"hard seed {seed}: oracle {oracle_ans} != canonical {canonical_ans}"
