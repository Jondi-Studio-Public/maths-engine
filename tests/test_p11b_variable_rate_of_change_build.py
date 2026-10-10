"""Focused contract tests for the variable-rate-of-change skill
(senior-examine-examples-of-variable-rates-of-change-of, ACMMM087)."""

from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p11b_variable_rate_of_change_build import _verify_average_rate

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "p11b_variable_rate_of_change"
TIERS = ("easy", "medium", "hard")
SEEDS = range(40)


# --- Determinism and answer shape ---


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_plain_int(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, int)
        assert q.answer_text == str(q.answer)
        # No decimal point ever -- the average rate of change of these
        # families is always an exact integer.
        assert "." not in q.answer_text


@pytest.mark.parametrize("tier", TIERS)
def test_interval_width_is_at_least_two(tier):
    import re

    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        m = re.search(r"x = (-?\d+) to x = (-?\d+)", q.text)
        assert m is not None, q.text
        x1, x2 = int(m.group(1)), int(m.group(2))
        assert x2 - x1 >= 2, f"seed {seed}/{tier}: width {x2 - x1} < 2"


# Every coefficient is stated concretely ---


@pytest.mark.parametrize("tier", TIERS)
def test_full_polynomial_is_stated_in_the_question(tier):
    """Every question states f(x)'s complete formula: leading term, a linear term and a constant term, all with
    concrete integers -- never a partially-specified function."""
    import re

    degree = "x^3" if tier == "hard" else "x^2"
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        m = re.search(r"f\(x\) = (.+?)\. Find", q.text)
        assert m is not None, q.text
        poly = m.group(1)
        assert degree in poly
        # A linear term ("...x" not part of x^2/x^3) and a constant term
        # are both present -- three terms total, none of them dropped.
        assert re.search(r"[+-]\s*\d*x(?!\^)", poly), poly
        assert re.search(r"[+-]\s*\d+$", poly), poly


# --- Distractor hygiene ---


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_pairwise_distinct_and_non_empty(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert len(q.distractors) == 3, q.distractors
        values = [q.answer_text] + list(q.distractors)
        assert len(set(values)) == 4, f"seed {seed}/{tier}: collision in {values}"


# --- Real grader accept/reject ---


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_canonical_and_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"seed {seed}/{tier}: canonical {q.answer_text!r} rejected"
        for d in q.distractors:
            result = grade_number(q, d)
            assert not result.correct, f"seed {seed}/{tier}: distractor {d!r} accepted"


@pytest.mark.parametrize("tier", TIERS)
def test_grader_rejects_a_deliberately_wrong_answer(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        wrong = str(q.answer + 1000)
        result = grade_number(q, wrong)
        assert not result.correct, f"seed {seed}/{tier}: wrong answer {wrong!r} accepted"


# --- Independent check fires on a corrupted answer ---


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        # f(x) = x^2 + 5x - 3 over [2, 4]: true rate is 11, not 12.
        _verify_average_rate("quadratic", a=1, b=5, c=-3, x1=2, x2=4, rate=12)
    with pytest.raises(VerificationError):
        # f(x) = x^3 + 2x + 5 over [-3, -1]: true rate is 15, not 16.
        _verify_average_rate("cubic", a=1, b=2, c=5, x1=-3, x2=-1, rate=16)


def test_independent_check_passes_on_correct_answer():
    _verify_average_rate("quadratic", a=1, b=5, c=-3, x1=2, x2=4, rate=11)
    _verify_average_rate("cubic", a=1, b=2, c=5, x1=-3, x2=-1, rate=15)


# --- Distinct-question-pool floor (>= 20 distinct prompts per tier / 60 seeds) ---

# --- Zero-rate / collision resample measurement (guard cannot loop forever) ---


def test_zero_rate_resample_guard_never_exhausts_50_attempts():
    """The generator's own resample loop already raises if it cannot find a
    non-degenerate `b` within 50 attempts; running several thousand seeds
    here is the empirical proof that this never happens in practice, not
    merely "should be rare"."""
    for tier in TIERS:
        for seed in range(3000):
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert q.answer != 0


# --- Skill contract loading ---


def test_skill_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    skill = by_id.get("senior-examine-examples-of-variable-rates-of-change-of")
    assert skill is not None, "skill not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert skill.prerequisites == ["function-notation"]
    difficulties = {s.difficulty for s in skill.sources}
    assert difficulties == {"easy", "medium", "hard"}
