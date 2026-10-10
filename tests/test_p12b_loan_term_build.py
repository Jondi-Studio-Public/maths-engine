"""Effect of interest rate and repayment on loan term.

Tests for senior-investigate-the-effect-of-the-interest-rate-and (ACMEM175),
topic p12b_loan_term_from_repayment.
"""

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.generators.p12b_loan_term_build import _n_from_P_r_M, _r_from_P_n_M

TOPIC = "p12b_loan_term_from_repayment"
SKILL_ID = "senior-investigate-the-effect-of-the-interest-rate-and"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
MODES = {"easy": "years", "medium": "months", "hard": "term_difference", "exam": "max_rate"}
SEEDS = range(40)


def _make_from_sources(skills_dict, difficulty, seed):
    """Generate through the skill's own `sources` entry, params-driven, not a
    bare `make(topic, seed, difficulty)` call."""
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene --------------------------------------------------

# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        wrong = str(int(q.answer) + 1)
        assert not grade_number(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


# --- Answer is a plain, exact integer (never a decimal point) ------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_is_exact_integer_no_decimal_point(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert isinstance(q.answer, int), (difficulty, seed, q.answer)
        assert "." not in q.answer_text, (difficulty, seed, q.answer_text)


# --- Term bounds per tier: easy years 2-6, medium months two-digit
# non-multiple-of-12, hard difference small positive, exam rate 3-15 -----


def test_easy_years_bounds():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy", mode="years")
        assert 2 <= q.answer <= 6, (seed, q.answer)


def test_medium_months_bounds_and_not_multiple_of_12():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="medium", mode="months")
        assert 10 <= q.answer <= 99, (seed, q.answer)
        assert q.answer % 12 != 0, (seed, q.answer)


def test_hard_term_difference_bounds():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard", mode="term_difference")
        assert 1 <= q.answer <= 5, (seed, q.answer)


def test_exam_max_rate_bounds():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", mode="max_rate")
        assert 3 <= q.answer <= 15, (seed, q.answer)


# --- Every distractor from the SAME question's construction is a genuinely
# different number from the answer AND from each other (checked already by
# construction, re-proven here as a black-box sweep) ----------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractors_are_distinct_numeric_values(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        values = [Fraction(q.answer_text)] + [Fraction(d) for d in q.distractors]
        assert len(values) == len(set(values)), (difficulty, seed, q.answer_text, q.distractors)


# --- At least 20 distinct rendered questions per tier over 60 seeds ------


@pytest.mark.parametrize("difficulty,mode", MODES.items())
def test_at_least_20_distinct_questions_per_tier(difficulty, mode):
    seen = {make(TOPIC, seed=s, difficulty=difficulty, mode=mode).text for s in range(60)}
    assert len(seen) >= 20, (difficulty, mode, len(seen))


# --- ValueError on a nonsensical preset -----------------------------------


def test_raises_value_error_on_unknown_mode():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, mode="bogus")


# --- The independent forward-solve helpers agree with a hand-computed
# example, and disagree with a corrupted one -------------------------------


def test_n_from_P_r_M_matches_hand_computation():
    # P=$24000, r=5%, n=4 years -> denom=100+5*4=120, k=P/(1200*4)=5, M=k*120=600
    P, r, M = Fraction(24000), Fraction(5), Fraction(600)
    assert _n_from_P_r_M(P, r, M) == Fraction(4)
    # A corrupted M must NOT reproduce n=4.
    assert _n_from_P_r_M(P, r, Fraction(601)) != Fraction(4)


def test_r_from_P_n_M_matches_hand_computation():
    # Same scenario solved for r instead: P=24000, n=4, M=600 -> r=5.
    P, n, M = Fraction(24000), Fraction(4), Fraction(600)
    assert _r_from_P_n_M(P, n, M) == Fraction(5)
    assert _r_from_P_n_M(P, n, Fraction(601)) != Fraction(5)


# --- The skill's own YAML wires 4 sources, one per tier, exam under
# stages: [exam] ------------------------------------------------------------


def test_skill_sources_and_stages(skills_dict):
    skill = skills_dict[SKILL_ID]
    assert skill.answer == "number"
    by_difficulty = {s.difficulty: s for s in skill.sources}
    assert set(by_difficulty) == {"easy", "medium", "hard", "exam"}
    assert by_difficulty["exam"].stages == ["exam"]
    assert by_difficulty["exam"].stages != []
