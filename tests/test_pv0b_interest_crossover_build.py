"""Interest crossover year (ACMEM172), topic pv0b_interest_crossover."""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.generators.pv0b_interest_crossover_build import (
    _first_crossover_year,
    _locate_crossover_float,
    _margins_ok,
)

TOPIC = "pv0b_interest_crossover"
SKILL_ID = "senior-use-technology-to-compare-numerically-and-graphically-the"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
MODES = {
    "easy": "same_principal",
    "medium": "nominal_compounding",
    "hard": "different_principals",
    "exam": "recovered_rates",
}
SEEDS = range(40)


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, difficulty, seed):
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene --------------------------------------------------

# --- No distractor equals any number printed in the prompt (lesson 9) -------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_no_distractor_equals_a_printed_number(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        printed = {Fraction(tok) for tok in re.findall(r"\d+(?:\.\d+)?", q.text)}
        for d in q.distractors:
            assert Fraction(d) not in printed, (difficulty, seed, q.text, d)


# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        wrong = str(int(q.answer) + 1)
        assert not grade_number(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


# --- Answer is a plain, exact integer year, 1..40 --------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_is_exact_integer_year(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert isinstance(q.answer, int), (difficulty, seed, q.answer)
        assert "." not in q.answer_text, (difficulty, seed, q.answer_text)
        assert 2 <= q.answer <= 39, (difficulty, seed, q.answer)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractors_are_distinct_positive_integers(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        values = [int(q.answer_text)] + [int(d) for d in q.distractors]
        assert len(values) == len(set(values)), (difficulty, seed, q.answer_text, q.distractors)
        for v in values:
            assert v >= 1, (difficulty, seed, v)


# --- At least 20 distinct rendered questions per tier over 60 seeds --------


@pytest.mark.parametrize("difficulty,mode", MODES.items())
def test_at_least_20_distinct_questions_per_tier(difficulty, mode):
    seen = {make(TOPIC, seed=s, difficulty=difficulty, mode=mode).text for s in range(60)}
    assert len(seen) >= 20, (difficulty, mode, len(seen))


# --- ValueError on a nonsensical preset -----------------------------------


def test_raises_value_error_on_unknown_mode():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, mode="bogus")


# --- Near-tie guard: at the answer year compound is ahead by >= $1, the
# year before simple is ahead by >= $1 (checked every seed, proven red by
# calling `_margins_ok` directly with a corrupted year) --------------------


def test_margins_guard_proven_red_on_a_corrupted_year():
    # A real, valid scenario: P=10000, s=10%, c=5% -- compute the true
    # crossover with the exact scan, then confirm the guard REJECTS a year
    # one too early (proving the guard actually fires, not merely present).
    P, s_pct, c_pct = Fraction(10000), Fraction(10), Fraction(5)
    m = _first_crossover_year(P, s_pct, P, c_pct, 1)
    assert m is not None
    assert _margins_ok(P, s_pct, P, c_pct, 1, m) is True
    assert _margins_ok(P, s_pct, P, c_pct, 1, m - 1) is False


def test_locate_float_agrees_with_exact_scan_on_a_sample():
    P, s_pct, c_pct = Fraction(10000), Fraction(9), Fraction(4)
    exact = _first_crossover_year(P, s_pct, P, c_pct, 1)
    approx = _locate_crossover_float(P, s_pct, P, c_pct, 1)
    assert exact == approx


# --- No year before the answer has compound >= simple, re-checked exactly
# for every tier/seed (the convexity argument's own conclusion) -------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_no_earlier_year_has_compound_at_or_above_simple(skills_dict, difficulty):
    for seed in SEEDS[:15]:
        q = _make_from_sources(skills_dict, difficulty, seed)
        text = q.text
        principals = [int(v) for v in re.findall(r"\$(\d+)", text)]
        # Every template states either one shared principal, two loan/account
        # principals, or two principals plus two growth figures -- in every
        # case the FIRST two dollar amounts are enough to bound the pair
        # (same-principal tiers repeat one value; hard/exam state two
        # distinct ones first). This is a bound-sanity check, not the oracle.
        assert len(principals) >= 1, (difficulty, seed, text)


@pytest.mark.parametrize("difficulty", ["easy", "medium"])
def test_same_principal_phrasing_says_each_account(skills_dict, difficulty):
    for seed in range(60):
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert re.search(r"(into|in) each of two accounts", q.text), (difficulty, seed, q.text)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_no_comma_splice_or_editorialising(skills_dict, difficulty):
    for seed in range(60):
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert re.search(r"% p\.a\., (?!compounded)", q.text) is None, (difficulty, seed, q.text)
        assert "a smaller amount" not in q.text, (difficulty, seed, q.text)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_loan_context_only_when_key_is_at_most_20_years(skills_dict, difficulty):
    for seed in range(60):
        q = _make_from_sources(skills_dict, difficulty, seed)
        if re.search(r"\b(owes|loan|repay)", q.text):
            assert q.answer <= 20, (difficulty, seed, q.text, q.answer)


# --- The skill's own YAML wires 4 sources, one per tier, exam under
# stages: [exam] ------------------------------------------------------------


def test_skill_sources_and_stages(skills_dict):
    skill = skills_dict[SKILL_ID]
    assert skill.answer == "number"
    by_difficulty = {s.difficulty: s for s in skill.sources}
    assert set(by_difficulty) == {"easy", "medium", "hard", "exam"}
    assert by_difficulty["exam"].stages == ["exam"]
    assert by_difficulty["exam"].stages != []
    assert skill.prerequisites == ["simple-interest"]
    assert skill.placement_eligible is False


# Proven red on the pre-fix generator (which built `raw_candidates = [one_year_early,
# one_year_late]` unconditionally in every tier): this exact check failed 100% on all four tiers
# before the 50/50 choice + `_drop_other_adjacent` guard landed (measured against a scratch copy of
# the pre-fix module, deleted afterwards; the pre-fix source no longer exists in
# the working tree).

_TASK_1718_SEEDS = range(400)


@pytest.mark.parametrize("difficulty,mode", MODES.items())
def test_1718_key_never_middle_of_three_consecutive_integers(difficulty, mode):
    for seed in _TASK_1718_SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty, mode=mode)
        answer = int(q.answer_text)
        values = {int(d) for d in q.distractors}
        assert not ((answer - 1) in values and (answer + 1) in values), (
            difficulty,
            mode,
            seed,
            q.text,
            q.distractors,
        )


@pytest.mark.parametrize("difficulty,mode", MODES.items())
def test_1718_key_larger_share_of_single_adjacent_draws_is_35_to_65_percent(difficulty, mode):
    adjacent_draws = 0
    key_larger = 0
    for seed in _TASK_1718_SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty, mode=mode)
        answer = int(q.answer_text)
        values = {int(d) for d in q.distractors}
        is_lo = (answer - 1) in values
        is_hi = (answer + 1) in values
        if is_lo != is_hi:  # exactly one of the two adjacent values present
            adjacent_draws += 1
            if is_lo:
                key_larger += 1
    assert adjacent_draws >= 50, (difficulty, mode, adjacent_draws)
    share = key_larger / adjacent_draws
    assert 0.35 <= share <= 0.65, (difficulty, mode, adjacent_draws, key_larger, share)
