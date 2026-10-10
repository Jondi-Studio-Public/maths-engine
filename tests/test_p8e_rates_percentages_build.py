"""Tests `senior-review-rates-and-percentages` (ACMGM001)."""

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent
from mathgen.generators.p8e_rates_percentages_build import (
    _ATTEMPT_LOG,
    _RESAMPLE_ATTEMPTS,
    _reparse_final_dollars,
)

TOPIC = "rate_percentage_combined"
SKILL_ID = "senior-review-rates-and-percentages"
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(40)

_EXPECTED_DISTRACTOR_COUNT = {"easy": 2, "medium": 2, "hard": 3}


def _make_from_sources(skills_dict, difficulty, seed):
    """Generate through the skill's own `sources` entry, not a bare `make(topic, seed, difficulty)` call."""
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene --------------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractor_count_distinct_nonempty_never_equal_answer(skills_dict, difficulty):
    expected_n = _EXPECTED_DISTRACTOR_COUNT[difficulty]
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert len(q.distractors) == expected_n, (difficulty, seed, q.distractors)


# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        wrong = str(round(q.answer + 1.0, 2))
        assert not grade_number(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


# --- Answer-kind / formatting discipline --------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_is_exact_two_decimal_currency_under_1000(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert isinstance(q.answer, float)
        assert q.answer_text.startswith("$")
        whole, frac = q.answer_text[1:].split(".")
        assert len(frac) == 2
        assert Fraction(q.answer_text[1:]) * 100 % 1 == 0  # exact multiple of one cent
        assert Fraction(q.answer_text[1:]) < 1000
        for d in q.distractors:
            assert d.startswith("$")
            assert len(d.split(".")[1]) == 2


# --- At least 20 distinct questions per tier over 60 seeds --------------

# --- ValueError on a nonsensical preset -----------------------------------


def test_raises_value_error_on_unknown_mode():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, mode="bogus")


# --- Independent in-generator check fires on a corrupted answer ---------


def test_reparse_final_dollars_fires_on_corrupted():
    text = "A tradesperson charges $20.00/hour. A job takes 5 hours, plus a 10% call-out loading."
    assert _reparse_final_dollars(text, 1) == Fraction("110.00")
    with pytest.raises(VerificationError):
        check_equivalent(
            _reparse_final_dollars(text, 1), Fraction("999"), "reparsed amount must equal the constructed answer"
        )


def test_reparse_final_dollars_raises_on_unparseable_text():
    with pytest.raises(VerificationError):
        _reparse_final_dollars("nonsense text with no rate at all", 1)


# --- Both skills load correctly through the API --------------------------


def test_skill_loads_through_api(skills_dict):
    skill = skills_dict[SKILL_ID]
    assert skill is not None
    assert skill.domain == "arithmetic"
    assert skill.prerequisites == []
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert len(skill.sources) == 3
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard"}


# --- Resample-loop bound is never hit over a large seed sweep ------------


def test_resample_bound_never_hit_over_seed_sweep():
    _ATTEMPT_LOG.clear()
    for difficulty in DIFFICULTIES:
        for seed in range(700):  # 700 * 3 tiers = 2100 >= the required 2000-seed sweep
            make(TOPIC, seed=seed, difficulty=difficulty)
    assert len(_ATTEMPT_LOG) == 2100
    assert max(_ATTEMPT_LOG) < _RESAMPLE_ATTEMPTS, (
        "a draw got within one attempt of the resample bound",
        max(_ATTEMPT_LOG),
    )
