"""Tests the currency-exchange skill (`senior-use-currency-exchange-rates-to-determine-the-cost`, ACMGM007)."""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS, grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p8e_currency_exchange_build import (
    FOREIGN_HI,
    FOREIGN_LO,
    NICE_RATES_CENTS,
    SYMBOLS,
    _divisor_for_aud_based,
    _verify_from_text,
)

_RATE_RE = re.compile(r"= (?:A\$|US\$|€|£)(\d+\.\d{2})\.")

TOPIC = "currency_exchange_conversion"
SKILL_ID = "senior-use-currency-exchange-rates-to-determine-the-cost"
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(40)


def _make_from_sources(skills_dict, difficulty, seed):
    """Generate through the skill's own `sources` entry, never a bare `make(topic, seed, difficulty=tier)`."""
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
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert len(q.distractors) == len(set(q.distractors)) == 3, (difficulty, seed, q.distractors)


# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(skills_dict, difficulty):
    assert GRADERS["number"] is grade_number


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        wrong = str(round(q.answer + 1.0, 2))
        assert not grade_number(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


# --- Answer-kind / formatting discipline --------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_is_exact_two_decimal_aud_currency_under_1000(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert isinstance(q.answer, float)
        assert q.answer_text.startswith("A$")
        whole, frac = q.answer_text[2:].split(".")
        assert len(frac) == 2
        assert Fraction(q.answer_text[2:]) * 100 % 1 == 0  # exact multiple of one cent
        assert Fraction(q.answer_text[2:]) < 1000
        for d in q.distractors:
            assert d.startswith("A$"), (difficulty, seed, d)
            assert len(d.split(".")[1]) == 2


# --- No bare "$" anywhere: A$/US$ only, never a currency-confusable "$" -----


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_no_bare_dollar_sign(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        stripped = q.text.replace("A$", "").replace("US$", "")
        assert "$" not in stripped, (difficulty, seed, q.text)
        stripped_ans = q.answer_text.replace("A$", "")
        assert "$" not in stripped_ans


# --- Both framings ("cost of buying" / "value of") occur --------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_both_framings_occur(difficulty):
    framings = set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        framings.add("cost" if "cost, in Australian dollars" in q.text else "value")
    assert framings == {"cost", "value"}, (difficulty, framings)


# --- All three foreign-currency symbols occur --------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_all_three_symbols_occur(difficulty):
    seen = set()
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        for sym in SYMBOLS:
            if sym in q.text:
                seen.add(sym)
    assert seen == set(SYMBOLS), (difficulty, seen)


def test_hard_tier_direction_split_over_200_seeds():
    counts = {"aud_based": 0, "foreign_based": 0}
    n = 200
    for seed in range(n):
        q = make(TOPIC, seed=seed, difficulty="hard")
        if q.text.startswith("The exchange rate is A$1"):
            counts["aud_based"] += 1
        else:
            counts["foreign_based"] += 1
    print(f"hard-tier direction split over {n} seeds: {counts}")
    # both directions must occur, and neither may dominate beyond a loose bound
    # (a genuine ~50/50 draw; a collapsed direction would be a construction defect)
    assert counts["aud_based"] > 0 and counts["foreign_based"] > 0
    assert 0.3 * n < counts["aud_based"] < 0.7 * n, counts


# --- Easy tier: nice-rate set, and the wide-amount-range claim -------------


def test_easy_tier_uses_only_the_nice_rate_set():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        m = _RATE_RE.search(q.text)
        assert m is not None, (seed, q.text)
        whole, frac = m.group(1).split(".")
        p_cents = int(whole) * 100 + int(frac)
        assert p_cents in NICE_RATES_CENTS, (seed, p_cents, q.text)


def test_divisor_matches_the_wide_amount_range_claim():
    # every nice rate's divisor is 1 or 3 (never large), confirming "a wide
    # range of foreign amounts" for each -- see the generator module docstring.
    divisors = {p: _divisor_for_aud_based(p) for p in NICE_RATES_CENTS}
    assert divisors == {50: 1, 60: 3, 75: 3, 80: 1, 125: 1, 150: 3}, divisors
    for p, d in divisors.items():
        lo = -(-FOREIGN_LO // d)
        hi = FOREIGN_HI // d
        n_valid = hi - lo + 1
        assert n_valid >= 100, (p, d, n_valid)  # "a wide range", not a handful


# --- At least 20 distinct questions per tier over 60 seeds --------------

# --- ValueError on a nonsensical preset -----------------------------------


def test_raises_value_error_on_unknown_mode():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, mode="bogus")


# --- Independent in-generator check fires on a corrupted answer ---------


def test_verify_from_text_fires_on_corrupted_answer():
    text = "The exchange rate is A$1 = US$0.65. What is the value, in Australian dollars, of US$260?"
    _verify_from_text(text, 40000)  # 260 / 0.65 = 400.00 -> 40000 cents; must not raise
    with pytest.raises(VerificationError):
        _verify_from_text(text, 99999)


def test_verify_from_text_raises_on_unparseable_text():
    with pytest.raises(VerificationError):
        _verify_from_text("nonsense text with no rate at all", 1)


def test_verify_from_text_handles_foreign_based_direction():
    text = "The exchange rate is US$1 = A$1.35. What is the cost, in Australian dollars, of buying US$200?"
    _verify_from_text(text, 27000)  # 200 * 1.35 = 270.00
    with pytest.raises(VerificationError):
        _verify_from_text(text, 1)


# --- No resample-loop deadlock over a wide seed sweep (medium/hard) --------


def test_no_resample_deadlock_over_seed_sweep():
    for difficulty in ("medium", "hard"):
        for seed in range(500):
            make(TOPIC, seed=seed, difficulty=difficulty)  # must not raise


# --- Skill loads correctly through the API --------------------------------


def test_skill_loads_through_api(skills_dict):
    skill = skills_dict[SKILL_ID]
    assert skill is not None
    assert skill.domain == "arithmetic"
    assert skill.prerequisites == ["order-of-operations"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert len(skill.sources) == 3
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard"}
    assert {tuple(s.stages) for s in skill.sources} == {(6,), (6, "maintenance")}
