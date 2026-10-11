"""Examine the approximate normality of the distribution of p-hat
(senior-examine-the-approximate-normality-of-the-distribution-of)."""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.generators.p8c_normality_approx_build import (
    ALL_CATEGORIES,
    CATEGORY_BOTH_FAIL,
    CATEGORY_P_FAILS,
    CATEGORY_Q_FAILS,
    CATEGORY_VALID,
    _independent_check_borderline,
    _independent_check_counts,
    _independent_check_direct,
)
from mathgen.core import VerificationError

TOPIC = "p8c_normality_approx_phat"
SEEDS = range(40)
TIERS = ["easy", "medium", "hard"]


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_are_exactly_the_other_three_categories(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q.answer_text in ALL_CATEGORIES
        assert len(q.distractors) == 3
        assert set(q.distractors) == set(ALL_CATEGORIES) - {q.answer_text}


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_mcq_grader_accepts_answer_rejects_distractors(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_idx)).correct
        for i, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, str(i)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_deliberately_wrong_answer_rejected(difficulty):
    for seed in SEEDS[:10]:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        bogus_idx = len(options) + 5
        assert not grade_mcq(q, str(bogus_idx)).correct
        assert not grade_mcq(q, "not-a-number").correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_rule_stated_explicitly_in_every_question(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert "n×p-hat" in q.text and "n×(1-p-hat)" in q.text
        assert "at least 10" in q.text


@pytest.mark.parametrize("difficulty", TIERS)
def test_no_literal_asterisk_anywhere_learner_facing(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        for field in [q.text, q.answer_text, *q.distractors]:
            assert "*" not in field


def test_easy_gives_phat_directly_not_counts():
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        assert "sample proportion of p-hat =" in q.text
        assert "x =" not in q.text


def test_medium_and_hard_give_counts_not_phat_directly():
    for difficulty in ["medium", "hard"]:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            assert re.search(r"x = \d+ success", q.text)
            assert "sample proportion of p-hat =" not in q.text


def test_hard_is_genuinely_borderline_never_exactly_on_threshold():
    """At least one of n*phat / n*(1-phat) sits within 1 or 2 of 10, and
    neither ever lands exactly on 10."""
    close_count = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        n = int(re.search(r"n = (\d+)", q.text).group(1))
        x = int(re.search(r"x = (\d+)", q.text).group(1))
        prod_p, prod_q = x, n - x
        assert prod_p != 10 and prod_q != 10
        min_diff = min(abs(prod_p - 10), abs(prod_q - 10))
        assert min_diff in (1, 2)
        close_count += 1
    assert close_count == len(SEEDS)


def test_easy_and_medium_are_confidently_clear_of_the_threshold():
    for difficulty in ["easy", "medium"]:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=difficulty)
            if difficulty == "easy":
                n = int(re.search(r"n = (\d+)", q.text).group(1))
                phat = Fraction(re.search(r"p-hat = (\d+\.\d+)", q.text).group(1))
            else:
                n = int(re.search(r"n = (\d+)", q.text).group(1))
                x = int(re.search(r"x = (\d+)", q.text).group(1))
                phat = Fraction(x, n)
            prod_p, prod_q = n * phat, n * (1 - phat)
            assert abs(prod_p - 10) > 3
            assert abs(prod_q - 10) > 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_all_four_categories_occur_over_300_seeds(difficulty):
    """Sweep wider than the 40-seed determinism check to confirm the full
    draw space actually reaches all four categories at every tier."""
    seen = {make(TOPIC, seed=s, difficulty=difficulty).answer_text for s in range(300)}
    assert seen == set(ALL_CATEGORIES), f"{difficulty}: only saw {seen}"


def test_independent_check_direct_fires_on_corrupted_category():
    n, phat = 40, Fraction(1, 2)
    phat_display = "0.50"
    _independent_check_direct(n, phat, phat_display, CATEGORY_VALID)  # true category, no raise
    with pytest.raises(VerificationError):
        _independent_check_direct(n, phat, phat_display, CATEGORY_BOTH_FAIL)


def test_independent_check_counts_fires_on_corrupted_category():
    n, x = 40, 20
    _independent_check_counts(n, x, CATEGORY_VALID)  # true category, no raise
    with pytest.raises(VerificationError):
        _independent_check_counts(n, x, CATEGORY_P_FAILS)


def test_independent_check_borderline_fires_on_corrupted_category():
    n, x = 20, 8  # x=8 -> prod_p=8<10, prod_q=12>=10 -> CATEGORY_P_FAILS
    _independent_check_borderline(n, x, CATEGORY_P_FAILS)  # true category, no raise
    with pytest.raises(VerificationError):
        _independent_check_borderline(n, x, CATEGORY_Q_FAILS)


def test_skill_loads_with_expected_shape():
    _, skills = load_skills()
    by_id = {s.id: s for s in skills}
    skill = by_id["senior-examine-the-approximate-normality-of-the-distribution-of"]
    assert skill.prerequisites == ["simple-event-probability"]
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}


def test_hard_tier_sources_reach_all_four_categories():
    """The skill's hard tier draws from both the borderline and the from_failures source; the failures source
    alone only ever gives the two single-product categories."""
    sk = next(s for s in load_skills()[1] if s.id == "senior-examine-the-approximate-normality-of-the-distribution-of")
    hard = [src for src in sk.sources if src.difficulty == "hard"]
    assert {src.params["variant"] for src in hard} == {"borderline", "from_failures"}
    answers = {make(src.topic, seed=s, difficulty="hard", **src.params).answer_text for src in hard for s in range(60)}
    assert len(answers) == 4, answers
