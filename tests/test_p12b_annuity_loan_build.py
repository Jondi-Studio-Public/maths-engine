"""Reducing-balance loan repayment and annuity/perpetuity present value.

senior-with-the-aid-of-a-financial-calculator-or (ACMGM098, topic
p12b_reducing_balance_loan, answer: number) and
senior-with-the-aid-of-a-financial-calculator-or-acmgm100 (ACMGM100, topic
p12b_annuity_perpetuity, answer: number). See the generator module's own
docstring for the exact-arithmetic construction and the boundary
confirmation against compound-interest, flat-loan-repayment and the
neighbouring skills.
"""

from __future__ import annotations

import collections

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError, check_equivalent

TIERS = ("easy", "medium", "hard", "exam")
SEEDS = range(60)
CONCENTRATION_SEEDS = range(200)

RBL_TOPIC = "p12b_reducing_balance_loan"
RBL_SKILL = "senior-with-the-aid-of-a-financial-calculator-or"

ANN_TOPIC = "p12b_annuity_perpetuity"
ANN_SKILL = "senior-with-the-aid-of-a-financial-calculator-or-acmgm100"

TOPICS = (RBL_TOPIC, ANN_TOPIC)


# --- Determinism ------------------------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_generator_is_deterministic(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        repeat = make(topic, seed=seed, difficulty=tier)
        assert q.to_json() == repeat.to_json()


# --- Distractor hygiene ------------------------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors, f"seed={seed}: answer in distractors"
        assert len(q.distractors) == len(set(q.distractors)), f"seed={seed}: dup distractors"
        assert len(q.distractors) >= 2


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_answer_and_distractors_formatted_alike(topic, tier):
    """No option is identifiable by its shape: the money-valued tiers all
    carry exactly 2 decimal places (answer and every distractor); the
    annuity hard tier (answer is a whole number of years) carries none."""
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        is_year_count = topic == ANN_TOPIC and tier == "hard"
        options = [q.answer_text, *q.distractors]
        for opt in options:
            has_dp = "." in opt
            assert has_dp != is_year_count, (
                f"seed={seed} {topic}/{tier}: option {opt!r} decimal-point shape inconsistent with the rest of the tier"
            )


# --- Real grader: accepts canonical, rejects distractors and a wrong value --


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_and_rejects(topic, tier):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        result = grade_number(q, q.answer_text)
        assert result.correct, f"seed={seed} {topic}/{tier}: canonical not accepted"
        for d in q.distractors:
            result = grade_number(q, d)
            assert not result.correct, f"seed={seed} {topic}/{tier}: distractor {d!r} accepted"
        # A deliberately wrong value, offset by more than the stated precision.
        if "." in q.answer_text:
            wrong = str(round(float(q.answer_text) + 1.0, 2))
        else:
            wrong = str(int(q.answer_text) + 1)
        result = grade_number(q, wrong)
        assert not result.correct, f"seed={seed} {topic}/{tier}: deliberately wrong {wrong!r} accepted"


# --- Answer kind -------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_rbl_answer_is_money_to_two_decimal_places(tier):
    for seed in SEEDS:
        q = make(RBL_TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, (int, float))
        text = q.answer_text
        assert "," not in text
        if "." in text:
            assert len(text.split(".")[1]) == 2, f"seed={seed} {tier}: {text!r} not 2 d.p."


@pytest.mark.parametrize("tier", ("easy", "medium", "exam"))
def test_annuity_money_tiers_answer_is_money_to_two_decimal_places(tier):
    for seed in SEEDS:
        q = make(ANN_TOPIC, seed=seed, difficulty=tier)
        assert isinstance(q.answer, (int, float))
        text = q.answer_text
        assert "," not in text
        if "." in text:
            assert len(text.split(".")[1]) == 2, f"seed={seed} {tier}: {text!r} not 2 d.p."


def test_annuity_hard_answer_is_a_bare_positive_integer_year_count():
    for seed in SEEDS:
        q = make(ANN_TOPIC, seed=seed, difficulty="hard")
        assert isinstance(q.answer, int)
        assert q.answer > 0
        assert "." not in q.answer_text
        assert "," not in q.answer_text


# --- Distinct rendered questions ----------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_at_least_20_distinct_questions_per_tier(topic, tier):
    texts = {make(topic, seed=seed, difficulty=tier).text for seed in SEEDS}
    assert len(texts) >= 20, f"{topic}/{tier}: only {len(texts)} distinct questions over 60 seeds"


# --- Answer-concentration measure (200 seeds) --------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", TIERS)
def test_answer_concentration_under_25_percent_at_200_seeds(topic, tier):
    """No single answer value above 25% of draws at 200 seeds."""
    counts = collections.Counter()
    for seed in CONCENTRATION_SEEDS:
        q = make(topic, seed=seed, difficulty=tier)
        counts[q.answer_text] += 1
    total = sum(counts.values())
    top_text, top_n = counts.most_common(1)[0]
    share = top_n / total
    assert share <= 0.25, f"{topic}/{tier}: top answer {top_text!r} is {share:.1%} of {total} draws"


# --- Independent-check tripwire ------------------------------------------


def test_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        check_equivalent(5, 6, "should fail")


# --- Skill loading -------------------------------------------------------


def test_rbl_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == RBL_SKILL), None)
    assert skill is not None, f"{RBL_SKILL} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["simple-interest"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        assert source.topic == RBL_TOPIC
    difficulties = {source.difficulty for source in skill.sources}
    assert difficulties == set(TIERS)
    exam_source = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_source.stages == ["exam"]


def test_annuity_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == ANN_SKILL), None)
    assert skill is not None, f"{ANN_SKILL} not found"
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["simple-interest"]
    assert len(skill.sources) == 4
    for source in skill.sources:
        if source.difficulty != "exam":  # the exam source is a multi-part chain with its own module
            assert source.topic == ANN_TOPIC
    difficulties = {source.difficulty for source in skill.sources}
    assert difficulties == set(TIERS)
    exam_source = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_source.stages == ["exam"]


# --- Tie-boundary proof: finite annuity PV never collides with the
# perpetuity PV at the same rate -----------------------------------------


def test_finite_annuity_pv_never_ties_the_perpetuity_pv():
    """The medium tier's own construction loop already rejects any draw
    where the finite PV is within a cent of the perpetuity PV (see
    `_ann_medium`'s `check` inside the generator); this test re-derives the
    same guard from the OUTSIDE, independently, over every medium-tier seed
    -- confirming the guard fires as a property of every shipped question,
    not merely as an internal assertion that could silently stop firing."""
    from fractions import Fraction
    import re

    pattern = re.compile(
        r"\$(\d+)[^$]*?(\d+) years[^$]*?([\d.]+)% per annum",
        re.DOTALL,
    )
    for seed in SEEDS:
        q = make(ANN_TOPIC, seed=seed, difficulty="medium")
        m = pattern.search(q.text)
        assert m, f"seed={seed}: unparseable medium question: {q.text!r}"
        income, term_years, rate = m.groups()
        i = Fraction(rate) / 100 / 12
        perp_pv = Fraction(income) / i
        finite_pv = Fraction(q.answer_text)
        assert perp_pv - finite_pv > Fraction(1, 100), (
            f"seed={seed}: finite PV {finite_pv} within a cent of perpetuity PV {perp_pv} -- tie-boundary guard failed"
        )
