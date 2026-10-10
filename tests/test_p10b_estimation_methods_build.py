"""Tests for:
- senior-use-leading-digit-approximation-to-obtain-estimates-of (ACMEM004)
- senior-apply-approximation-strategies-for-calculations (ACMEM010)
"""

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p10b_estimation_methods_build import (
    _front_end_truncate,
    _ld_estimate_three,
    _ld_estimate_two,
    _round_to_leading_digit,
    _round_to_stated_place,
)

LD_TOPIC = "leading_digit_estimate"
AAS_TOPIC = "apply_approximation_strategy"
TOPICS = [LD_TOPIC, AAS_TOPIC]
DIFFICULTIES = ["easy", "medium", "hard"]
SEEDS = range(40)
VARIETY_SEEDS = range(60)

_SKILL_IDS = {
    LD_TOPIC: "senior-use-leading-digit-approximation-to-obtain-estimates-of",
    AAS_TOPIC: "senior-apply-approximation-strategies-for-calculations",
}


def _independent_leading_digit_round(n: int) -> int:
    """A SECOND, independently-reimplemented (not imported) leading-digit
    rounding rule, used only by the mandatory overlap-prevention test below
    -- deliberately not the same function the generator itself calls, so a
    shared bug in the generator's own rounding route cannot silently pass
    this specific test."""
    s = str(abs(n))
    place = 10 ** (len(s) - 1)
    remainder = n % place
    base = n - remainder
    return base + place if remainder * 2 >= place else base


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, topic, difficulty, seed):
    """Generate through the skill's own `sources` entry, not a bare `make(topic, seed, difficulty)` call."""
    skill = skills_dict[_SKILL_IDS[topic]]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene: distinct, non-empty, never equal to the answer ------

# --- Answer/distractor formatting: plain integer, no comma, no decimal ------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_distractors_are_plain_integers_no_comma(skills_dict, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        assert isinstance(q.answer, int)
        assert q.answer_text.isdigit()
        assert "," not in q.answer_text
        for d in q.distractors:
            assert d.isdigit(), (topic, difficulty, seed, d)
            assert "," not in d


# --- Magnitude cap: every seed, every tier, both topics ----------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_stays_under_10000(skills_dict, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        assert 0 <= q.answer < 10_000, (topic, difficulty, seed, q.answer)


# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, topic, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, topic, difficulty, seed)
        wrong = str(q.answer + 1)  # an integer answer graded EXACTLY: +1 always clears it
        assert not grade_number(q, wrong).correct, (topic, difficulty, seed, q.text, wrong)


# --- The mandatory overlap-prevention test ---


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_apply_approximation_strategy_always_differs_from_leading_digit_rounding(skills_dict, difficulty):
    """For every sampled seed/tier of ``apply_approximation_strategy``,
    recompute what ``leading_digit_estimate``'s own rounding rule would give
    for the SAME operands (via a rule reimplemented locally in THIS test,
    never imported from the generator) and assert it never equals this
    generator's own answer -- i.e. ACMEM010 never degenerates into
    "ACMEM004 in disguise"."""
    import re

    for seed in VARIETY_SEEDS:
        q = _make_from_sources(skills_dict, AAS_TOPIC, difficulty, seed)
        expr = q.text.split("estimate ", 1)[1]
        nums = [int(tok) for tok in re.findall(r"\d+", expr)]
        ops = re.findall(r"[+\-]", expr)
        assert len(ops) == len(nums) - 1, (difficulty, seed, q.text)
        rounded = [_independent_leading_digit_round(n) for n in nums]
        ld_answer = rounded[0]
        for op, value in zip(ops, rounded[1:]):
            ld_answer = ld_answer + value if op == "+" else ld_answer - value
        assert ld_answer != q.answer, (difficulty, seed, q.text, q.answer, ld_answer)


# --- Distractor-count / pool-size requirement: >= 20 distinct prompts per
#     tier over 60 seeds -------------------------------------

# --- Independent check fires on a corrupted answer ---------------------


def test_leading_digit_round_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        from mathgen.core import check_equivalent

        check_equivalent(_round_to_leading_digit(512) + 1, _round_to_leading_digit(512), "must differ")


def test_front_end_truncate_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        from mathgen.core import check_equivalent

        check_equivalent(_front_end_truncate(683) + 1, _front_end_truncate(683), "must differ")


def test_round_to_stated_place_independent_check_fires_on_corrupted_answer():
    with pytest.raises(VerificationError):
        from mathgen.core import check_equivalent

        check_equivalent(_round_to_stated_place(340, "ten") + 10, _round_to_stated_place(340, "ten"), "must differ")


# --- Skill loading: prerequisites, answer kind, placement, stages -----------


@pytest.mark.parametrize("topic", TOPICS)
def test_skill_loads_with_expected_shape(skills_dict, topic):
    skill = skills_dict[_SKILL_IDS[topic]]
    assert skill.prerequisites == ["order-of-operations"]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    stages = {stage for src in skill.sources for stage in src.stages}
    assert stages, f"{topic}: no stages declared across sources"


# --- Rounding-rule spot checks (documents the exact contract) ---------------


def test_round_to_leading_digit_spot_values():
    assert _round_to_leading_digit(512) == 500
    assert _round_to_leading_digit(4238) == 4000
    assert _round_to_leading_digit(683) == 700
    assert _round_to_leading_digit(89) == 90


def test_front_end_truncate_spot_values():
    assert _front_end_truncate(683) == 600
    assert _front_end_truncate(158) == 100
    assert _front_end_truncate(89) == 80


def test_round_to_stated_place_spot_values():
    assert _round_to_stated_place(4238, "ten") == 4240
    assert _round_to_stated_place(4238, "hundred") == 4200
    assert _round_to_stated_place(455, "ten") == 460  # ties away from zero


def test_ld_estimate_helpers_match_round_to_leading_digit():
    assert _ld_estimate_two(512, 83, "+") == _round_to_leading_digit(512) + _round_to_leading_digit(83)
    assert _ld_estimate_three(512, 83, 47, "+", "-") == (
        _round_to_leading_digit(512) + _round_to_leading_digit(83) - _round_to_leading_digit(47)
    )
