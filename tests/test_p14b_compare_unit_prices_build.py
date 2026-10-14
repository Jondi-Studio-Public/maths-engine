"""Tests `p14b_compare_unit_prices`: no generator errors, distractor shape, grading and decision-answer checks.

compare-unit-prices (ACMEM016, ACMGM005), domain arithmetic, answer: mcq."""

from dataclasses import replace
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p14b_compare_unit_prices_build import _decide, _verify_decision

TOPIC = "p14b_compare_unit_prices"
SKILL_ID = "compare-unit-prices"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)

# easy/medium/exam are 2-pack: 3-option mcq (answer + 2 distractors).
# hard is 3-pack: 4-option mcq (answer + 3 distractors).
_EXPECTED_DISTRACTORS = {"easy": 2, "medium": 2, "hard": 3, "exam": 2}


def _make_from_sources(skills_dict, difficulty, seed):
    """Generate through the skill's own `sources` entry, not a bare `make(topic, seed, difficulty)` call."""
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    if difficulty == "exam":
        src = replace(src, topic="p14b_compare_unit_prices", params={})
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


# --- Determinism -----------------------------------------------------------

# --- No generator exceptions over a wide seed range -------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_no_generator_exceptions(difficulty):
    for seed in range(400):
        make(TOPIC, seed=seed, difficulty=difficulty)


# --- Distractor hygiene: distinct, non-empty, never equal to the answer,
# and every applicable option present on EVERY draw (never a filtered-down
# 2-option mcq for the 3-pack hard tier, never a filtered-down 2-option mcq
# missing the tie option for the 2-pack tiers). ------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractor_count_distinct_nonempty_never_equal_answer(skills_dict, difficulty):
    expected_n = _EXPECTED_DISTRACTORS[difficulty]
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert len(q.distractors) == expected_n, (difficulty, seed, q.distractors)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_every_draw_offers_every_applicable_option(difficulty):
    """A generic distractor check only requires >=2 distractors on 85% of draws; this skill's answer space
    is a small FIXED enumeration, so every draw must offer every option, not
    just 85% of them."""
    expected_total = _EXPECTED_DISTRACTORS[difficulty] + 1
    for seed in range(100):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert len(q.distractors) + 1 == expected_total, (difficulty, seed, q.distractors)


# --- Real grader accept/reject, via the actual mcq grader -------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_index)).correct, (difficulty, seed, q.text)
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, str(i)).correct, (difficulty, seed, q.text, opt)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_index(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        wrong_index = (correct_index + 1) % len(options)
        assert not grade_mcq(q, str(wrong_index)).correct, (difficulty, seed, q.text)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_an_out_of_range_index(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert not grade_mcq(q, "99").correct, (difficulty, seed)
        assert not grade_mcq(q, "not a number").correct, (difficulty, seed)


# --- Answer-kind discipline: the answer is always one of the fixed labels --

_TWO_PACK_OPTIONS = {"Pack A", "Pack B", "Tie -- the same price per unit"}
_THREE_PACK_OPTIONS = {"Pack A", "Pack B", "Pack C", "Tie -- two packs share the cheapest price"}


@pytest.mark.parametrize("difficulty", ["easy", "medium", "exam"])
def test_two_pack_answer_is_always_one_of_the_fixed_labels(difficulty):
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q.answer_text in _TWO_PACK_OPTIONS
        assert set(q.distractors) | {q.answer_text} == _TWO_PACK_OPTIONS


def test_three_pack_answer_is_always_one_of_the_fixed_labels():
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="hard")
        assert q.answer_text in _THREE_PACK_OPTIONS
        assert set(q.distractors) | {q.answer_text} == _THREE_PACK_OPTIONS


# --- Never a numeric unit price or numeric gap in the graded answer --------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_text_is_never_a_bare_number(difficulty):
    """The kernel that separates this skill from `unit-pricing` and
    `senior-use-rates-to-make-comparisons`'s hard tier: the graded answer is
    a DECISION label, never a numeric unit price or a numeric gap."""
    for seed in range(100):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q.answer_text.startswith("Pack") or q.answer_text.startswith("Tie")
        try:
            float(q.answer_text)
        except ValueError:
            pass
        else:
            pytest.fail(f"answer_text {q.answer_text!r} parses as a bare number")


# This just guards against a construction regression that collapses the split (e.g. "tie" becoming
# unreachable, or one pack always winning).


@pytest.mark.parametrize("difficulty", ["easy", "medium", "exam"])
def test_two_pack_answer_concentration_is_roughly_even(difficulty):
    from collections import Counter

    counts = Counter(make(TOPIC, seed=s, difficulty=difficulty).answer_text for s in range(200))
    for label in _TWO_PACK_OPTIONS:
        share = counts.get(label, 0) / 200
        assert 0.15 <= share <= 0.55, (difficulty, label, share, dict(counts))


def test_three_pack_answer_concentration_is_roughly_even():
    from collections import Counter

    counts = Counter(make(TOPIC, seed=s, difficulty="hard").answer_text for s in range(200))
    for label in _THREE_PACK_OPTIONS:
        share = counts.get(label, 0) / 200
        assert 0.10 <= share <= 0.45, (label, share, dict(counts))


# --- At least 20 distinct questions per tier over 60 seeds ------------------

# --- ValueError on a nonsensical mode ---------------------------------------


def test_raises_value_error_on_unknown_mode():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, mode="bogus")


# --- The independent in-generator decision check fires on a corrupted case -


def test_decide_two_way():
    assert _decide({"A": Fraction(1, 2), "B": Fraction(1, 3)}) == "B"
    assert _decide({"A": Fraction(1, 3), "B": Fraction(1, 2)}) == "A"
    assert _decide({"A": Fraction(1, 2), "B": Fraction(1, 2)}) == "tie"


def test_decide_three_way_two_tied():
    assert _decide({"A": Fraction(1, 2), "B": Fraction(1, 3), "C": Fraction(1, 3)}) == "tie"
    assert _decide({"A": Fraction(1, 4), "B": Fraction(1, 3), "C": Fraction(1, 2)}) == "A"


def test_verify_decision_fires_on_corrupted():
    prices = {"A": Fraction(1, 3), "B": Fraction(1, 2)}
    _verify_decision(prices, "A")  # does not raise: A's rate is genuinely smaller
    with pytest.raises(VerificationError):
        _verify_decision(prices, "B")
    with pytest.raises(VerificationError):
        _verify_decision(prices, "tie")


# --- The skill loads correctly through the API ------------------------------


def test_skill_loads_through_api(skills_dict):
    skill = skills_dict[SKILL_ID]
    assert skill is not None
    assert skill.domain == "arithmetic"
    assert skill.prerequisites == ["ratio-rate-proportion", "unit-pricing"]
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    difficulties = {s.difficulty for s in skill.sources}
    assert difficulties == {"easy", "medium", "hard", "exam"}
    exam_source = next(s for s in skill.sources if s.difficulty == "exam")
    assert exam_source.stages == ["exam"]
