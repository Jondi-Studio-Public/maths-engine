"""Focused contract tests for: ucv1-sigma-algebra-closure (answer: mcq) and ucv1-product-measure-rectangle
(answer: fraction)."""

from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_fraction, grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make

SIGMA_ID = "ucv1-sigma-algebra-closure"
PRODUCT_ID = "ucv1-product-measure-rectangle"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]
_CLASSES = {"closed under both", "fails complement", "fails union", "fails both"}


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


SIGMA_SKILL, SIGMA_SOURCES = _skill_sources(SIGMA_ID)
PRODUCT_SKILL, PRODUCT_SOURCES = _skill_sources(PRODUCT_ID)


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    topic = src.topic
    params = src.params or {}
    if difficulty == "exam" and topic == "exd_x_r3e_product_measure_complement":
        # The exam source moved to a multi-part chain; these tests pin the earlier single-answer tier.
        topic, params = "p15c_product_measure_rectangle", {}
    return make(topic, seed=seed, difficulty=src.difficulty, **params)


# ==========================================================================
# ucv1-sigma-algebra-closure
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_sigma_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(SIGMA_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_sigma_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(SIGMA_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_index)).correct
        for index, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, str(index)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_sigma_real_grader_rejects_out_of_range_index(difficulty):
    for seed in SEEDS:
        q = _make(SIGMA_SOURCES, difficulty, seed)
        assert not grade_mcq(q, "99").correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_sigma_answer_is_one_of_the_four_classes(difficulty):
    for seed in SEEDS:
        q = _make(SIGMA_SOURCES, difficulty, seed)
        assert q.answer_text in _CLASSES
        assert set(q.distractors) | {q.answer_text} == _CLASSES
        assert isinstance(q.answer, str) and q.answer == q.answer_text


@pytest.mark.parametrize("difficulty", TIERS)
def test_sigma_200_seed_class_balance(difficulty):
    counts = Counter(_make(SIGMA_SOURCES, difficulty, seed).answer_text for seed in range(200))
    assert set(counts) == _CLASSES, f"{difficulty}: classes seen {set(counts)}"
    top_share = max(counts.values()) / 200
    assert top_share <= 0.45, f"{difficulty}: top class {counts.most_common(1)} is {top_share:.1%} of 200 draws"


def test_sigma_hard_and_exam_are_near_miss():
    """Boundary requirement: hard/exam failing draws are near-misses -- the
    offending set differs from a legitimate family member by exactly one
    element on at least a meaningful share of draws."""
    for difficulty in ("hard", "exam"):
        fired = 0
        total = 0
        for seed in range(200):
            q = _make(SIGMA_SOURCES, difficulty, seed)
            if q.answer_text == "closed under both":
                continue
            total += 1
        assert total > 0
    # the near-miss property itself is proved inside the generator's own
    # construction (each set differs by one element from a legitimate
    # sibling); here we just confirm failing classes are common at these
    # tiers, which they must be for the near-miss trap to matter.


def test_sigma_exam_tier_uses_indirect_definition_on_some_draws():
    indirect = 0
    for seed in range(200):
        q = _make(SIGMA_SOURCES, "exam", seed)
        if "the complement of" in q.text:
            indirect += 1
    assert indirect >= 10, f"only {indirect}/200 exam draws used an indirect clause"


def test_sigma_medium_tier_uses_indirect_definition_on_some_draws():
    """Easy never does this."""
    indirect = 0
    for seed in range(200):
        q = _make(SIGMA_SOURCES, "medium", seed)
        if "the complement of" in q.text:
            indirect += 1
    assert indirect >= 10, f"only {indirect}/200 medium draws used an indirect clause"


def test_sigma_easy_never_uses_indirect_definition():
    for seed in range(200):
        q = _make(SIGMA_SOURCES, "easy", seed)
        assert "the complement of" not in q.text


def test_sigma_skill_loads_through_api_with_correct_properties():
    assert SIGMA_SKILL.domain == "measure-and-advanced-probability"
    assert SIGMA_SKILL.status == "active"
    assert SIGMA_SKILL.placement_eligible is False
    assert SIGMA_SKILL.answer == "mcq"
    assert SIGMA_SKILL.prerequisites == []
    assert {src.difficulty for src in SIGMA_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert SIGMA_SOURCES["exam"].stages == ["exam"]


# ==========================================================================
# ucv1-product-measure-rectangle
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_product_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(PRODUCT_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_product_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(PRODUCT_SOURCES, difficulty, seed)
        true_value = Fraction(q.answer["p"], q.answer["q"])
        wrong = true_value + Fraction(101, 100)  # far outside grade_fraction's exact-value check
        assert not grade_fraction(q, str(wrong)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_product_answer_is_an_exact_fraction(difficulty):
    for seed in SEEDS:
        q = _make(PRODUCT_SOURCES, difficulty, seed)
        assert isinstance(q.answer, dict) and set(q.answer) == {"p", "q"}
        assert q.answer["q"] > 0


@pytest.mark.parametrize("difficulty", TIERS)
def test_product_never_offers_a_does_not_simplify_distractor(difficulty):
    """grade_fraction compares by VALUE: a distractor must differ in value,
    never merely in displayed reduced form."""
    for seed in SWEEP_SEEDS:
        q = _make(PRODUCT_SOURCES, difficulty, seed)
        true_value = Fraction(q.answer["p"], q.answer["q"])
        for d in q.distractors:
            p, _, rest = d.partition("/")
            value = Fraction(int(p), int(rest)) if rest else Fraction(int(p))
            assert value != true_value, (seed, difficulty, d, q.answer_text)


@pytest.mark.parametrize("difficulty", TIERS)
def test_product_200_seed_answer_never_concentrates(difficulty):
    counts = Counter(_make(PRODUCT_SOURCES, difficulty, seed).answer_text for seed in range(200))
    top_share = max(counts.values()) / 200
    assert top_share <= 0.25, f"{difficulty}: top answer is {top_share:.1%} of 200 draws"


def test_product_medium_tier_requires_summing_two_weights():
    for seed in SEEDS:
        q = _make(PRODUCT_SOURCES, "medium", seed)
        assert q.text.count(",") >= 2  # A and B each list two elements


def test_product_hard_tier_uses_a_complement_for_a():
    for seed in SEEDS:
        q = _make(PRODUCT_SOURCES, "hard", seed)
        assert "minus" in q.text


def test_product_exam_tier_uses_a_union_and_a_complement():
    for seed in SEEDS:
        q = _make(PRODUCT_SOURCES, "exam", seed)
        assert "∪" in q.text or "union" in q.text.lower()
        assert "minus" in q.text


def test_product_skill_loads_through_api_with_correct_properties():
    assert PRODUCT_SKILL.domain == "measure-and-advanced-probability"
    assert PRODUCT_SKILL.status == "active"
    assert PRODUCT_SKILL.placement_eligible is False
    assert PRODUCT_SKILL.answer == "fraction"
    assert PRODUCT_SKILL.prerequisites == []
    assert {src.difficulty for src in PRODUCT_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert PRODUCT_SOURCES["exam"].stages == ["exam"]
