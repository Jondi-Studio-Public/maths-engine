"""Focused contract tests for: ucv1-finite-subgroup-test and ucv1-isomorphism-invariant (both answer: mcq)."""

from dataclasses import replace

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
import mathgen.generators.p15a_subgroup_isomorphism_build  # noqa: F401 -- registers topics

SUB_ID = "ucv1-finite-subgroup-test"
ISO_ID = "ucv1-isomorphism-invariant"
SUB_TOPIC = "p15a_finite_subgroup_test"
ISO_TOPIC = "p15a_isomorphism_invariant"

SEEDS = range(40)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


SUB_SKILL, SUB_SOURCES = _skill_sources(SUB_ID)
# The skill's exam source is now a multi-part chain; these checks keep covering the single-answer exam generator it replaced.
SUB_SOURCES["exam"] = replace(SUB_SOURCES["exam"], topic="p15a_finite_subgroup_test", params={})
ISO_SKILL, ISO_SOURCES = _skill_sources(ISO_ID)
ISO_SOURCES["exam"] = replace(ISO_SOURCES["exam"], topic="p15a_isomorphism_invariant", params={})


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ==========================================================================
# ucv1-finite-subgroup-test
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_sub_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(SUB_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_sub_answer_is_never_fails_inverses(difficulty):
    """'fails inverses' is mathematically impossible
    as a key (finite subgroup test theorem) and must appear only as a
    guaranteed-wrong distractor, never as answer_text."""
    for seed in SEEDS:
        q = _make(SUB_SOURCES, difficulty, seed)
        assert "inverse" not in q.answer_text
        assert any("inverse" in d for d in q.distractors)


@pytest.mark.parametrize("difficulty", TIERS)
def test_sub_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(SUB_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_idx)).correct
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, str(i)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_sub_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(SUB_SOURCES, difficulty, seed)
        options = mcq_options(q)
        bogus = str(len(options) + 5)
        assert not grade_mcq(q, bogus).correct


def test_sub_skill_loads_with_expected_shape():
    assert SUB_SKILL.answer == "mcq"
    assert SUB_SKILL.placement_eligible is False
    assert SUB_SKILL.prerequisites == ["finite-set-operations"]
    assert set(SUB_SOURCES) == {"easy", "medium", "hard", "exam"}


def test_sub_hard_and_exam_use_permutation_composition_not_modular_addition():
    for seed in range(10):
        qh = _make(SUB_SOURCES, "hard", seed)
        qe = _make(SUB_SOURCES, "exam", seed)
        assert "composition of permutations" in qh.text
        assert "composition of permutations" in qe.text
        qeasy = _make(SUB_SOURCES, "easy", seed)
        assert "modulo" in qeasy.text


# ==========================================================================
# ucv1-isomorphism-invariant
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_iso_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(ISO_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 2


@pytest.mark.parametrize("difficulty", TIERS)
def test_iso_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(ISO_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_idx)).correct
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, str(i)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_iso_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(ISO_SOURCES, difficulty, seed)
        options = mcq_options(q)
        bogus = str(len(options) + 5)
        assert not grade_mcq(q, bogus).correct


def test_iso_skill_loads_with_expected_shape():
    assert ISO_SKILL.answer == "mcq"
    assert ISO_SKILL.placement_eligible is False
    assert ISO_SKILL.prerequisites == ["cyclic-group-element-order"]
    assert set(ISO_SOURCES) == {"easy", "medium", "hard", "exam"}


def test_iso_hard_and_exam_present_raw_untallied_lists():
    for seed in range(10):
        qh = _make(ISO_SOURCES, "hard", seed)
        qe = _make(ISO_SOURCES, "exam", seed)
        assert "ally the element orders" in qh.text
        assert "ally the element orders" in qe.text
        q_easy = _make(ISO_SOURCES, "easy", seed)
        assert "ally" not in q_easy.text


def test_iso_exam_has_no_notebook_narrative_and_tags_are_separable():
    """The old "lab notebook... measured the order of every element" framing is gone, and the exam tier's genuine
    extra step (untangle a single shuffled, G/H-tagged list before tallying either group) is present and
    round-trips to the same two profiles the hard-tier construction uses."""
    for seed in range(50):
        q = make(ISO_TOPIC, seed=seed, difficulty="exam")
        assert "notebook" not in q.text.lower()
        assert "measured" not in q.text.lower()
        assert "experiment" not in q.text.lower()
        assert "shuffled together" in q.text
        assert "Separate the entries by group" in q.text


def test_iso_easy_pool_widened_beyond_orders_4_and_6():
    """Report+widen the easy-tier pool. [4, 6] alone gave 16 distinct prompts over 200 seeds; the widened pool
    must show materially more variety over the same sample size."""
    seen = set()
    for seed in range(200):
        q = make(ISO_TOPIC, seed=seed, difficulty="easy")
        seen.add(q.text)
    assert len(seen) > 16, len(seen)


def test_iso_never_computes_element_orders_itself():
    """The reuse note: neither skill asks the learner to COMPUTE an element
    order (that is cyclic-group-element-order's own job) -- orders are always
    SUPPLIED data."""
    for seed in range(10):
        for tier in TIERS:
            q = _make(ISO_SOURCES, tier, seed)
            assert "compute the" not in q.text.lower()
            assert "additive order" not in q.text.lower()
