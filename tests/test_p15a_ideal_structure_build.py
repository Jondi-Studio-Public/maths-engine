"""Focused contract tests for: ucv1-ideal-membership and ucv1-structure-classification (both answer: mcq)."""

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from tests.exd_x_r2h_support import legacy_exam_sources
from mathgen import make
import mathgen.generators.p15a_ideal_structure_build  # noqa: F401 -- registers topics

IDEAL_ID = "ucv1-ideal-membership"
STRUCT_ID = "ucv1-structure-classification"
IDEAL_TOPIC = "p15a_ideal_membership"
STRUCT_TOPIC = "p15a_structure_classification"

SEEDS = range(40)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, legacy_exam_sources(skill.id, {src.difficulty: src for src in skill.sources})


IDEAL_SKILL, IDEAL_SOURCES = _skill_sources(IDEAL_ID)
STRUCT_SKILL, STRUCT_SOURCES = _skill_sources(STRUCT_ID)


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ==========================================================================
# ucv1-ideal-membership
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_ideal_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(IDEAL_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3
        assert q1.answer_text in ("A", "B", "C", "D")
        assert set(q1.distractors) | {q1.answer_text} == {"A", "B", "C", "D"}


@pytest.mark.parametrize("difficulty", TIERS)
def test_ideal_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(IDEAL_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_idx)).correct
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, str(i)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_ideal_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(IDEAL_SOURCES, difficulty, seed)
        assert not grade_mcq(q, "9").correct


def test_ideal_skill_loads_with_expected_shape():
    assert IDEAL_SKILL.answer == "mcq"
    assert IDEAL_SKILL.placement_eligible is False
    assert IDEAL_SKILL.domain == "advanced-algebra"
    assert IDEAL_SKILL.prerequisites == ["integer-polynomial-quotient-remainder"]


def test_ideal_hard_shows_unexpanded_form_and_exam_uses_a_finite_field():
    for seed in range(10):
        qh = _make(IDEAL_SOURCES, "hard", seed)
        assert "expand it first" in qh.text
        qe = _make(IDEAL_SOURCES, "exam", seed)
        assert "reduced modulo" in qe.text
        qm = _make(IDEAL_SOURCES, "medium", seed)
        assert "expand it first" not in qm.text
        assert "reduced modulo" not in qm.text


def test_ideal_hard_key_not_identifiable_by_written_form():
    """The hard tier writes every candidate as `(q)(g) + (r)`; the key must not be the only option written as a
    bare product with no `+ (` group -- every option's count of `+ (` groups must be equal (here: exactly one,
    always)."""
    import re

    for seed in range(100):
        q = _make(IDEAL_SOURCES, "hard", seed)
        m = re.search(r"A: (.*?)   B: (.*?)   C: (.*?)   D: (.*)$", q.text)
        assert m, q.text
        options = [m.group(i) for i in range(1, 5)]
        plus_counts = [opt.count("+ (") for opt in options]
        assert len(set(plus_counts)) == 1, (seed, options, plus_counts)
        assert plus_counts[0] == 1, (seed, options, plus_counts)


def test_ideal_never_asks_for_the_remainder_itself():
    """Reuse note: an option list of remainders would land on the
    prerequisite (integer-polynomial-quotient-remainder); every option here
    is a bare letter A-D, never a polynomial/remainder value."""
    for seed in range(10):
        for tier in TIERS:
            q = _make(IDEAL_SOURCES, tier, seed)
            assert q.answer_text in ("A", "B", "C", "D")


# ==========================================================================
# ucv1-structure-classification
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_struct_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(STRUCT_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3
        assert set(q1.distractors) | {q1.answer_text} == {"A", "B", "C", "D"}


@pytest.mark.parametrize("difficulty", TIERS)
def test_struct_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(STRUCT_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_idx)).correct
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, str(i)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_struct_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(STRUCT_SOURCES, difficulty, seed)
        assert not grade_mcq(q, "9").correct


def test_struct_skill_loads_with_expected_shape():
    assert STRUCT_SKILL.answer == "mcq"
    assert STRUCT_SKILL.placement_eligible is False
    assert STRUCT_SKILL.domain == "advanced-algebra"
    assert STRUCT_SKILL.prerequisites == ["cyclic-group-element-order"]


def test_struct_exam_presents_raw_untallied_list():
    for seed in range(10):
        qe = _make(STRUCT_SOURCES, "exam", seed)
        assert "Tally the element orders" in qe.text
        qh = _make(STRUCT_SOURCES, "hard", seed)
        assert "Tally" not in qh.text


def test_struct_hard_tier_uses_order_12_widest_option_set():
    for seed in range(10):
        qh = _make(STRUCT_SOURCES, "hard", seed)
        assert "order 12" in qh.text


def test_struct_never_computes_an_element_order():
    for seed in range(10):
        for tier in TIERS:
            q = _make(STRUCT_SOURCES, tier, seed)
            assert "compute the" not in q.text.lower()
