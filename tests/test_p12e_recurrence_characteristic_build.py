"""Focused contract tests for: ucv1-recurrence-two-step (answer: number) and
ucv1-second-order-characteristic-roots (answer: pair, labelled)."""

import re

import pytest

from mathgen.grading.typed import grade_number, grade_pair
from mathgen.skills import load_skills
from mathgen import make

RECURRENCE_ID = "ucv1-recurrence-two-step"
ROOTS_ID = "ucv1-second-order-characteristic-roots"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_NUMBER_RE = re.compile(r"-?\d+")
_LABELLED_PAIR_RE = re.compile(r"(\w+) = (-?\d+), (\w+) = (-?\d+)")


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


RECURRENCE_SKILL, RECURRENCE_SOURCES = _skill_sources(RECURRENCE_ID)
ROOTS_SKILL, ROOTS_SOURCES = _skill_sources(ROOTS_ID)


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ==========================================================================
# ucv1-recurrence-two-step (answer: number)
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_recurrence_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(RECURRENCE_SOURCES, difficulty, seed)
        wrong = str(int(q.answer) + 1000)
        assert not grade_number(q, wrong).correct, f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"


@pytest.mark.parametrize("difficulty", TIERS)
def test_recurrence_answer_is_an_exact_integer(difficulty):
    for seed in SEEDS:
        q = _make(RECURRENCE_SOURCES, difficulty, seed)
        assert _NUMBER_RE.fullmatch(q.answer_text), q.answer_text
        assert isinstance(q.answer, int)


@pytest.mark.parametrize("difficulty", TIERS)
def test_recurrence_every_distractor_formatted_like_the_answer(difficulty):
    for seed in SEEDS:
        q = _make(RECURRENCE_SOURCES, difficulty, seed)
        for d in q.distractors:
            assert _NUMBER_RE.fullmatch(d), d


def test_recurrence_hard_tier_is_an_inverse_shape_not_a_forward_one():
    """hard asks for a1 given a2/a3 -- the sentence never says "Find a_3"/"Find
    a_4" (the forward tiers' shape); exam similarly always asks for a4 via a
    pair-consistency check, never restating hard's "Find a_1"."""
    for seed in SEEDS:
        hard_q = _make(RECURRENCE_SOURCES, "hard", seed)
        assert re.search(r"a_1[.?]\s*$", hard_q.text), hard_q.text
        assert "a_2 = " in hard_q.text and "a_3 = " in hard_q.text
        assert "a_1 = " not in hard_q.text  # a_1 is solved for, never stated
        exam_q = _make(RECURRENCE_SOURCES, "exam", seed)
        assert "a_4" in exam_q.text
        assert "Pair A" in exam_q.text and "Pair B" in exam_q.text


def test_recurrence_skill_loads_through_api_with_correct_properties():
    assert RECURRENCE_SKILL.domain == "discrete-mathematics"
    assert RECURRENCE_SKILL.status == "active"
    assert RECURRENCE_SKILL.placement_eligible is False
    assert RECURRENCE_SKILL.answer == "number"
    assert RECURRENCE_SKILL.prerequisites == []
    assert {src.difficulty for src in RECURRENCE_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = RECURRENCE_SOURCES["exam"]
    assert exam_src.stages == ["exam"]


# ==========================================================================
# ucv1-second-order-characteristic-roots (answer: pair, labelled)
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_roots_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(ROOTS_SOURCES, difficulty, seed)
        labels = list(q.answer.keys())
        wrong = f"{labels[0]} = {int(q.answer[labels[0]]) + 1000}, {labels[1]} = {int(q.answer[labels[1]]) - 1000}"
        assert not grade_pair(q, wrong).correct, f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"


@pytest.mark.parametrize("difficulty", TIERS)
def test_roots_labelled_order_independence_accepted(difficulty):
    """grade_pair matches by LABEL name, not position
    (mathgen/grading/typed.py's _labelled_pair_verdict) -- swapping the SEGMENT order in
    the learner's typed answer must still be accepted."""
    for seed in SEEDS:
        q = _make(ROOTS_SOURCES, difficulty, seed)
        m = _LABELLED_PAIR_RE.fullmatch(q.answer_text)
        assert m, q.answer_text
        label1, value1, label2, value2 = m.groups()
        swapped = f"{label2} = {value2}, {label1} = {value1}"
        assert grade_pair(q, swapped).correct, f"seed {seed} {difficulty}: label-order-swapped {swapped} not accepted"


@pytest.mark.parametrize("difficulty", TIERS)
def test_roots_answer_is_an_exact_integer_pair(difficulty):
    for seed in SEEDS:
        q = _make(ROOTS_SOURCES, difficulty, seed)
        assert _LABELLED_PAIR_RE.fullmatch(q.answer_text), q.answer_text
        assert isinstance(q.answer, dict)
        for v in q.answer.values():
            assert isinstance(v, int)


@pytest.mark.parametrize("difficulty", TIERS)
def test_roots_every_distractor_formatted_like_the_answer(difficulty):
    for seed in SEEDS:
        q = _make(ROOTS_SOURCES, difficulty, seed)
        for d in q.distractors:
            assert _LABELLED_PAIR_RE.fullmatch(d), d


def test_roots_labels_differ_by_tier_shape():
    """r1/r2 for the real-root tiers, real_part/imaginary_part for hard's
    complex roots, b/c for exam's inverse -- three distinct label vocabularies,
    never a mix."""
    for seed in SEEDS:
        easy_q = _make(ROOTS_SOURCES, "easy", seed)
        assert set(easy_q.answer.keys()) == {"r1", "r2"}
        medium_q = _make(ROOTS_SOURCES, "medium", seed)
        assert set(medium_q.answer.keys()) == {"r1", "r2"}
        assert medium_q.answer["r1"] == medium_q.answer["r2"]
        hard_q = _make(ROOTS_SOURCES, "hard", seed)
        assert set(hard_q.answer.keys()) == {"real_part", "imaginary_part"}
        assert hard_q.answer["imaginary_part"] > 0
        exam_q = _make(ROOTS_SOURCES, "exam", seed)
        assert set(exam_q.answer.keys()) == {"b", "c"}


def test_roots_skill_loads_through_api_with_correct_properties():
    assert ROOTS_SKILL.domain == "ordinary-differential-equations"
    assert ROOTS_SKILL.status == "active"
    assert ROOTS_SKILL.placement_eligible is False
    assert ROOTS_SKILL.answer == "pair"
    assert ROOTS_SKILL.prerequisites == ["solve-ode-separable-poly"]
    assert {src.difficulty for src in ROOTS_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = ROOTS_SOURCES["exam"]
    assert exam_src.stages == ["exam"]
