"""`f10-transformations-invariants`: which properties of a shape stay the same under a named translation,
reflection or rotation (or a short sequence of them). Answers are re-derived here from the question text alone."""

import re
from collections import Counter
from pathlib import Path

import pytest

from mathgen import make
from mathgen.core import mcq_options
from mathgen.grading.typed import grade_mcq
from mathgen.skills import load_skills

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]
SKILL_ID = "f10-transformations-invariants"


def _questions(tier):
    return [make("transformations_invariants", seed=s, difficulty=tier) for s in SEEDS]


def _index(question, option):
    return mcq_options(question).index(option)


def _skill():
    _, skills = load_skills(ROOT / "skills")
    return next(s for s in skills if s.id == SKILL_ID)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_options_well_formed(tier):
    for seed in SEEDS:
        q = make("transformations_invariants", seed=seed, difficulty=tier)
        assert q.to_json() == make("transformations_invariants", seed=seed, difficulty=tier).to_json()
        assert len(q.distractors) == 3
        assert len(set(q.distractors) | {q.answer_text}) == 4
        assert set(q.option_notes) == set(q.distractors)


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_answer_and_rejects_every_distractor(tier):
    for q in _questions(tier):
        assert grade_mcq(q, _index(q, q.answer_text)).correct
        for d in q.distractors:
            assert not grade_mcq(q, _index(q, d)).correct
        assert not grade_mcq(q, "9").correct
        assert not grade_mcq(q, "").correct


@pytest.mark.parametrize("tier", TIERS)
def test_every_option_note_is_a_declared_misconception(tier):
    declared = {m.id for m in _skill().misconceptions}
    for q in _questions(tier):
        assert set(q.option_notes.values()) <= declared


def test_no_exam_tier_and_three_sources():
    sk = _skill()
    assert [s.difficulty for s in sk.sources] == TIERS
    assert sk.answer == "mcq" and sk.status == "active"


def test_easy_answer_follows_the_named_transformation():
    seen = Counter()
    for q in _questions("easy"):
        reflected = "reflected" in q.text
        seen[reflected] += 1
        assert "Side lengths: the same. Angles: the same." in q.answer_text
        assert ("The image is a mirror image" in q.answer_text) == reflected
    assert all(seen.values())


def test_medium_answer_repeats_the_stated_length_and_angle_and_reverses_only_for_reflection():
    for q in _questions("medium"):
        side = int(re.search(r"has [A-E]{2} = (\d+) cm", q.text).group(1))
        angle = int(re.search(r"angle [A-E]{3} = (\d+)°", q.text).group(1))
        assert f"= {side} cm" in q.answer_text and f"= {angle}°" in q.answer_text
        assert ("is a mirror image" in q.answer_text) == ("reflected" in q.text)
        for d in q.distractors:
            assert d != q.answer_text


def test_hard_reversal_is_the_parity_of_reflections():
    flips = Counter()
    for q in _questions("hard"):
        n_reflections = q.text.count("reflected")
        reversed_ = n_reflections % 2 == 1
        assert ("and it is a mirror image" in q.answer_text) == reversed_
        assert "are the same as" in q.answer_text
        flips[n_reflections] += 1
    assert set(flips) >= {0, 1, 2}


def test_hard_symmetry_matches_the_named_shape():
    table = {
        "scalene triangle": (0, 1),
        "isosceles triangle that is not equilateral": (1, 1),
        "equilateral triangle": (3, 3),
        "rectangle that is not a square": (2, 2),
        "square": (4, 4),
        "regular pentagon": (5, 5),
        "regular hexagon": (6, 6),
        "regular octagon": (8, 8),
    }
    for q in _questions("hard"):
        name = next(n for n in sorted(table, key=len, reverse=True) if re.match(rf"An? {n} is ", q.text))
        lines, order = table[name]
        m_lines = re.search(r"it has (no|\d+) lines? of symmetry", q.answer_text)
        m_order = re.search(r"rotational symmetry of order (\d+)|it has (no) rotational symmetry", q.answer_text)
        if m_lines:
            assert (0 if m_lines.group(1) == "no" else int(m_lines.group(1))) == lines
        else:
            assert (1 if m_order.group(2) else int(m_order.group(1))) == order


@pytest.mark.parametrize("tier", TIERS)
def test_answer_pool_is_varied_and_not_positional(tier):
    qs = _questions(tier)
    answers = Counter(q.answer_text for q in qs)
    assert len(answers) >= 12
    assert max(answers.values()) / len(qs) <= 0.2
    positions = Counter(mcq_options(q).index(q.answer_text) for q in qs)
    assert len(positions) == 4 and max(positions.values()) / len(qs) <= 0.45
    assert len({q.text for q in qs}) >= 40


@pytest.mark.parametrize("tier", TIERS)
def test_some_distractor_shares_the_answers_reversal_clause(tier):
    """The correct option is never the only one with its reversal clause (the flipped components are drawn at
    random, never all three on one component)."""
    clause = lambda t: re.search(r"(is not|is) a mirror image", t).group(1)
    for q in _questions(tier):
        marks = [clause(d) == clause(q.answer_text) for d in q.distractors]
        assert any(marks)
