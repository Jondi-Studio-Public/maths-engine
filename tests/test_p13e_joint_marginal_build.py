"""Focused contract tests for: ucv1-joint-distribution-marginal (answer: fraction)."""

import pytest

from mathgen.grading.typed import grade_fraction
from mathgen.skills import load_skills
from mathgen import make

SKILL_ID = "ucv1-joint-distribution-marginal"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return skill, {src.difficulty: src for src in skill.sources}


SKILL, SOURCES = _skill_sources()


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        wrong = f"{q.answer['p'] + q.answer['q']}/{q.answer['q']}"
        assert not grade_fraction(q, wrong).correct, f"seed {seed} {difficulty}: deliberate wrong {wrong} not rejected"


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_a_fraction_in_lowest_terms(difficulty):
    from fractions import Fraction

    for seed in SEEDS:
        q = _make(difficulty, seed)
        p, qd = q.answer["p"], q.answer["q"]
        frac = Fraction(p, qd)
        assert frac.numerator == p and frac.denominator == qd, (
            f"seed {seed} {difficulty}: answer {p}/{qd} is not already reduced"
        )


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_option_is_a_proper_fraction_in_zero_one(difficulty):
    # A "probability" outside the open interval (0, 1) would be trivially
    # eliminable on sight; the exam tier's wrong-missing-cell misconception
    # is guarded to only ever land strictly between 0 and 1.
    from fractions import Fraction

    for seed in SWEEP_SEEDS:
        q = _make(difficulty, seed)
        assert 0 < Fraction(q.answer_text) < 1
        for d in q.distractors:
            assert 0 < Fraction(d) < 1, f"seed {seed} {difficulty}: distractor {d} outside (0,1)"


def test_easy_asks_for_the_x1_marginal():
    for seed in SEEDS:
        q = _make("easy", seed)
        assert "P(X = x1)" in q.text
        assert "|" not in q.text


def test_medium_asks_conditional_on_x1_row():
    for seed in SEEDS:
        q = _make("medium", seed)
        assert "P(Y = y2 | X = x1)" in q.text


def test_hard_asks_conditional_on_y3_column():
    for seed in SEEDS:
        q = _make("hard", seed)
        assert "P(X = x2 | Y = y3)" in q.text


def test_exam_hides_exactly_one_cell_and_states_the_total():
    import re

    cell_re = re.compile(r"P\(X=x\d,Y=y\d\)=\d+/\d+")
    for seed in SEEDS:
        q = _make("exam", seed)
        cells = cell_re.findall(q.text)
        assert len(cells) == 5, f"seed {seed}: exam table should show 5 of 6 cells, got {len(cells)}"
        assert "sum to 1" in q.text


def test_skill_loads_through_api_with_correct_properties():
    assert SKILL.domain == "probability"
    assert SKILL.status == "active"
    assert SKILL.placement_eligible is False
    assert SKILL.answer == "fraction"
    assert SKILL.prerequisites == ["conditional-probability-counts"]
    assert {src.difficulty for src in SKILL.sources} == {"easy", "medium", "hard", "exam"}
    exam_src = SOURCES["exam"]
    assert exam_src.stages == ["exam"]
