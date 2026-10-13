"""Tests the exam chain for solving a linear equation graphically: the crossing of two plans, in a real situation.

The oracle re-derives every part from the drawn figure (the two lines, read in real units) and the numbers the prompts
state (the two rules and the length of the stay), never from the key.
"""

from __future__ import annotations

import re
from fractions import Fraction as F

import pytest

from mathgen import figures, make
from mathgen.generators.pv1a_line_figure_build import read_intersection, read_line
from tests.exd_x_r2c_support import (
    canonical,
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through,
    check_grading,
    check_shapes,
    check_wired,
    exam,
)

TOPIC = "exd_x_w12c_solve_graphically"
SKILL_ID = "solve-linear-equations-graphically"
HARD_TOPIC = "fg1_solve_graphically"
HARD_KWARGS = {"tier": "hard"}


def oracle(q):
    """The crossing time, the cost there, the greatest whole number of periods the early-cheaper plan wins, the saving."""
    cross_x, cross_y = read_intersection(q.figure, 0, 1)
    (m_a, c_a), (m_b, c_b) = read_line(q.figure, 0), read_line(q.figure, 1)
    rules = re.findall(r"C = (\d+) \+ (\d+)t", q.parts[1].prompt)
    assert [(F(f), F(r)) for f, r in rules] == [(c_a, m_a), (c_b, m_b)], "rules do not match the drawn lines"
    stay = int(re.search(r"for (\d+) (?:weeks|hours|months)\.", q.parts[3].prompt).group(1))
    saving = abs((c_a + m_a * stay) - (c_b + m_b * stay))
    return cross_x, c_a + m_a * cross_x, cross_x - 1, saving, cross_y, m_a, m_b, stay


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_source_wired():
    check_wired(SKILL_ID, TOPIC)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    check_shapes(TOPIC, HARD_TOPIC, HARD_KWARGS)


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number"] * 4
        assert [p.marks for p in q.parts] == [1, 2, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.sensible_range is not None for p in q.parts)


def test_figure_is_valid_two_lines_and_hides_the_answers(questions):
    for q in questions[:100]:
        assert q.figure is not None and figures.validate(q.figure) == []
        assert [ln["label"] for ln in q.figure["lines"]] == ["A", "B"]
        figures.assert_no_leak(q.figure, [p.answer_text for p in q.parts])


def test_contexts_are_australian_and_vary(questions):
    units = {p for q in questions for p in ("weeks", "hours", "months") if f"in {p}" in q.parts[1].prompt}
    assert units == {"weeks", "hours", "months"}


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        cross_x, cost, greatest, saving, cross_y, m_a, m_b, stay = oracle(q)
        a_t, b_t, c_t, d_t = (p.answer_text for p in q.parts)
        ok = (
            cross_x.denominator == 1
            and F(a_t) == cross_x
            and F(b_t) == cost == cross_y
            and F(c_t) == greatest
            and F(d_t) == saving
            and stay > cross_x
            and m_a != m_b
            and saving > 0
        )
        if not ok:
            bad.append((q.stem, canonical(q)))
    assert bad == [], bad[:2]


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_answer_space():
    check_answer_space(TOPIC, distinct=100, share=0.05)


def test_follow_through_from_a_misread_crossing():
    """A crossing read late carries through the cost, the 'cheaper for' count and the saving."""

    def carried(q):
        cross_x, cost, greatest, saving, cross_y, m_a, m_b, stay = oracle(q)
        (m_a, c_a), (m_b, c_b) = read_line(q.figure, 0), read_line(q.figure, 1)
        wrong = next(w for w in (cross_x + 2, cross_x + 1, cross_x + 3) if 3 <= w < stay)
        later = {
            "b": str(c_a + m_a * wrong),
            "c": str(wrong - 1),
            "d": str(abs(m_a - m_b) * (stay - wrong)),
        }
        return "a", str(int(wrong)), later

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nonsense")


def test_working_reads_as_plain_english(questions):
    """The worked steps never say "1 months", "a hour", "1, 2, ... 1 months", or write the letter x for times."""
    for q in questions:
        text = " ".join(step for part in q.parts for step in part.working)
        assert not re.search(r"\b1 (?:weeks|hours|months)\b", text), text
        assert "1, 2, ... 1 " not in text, text
        assert "dollars a hour" not in text and "a hour" not in text, text
        assert not re.search(r"\d x \d", text), text
