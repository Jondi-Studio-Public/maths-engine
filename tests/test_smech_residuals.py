"""Tests `least-squares-residuals`: oracles re-derive every answer, and the least-squares line, from the text."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
from smech_support import (
    assert_exam_rules,
    assert_number_distractors_match_places,
    assert_single_answer_rules,
    fixed,
    frac,
    questions,
    skill,
)

from mathgen import make
from mathgen.engine import grade

SK = skill("least-squares-residuals")
SEEDS = range(60)
NUM = r"-?\d+(?:\.\d+)?"


def points(t):
    return [(int(x), int(y)) for x, y in re.findall(r"\((\d+), (\d+)\)", t)]


def line(t):
    m = re.search(rf"predicted [^=]+ = ({NUM}) ([+-]) ({NUM}) ×", t)
    a, b = frac(m.group(1)), frac(m.group(3))
    return a, (b if m.group(2) == "+" else -b)


def least_squares(pts):
    n = len(pts)
    xbar = Fraction(sum(x for x, _ in pts), n)
    ybar = Fraction(sum(y for _, y in pts), n)
    b = sum((x - xbar) * (y - ybar) for x, y in pts) / sum((x - xbar) ** 2 for x, _ in pts)
    return ybar - b * xbar, b


def oracle(q) -> str:
    t = q.text
    if "One observation has" in t:
        a, b = line(t)
        x, y = (int(v) for v in re.search(r"equal to (\d+) and .* equal to (\d+)\.", t).groups())
        return fixed(y - (a + b * x), 1)
    if "residual of greatest size" in t:
        pts = points(t)
        a, b = line(t)
        assert (a, b) == least_squares(pts), "the printed line is not the least-squares line"
        res = [y - (a + b * x) for x, y in pts]
        return fixed(max(res, key=abs), 1)
    if "four of the five observations" in t:
        listed = [frac(v) for v in re.findall(rf"x = \d+: ({NUM})", t)]
        assert len(listed) == 4
        return fixed(-sum(listed), 1)
    if "by eye" in t:
        pts = points(t)
        c, d = line(t)
        return fixed(sum(y - (c + d * x) for x, y in pts), 0)
    raise AssertionError(f"no oracle for {t}")


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_grading_rules(difficulty):
    assert_single_answer_rules(SK, difficulty, SEEDS)
    for q in questions(SK, difficulty, SEEDS):
        assert_number_distractors_match_places(q)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle_agrees_with_the_key(difficulty):
    for q in questions(SK, difficulty, range(200)):
        assert oracle(q) == q.answer_text, (q.text, q.answer_text)


def test_tiers_differ_in_the_decision_they_ask_for():
    easy = " ".join(q.text for q in questions(SK, "easy", range(60)))
    medium = " ".join(q.text for q in questions(SK, "medium", range(60)))
    hard = [q.text for q in questions(SK, "hard", range(80))]
    assert "One observation" in easy and "five observations" not in easy
    assert "greatest size" in medium
    assert any("four of the five" in t for t in hard) and any("by eye" in t for t in hard)


def test_the_printed_least_squares_line_is_the_least_squares_line():
    for q in questions(SK, "medium", range(100)):
        assert line(q.text) == least_squares(points(q.text))


def test_the_by_eye_sum_is_not_zero_and_the_zero_trap_is_offered():
    seen = 0
    for q in questions(SK, "hard", range(120)):
        if "by eye" in q.text:
            assert frac(q.answer_text) != 0
            assert any(frac(d) == 0 for d in q.distractors)
            seen += 1
    assert seen >= 20


def test_sign_matters_in_the_medium_tier():
    signs = {frac(q.answer_text) < 0 for q in questions(SK, "medium", range(80))}
    assert signs == {True, False}


def _exam():
    return [make("smech_residuals_exam", seed=s, difficulty="exam") for s in range(120)]


def exam_oracle(q):
    t = q.text
    pts = points(t)
    a, b = line(t)
    assert (a, b) == least_squares(pts)
    x = int(re.search(r"equal to (\d+),", q.parts[1].prompt + ",").group(1))
    y = dict(pts)[x]
    res = [yy - (a + b * xx) for xx, yy in pts]
    return {
        "a": fixed(a + b * x, 1),
        "b": fixed(y - (a + b * x), 1),
        "c": fixed(max(res, key=abs), 1),
        "d": fixed(sum(r * r for r in res), 2),
    }


def test_exam_chain():
    assert_exam_rules(SK, range(60), labels="abcd", marks=[1, 1, 2, 2])
    for q in _exam():
        assert exam_oracle(q) == {p.label: p.answer_text for p in q.parts}, q.text


def test_exam_follows_a_wrong_prediction_through():
    checked = 0
    for q in _exam()[:80]:
        x = int(re.search(r"equal to (\d+),", q.parts[1].prompt + ",").group(1))
        y = dict(points(q.text))[x]
        wrong = frac(q.parts[0].answer_text) + Fraction(7, 10)
        reply = {
            "a": fixed(wrong, 1),
            "b": fixed(y - wrong, 1),
            "c": q.parts[2].answer_text,
            "d": q.parts[3].answer_text,
        }
        if reply["b"] in q.parts[1].distractors:
            continue
        result = grade(SK, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == q.parts[1].marks and result.parts["b"].follow_through
        assert result.parts["c"].marks == q.parts[2].marks
        checked += 1
    assert checked >= 40


def test_exam_distractors_keep_the_precision_and_text_varies():
    for q in _exam()[:40]:
        for p in q.parts:
            assert_number_distractors_match_places(p)
    assert len({q.text for q in _exam()}) >= 60


def test_deterministic():
    for topic, d in (("smech_residuals", "hard"), ("smech_residuals_exam", "exam")):
        assert make(topic, seed=7, difficulty=d).to_dict() == make(topic, seed=7, difficulty=d).to_dict()
