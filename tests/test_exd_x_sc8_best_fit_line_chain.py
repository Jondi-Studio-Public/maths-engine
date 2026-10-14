"""Exam source of `senior-find-the-line-of-best-fit-by-eye`: gradient, intercept, prediction and residual."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc8_best_fit_line_chain"
SKILL_ID = "senior-find-the-line-of-best-fit-by-eye"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_NUM = r"(-?\d+(?:\.\d+)?)"


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _fmt(value: Fraction) -> str:
    """A value to 1 decimal place, rounded half up (the stated precision of every part)."""
    scaled = Fraction(value) * 10
    rounded = int(scaled + Fraction(1, 2)) if scaled >= 0 else -int(-scaled + Fraction(1, 2))
    return f"{rounded / 10:.1f}"


def _points(text):
    return [(Fraction(x), Fraction(y)) for x, y in re.findall(rf"\({_NUM}, {_NUM}\)", text)]


def oracle(q):
    """Re-derive every part's answer from the stem and part prompts alone."""
    pts = _points(q.text)
    data, (ax, ay), (bx, by) = pts[:6], pts[6], pts[7]
    m = (by - ay) / (bx - ax)
    c = ay - m * ax
    x0 = Fraction(re.search(r"predict y when x = (-?\d+)", q.parts[2].prompt).group(1))
    px = Fraction(re.search(r"data point with x = (-?\d+)", q.parts[3].prompt).group(1))
    py = dict(data)[px]
    return {"a": _fmt(m), "b": _fmt(c), "c": _fmt(m * x0 + c), "d": _fmt(py - (m * px + c))}, (m, c, x0, px, py)


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(120)]


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 1, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        expected, _ = oracle(q)
        assert expected == {p.label: p.answer_text for p in q.parts}, (seed, q.text)


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6, (q.text, q.answer_text)


def test_distractors_rejected_and_tagged(questions):
    for q in questions:
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            assert all(d in p.distractor_values for d in p.distractors)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_gradient(questions):
    for q in questions:
        _, (_, _, x0, px, py) = oracle(q)
        ax, ay = _points(q.text)[6]
        wrong_m = Fraction(q.parts[0].distractors[0])
        wrong_c = ay - wrong_m * ax
        reply = {
            "a": q.parts[0].distractors[0],
            "b": _fmt(wrong_c),
            "c": _fmt(wrong_m * x0 + wrong_c),
            "d": _fmt(py - (wrong_m * px + wrong_c)),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            part = result.parts[label]
            coincides = reply[label] == q.parts["abcd".index(label)].answer_text  # a wrong chain can land on the key
            assert part.marks == part.max_marks and (part.follow_through or coincides), (q.text, label, reply)
        assert result.marks == 4 and not result.correct


def test_follow_through_from_wrong_intercept_only(questions):
    for q in questions[:60]:
        expected, (m, c, x0, px, py) = oracle(q)
        wrong_c = Fraction(q.parts[1].distractors[0])
        reply = {
            "a": expected["a"],
            "b": q.parts[1].distractors[0],
            "c": _fmt(m * x0 + wrong_c),
            "d": _fmt(py - (m * px + wrong_c)),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 2 and result.parts["b"].marks == 0
        assert result.parts["c"].marks == 1 and result.parts["d"].marks == 2


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        assert grade(_SKILL, q, {"a": "999", "b": "777", "c": "888", "d": "555"}).marks == 0
