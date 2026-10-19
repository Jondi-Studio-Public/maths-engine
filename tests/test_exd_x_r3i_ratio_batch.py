"""Exam source of the ratio skill: decimal test amounts to a simplest ratio, a share and a full batch."""

from __future__ import annotations

import re
from fractions import Fraction

SKILL_ID = "senior-demonstrate-an-understanding-of-the-elementary-ideas-and"
TOPIC = "exd_x_r3i_ratio_batch"
LABELS = ["a", "b", "c", "d"]

from collections import Counter

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
SEEDS = range(300)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == LABELS
        assert 3 <= len(q.parts) <= 4
        assert all(p.distractors and len(p.distractors) >= 2 for p in q.parts)
        assert sum(1 for p in q.parts if p.follow_through) >= 1


def test_no_text_defects(questions):
    for q in questions[:100]:
        assert find_text_defects(q, include_working=True) == [], q.text


def test_key_rederived_from_prompt(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, q.text


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: str(d)})
                assert result.parts[p.label].marks == 0, (q.text, p.label, str(d))


def test_distractors_match_key_layout(questions):
    for q in questions:
        for p in q.parts:
            if p.answer_kind != "number":
                continue
            for d in p.distractors:
                assert ("." in d) == ("." in p.answer_text), (p.label, d, p.answer_text)
                if "." in d:
                    assert len(d.split(".")[1]) == len(p.answer_text.split(".")[1]), (p.label, d, p.answer_text)


def test_distractors_are_tagged(questions):
    from mathgen.misconceptions import CATALOGUE

    for q in questions[:100]:
        for p in q.parts:
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (q.text, p.label, d)


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        reply = {p.label: ("999999.5" if p.answer_kind == "number" else "999999/7") for p in q.parts}
        result = grade(_SKILL, q, reply)
        assert result.marks == 0 and not result.correct


def oracle(q):
    x, y = (Fraction(v) for v in re.findall(r"(\d+(?:\.\d)?) (?:litres?|kilograms)", q.stem)[:2])
    total = int(re.search(r"batch of (\d+)", q.parts[2].prompt).group(1))
    ratio = x / y
    share = ratio / (1 + ratio)
    first = share * total
    return {
        "a": f"{ratio.numerator}:{ratio.denominator}",
        "b": f"{share.numerator}/{share.denominator}" if share.denominator != 1 else str(share.numerator),
        "c": str(int(first)),
        "d": str(int(total - first)),
    }


def test_follow_through_from_wrong_ratio(questions):
    checked = 0
    for q in questions[:200]:
        total = int(re.search(r"batch of (\d+)", q.parts[2].prompt).group(1))
        wa = q.parts[0].distractors[0]
        num, den = (int(v) for v in wa.split(":"))
        share = Fraction(num, num + den)
        first = share * total
        if first.denominator != 1:
            continue
        reply = {
            "a": wa,
            "b": f"{share.numerator}/{share.denominator}",
            "c": str(int(first)),
            "d": str(int(total - first)),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks, (q.text, reply)
        assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, reply)
        assert result.parts["d"].marks == result.parts["d"].max_marks, (q.text, reply)
        checked += 1
    assert checked > 10
