"""Exam source of `compare-unit-prices`: two packs in different units, their gap, and a bulk saving."""

from __future__ import annotations

import re
from fractions import Fraction

SKILL_ID = "compare-unit-prices"
TOPIC = "exd_x_r3i_unit_price_saving"
LABELS = ["a", "b", "c", "d"]
_TO_HUNDREDS = {"g": Fraction(1, 100), "kg": Fraction(10), "mL": Fraction(1, 100), "L": Fraction(10)}

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


def _packs(text):
    found = re.findall(r"Pack ([AB]): (\d+(?:\.\d)?) (g|kg|mL|L) for \$(\d+\.\d\d)", text)
    return {lab: (Fraction(qty) * _TO_HUNDREDS[unit], Fraction(price)) for lab, qty, unit, price in found}


def oracle(q):
    packs = _packs(q.text)
    u = {lab: price / hundreds for lab, (hundreds, price) in packs.items()}
    order = int(re.search(r"needs (\d+) (?:kg|L)", q.text).group(1))
    gap = abs(u["A"] - u["B"])
    return {
        "a": f"{float(u['A']):.2f}",
        "b": f"{float(u['B']):.2f}",
        "c": f"{float(gap):.2f}",
        "d": f"{float(gap * order * 10):.2f}",
    }


def test_follow_through_from_wrong_unit_price(questions):
    checked = 0
    for q in questions[:150]:
        order = int(re.search(r"needs (\d+) (?:kg|L)", q.text).group(1))
        wa = Fraction(q.parts[0].distractors[0])
        b = Fraction(q.parts[1].answer_text)
        gap = abs(wa - b)
        if gap == 0:
            continue
        reply = {
            **_canonical(q),
            "a": q.parts[0].distractors[0],
            "c": f"{float(gap):.2f}",
            "d": f"{float(gap * order * 10):.2f}",
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, reply)
        assert result.parts["d"].marks == result.parts["d"].max_marks, (q.text, reply)
        checked += 1
    assert checked > 100
