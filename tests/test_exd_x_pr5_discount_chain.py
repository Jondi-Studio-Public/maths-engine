"""Tests the exam source of `f10-fraction-percent-quantity-strategy`: two successive discounts, the total saving and the matching single percentage.

The key of every part is re-derived from the rendered stem with independent code (exact fractions), and a wrong
earlier answer carried correctly into a later part is checked to earn that part's marks.
"""

from __future__ import annotations

import collections
import re
from fractions import Fraction

import pytest

import mathgen.generators.exd_x_pr5_discount_chain  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_pr5_discount_chain"
SKILL_ID = "f10-fraction-percent-quantity-strategy"
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _frac(text):
    return Fraction(text)


def oracle(q):
    price = int(re.search(r"\$(\d+)", q.stem).group(1))
    p1, p2 = (Fraction(v) for v in re.findall(r"(\d+(?:\.\d+)?)%", q.stem))
    sale = price * (1 - p1 / 100)
    final = sale * (1 - p2 / 100)
    saving = price - final
    one = saving * 100 / price
    return {"a": str(sale), "b": str(final), "c": str(saving), "d": f"{float(one):.1f}"}


def test_shape_and_marks(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 2, 1, 2]
        assert [p.answer_kind for p in q.parts] == ["number"] * 4
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_oracle_agrees_with_the_key_over_300_seeds(questions):
    bad = [(q.text, oracle(q), _canonical(q)) for q in questions if oracle(q) != _canonical(q)]
    assert bad == []


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, q.text


def test_every_distractor_rejected_over_300_seeds(questions):
    for q in questions[:300]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 3, (q.text, p.label, p.distractors)
            for d in p.distractors:
                reply = {**canon, p.label: d}
                result = grade(_SKILL, q, reply)
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_distractors_are_tagged(questions):
    for q in questions[:60]:
        for p in q.parts:
            assert set(p.distractors) <= set(p.option_notes), (q.text, p.label)
            assert all(d in p.distractor_values for d in p.distractors)


def test_follow_through_from_a_wrong_sale_price(questions):
    for q in questions:
        price = int(re.search(r"\$(\d+)", q.stem).group(1))
        p2 = Fraction(re.findall(r"(\d+(?:\.\d+)?)%", q.stem)[1])
        a = Fraction(q.parts[0].distractors[0])
        b = a * (1 - p2 / 100)
        c = price - b
        d = c * 100 / price
        if (
            min(b, c, d) <= 0
            or b.denominator != 1
            or c.denominator != 1
            or (d * 10).denominator != 1
            or len({a, b, c}) < 3
        ):
            continue
        reply = {"a": q.parts[0].distractors[0], "b": str(b), "c": str(c), "d": f"{float(d):.1f}"}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, reply)


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": "7777", "b": "7778", "c": "7779", "d": "7780.0"})
        assert result.marks == 0 and not result.correct
