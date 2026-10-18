"""Tests the four-part exam chain on compound interest with an unknown rate (growth factor, rate, a later balance, total interest)."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_r2d_compound_growth_chain"
SKILL_ID = "compound-interest"
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _hard(seed):
    src = [s for s in _SKILL.sources if s.difficulty != "exam"][-1]  # the hardest practice tier
    return make(src.topic, seed=seed, **src.params)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == LABELS
        assert [p.marks for p in q.parts] == MARKS
        assert [p.follow_through is not None for p in q.parts] == FT_FLAGS
        assert all(p.distractors and all(d in p.option_notes for d in p.distractors) for p in q.parts)


def test_prompt_shapes_beyond_the_hard_tier():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(_hard(s)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard


def test_oracle_agrees_with_the_rendered_prompt(questions):
    bad = [(q.text, oracle(q), _canonical(q)) for q in questions if oracle(q) != _canonical(q)]
    assert bad == [], bad[:2]


def test_full_marks_for_the_key(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(MARKS), q.text


def test_every_distractor_is_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_all_wrong_earns_nothing(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {p.label: f"9999{i}" for i, p in enumerate(q.parts)})
        assert result.marks == 0 and not result.correct


from decimal import ROUND_HALF_UP, Decimal

LABELS = ["a", "b", "c", "d"]
MARKS = [2, 1, 2, 1]
FT_FLAGS = [False, True, True, True]


def _q(x, places):
    return str(Decimal(x).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _facts(q):
    p = Decimal(re.search(r"\$([\d,]+) (?:in|into)", q.stem).group(1).replace(",", ""))
    t = int(re.search(r"(?:after|for) (\d+) years", q.stem).group(1))
    b = Decimal(re.search(r"balance (?:after \d+ years )?is \$([\d,]+\.\d\d)", q.stem).group(1).replace(",", ""))
    n = int(re.search(r"further (\d+) years", q.parts[2].prompt).group(1))
    return p, t, b, n


def oracle(q):
    p, t, b, n = _facts(q)
    factor = Decimal(str(round((float(b) / float(p)) ** (1 / t), 3)))
    rate = (factor - 1) * 100
    later = b * factor**n
    return {
        "a": _q(factor, 3),
        "b": _q(rate, 1),
        "c": _q(later, 2),
        "d": _q(Decimal(_q(later, 2)) - p, 2),
    }


def test_follow_through_from_a_wrong_factor(questions):
    for q in questions[:100]:
        p, t, b, n = _facts(q)
        wrong = Decimal(q.parts[0].distractors[0])
        rate = (wrong - 1) * 100
        later = b * wrong**n
        reply = {"a": str(wrong), "b": _q(rate, 1), "c": _q(later, 2), "d": _q(Decimal(_q(later, 2)) - p, 2)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            if reply[label] == q.parts["abcd".index(label)].answer_text:
                continue
            assert result.parts[label].marks == MARKS["abcd".index(label)], (q.text, label, reply)
            assert result.parts[label].follow_through is True


def test_a_rounded_but_correct_chain_is_full_marks(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, _canonical(q))
        assert result.marks == sum(MARKS)


def test_the_rate_is_a_multiple_of_half_a_percent(questions):
    for q in questions:
        r = Decimal(q.parts[1].answer_text)
        assert r * 2 == int(r * 2) and Decimal("2.5") <= r <= Decimal("7.5")
