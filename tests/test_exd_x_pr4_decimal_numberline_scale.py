"""Tests the multi-part exam source of `f10-decimal-order-numberline`: read a position on a number line with an unfamiliar scale, move on, and find the distance to a labelled mark.

Every key is re-derived from the rendered stem with independent code, then the real grader is run on the
canonical reply, on each distractor, and on replies that carry a wrong earlier answer forward.
"""

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction
from math import floor

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_pr4_decimal_numberline_scale"
SKILL_ID = "f10-decimal-order-numberline"
SEEDS = range(300)
FOLLOWS = "bc"  # the parts that carry a follow_through expression
CARRIED_FROM_A = "bc"  # the parts whose follow-through names part (a)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _num(text):
    return Fraction(Decimal(text.replace(",", "")))


def _fmt(value, dp):
    """`value` (a Fraction) at exactly `dp` decimal places, ties away from zero."""
    scaled = Fraction(value) * 10**dp
    n = floor(abs(scaled) + Fraction(1, 2)) * (1 if scaled >= 0 else -1)
    sign = "-" if n < 0 else ""
    digits = str(abs(n)).zfill(dp + 1)
    return sign + (digits[:-dp] + "." + digits[-dp:] if dp else digits)


def _scale(q):
    n = int(re.search(r"divided into (\d+) equal", q.stem).group(1))
    k, label = re.search(r"is (\d+) sections? past the ([\d.]+) km post", q.stem).groups()
    j = int(re.search(r"is (\d+) sections past P", q.stem).group(1))
    return n, int(k), Fraction(label), j


def _dp(n):
    return 3 if n == 4 else 2


def oracle(q):
    n, k, label, j = _scale(q)
    step = Fraction(1, 10 * n)
    p = label + k * step
    qv = p + j * step
    return {"a": _fmt(p, _dp(n)), "b": _fmt(qv, _dp(n)), "c": _fmt(label + Fraction(2, 10) - qv, _dp(n))}


def wrong_first(q):
    n, k, label, j = _scale(q)
    step = Fraction(1, 10 * n)
    p = label + (k - 1) * step  # one section short
    qv = p + j * step
    return {"a": _fmt(p, _dp(n)), "b": _fmt(qv, _dp(n)), "c": _fmt(label + Fraction(2, 10) - qv, _dp(n))}


def wrong_second(q):
    n, k, label, j = _scale(q)
    step = Fraction(1, 10 * n)
    p = label + k * step
    wrong_q = p + (j + 1) * step
    return {"a": _fmt(p, _dp(n)), "b": _fmt(wrong_q, _dp(n)), "c": _fmt(label + Fraction(2, 10) - wrong_q, _dp(n))}


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"] and exam[0].params == {"tier": "exam"}
    assert _SKILL.sources[-1] is exam[0]


def test_shape_and_marks(questions):
    for q in questions[:100]:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert all(p.answer_kind == "number" and p.marks >= 1 for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [label in FOLLOWS for label in "abc"]
        assert all(len(p.distractors) >= 2 for p in q.parts)


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_prompt_variety_and_new_shapes_over_hard_tier():
    hard_src = next(s for s in _SKILL.sources if s.difficulty == "hard")
    exam = {prompt_shape(_exam(s)) for s in range(80)}
    hard = {
        prompt_shape(
            make(
                hard_src.topic,
                seed=s,
                **{k: (tuple(v) if isinstance(v, list) else v) for k, v in hard_src.params.items()},
            )
        )
        for s in range(80)
    }
    assert len(exam) >= 3, exam
    assert exam - hard, "the exam tier adds no prompt shape beyond the hard tier"


def test_deterministic():
    for seed in (0, 1, 7, 99):
        assert _exam(seed).to_json() == _exam(seed).to_json()


def test_answer_space(questions):
    answers = [q.answer_text for q in questions]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_key_rederived_from_prompt(questions):
    bad = []
    for q in questions:
        if oracle(q) != _canonical(q):
            bad.append((q.text, oracle(q), _canonical(q)))
    assert bad == []


def test_composed_answer_full_marks(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, q.text
        result = grade(_SKILL, q, _canonical(q))
        assert result.correct, q.text


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d, canon)


def test_follow_through_from_wrong_first_part(questions):
    for q in questions[:150]:
        replies = wrong_first(q)
        result = grade(_SKILL, q, replies)
        assert result.parts["a"].marks == 0, (q.text, replies)
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, replies)
            assert bool(result.parts[label].follow_through) == (label in FOLLOWS and label in CARRIED_FROM_A)
        assert not result.correct


def test_wrong_value_not_carried_when_unrelated(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, {"a": "0", "b": "0", "c": "0"})
        assert result.marks == 0 and not result.correct


def test_chain_is_connected(questions):
    """A wrong (b) with a right (a) must not earn (b), and (c) must still follow from the learner's own (b)."""
    for q in questions[:100]:
        replies = wrong_second(q)
        result = grade(_SKILL, q, replies)
        assert result.parts["a"].marks == result.parts["a"].max_marks
        assert result.parts["b"].marks == 0, (q.text, replies)
        assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, replies)
