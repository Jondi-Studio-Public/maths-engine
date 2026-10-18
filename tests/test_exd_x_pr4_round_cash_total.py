"""Tests the multi-part exam source of `f10-round-decimals-accuracy`: an exact bill, the cash total rounded to the nearest 5 cents, and the change.

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

TOPIC = "exd_x_pr4_round_cash_total"
SKILL_ID = "f10-round-decimals-accuracy"
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


def _bill(q):
    m = re.search(r"buys (\d+) .+? at \$(\d+\.\d\d) each and (\d+) .+? at \$(\d+\.\d\d) each", q.stem)
    q1, p1, q2, p2 = int(m.group(1)), _num(m.group(2)), int(m.group(3)), _num(m.group(4))
    note = int(re.search(r"\$(\d+) note", q.stem).group(1))
    return q1 * p1 + q2 * p2, note


def _five(total):
    return floor(total * 20 + Fraction(1, 2)) * Fraction(1, 20)


def oracle(q):
    total, note = _bill(q)
    cash = _five(total)
    return {"a": _fmt(total, 2), "b": _fmt(cash, 2), "c": _fmt(note - cash, 2)}


def wrong_first(q):
    total, note = _bill(q)
    wrong = total - Fraction(3, 10)
    cash = _five(wrong)
    return {"a": _fmt(wrong, 2), "b": _fmt(cash, 2), "c": _fmt(note - cash, 2)}


def wrong_second(q):
    total, note = _bill(q)
    wrong = _five(total) + Fraction(1, 10)
    return {"a": _fmt(total, 2), "b": _fmt(wrong, 2), "c": _fmt(note - wrong, 2)}


def test_shape_and_marks(questions):
    for q in questions[:100]:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert all(p.answer_kind == "number" and p.marks >= 1 for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [label in FOLLOWS for label in "abc"]
        assert all(len(p.distractors) >= 2 for p in q.parts)


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
