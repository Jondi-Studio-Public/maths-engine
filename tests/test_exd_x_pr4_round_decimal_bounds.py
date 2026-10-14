"""Tests the multi-part exam source of `round-decimals`: the smallest and largest true values behind a rounded reading, then a derived smallest value.

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

TOPIC = "exd_x_pr4_round_decimal_bounds"
SKILL_ID = "round-decimals"
SEEDS = range(300)
FOLLOWS = "c"  # the parts that carry a follow_through expression
CARRIED_FROM_A = "c"  # the parts whose follow-through names part (a)

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


def _reading(q):
    stem = q.stem
    reading_text = re.search(r"(\d+(?:\.\d+)?) (?:kg|mm|m|L)\b", stem).group(1)
    places = 0 if "nearest whole number" in stem else 1
    diff = _num(re.findall(r"(\d+\.\d) (?:kg|mm|m|L)\b", stem)[-1])
    return _num(reading_text), places, diff


def _bounds(reading, places):
    finest = Fraction(1, 10 ** (places + 1))
    grid = [finest * m for m in range(int((reading - 1) / finest), int((reading + 1) / finest) + 1)]
    kept = [v for v in grid if floor(v * 10**places + Fraction(1, 2)) == reading * 10**places]
    return min(kept), max(kept)


def oracle(q):
    reading, places, diff = _reading(q)
    low, high = _bounds(reading, places)
    return {"a": _fmt(low, places + 1), "b": _fmt(high, places + 1), "c": _fmt(low - diff, places + 1)}


def wrong_first(q):
    reading, places, diff = _reading(q)
    low, high = _bounds(reading, places)
    wrong = reading  # the rounded value itself, taken as the smallest
    dp = places + 1
    return {"a": _fmt(wrong, dp), "b": _fmt(high, dp), "c": _fmt(wrong - diff, dp)}


def wrong_second(q):
    reading, places, diff = _reading(q)
    low, high = _bounds(reading, places)
    dp = places + 1
    return {"a": _fmt(low, dp), "b": _fmt(high + 10 * Fraction(1, 10**dp), dp), "c": _fmt(low - diff, dp)}


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
