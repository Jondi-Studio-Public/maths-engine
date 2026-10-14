"""Tests for the exam source of senior-use-relative-frequencies-obtained-from-data-as-point.

The exam tier is a four-part chain on a frequency table with one count lost: the missing count, the relative-frequency
estimate of a two-category event, its projection onto future trials, and the chance of at least one in two trials.
Every key is re-derived from the rendered prompt with exact fractions.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import round_half_up
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2l_relfreq_chain"
SKILL_ID = "senior-use-relative-frequencies-obtained-from-data-as-point"
SEEDS = range(320)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _show(v: Fraction) -> str:
    return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)


def _facts(q):
    total = int(re.match(r"(\d+) ", q.text).group(1))
    body = q.text.split("The counts were ")[1].split(". (a)")[0]
    counts = {}
    for item in body.split(", "):
        name, value = item.rsplit(": ", 1)
        counts[name] = None if value == "not shown" else int(value)
    event = re.search(r"recorded as (.+?) or (.+?)\. Give", q.parts[1].prompt).groups()
    horizon = int(re.search(r"next (\d+) ", q.parts[2].prompt).group(1))
    return total, counts, event, horizon


def oracle(q) -> dict[str, str]:
    total, counts, event, horizon = _facts(q)
    missing = total - sum(v for v in counts.values() if v is not None)
    full = {k: (missing if v is None else v) for k, v in counts.items()}
    p = Fraction(sum(full[name] for name in event), total)
    both = Fraction(0)
    for first in (True, False):
        for second in (True, False):
            if first or second:
                both += (p if first else 1 - p) * (p if second else 1 - p)
    return {"a": str(missing), "b": _show(p), "c": str(int(round_half_up(horizon * p))), "d": _show(both)}


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert [p.answer_kind for p in q.parts] == ["number", "fraction", "number", "fraction"]


def test_oracle_agrees_over_300_seeds(questions):
    bad = [(q.text, oracle(q), _canonical(q)) for q in questions if oracle(q) != _canonical(q)]
    assert bad == []


def test_event_always_includes_the_missing_category(questions):
    for q in questions:
        _, counts, event, _ = _facts(q)
        missing_name = next(k for k, v in counts.items() if v is None)
        assert missing_name in event


def test_composed_answer_full_marks(questions):
    for q in questions[:120]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6


def test_every_distractor_rejected_and_tagged(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) == 3 and len(set(p.distractors)) == 3
            assert all(p.option_notes.get(d) in CATALOGUE for d in p.distractors), (p.label, p.option_notes)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_missing_count(questions):
    checked = 0
    for q in questions[:150]:
        total, counts, event, horizon = _facts(q)
        missing_name = next(k for k, v in counts.items() if v is None)
        other = next(name for name in event if name != missing_name)
        wrong_a = int(q.parts[0].answer_text) + 1
        p = Fraction(wrong_a + counts[other], total)
        projected = horizon * p
        if projected % 1 == Fraction(1, 2) or not 0 < p < 1:
            continue
        reply = {
            "a": str(wrong_a),
            "b": _show(p),
            "c": str(int(round_half_up(projected))),
            "d": _show(1 - (1 - p) ** 2),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, reply)
            if reply[label] != q.parts["abcd".index(label)].answer_text:  # a rounded projection can coincide
                assert result.parts[label].follow_through is True
        assert result.marks == 5 and not result.correct
        checked += 1
    assert checked >= 140


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "999", "b": "1/1000", "c": "5000", "d": "3/1000"})
        assert result.marks == 0
