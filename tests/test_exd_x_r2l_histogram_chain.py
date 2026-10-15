"""Tests for the exam source of senior-use-relative-frequencies-and-histograms-obtained-from-data.

The exam tier is a three-part chain on a histogram with one frequency missing: the missing frequency, an interval
probability with both cut values inside classes (uniform spread), and a conditional probability. Every key is
re-derived from the rendered prompt with exact fractions.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2l_histogram_chain"
SKILL_ID = "senior-use-relative-frequencies-and-histograms-obtained-from-data"
SEEDS = range(320)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _show(v: Fraction) -> str:
    return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)


def _table(q):
    n = int(re.search(r"(\d+) (?:patients|eggs|swimmers|torch batteries|javelin throws)", q.text).group(1))
    rows = re.findall(r"\[(\d+), (\d+)\): (not shown|\d+)", q.text)
    return n, [(int(a), int(b), None if f == "not shown" else int(f)) for a, b, f in rows]


def _mass(rows, freqs, lo, hi) -> Fraction:
    """Frequency-weighted overlap of [lo, hi) with each class, as a count (uniform within a class)."""
    total = Fraction(0)
    for (a, b, _), f in zip(rows, freqs):
        overlap = max(0, min(b, hi) - max(a, lo))
        total += Fraction(f * overlap, b - a)
    return total


def oracle(q) -> dict[str, str]:
    n, rows = _table(q)
    missing = n - sum(f for _, _, f in rows if f is not None)
    freqs = [missing if f is None else f for _, _, f in rows]
    lo, hi = (int(v) for v in re.search(r"at least (\d+) \w+ and less than (\d+)", q.parts[1].prompt).groups())
    joint = _mass(rows, freqs, lo, hi) / n
    tail = _mass(rows, freqs, lo, rows[-1][1]) / n
    return {"a": str(missing), "b": _show(joint), "c": _show(joint / tail)}


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [1, 3, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert [p.answer_kind for p in q.parts] == ["number", "fraction", "fraction"]


def test_oracle_agrees_over_300_seeds(questions):
    bad = [(q.text, oracle(q), _canonical(q)) for q in questions if oracle(q) != _canonical(q)]
    assert bad == []


def test_both_cut_values_fall_inside_classes(questions):
    for q in questions:
        _, rows = _table(q)
        lo, hi = (int(v) for v in re.search(r"at least (\d+) \w+ and less than (\d+)", q.parts[1].prompt).groups())
        edges = {a for a, _, _ in rows} | {rows[-1][1]}
        assert lo not in edges and hi not in edges


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


def test_follow_through_from_wrong_missing_frequency(questions):
    checked = 0
    for q in questions[:150]:
        n, rows = _table(q)
        wrong_a = int(q.parts[0].answer_text) + 2
        freqs = [wrong_a if f is None else f for _, _, f in rows]
        lo, hi = (int(v) for v in re.search(r"at least (\d+) \w+ and less than (\d+)", q.parts[1].prompt).groups())
        b = _mass(rows, freqs, lo, hi) / n
        tail = _mass(rows, freqs, lo, rows[-1][1]) / n
        if not 0 < b < 1 or tail == 1:  # a reply that repeats an earlier one earns no follow-through
            continue
        result = grade(_SKILL, q, {"a": str(wrong_a), "b": _show(b), "c": _show(b / tail)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 3 and result.parts["b"].follow_through is True
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True
        assert result.marks == 5 and not result.correct
        checked += 1
    assert checked >= 140


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "999", "b": "1/1000", "c": "3/1000"})
        assert result.marks == 0
