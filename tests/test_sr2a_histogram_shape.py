"""Tests for `histogram-shape-and-spread` (ACMGM029, ACMEM060): modal class, spread and middle of a drawn histogram."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from mathgen import figures, make
from tests.sr2a_support import (
    TIERS,
    check_figure,
    check_follow_through,
    check_question_grading,
    check_tagged,
    questions,
    top_share,
)

TOPIC = "sr2a_histogram_shape"
CHAIN = "sr2a_histogram_chain"


def _panel(q):
    p = q.figure["panels"][0]
    return p["edges"], p["frequencies"]


def _median_class(f):
    n = sum(f)
    cum = 0
    for i, v in enumerate(f):
        if cum + v >= n / 2 and v:
            return i
        cum += v


@pytest.mark.parametrize("tier", TIERS)
def test_answers_grade_correct_and_distractors_wrong(tier):
    for q in questions(TOPIC, tier):
        check_question_grading(q)
        check_tagged(q)
        check_figure(q)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_varied(tier):
    assert make(TOPIC, seed=3, difficulty=tier).to_json() == make(TOPIC, seed=3, difficulty=tier).to_json()
    assert len({repr(make(TOPIC, seed=s, difficulty=tier).figure) for s in range(1, 41)}) >= 38
    assert top_share(TOPIC, tier) <= 0.3


@pytest.mark.parametrize("tier", TIERS)
def test_the_figure_has_one_clear_modal_class_and_no_stated_frequencies(tier):
    for q in questions(TOPIC, tier, range(1, 41)):
        _edges, f = _panel(q)
        assert f.count(max(f)) == 1
        assert not re.search(r"\b[Ff]requenc(y|ies) (of|is|are) \d", q.text)


def test_easy_reads_the_modal_class():
    kinds = set()
    for q in questions(TOPIC, "easy", range(1, 61)):
        edges, f = _panel(q)
        i = f.index(max(f))
        if "frequency of the modal" in q.text or "How many values are in the modal" in q.text:
            assert float(q.answer_text) == max(f)
            kinds.add("frequency")
        else:
            assert float(q.answer_text) == edges[i]
            kinds.add("boundary")
    assert kinds == {"frequency", "boundary"}


def test_medium_range_and_count_below():
    kinds = set()
    for q in questions(TOPIC, "medium", range(1, 81)):
        edges, f = _panel(q)
        if "range" in q.text:
            nz = [i for i, v in enumerate(f) if v]
            assert float(q.answer_text) == edges[nz[-1] + 1] - edges[nz[0]]
            kinds.add("range")
        else:
            x = float(re.findall(r"(\d+(?:\.\d+)?)\?", q.text)[-1])
            assert x in edges
            assert int(q.answer_text) == sum(v for v, lo in zip(f, edges) if lo < x)
            kinds.add("below")
    assert kinds == {"range", "below"}


def test_some_ranges_are_shorter_than_the_axis():
    short = 0
    for q in questions(TOPIC, "medium", range(1, 81)):
        edges, f = _panel(q)
        if "range" in q.text and 0 in (f[0], f[-1]):
            short += 1
            assert float(q.answer_text) < edges[-1] - edges[0]
    assert short >= 3


def test_hard_median_class_and_interpolation():
    kinds = set()
    for q in questions(TOPIC, "hard", range(1, 81)):
        edges, f = _panel(q)
        if "median" in q.text:
            assert float(q.answer_text) == edges[_median_class(f)]
            n = sum(f)
            lo, hi = (n + 1) // 2, n // 2 + 1
            cum = [sum(f[: i + 1]) for i in range(len(f))]
            c1 = next(i for i, c in enumerate(cum) if c >= lo)
            c2 = next(i for i, c in enumerate(cum) if c >= hi)
            assert c1 == c2
            kinds.add("median")
        else:
            x = float(re.findall(r"(?:than|below) (\d+(?:\.\d+)?)\.", q.text)[-1])
            w = edges[1] - edges[0]
            est = sum(v * min(max((x - lo) / w, 0), 1) for v, lo in zip(f, edges))
            assert est == int(est) == int(q.answer_text)
            kinds.add("interp")
    assert kinds == {"median", "interp"}


def test_exam_chain_values_follow_through_and_shapes():
    honoured = 0
    shapes = Counter()
    for q in questions(CHAIN, "exam", range(1, 61)):
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "mcq"]
        check_question_grading(q)
        check_tagged(q)
        honoured += check_follow_through(q)
        assert not figures.validate(q.figure)
        assert figures.get_reader(CHAIN)(q) == "; ".join(f"({p.label}) {p.answer_text}" for p in q.parts)
        edges, f = _panel(q)
        n = sum(f)
        assert n % 2 == 1 and int(q.parts[0].answer_text) == n and int(q.parts[1].answer_text) == (n + 1) // 2
        assert float(q.parts[2].answer_text) == edges[_median_class(f)]
        shapes[q.parts[3].answer_text] += 1
    assert honoured > 20
    assert len(shapes) == 3
