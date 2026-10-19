"""Tests for `parallel-box-plots-outlier-fences` (ACMGM031, ACMEM059): fences and outliers from two drawn box plots."""

from __future__ import annotations

import re

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

TOPIC = "sr2a_box_plots"
CHAIN = "sr2a_box_plots_chain"


def _plots(q):
    return {p["label"]: p for p in q.figure["plots"]}


def _iqr(p):
    return p["q3"] - p["q1"]


def _up(p):
    return p["q3"] + 1.5 * _iqr(p)


def _low(p):
    return p["q1"] - 1.5 * _iqr(p)


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
def test_fences_are_whole_numbers_and_plots_are_valid(tier):
    for q in questions(TOPIC, tier, range(1, 41)):
        assert not figures.validate(q.figure)
        for p in _plots(q).values():
            assert _iqr(p) % 2 == 0 and _up(p) == int(_up(p)) and _low(p) == int(_low(p))
            assert all(v % 2 == 0 for v in (p["min"], p["q1"], p["median"], p["q3"], p["max"], *p["outliers"]))


def test_easy_iqr_and_median_difference():
    kinds = set()
    for q in questions(TOPIC, "easy", range(1, 61)):
        plots = _plots(q)
        if "median" in q.text:
            m = re.search(r"median of ([\w ]+?) (?:than|greater than) the median of ([\w ]+?)\?", q.text)
            assert int(q.answer_text) == plots[m.group(1)]["median"] - plots[m.group(2)]["median"] > 0
            kinds.add("median")
        else:
            g = re.search(r"interquartile range (?:of|for) ([\w ]+?)[?.]", q.text).group(1)
            assert int(q.answer_text) == _iqr(plots[g])
            kinds.add("iqr")
    assert kinds == {"median", "iqr"}


def test_medium_upper_and_lower_fences():
    kinds = set()
    for q in questions(TOPIC, "medium", range(1, 61)):
        plots = _plots(q)
        g = next(n for n in plots if n in q.text.split("below Q1.")[1])
        if "upper fence" in q.text:
            assert int(q.answer_text) == _up(plots[g])
            kinds.add("upper")
        else:
            assert int(q.answer_text) == _low(plots[g]) >= 0
            kinds.add("lower")
    assert kinds == {"upper", "lower"}


def test_hard_beyond_the_fence_and_smallest_outlier():
    kinds = set()
    for q in questions(TOPIC, "hard", range(1, 81)):
        plots = _plots(q)
        g = next(n for n in plots if n in q.text.split("below Q1.")[1])
        p = plots[g]
        if p["outliers"]:
            o = p["outliers"][0]
            expected = o - _up(p) if o > p["q3"] else _low(p) - o
            assert int(q.answer_text) == expected > 0
            kinds.add("beyond")
        else:
            assert int(q.answer_text) == _up(p) + 1
            assert int(q.answer_text) > p["max"]
            kinds.add("smallest")
    assert kinds == {"beyond", "smallest"}


def test_exam_chain_values_and_the_range_trap():
    honoured = 0
    for q in questions(CHAIN, "exam"):
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "mcq"]
        check_question_grading(q)
        check_tagged(q)
        honoured += check_follow_through(q)
        assert not figures.validate(q.figure)
        assert figures.get_reader(CHAIN)(q) == "; ".join(f"({p.label}) {p.answer_text}" for p in q.parts)
        plots = _plots(q)
        g = next(n for n, p in plots.items() if p["outliers"])
        h = next(n for n in plots if n != g)
        p, other = plots[g], plots[h]
        o = p["outliers"][0]
        assert int(q.parts[0].answer_text) == _iqr(p)
        fence = _up(p) if o > p["q3"] else _low(p)
        assert int(q.parts[1].answer_text) == fence
        assert int(q.parts[2].answer_text) == abs(o - fence)
        assert max(p["max"], o) - min(p["min"], o) > other["max"] - other["min"]
        assert _iqr(other) > _iqr(p)
        assert q.parts[3].answer_text.startswith(h)
    assert honoured > 20
