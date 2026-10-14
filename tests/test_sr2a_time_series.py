"""Tests for `time-series-features` (ACMGM088, reading half of ACMGM087): features read from a drawn line graph."""

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

TOPIC = "sr2a_time_series"
CHAIN = "sr2a_time_series_chain"


def _ys(q):
    return {int(p["at"][0]): p["at"][1] for p in q.figure["points"]}


def _gaps(ys, p):
    return {t: ys[t + p] - ys[t] for t in range(1, len(ys) - p + 1)}


def _cycle(ys):
    """The cycle length (4, 6 or 7) for which at most two same-season gaps differ from the common one."""
    for p in (4, 6, 7):
        if len(ys) % p == 0 and len(ys) >= 2 * p:
            g = list(_gaps(ys, p).values())
            if max(g.count(v) for v in set(g)) >= len(g) - 2:
                return p
    raise AssertionError("no cycle found")


@pytest.mark.parametrize("tier", TIERS)
def test_answers_grade_correct_and_distractors_wrong(tier):
    for q in questions(TOPIC, tier):
        check_question_grading(q)
        check_tagged(q)
        check_figure(q)


@pytest.mark.parametrize("tier", TIERS)
def test_figure_is_valid_with_every_value_on_a_gridline(tier):
    for q in questions(TOPIC, tier, range(1, 31)):
        assert not figures.validate(q.figure)
        step = q.figure["y_grid"]
        assert all(y % step == 0 and y >= step for y in _ys(q).values())
        assert len(q.figure["segments"]) == len(q.figure["points"]) - 1
        assert figures.get_reader(TOPIC)(q) == q.answer_text


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_varied(tier):
    assert make(TOPIC, seed=3, difficulty=tier).to_json() == make(TOPIC, seed=3, difficulty=tier).to_json()
    assert len({repr(make(TOPIC, seed=s, difficulty=tier).figure) for s in range(1, 41)}) >= 38
    assert top_share(TOPIC, tier) <= 0.35


def test_easy_difference_from_the_graph():
    for q in questions(TOPIC, "easy", range(1, 61)):
        ys = _ys(q)
        later, earlier = (int(x) for x in re.findall(r"(?:quarter|bill|day) (\d+)", q.text)[-2:])
        assert int(q.answer_text) == ys[later] - ys[earlier] > 0


def test_medium_prediction_follows_pattern_and_step():
    for q in questions(TOPIC, "medium", range(1, 61)):
        ys = _ys(q)
        p = int(re.search(r"(?:[Ee]very|repeats every) (\d+)", q.text).group(1))
        n = len(ys)
        assert n % p == 0 and len(set(_gaps(ys, p).values())) == 1, "a medium series has no one-off value"
        step = ys[n + 1 - p] - ys[n + 1 - 2 * p]
        assert int(q.answer_text) == ys[n + 1 - p] + step


def test_hard_which_removes_the_only_break():
    for q in questions(TOPIC, "hard", range(1, 81)):
        ys = _ys(q)
        p = _cycle(ys)
        if "number was it" in q.text or "did this happen" in q.text:
            t = int(q.answer_text)
            others = [g for k, g in _gaps(ys, p).items() if k != t and k + p != t]
            assert len(set(others)) == 1, "the pattern is not otherwise regular"
            assert ys[t + p] - ys[t] not in others or t + p > len(ys)
        else:
            n = len(ys)
            step = ys[n + 2 - p] - ys[n + 2 - 2 * p]
            assert int(q.answer_text) == ys[n + 1 - 2 * p] + 2 * step
            assert ys[n + 1 - p] != ys[n + 1 - 2 * p] + step, "the nearest season must be the broken one"


def test_hard_both_variants_occur():
    kinds = {("number was it" in q.text or "did this happen" in q.text) for q in questions(TOPIC, "hard", range(1, 61))}
    assert kinds == {True, False}


def test_exam_chain_graph_and_follow_through():
    honoured = 0
    for q in questions(CHAIN, "exam"):
        assert [p.answer_kind for p in q.parts] == ["number"] * 4
        check_question_grading(q)
        honoured += check_follow_through(q)
        assert not figures.validate(q.figure)
        assert figures.get_reader(CHAIN)(q) == "; ".join(f"({p.label}) {p.answer_text}" for p in q.parts)
        ys = _ys(q)
        p = _cycle(ys)
        assert int(q.parts[0].answer_text) == p
    assert honoured > 20
