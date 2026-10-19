"""Tests for `scatterplot-fit-line-read-predict` (ACMGM053, ACMGM054): reading a drawn line of best fit."""

from __future__ import annotations

import re

import pytest

from mathgen import figures, make
from mathgen.generators._sr2a_kit import corr
from tests.sr2a_support import (
    TIERS,
    check_figure,
    check_follow_through,
    check_question_grading,
    check_tagged,
    questions,
    top_share,
)

TOPIC = "sr2a_scatter_fit"
CHAIN = "sr2a_scatter_fit_chain"


def _line(q):
    (x1, y1), (x2, y2) = q.figure["lines"][0]["through"]
    m = (y2 - y1) / (x2 - x1)
    return y1 - m * x1, m


def _pts(q):
    return [(p["at"][0], p["at"][1], p.get("label")) for p in q.figure["points"]]


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
    assert top_share(TOPIC, tier) <= 0.4


def test_the_drawn_line_is_close_to_the_least_squares_line():
    for tier in TIERS:
        for q in questions(TOPIC, tier, range(1, 41)):
            c, m = _line(q)
            xs = [p[0] for p in _pts(q)]
            ys = [p[1] for p in _pts(q)]
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            om = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            assert abs(om - m) <= 0.2 and abs(my - om * mx - c) <= 1.0
            assert abs(corr(xs, ys)) >= 0.9


def test_easy_is_the_gradient():
    signs = set()
    for q in questions(TOPIC, "easy", range(1, 61)):
        _c, m = _line(q)
        assert float(q.answer_text) == m
        signs.add(m > 0)
    assert signs == {True, False}


def test_medium_prediction_differs_from_the_plotted_point():
    for q in questions(TOPIC, "medium", range(1, 61)):
        c, m = _line(q)
        k = int(re.findall(r"(\d+)", q.text.split("best fit", 1)[1])[0])
        point = next(y for x, y, _ in _pts(q) if x == k)
        assert float(q.answer_text) == c + m * k and point != c + m * k


def test_hard_both_variants_and_values():
    seen = set()
    for q in questions(TOPIC, "hard", range(1, 81)):
        c, m = _line(q)
        if "residual" in q.text:
            x, y, _ = next(p for p in _pts(q) if p[2] == "P")
            assert float(q.answer_text) == y - (c + m * x)
            assert abs(float(q.answer_text)) >= 2
            seen.add("residual")
        else:
            target = int(re.findall(r"(\d+)", q.text.split("predict", 1)[1])[0])
            assert c + m * float(q.answer_text) == target
            seen.add("inverse")
    assert seen == {"residual", "inverse"}


def test_exam_chain_values_and_follow_through():
    honoured = 0
    for q in questions(CHAIN, "exam"):
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "mcq"]
        check_question_grading(q)
        check_tagged(q)
        honoured += check_follow_through(q)
        assert not figures.validate(q.figure)
        assert figures.get_reader(CHAIN)(q) == "; ".join(f"({p.label}) {p.answer_text}" for p in q.parts)
        c, m = _line(q)
        x, y, _ = next(p for p in _pts(q) if p[2] == "P")
        assert float(q.parts[0].answer_text) == m
        assert float(q.parts[1].answer_text) == c + m * x
        assert float(q.parts[2].answer_text) == y - (c + m * x)
    assert honoured > 20
