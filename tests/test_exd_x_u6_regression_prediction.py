"""Exam source of `least-squares-line`: fit from raw data, predict, and find a residual."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u6_regression_prediction"
SKILL_ID = "least-squares-line"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _rows(stem):
    lists = re.findall(r": ((?:-?\d+(?:\.\d+)?, ){4}-?\d+(?:\.\d+)?)\.", stem)
    assert len(lists) == 2, stem
    return [[Fraction(v) for v in row.split(", ")] for row in lists]


def _oracle(q):
    xs, ys = _rows(q.stem)
    n = len(xs)
    xbar, ybar = sum(xs) / n, sum(ys) / n
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum((x - xbar) ** 2 for x in xs)
    icpt = ybar - slope * xbar
    x_pred = Fraction(re.search(r"when .*? is (\d+)", q.text).group(1))
    x_obs, y_obs = (
        Fraction(v) for v in re.search(r"was (\d+)[^,]*, the observed value was (\d+(?:\.\d+)?)", q.text).groups()
    )
    resid = y_obs - (icpt + slope * x_obs)
    return [slope, icpt, icpt + slope * x_pred, resid]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("least_squares_line", seed=s, kind="summary")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        got = [Fraction(p.answer_text) for p in q.parts]
        assert got == _oracle(q), (seed, q.text)


def test_grader_accepts_key_and_rejects_distractors():
    shown = 0
    for seed in range(100):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 8
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            for d in p.distractors:
                shown += 1
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d)
    assert shown >= 300


def test_follow_through_from_a_wrong_slope():
    checked = 0
    for seed in range(100):
        q = _exam(seed)
        xs, ys = _rows(q.stem)
        slope = Fraction(q.parts[0].distractors[0])
        icpt = sum(ys) / 5 - slope * sum(xs) / 5
        x_pred = Fraction(re.search(r"when .*? is (\d+)", q.text).group(1))
        x_obs, y_obs = (
            Fraction(v) for v in re.search(r"was (\d+)[^,]*, the observed value was (\d+(?:\.\d+)?)", q.text).groups()
        )
        pred = icpt + slope * x_pred
        resid = y_obs - (icpt + slope * x_obs)
        fmt = lambda v: f"{float(v):.1f}"  # noqa: E731 - every value here is on the 0.1 grid or rounds to it
        if any(abs(v * 10 - round(v * 10)) > Fraction(1, 1000) for v in (icpt, pred, resid)):
            continue
        reply = {"a": q.parts[0].distractors[0], "b": fmt(icpt), "c": fmt(pred), "d": fmt(resid)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            value = Fraction(reply[label])
            if value in (slope, icpt, pred) and label != "b":
                continue
            assert result.parts[label].marks == result.parts[label].max_marks, (seed, label, reply)
        checked += 1
    assert checked >= 30


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
