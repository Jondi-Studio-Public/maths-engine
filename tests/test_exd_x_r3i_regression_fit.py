"""Exam source of `ucv1-regression-residual-diagnostic`: residual, SSE, SST and r^2 for five cases."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.core import round_half_up

SKILL_ID = "ucv1-regression-residual-diagnostic"
TOPIC = "exd_x_r3i_regression_fit"
LABELS = ["a", "b", "c", "d"]

from collections import Counter

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
SEEDS = range(300)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == LABELS
        assert 3 <= len(q.parts) <= 4
        assert all(p.distractors and len(p.distractors) >= 2 for p in q.parts)
        assert sum(1 for p in q.parts if p.follow_through) >= 1


def test_prompt_variety_and_new_shapes():
    hard_src = next(s for s in _SKILL.sources if s.difficulty == "hard")
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make(hard_src.topic, seed=s, **(hard_src.params or {}))) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects(questions):
    for q in questions[:100]:
        assert find_text_defects(q, include_working=True) == [], q.text


def test_key_rederived_from_prompt(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, q.text


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: str(d)})
                assert result.parts[p.label].marks == 0, (q.text, p.label, str(d))


def test_distractors_match_key_layout(questions):
    for q in questions:
        for p in q.parts:
            if p.answer_kind != "number":
                continue
            for d in p.distractors:
                assert ("." in d) == ("." in p.answer_text), (p.label, d, p.answer_text)
                if "." in d:
                    assert len(d.split(".")[1]) == len(p.answer_text.split(".")[1]), (p.label, d, p.answer_text)


def test_distractors_are_tagged(questions):
    from mathgen.misconceptions import CATALOGUE

    for q in questions[:100]:
        for p in q.parts:
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (q.text, p.label, d)


def test_answer_space(questions):
    tuples = Counter(tuple(_canonical(q).values()) for q in questions)
    assert len(tuples) >= 100, len(tuples)
    assert max(tuples.values()) / len(questions) <= 0.05


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        reply = {p.label: ("999999.5" if p.answer_kind == "number" else "999999/7") for p in q.parts}
        result = grade(_SKILL, q, reply)
        assert result.marks == 0 and not result.correct


def test_deterministic():
    assert _exam(7).to_dict() == _exam(7).to_dict()


def _facts(q):
    pts = [(int(x), int(y)) for x, y in re.findall(r"\((\d), (\d+)\)", q.stem)]
    m = re.search(r"ŷ = (\d+\.\d) ([+-]) (\d+\.\d)x", q.stem)
    a, b = Fraction(m.group(1)), Fraction(m.group(3)) * (1 if m.group(2) == "+" else -1)
    k = int(re.search(r"x = (\d)", q.parts[0].prompt).group(1))
    return pts, a, b, k


def oracle(q):
    pts, a, b, k = _facts(q)
    resid = {x: y - (a + b * x) for x, y in pts}
    sse = sum(e * e for e in resid.values())
    ybar = Fraction(sum(y for _, y in pts), len(pts))
    sst = sum((y - ybar) ** 2 for _, y in pts)
    r2 = round_half_up(1 - sse / sst, 3)
    return {"a": f"{float(resid[k]):.1f}", "b": f"{float(sse):.2f}", "c": f"{float(sst):.2f}", "d": f"{float(r2):.3f}"}


def test_follow_through_from_wrong_sse(questions):
    checked = 0
    for q in questions[:150]:
        wb = Fraction(q.parts[1].distractors[0])
        sst = Fraction(q.parts[2].answer_text)
        if wb <= 0:
            continue
        r2 = round_half_up(1 - wb / sst, 3)
        reply = {**_canonical(q), "b": q.parts[1].distractors[0], "d": f"{float(r2):.3f}"}
        result = grade(_SKILL, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["d"].marks == result.parts["d"].max_marks, (q.text, reply)
        checked += 1
    assert checked > 100
