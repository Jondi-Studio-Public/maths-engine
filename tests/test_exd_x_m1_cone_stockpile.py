"""Tests the exam source of `cone-volume`: height from the slant height, volume, and truck loads."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_m1_cone_stockpile"
SKILL_ID = "cone-volume"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_PI = Fraction(314159, 100000)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _dp1(x: Fraction) -> Fraction:
    return Fraction(math.floor(x * 10 + Fraction(1, 2)), 10)


def _read(text: str):
    d, s = (int(v) for v in re.search(r"base diameter of (\d+) m and a slant height of (\d+) m", text).groups())
    cap = int(re.search(r"carries (\d+) m³", text).group(1))
    return d, s, cap


def _oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the rendered prompt alone."""
    d, s, cap = _read(q.text)
    r = Fraction(d, 2)
    h = math.isqrt(s * s - int(r) ** 2)
    assert h * h == s * s - int(r) ** 2
    v = _dp1(_PI * r * r * h / 3)
    return {"a": str(h), "b": f"{float(v):.1f}", "c": str(math.ceil(v / cap))}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("cone_volume", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_shape():
    q = _exam(1)
    assert [p.label for p in q.parts] == ["a", "b", "c"]
    assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        assert _oracle(q) == _canonical(q), (seed, q.text)


def test_exact_route_agrees_with_rounded_route():
    for seed in range(300):
        q = _exam(seed)
        d, s, cap = _read(q.text)
        r = Fraction(d, 2)
        h = math.isqrt(s * s - int(r) ** 2)
        assert str(math.ceil(_PI * r * r * h / 3 / cap)) == q.parts[2].answer_text, (seed, q.text)


def test_prompt_does_not_contain_a_key():
    for seed in range(300):
        q = _exam(seed)
        numbers = set(re.findall(r"\d+(?:\.\d+)?", q.stem + " ".join(p.prompt for p in q.parts)))
        assert not numbers & {p.answer_text for p in q.parts}, (seed, q.text)


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(120)]


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5


def test_distractors_rejected(questions):
    shown = 0
    for q in questions:
        canon = _canonical(q)
        for p in q.parts:
            assert p.distractors, (q.text, p.label)
            for d in p.distractors:
                shown += 1
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (q.text, p.label, d)
    assert shown >= 300


def test_follow_through_from_wrong_height(questions):
    for q in questions:
        d, _, cap = _read(q.text)
        h_wrong = q.parts[0].distractors[0]
        v = _dp1(_PI * Fraction(d, 2) ** 2 * Fraction(h_wrong) / 3)
        loads = math.ceil(v / cap)
        result = grade(_SKILL, q, {"a": h_wrong, "b": f"{float(v):.1f}", "c": str(loads)})
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, h_wrong, v, loads)
            if {"b": f"{float(v):.1f}", "c": str(loads)}[label] != q.parts["abc".index(label)].answer_text:
                assert result.parts[label].follow_through is True
        assert not result.correct


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
