"""Tests the exam sources of the three metric conversion skills: a total shared into portions, with the leftover."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

_SKILLS = {s.id: s for s in load_skills()[1]}
_KINDS = {
    "length": "convert-metric-length",
    "capacity": "convert-metric-capacity",
    "mass": "convert-metric-mass",
}
_FACTORS = {("L", "mL"): 1000, ("kg", "g"): 1000, ("m", "cm"): 100, ("cm", "mm"): 10, ("km", "m"): 1000}
_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?) (mL|mm|km|kg|cm|L|g|m)\b")


def _exam(kind, seed):
    return make(f"exd_x_m1_metric_{kind}", seed=seed, difficulty="exam")


def _oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the rendered prompt alone."""
    (big, big_unit), (s, small_unit) = _UNIT_RE.findall(q.stem)[:2]
    factor = _FACTORS[(big_unit, small_unit)]
    total = Fraction(big) * factor
    s = int(s)
    whole = int(total) // s
    left = (total - s * whole) / factor
    text = f"{float(left):.{len(str(factor)) - 1}f}"
    return {"a": str(int(total)), "b": str(whole), "c": text}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


@pytest.mark.parametrize("kind", _KINDS)
def test_source_wired(kind):
    exam = [s for s in _SKILLS[_KINDS[kind]].sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [f"exd_x_m1_metric_{kind}"]
    assert exam[0].stages == ["exam"]


@pytest.mark.parametrize("kind", _KINDS)
def test_prompt_variety_and_new_shapes(kind):
    exam = {prompt_shape(_exam(kind, s)) for s in range(60)}
    hard = {
        prompt_shape(
            make("metric_conversion", seed=s, difficulty="hard", kind=kind, max_steps=1, value_hi=30, combine=True)
        )
        for s in range(60)
    }
    assert len(exam) >= 3
    assert exam - hard


@pytest.mark.parametrize("kind", _KINDS)
def test_no_text_defects(kind):
    for seed in range(60):
        assert find_text_defects(_exam(kind, seed)) == []


@pytest.mark.parametrize("kind", _KINDS)
def test_key_rederived_from_prompt(kind):
    for seed in range(300):
        q = _exam(kind, seed)
        assert _oracle(q) == _canonical(q), (seed, q.text)


@pytest.mark.parametrize("kind", _KINDS)
def test_prompt_does_not_contain_a_key(kind):
    for seed in range(300):
        q = _exam(kind, seed)
        numbers = set(re.findall(r"\d+(?:\.\d+)?", q.stem + " ".join(p.prompt for p in q.parts)))
        assert not numbers & {p.answer_text for p in q.parts}, (seed, q.text)


@pytest.mark.parametrize("kind", _KINDS)
def test_grader_accepts_key_and_rejects_distractors(kind):
    skill = _SKILLS[_KINDS[kind]]
    shown = 0
    for seed in range(120):
        q = _exam(kind, seed)
        result = grade(skill, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5
        canon = _canonical(q)
        for p in q.parts:
            assert p.distractors, (q.text, p.label)
            for d in p.distractors:
                shown += 1
                assert grade(skill, q, {**canon, p.label: d}).parts[p.label].marks == 0, (q.text, p.label, d)
    assert shown >= 300


@pytest.mark.parametrize("kind", _KINDS)
def test_follow_through_from_wrong_conversion(kind):
    skill = _SKILLS[_KINDS[kind]]
    for seed in range(120):
        q = _exam(kind, seed)
        (big, big_unit), (s, small_unit) = _UNIT_RE.findall(q.stem)[:2]
        factor = _FACTORS[(big_unit, small_unit)]
        s = int(s)
        a_wrong = q.parts[0].distractors[0]  # total mis-converted by a power of ten
        a = Fraction(a_wrong)
        b = math.floor(a / s)
        c = (a - s * b) / factor
        reply = {"a": a_wrong, "b": str(b), "c": f"{float(c):.{len(str(factor)) - 1}f}"}
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bc":
            if reply[label] == q.parts["abc".index(label)].answer_text:
                continue
            if a == 0 or (label == "b" and b == 0) or a < 0:
                continue
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, reply)


@pytest.mark.parametrize("kind", _KINDS)
def test_follow_through_from_wrong_count(kind):
    """A wrong count of portions used correctly in (c) earns (c)."""
    skill = _SKILLS[_KINDS[kind]]
    checked = 0
    for seed in range(120):
        q = _exam(kind, seed)
        (big, big_unit), (s, small_unit) = _UNIT_RE.findall(q.stem)[:2]
        factor = _FACTORS[(big_unit, small_unit)]
        a = int(q.parts[0].answer_text)
        b_wrong = int(q.parts[1].answer_text) - 1
        c = Fraction(a - int(s) * b_wrong, factor)
        result = grade(skill, q, {"a": str(a), "b": str(b_wrong), "c": f"{float(c):.{len(str(factor)) - 1}f}"})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, b_wrong, c)
        assert result.parts["c"].follow_through is True
        checked += 1
    assert checked == 120


@pytest.mark.parametrize("kind", _KINDS)
def test_answer_space(kind):
    answers = [_exam(kind, s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


@pytest.mark.parametrize("kind", _KINDS)
def test_deterministic(kind):
    assert _exam(kind, 7).to_json() == _exam(kind, 7).to_json()
