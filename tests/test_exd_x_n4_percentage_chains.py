"""Exam tiers of `percentage-change` and `reverse-percentage`: two successive percentage changes, as a chain."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

FORWARD = "exd_x_n4_percentage_change"
REVERSE = "exd_x_n4_reverse_percentage"
SKILLS = {s.id: s for s in load_skills()[1]}
_SIGN = {"increased": 1, "marked up": 1, "raised": 1, "reduced": -1, "discounted": -1, "cut": -1}
_CHANGE_RE = re.compile(r"(increased|marked up|raised|reduced|discounted|cut) by (\d+)%")


def _exam(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


def _round(value: Fraction, places: int) -> str:
    scaled = value * 10**places
    rounded = Fraction(int(abs(scaled) + Fraction(1, 2)), 10**places) * (1 if scaled >= 0 else -1)
    return f"{float(rounded):.{places}f}"


def _changes(stem):
    found = [(_SIGN[w], int(p)) for w, p in _CHANGE_RE.findall(stem)]
    if "GST was then added" in stem:
        found.append((1, 10))
    return found


def forward_oracle(q):
    price = Fraction(re.search(r"\$(\d+)\.?", q.stem).group(1))
    (s1, p1), (s2, p2) = _changes(q.stem)
    a = price * (100 + s1 * p1) / 100
    b = a * (100 + s2 * p2) / 100
    return {"a": _round(a, 2), "b": _round(b, 2), "c": _round((b - price) * 100 / price, 1)}


def reverse_oracle(q):
    final = Fraction(re.search(r"paid \$(\d+\.\d\d)", q.stem).group(1))
    (s1, p1), (s2, p2) = _changes(q.stem)
    middle = final / Fraction(100 + s2 * p2, 100)
    original = middle / Fraction(100 + s1 * p1, 100)
    return {"a": _round(middle, 2), "b": _round(original, 2), "c": _round(abs(final - original), 2)}


CASES = [
    ("percentage-change", FORWARD, forward_oracle),
    ("reverse-percentage", REVERSE, reverse_oracle),
]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


@pytest.mark.parametrize("skill_id, topic, oracle", CASES)
def test_source_wired(skill_id, topic, oracle):
    exam = [s for s in SKILLS[skill_id].sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [topic]
    assert exam[0].stages == ["exam"]


@pytest.mark.parametrize("skill_id, topic, oracle", CASES)
def test_prompt_shapes_are_new(skill_id, topic, oracle):
    exam = {prompt_shape(_exam(topic, s)) for s in range(60)}
    hard = {
        prompt_shape(make("percentage_change", seed=s, reverse=skill_id == "reverse-percentage")) for s in range(60)
    }
    assert len(exam) >= 3
    assert exam - hard


@pytest.mark.parametrize("skill_id, topic, oracle", CASES)
def test_key_rederived_from_prompt(skill_id, topic, oracle):
    for seed in range(300):
        q = _exam(topic, seed)
        assert oracle(q) == _canonical(q), (seed, q.text)


@pytest.mark.parametrize("skill_id, topic, oracle", CASES)
def test_shape_and_marks(skill_id, topic, oracle):
    q = _exam(topic, 0)
    assert [p.label for p in q.parts] == ["a", "b", "c"]
    assert all(p.answer_kind == "number" and p.distractors for p in q.parts)
    assert [bool(p.follow_through) for p in q.parts] == [False, True, True]


@pytest.mark.parametrize("skill_id, topic, oracle", CASES)
def test_answer_space(skill_id, topic, oracle):
    answers = [_exam(topic, s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


@pytest.mark.parametrize("skill_id, topic, oracle", CASES)
def test_key_accepted_and_distractors_rejected(skill_id, topic, oracle):
    skill = SKILLS[skill_id]
    shown = 0
    for seed in range(120):
        q = _exam(topic, seed)
        canon = _canonical(q)
        full = grade(skill, q, canon)
        assert full.correct and full.marks == full.max_marks, (seed, q.text)
        for part in q.parts:
            assert len(part.distractors) >= 2
            for d in part.distractors:
                shown += 1
                result = grade(skill, q, {**canon, part.label: d})
                assert result.parts[part.label].marks == 0, (seed, part.label, d)
                assert part.option_notes.get(d)
    assert shown >= 600


def test_forward_follow_through_from_wrong_first_part():
    skill = SKILLS["percentage-change"]
    for seed in range(100):
        q = _exam(FORWARD, seed)
        (s1, p1), (s2, p2) = _changes(q.stem)
        price = Fraction(re.search(r"\$(\d+)", q.stem).group(1))
        wrong_a = Fraction(q.parts[0].distractors[0])
        wrong_b = wrong_a * (100 + s2 * p2) / 100
        wrong_c = (wrong_b - price) * 100 / price
        reply = {"a": q.parts[0].distractors[0], "b": _round(wrong_b, 2), "c": _round(wrong_c, 1)}
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == q.parts[1].marks, (seed, q.text, reply)
        assert result.parts["b"].follow_through
        assert result.marks < result.max_marks and not result.correct


def test_reverse_follow_through_from_wrong_first_part():
    skill = SKILLS["reverse-percentage"]
    carried = 0
    for seed in range(100):
        q = _exam(REVERSE, seed)
        (s1, p1), (s2, p2) = _changes(q.stem)
        final = Fraction(re.search(r"paid \$(\d+\.\d\d)", q.stem).group(1))
        wrong_a = Fraction(q.parts[0].distractors[0])
        wrong_b = wrong_a / Fraction(100 + s1 * p1, 100)
        gap = final - wrong_b if "more than" in q.text else wrong_b - final
        reply = {"a": q.parts[0].distractors[0], "b": _round(wrong_b, 2), "c": _round(gap, 2)}
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == q.parts[1].marks, (seed, q.text, reply)
        if gap > 0 and gap != wrong_b:  # a negative gap, or one that merely repeats (b), earns no carry-forward
            carried += 1
            assert result.parts["c"].marks == q.parts[2].marks, (seed, q.text, reply)
            assert result.parts["c"].follow_through
    assert carried >= 50


def test_all_wrong_earns_nothing():
    for topic, skill_id in ((FORWARD, "percentage-change"), (REVERSE, "reverse-percentage")):
        q = _exam(topic, 3)
        result = grade(SKILLS[skill_id], q, {"a": "999999.99", "b": "1.11", "c": "2.2"})
        assert result.marks == 0 and not result.correct


def test_deterministic():
    for topic in (FORWARD, REVERSE):
        assert _exam(topic, 7).to_json() == _exam(topic, 7).to_json()
