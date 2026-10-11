"""Tests for the multi-part exam tier `exam_parts_square_pyramid` (skill square-pyramid-volume-and-surface-area).

The oracle re-derives every part's answer from the question text alone (it does not import the generator).
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade
from mathgen.skills import load_skills

TOPIC = "exam_parts_square_pyramid"
SKILL_ID = "square-pyramid-volume-and-surface-area"
SEEDS = range(120)
K_WORDS = {"twice": Fraction(2), "three times": Fraction(3), "half": Fraction(1, 2)}


@pytest.fixture(scope="module")
def skill():
    _, skills = load_skills()
    return next(s for s in skills if s.id == SKILL_ID)


def _question(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _round1(value: Fraction) -> Fraction:
    scaled = value * 10
    return Fraction(math.floor(scaled + Fraction(1, 2)), 10)


def _fmt1(value: Fraction) -> str:
    scaled = int(value * 10)
    return f"{scaled // 10}.{scaled % 10}"


def _parse(text: str):
    s = int(re.search(r"side length (\d+) (?:cm|m)\b", text).group(1))
    l = int(re.search(r"slant height of (\d+) (?:cm|m)\b", text).group(1))
    h = Fraction(re.search(r"(\d+\.\d) (?:cm|m), correct to 1 decimal place", text).group(1))
    k = next(v for w, v in K_WORDS.items() if re.search(rf"\b{w} the corresponding length", text))
    return s, l, h, k


def _expected(text: str) -> dict:
    s, l, h, k = _parse(text)
    h_squared = l * l - (s // 2) ** 2
    root = math.isqrt(h_squared)
    assert root * root != h_squared
    # the stated h really is sqrt(h²) to 1 decimal place
    assert abs(h * 10 - Fraction(math.sqrt(h_squared) * 10).limit_denominator(10**9)) <= Fraction(1, 2)
    volume = _round1(Fraction(s * s) * h / 3)
    return dict(
        a=str(h_squared),
        b=_fmt1(volume),
        c=str(s * s + 2 * s * l),
        d=_fmt1(_round1(volume * k**3)),
        k=k,
        volume=volume,
    )


def _reply(q, **override):
    labels = [p.label for p in q.parts]
    return {label: override.get(label, p.answer_text) for label, p in zip(labels, q.parts)}


def test_shape():
    q = _question(0)
    assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
    assert [p.marks for p in q.parts] == [1, 2, 2, 1]
    assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]
    assert all(p.answer_kind == "number" for p in q.parts)


def test_full_marks_and_distractors(skill):
    for seed in SEEDS:
        q = _question(seed)
        result = grade(skill, q, q.answer_text)
        assert result.marks == result.max_marks == 6, seed
        assert result.correct
        for part in q.parts:
            assert part.distractors, (seed, part.label)
            for d in part.distractors:
                graded = grade(skill, q, _reply(q, **{part.label: d}))
                assert not graded.parts[part.label].correct, (seed, part.label, d)


def test_oracle_agrees_with_each_part():
    for seed in SEEDS:
        q = _question(seed)
        e = _expected(q.text)
        assert [p.answer_text for p in q.parts] == [e["a"], e["b"], e["c"], e["d"]], (seed, q.text)


def test_part_d_follow_through_expression_uses_numeric_k():
    for seed in SEEDS:
        q = _question(seed)
        e = _expected(q.text)
        ft = q.parts[3].follow_through
        assert {Fraction(2): "8*part_b", Fraction(3): "27*part_b", Fraction(1, 2): "part_b/8"}[e["k"]] == ft


def test_wrong_a_costs_only_part_a(skill):
    for seed in SEEDS:
        q = _question(seed)
        wrong = q.parts[0].distractors[0]
        result = grade(skill, q, _reply(q, a=wrong))
        assert not result.parts["a"].correct
        assert result.parts["b"].correct and result.parts["c"].correct and result.parts["d"].correct
        assert result.marks == 5 < result.max_marks


def test_follow_through_from_wrong_b(skill):
    for seed in SEEDS:
        q = _question(seed)
        e = _expected(q.text)
        wrong_b = e["volume"] + 5
        wrong_d = _fmt1(_round1(wrong_b * e["k"] ** 3))
        assert wrong_d != e["d"]
        result = grade(skill, q, _reply(q, b=_fmt1(wrong_b), d=wrong_d))
        assert not result.parts["b"].correct
        assert result.parts["d"].correct and result.parts["d"].follow_through is True, seed
        assert result.marks == 4 < result.max_marks
        # the canonical d still earns its mark, but as a plain correct answer, not as follow-through
        plain = grade(skill, q, _reply(q, b=_fmt1(wrong_b)))
        assert plain.parts["d"].correct and plain.parts["d"].follow_through is False


def test_wrong_d_is_not_rescued_by_correct_b(skill):
    for seed in SEEDS:
        q = _question(seed)
        for d in q.parts[3].distractors:
            assert not grade(skill, q, _reply(q, d=d)).parts["d"].correct


def test_wrong_everywhere_earns_nothing(skill):
    for seed in SEEDS:
        q = _question(seed)
        reply = {"a": "999991", "b": "999993.1", "c": "999997", "d": "999999.3"}
        result = grade(skill, q, reply)
        assert result.marks == 0
        assert not result.correct


def test_variety():
    stems = {_question(seed).stem for seed in range(60)}
    assert len(stems) >= 40
