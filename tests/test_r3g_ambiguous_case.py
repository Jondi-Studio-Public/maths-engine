"""Tests the ambiguous-case-triangle skill: keys re-derived from the cosine rule, grading, and the exam chain."""

from __future__ import annotations

import math
import re

import pytest

from mathgen.engine import grade
from tests import r3g_support as sup

SKILL = "ambiguous-case-triangle"
TOPIC = "r3g_ambiguous_case"
EXAM = "r3g_exam_ambiguous_case"
_ANGLE = re.compile(r"angle A = (\d+)°")
_SIDE_A = re.compile(r"\ba = ([\d.]+) cm")
_SIDE_B = re.compile(r"\bb = ([\d.]+) cm")


def _ssa(text: str):
    """(A, a, b) from any of the phrasings of the stem."""
    return (
        float(_ANGLE.search(text).group(1)),
        float(_SIDE_A.search(text).group(1)),
        float(_SIDE_B.search(text).group(1)),
    )


_ASK = re.compile(r"Find the (two possible sizes of angle B|two possible lengths of side c|two possible areas)")


def _triangles(A: float, a: float, b: float):
    """(B, c, area) of both triangles from the quadratic c² - 2bc cos A + b² - a² = 0 (the cosine rule)."""
    cA = math.cos(math.radians(A))
    root = math.sqrt(a * a - (b * math.sin(math.radians(A))) ** 2)
    out = []
    for c in sorted((b * cA - root, b * cA + root)):
        B = math.degrees(math.acos((a * a + c * c - b * b) / (2 * a * c)))
        out.append((B, c, 0.5 * b * c * math.sin(math.radians(A))))
    return out


def _numbers(text: str) -> list[float]:
    return sorted(float(v) for v in text.strip("{}").split(","))


def test_wired():
    sup.assert_wired(SKILL, TOPIC, EXAM, "set", ["sine-rule", "triangle-area-sine"])


def test_set_answers_match_an_independent_derivation():
    seen = set()
    for tier in ("easy", "medium", "hard"):
        for seed in range(100):
            q = sup.tier_question(TOPIC, tier, seed)
            ask = _ASK.search(q.text)
            if not ask:
                continue
            A, a, b = _ssa(q.text)
            t1, t2 = _triangles(A, a, b)
            index = {"two possible sizes of angle B": 0, "two possible lengths of side c": 1, "two possible areas": 2}[
                ask.group(1)
            ]
            expected = sorted((t1[index], t2[index]))
            got = _numbers(q.answer_text)
            assert all(abs(g - e) <= 0.0501 for g, e in zip(got, expected)) and len(got) == 2, (
                seed,
                q.text,
                got,
                expected,
            )
            seen.add(index)
    assert seen == {0, 1, 2}


def test_the_limits_and_the_whole_numbers():
    seen = set()
    for seed in range(150):
        q = sup.tier_question(TOPIC, "hard", seed)
        found = re.search(r"angle A = (\d+)° and side b = ([\d.]+) cm", q.text)
        if "Find the two limits" in q.text:
            A, b = int(found.group(1)), float(found.group(2))
            lo, hi = sorted(_numbers(q.answer_text))
            assert abs(lo - b * math.sin(math.radians(A))) <= 0.0051 and abs(hi - b) <= 0.0051, q.text
            seen.add("limits")
        elif "Find every value of a" in q.text:
            A, b = int(found.group(1)), float(found.group(2))
            members = {int(v) for v in _numbers(q.answer_text)}
            expected = {a for a in range(1, 40) if b * math.sin(math.radians(A)) < a < b}
            assert members == expected, q.text
            seen.add("whole")
    assert seen == {"limits", "whole"}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=20, top=0.1)


def test_tiers_differ():
    easy = " ".join(sup.tier_question(TOPIC, "easy", s).text for s in range(60))
    medium = " ".join(sup.tier_question(TOPIC, "medium", s).text for s in range(60))
    hard = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(90))
    assert "sizes of angle B" in easy and "lengths of side c" not in easy
    assert "lengths of side c" in medium and "sizes of angle B" not in medium
    assert "areas" in hard and "the two limits" in hard and "every value of a" in hard
    sup.assert_tier_shapes(TOPIC, EXAM, minimum_hard=2)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["number"] * 4, [2, 1, 2, 2], [False, True, True, True])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=60, top=0.05)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(80):
        q = sup.exam_question(EXAM, seed)
        A, a, b = _ssa(q.text)
        (B1, c1, _), (B2, c2, _) = sorted(_triangles(A, a, b), key=lambda t: t[0])
        got = [float(p.answer_text) for p in q.parts]
        for g, e in zip(got, [B1, B2, c1, c2]):
            assert abs(g - e) <= 0.0501, (seed, got, [B1, B2, c1, c2])


def test_sides_follow_through_from_a_wrong_angle():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        A, a, b = _ssa(q.text)
        canon = {p.label: p.answer_text for p in q.parts}
        wrong_b = 40.0
        c_from_wrong = a * math.sin(math.radians(180 - A - wrong_b)) / math.sin(math.radians(A))
        replies = {**canon, "a": "40.0", "b": "140.0", "c": f"{c_from_wrong:.1f}"}
        result = grade(sk, q, replies)
        assert result.parts["b"].marks == q.parts[1].marks and result.parts["c"].marks == q.parts[2].marks, seed
