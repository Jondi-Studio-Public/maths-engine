"""Tests the multi-part exam tier of `senior-determine-and-examine-the-n-th-roots-of-acmsm088`."""

from __future__ import annotations

import cmath
import math
import re

import pytest

from mathgen import make
from mathgen.engine import grade
from tests import exd_x_r3b_support as S

TOPIC = "exd_x_r3b_nth_roots"
SKILL_ID = "senior-determine-and-examine-the-n-th-roots-of-acmsm088"
_SKILL = S.skill(SKILL_ID)
_NAMES = {"cube": 3, "fourth": 4}
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_W = re.compile(r"w = (-?[\d√]+) ([+-]) \(?([\d√]+)\)?i")


@pytest.fixture(scope="module")
def questions():
    return S.exam_questions(TOPIC, 300)


def _real(text: str) -> float:
    sign = -1 if text.startswith("-") else 1
    text = text.lstrip("-")
    if "√" in text:
        coeff, radicand = text.split("√")
        return sign * (int(coeff) if coeff else 1) * math.sqrt(int(radicand))
    return sign * int(text)


def _facts(q):
    x, sign, y = _W.search(q.stem).groups()
    w = complex(_real(x), (-1 if sign == "-" else 1) * _real(y))
    n = next(v for k, v in _NAMES.items() if f"{k} root" in q.stem)
    quadrant = _ORDINALS[re.search(r"in the (\w+) quadrant", q.parts[3].prompt).group(1)]
    return w, n, quadrant


def test_source_wired():
    S.check_wiring(TOPIC, SKILL_ID)


def test_prompt_variety_and_new_shape_over_hard_tier():
    S.check_shapes(TOPIC, SKILL_ID)


def test_structure():
    S.check_structure(TOPIC, "abcd", [1, 1, 1, 2])


def test_keys_match_cmath_over_300_seeds(questions):
    for q in questions:
        w, n, quadrant = _facts(q)
        a, b, c, d = (int(p.answer_text) for p in q.parts)
        assert abs(abs(w) - a) < 1e-6, q.stem
        assert round(math.degrees(cmath.phase(w)) % 360, 6) == b, q.stem
        roots = [abs(w) ** (1 / n) * cmath.exp(1j * (cmath.phase(w) + 2 * math.pi * k) / n) for k in range(n)]
        hits = [z for z in roots if int(math.degrees(cmath.phase(z)) % 360) // 90 + 1 == quadrant]
        assert len(hits) == 1, q.stem
        assert abs(abs(hits[0]) - c) < 1e-6 and round(math.degrees(cmath.phase(hits[0])) % 360, 6) == d, q.stem


def test_grading_accepts_key_and_rejects_every_distractor():
    S.check_grading(TOPIC, SKILL_ID, 120)


def test_distractors_are_tagged_and_formatted():
    S.check_tags(TOPIC)


def test_follow_through_from_a_wrong_argument(questions):
    checked = 0
    for q in questions[:100]:
        _, n, _ = _facts(q)
        bad_b = int(q.parts[1].distractors[0])
        k = (int(q.parts[3].answer_text) * n - int(q.parts[1].answer_text)) // 360
        d_wrong = (bad_b + 360 * k) / n
        if d_wrong != int(d_wrong) or d_wrong == bad_b:
            continue
        reply = {"a": q.parts[0].answer_text, "b": str(bad_b), "c": q.parts[2].answer_text, "d": str(int(d_wrong))}
        result = grade(_SKILL, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (q.stem, reply)
        checked += 1
    assert checked >= 20


def test_follow_through_from_a_wrong_modulus(questions):
    checked = 0
    for q in questions[:100]:
        _, n, _ = _facts(q)
        a_wrong = int(q.parts[0].distractors[0]) ** 1  # the square, which has an integer n-th root only sometimes
        root = round(a_wrong ** (1 / n))
        if root**n != a_wrong:
            continue
        reply = {"a": str(a_wrong), "b": q.parts[1].answer_text, "c": str(root), "d": q.parts[3].answer_text}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True
        checked += 1
    assert checked >= 1


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "7777", "b": "13", "c": "99", "d": "11"})
        assert result.marks == 0 and not result.correct


def test_answer_space_over_300_seeds():
    S.check_answer_space(TOPIC)


def test_clean_and_deterministic():
    S.check_clean_and_deterministic(TOPIC)
    assert make(TOPIC, seed=7, difficulty="exam").to_dict() == make(TOPIC, seed=7, difficulty="exam").to_dict()
