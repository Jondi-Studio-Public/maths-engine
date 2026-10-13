"""Tests the multi-part exam tier of `simple-interest`: interest over months, total value, a rate for a shorter term."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade
from tests import exd_x_r3b_support as S

TOPIC = "exd_x_r3b_simple_interest"
SKILL_ID = "simple-interest"
_SKILL = S.skill(SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return S.exam_questions(TOPIC, 300)


def _facts(q):
    p = int(re.search(r"\$(\d+)", q.stem).group(1))
    rate = Fraction(re.search(r"(\d+(?:\.\d+)?)% p\.a\.", q.stem).group(1))
    months = int(re.search(r"(\d+) months", q.stem).group(1))
    shorter = int(re.search(r"in (\d+) months instead", q.parts[2].prompt).group(1))
    return p, rate, months, shorter


def _half_up(value: Fraction, places: int) -> Fraction:
    scaled = value * 10**places
    return Fraction(int(scaled + Fraction(1, 2)), 10**places)


def _text(value: Fraction) -> str:
    return f"{float(value):.2f}"


def test_source_wired():
    S.check_wiring(TOPIC, SKILL_ID)


def test_prompt_variety_and_new_shape_over_hard_tier():
    S.check_shapes(TOPIC, SKILL_ID)


def test_structure():
    S.check_structure(TOPIC, "abc", [2, 1, 2])


def test_keys_match_independent_arithmetic_over_300_seeds(questions):
    for q in questions:
        p, rate, months, shorter = _facts(q)
        interest = p * rate / 100 * Fraction(months, 12)
        assert interest.denominator in (1, 2, 4, 5, 10, 20, 25, 50, 100)
        assert q.parts[0].answer_text == _text(interest), q.stem
        assert q.parts[1].answer_text == _text(p + interest), q.stem
        new_rate = interest * 100 / (p * Fraction(shorter, 12))
        assert q.parts[2].answer_text == _text(_half_up(new_rate, 2)), q.stem
        assert 0 < new_rate <= 10


def test_grading_accepts_key_and_rejects_every_distractor():
    S.check_grading(TOPIC, SKILL_ID, 120)


def test_distractors_are_tagged_and_formatted():
    S.check_tags(TOPIC)


def test_follow_through_from_interest_with_months_taken_as_years(questions):
    checked = 0
    for q in questions[:100]:
        p, _, _, shorter = _facts(q)
        wrong = Fraction(q.parts[0].distractors[0])
        reply = {
            "a": q.parts[0].distractors[0],
            "b": _text(p + wrong),
            "c": _text(_half_up(wrong * 1200 / (p * shorter), 2)),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (q.stem, reply)
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (q.stem, reply)
        checked += 1
    assert checked == 100


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "1.11", "b": "2.22", "c": "3.33"})
        assert result.marks == 0 and not result.correct


def test_answer_space_over_300_seeds():
    S.check_answer_space(TOPIC)


def test_clean_and_deterministic():
    S.check_clean_and_deterministic(TOPIC)
    assert make(TOPIC, seed=7, difficulty="exam").to_dict() == make(TOPIC, seed=7, difficulty="exam").to_dict()
