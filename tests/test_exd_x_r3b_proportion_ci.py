"""Tests the multi-part exam tier of `ucv1-proportion-confidence-interval`: work back from a printed interval."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade
from tests import exd_x_r3b_support as S

TOPIC = "exd_x_r3b_proportion_ci"
SKILL_ID = "ucv1-proportion-confidence-interval"
_SKILL = S.skill(SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return S.exam_questions(TOPIC, 300)


def _facts(q):
    lo, hi = (Fraction(v) for v in re.search(r"\((0\.\d{4}), (0\.\d{4})\)", q.stem).groups())
    z = Fraction(re.search(r"z = (\d\.\d{3})", q.stem).group(1))
    target = Fraction(re.search(r"at most (0\.\d{4})", q.parts[3].prompt).group(1))
    return lo, hi, z, target


def _half_up(value: Fraction, places: int) -> Fraction:
    scale = 10**places
    return Fraction(math.floor(value * scale + Fraction(1, 2)), scale)


def _text(value: Fraction, places: int) -> str:
    return f"{float(value):.{places}f}"


def test_source_wired():
    S.check_wiring(TOPIC, SKILL_ID)


def test_prompt_variety_and_new_shape_over_hard_tier():
    S.check_shapes(TOPIC, SKILL_ID)


def test_structure():
    S.check_structure(TOPIC, "abcd", [1, 1, 2, 2])


def test_keys_match_independent_inversion_over_300_seeds(questions):
    for q in questions:
        lo, hi, z, target = _facts(q)
        p_hat, margin = (lo + hi) / 2, (hi - lo) / 2
        assert q.parts[0].answer_text == _text(p_hat, 4), q.stem
        assert q.parts[1].answer_text == _text(margin, 4), q.stem
        n = z * z * p_hat * (1 - p_hat) / margin**2
        assert int(q.parts[2].answer_text) == _half_up(n, 0), q.stem
        n_big = z * z * p_hat * (1 - p_hat) / target**2
        assert int(q.parts[3].answer_text) == math.ceil(n_big), q.stem
        assert math.ceil(n_big) > int(q.parts[2].answer_text)


def test_grading_accepts_key_and_rejects_every_distractor():
    S.check_grading(TOPIC, SKILL_ID, 120)


def test_distractors_are_tagged_and_formatted():
    S.check_tags(TOPIC)


def test_follow_through_from_a_wrong_margin(questions):
    checked = 0
    for q in questions[:100]:
        lo, hi, z, target = _facts(q)
        p_hat = (lo + hi) / 2
        wrong_m = Fraction(q.parts[1].distractors[0])  # the full width of the interval
        n_wrong = _half_up(z * z * p_hat * (1 - p_hat) / wrong_m**2, 0)
        reply = {
            "a": q.parts[0].answer_text,
            "b": q.parts[1].distractors[0],
            "c": str(int(n_wrong)),
            "d": q.parts[3].answer_text,
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (q.stem, reply)
        assert result.parts["d"].marks == 2
        checked += 1
    assert checked == 100


def test_follow_through_from_a_wrong_p_hat(questions):
    checked = 0
    for q in questions[:100]:
        lo, hi, z, target = _facts(q)
        margin = (hi - lo) / 2
        wrong_p = Fraction(q.parts[0].distractors[0])  # a bound
        n_wrong = _half_up(z * z * wrong_p * (1 - wrong_p) / margin**2, 0)
        big_wrong = math.ceil(z * z * wrong_p * (1 - wrong_p) / target**2)
        reply = {
            "a": q.parts[0].distractors[0],
            "b": q.parts[1].answer_text,
            "c": str(int(n_wrong)),
            "d": str(big_wrong),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == 2, (q.stem, reply)
        assert result.parts["d"].marks == 2, (q.stem, reply)
        checked += 1
    assert checked == 100


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "0.9999", "b": "0.7777", "c": "7", "d": "9"})
        assert result.marks == 0 and not result.correct


def test_answer_space_over_300_seeds():
    S.check_answer_space(TOPIC)


def test_clean_and_deterministic():
    S.check_clean_and_deterministic(TOPIC)
    assert make(TOPIC, seed=7, difficulty="exam").to_dict() == make(TOPIC, seed=7, difficulty="exam").to_dict()
