"""Tests the multi-part exam tier of `senior-use-the-scale-factor-for-two-similar-figures`."""

from __future__ import annotations

import re
from fractions import Fraction
from math import isqrt

import pytest

from mathgen import make
from mathgen.engine import grade
from tests import exd_x_r3b_support as S

TOPIC = "exd_x_r3b_similar_scale"
SKILL_ID = "senior-use-the-scale-factor-for-two-similar-figures"
_SKILL = S.skill(SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return S.exam_questions(TOPIC, 300)


def _facts(q):
    a1, a2 = (int(v) for v in re.findall(r"(\d+) cm²", q.stem))
    h1 = int(re.search(r"(\d+) cm(?![²³\w])", q.stem).group(1))
    v1 = int(re.search(r"(\d+) cm³", q.stem).group(1))
    return a1, a2, h1, v1


def _text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def test_source_wired():
    S.check_wiring(TOPIC, SKILL_ID)


def test_prompt_variety_and_new_shape_over_hard_tier():
    S.check_shapes(TOPIC, SKILL_ID)


def test_structure():
    S.check_structure(TOPIC, "abc", [2, 1, 2])


def test_keys_match_independent_scaling_over_300_seeds(questions):
    for q in questions:
        a1, a2, h1, v1 = _facts(q)
        ratio = Fraction(a2, a1)
        k = Fraction(isqrt(ratio.numerator), isqrt(ratio.denominator))
        assert k * k == ratio and k > 1
        assert q.parts[0].answer_text == _text(k), q.stem
        assert int(q.parts[1].answer_text) == k * h1, q.stem
        assert int(q.parts[2].answer_text) == k**3 * v1, q.stem


def test_grading_accepts_key_and_rejects_every_distractor():
    S.check_grading(TOPIC, SKILL_ID, 120)


def test_distractors_are_tagged_and_formatted():
    S.check_tags(TOPIC)


def _half_up(value: Fraction) -> int:
    return int(value + Fraction(1, 2)) if value >= 0 else -int(-value + Fraction(1, 2))


def test_follow_through_from_the_area_ratio_used_as_the_scale_factor(questions):
    """A learner who gives the area ratio as the scale factor and carries it through earns (b) and (c)."""
    checked = 0
    for q in questions[:100]:
        _, _, h1, v1 = _facts(q)
        wrong = Fraction(q.parts[0].distractors[0])  # first offered: the area ratio
        reply = {
            "a": q.parts[0].distractors[0],
            "b": str(_half_up(wrong * h1)),
            "c": str(_half_up(wrong**3 * v1)),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (q.stem, reply)
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (q.stem, reply)
        checked += 1
    assert checked == 100


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "9/4", "b": "17", "c": "23"})
        assert result.marks == 0 and not result.correct


def test_answer_space_over_300_seeds():
    S.check_answer_space(TOPIC)


def test_clean_and_deterministic():
    S.check_clean_and_deterministic(TOPIC)
    assert make(TOPIC, seed=7, difficulty="exam").to_dict() == make(TOPIC, seed=7, difficulty="exam").to_dict()
