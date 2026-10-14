"""Exam chain of the population-compounding skill: rate, starting value, then the first period past a threshold."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from tests import exd_x_r3d_support as S

TOPIC = "exd_x_r3d_compound_two_readings"
SKILL = S.load_skill("senior-consider-similar-problems-involving-compounding-for-example-population")


def _read(q):
    p1, p3 = (
        int(v.replace(",", "")) for v in re.search(r"is ([\d,]+) after 1 \w+,? and ([\d,]+) after 3", q.stem).groups()
    )
    prompt_c = q.parts[2].prompt
    threshold = int(re.search(r"(?:exceed|fall below) ([\d,]+)\?", prompt_c).group(1).replace(",", ""))
    up = "increases" in q.parts[0].prompt
    return p1, p3, threshold, up


def _root(ratio: Fraction) -> Fraction:
    num, den = ratio.numerator, ratio.denominator
    from math import isqrt

    assert isqrt(num) ** 2 == num and isqrt(den) ** 2 == den, ratio
    return Fraction(isqrt(num), isqrt(den))


def oracle(q):
    p1, p3, threshold, up = _read(q)
    mult = _root(Fraction(p3, p1))
    rate = (mult - 1) * 100 if up else (1 - mult) * 100
    start = Fraction(p1) / mult
    value, n = start, 0
    while (value <= threshold) if up else (value >= threshold):
        value *= mult
        n += 1
        assert n < 100
    return {"a": str(int(rate)), "b": str(int(start)), "c": str(n)}


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_growth_and_decay_both_occur(questions):
    assert {_read(q)[3] for q in questions} == {True, False}


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_rate(questions):
    import math

    def chain(q):
        p1, p3, threshold, up = _read(q)
        rate = next((int(d) for d in q.parts[0].distractors if 0 < int(d) < 100), None)
        if rate is None:
            return None
        mult = 1 + rate / 100 if up else 1 - rate / 100
        start = math.floor(p1 / mult + 0.5)
        mult_exact = Fraction(100 + rate if up else 100 - rate, 100)
        value, n = Fraction(start), 0
        while (value <= threshold) if up else (value >= threshold):
            value *= mult_exact
            n += 1
            assert n < 200
        if n < 1 or start in (rate, p1) or n in (rate, start):
            return None
        return {"a": str(rate), "b": str(start), "c": str(n)}, {"b", "c"}

    S.check_follow_through(SKILL, questions, chain, n=150)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "999", "b": "998", "c": "997"})


def test_deterministic():
    S.check_deterministic(TOPIC)
