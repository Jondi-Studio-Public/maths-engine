"""Exam-tier checks for `f10-enlargement-invariants`: scale factor from two areas, image perimeter, gap PP'."""

import re
from fractions import Fraction

import sympy

from mathgen import make
from tests.exd_x_sc2_support import (
    check_all_wrong_earns_nothing,
    check_answer_space,
    check_deterministic,
    check_follow_through,
    check_key_accepted_distractors_rejected,
    check_no_text_defects,
    check_oracle,
    check_variety,
    check_wired,
    decimal_text,
    text_of,
)

TOPIC = "exd_x_sc2_enlargement_chain"
SKILL = "f10-enlargement-invariants"


def _facts(q):
    text = text_of(q)
    length, width = (int(n) for n in re.search(r"measuring (\d+) cm by (\d+) cm", text).groups())
    area = Fraction(re.search(r"area of (\d+\.\d\d) cm²", text).group(1))
    dist = int(re.search(r"(\d+) cm from O", text).group(1))
    return length, width, area, dist


def _k(length, width, area):
    root = sympy.sqrt(sympy.Rational(area.numerator, area.denominator) / (length * width))
    assert root.is_Rational, root
    return Fraction(int(root.p), int(root.q))


def oracle(q):
    length, width, area, dist = _facts(q)
    k = _k(length, width, area)
    return {
        "a": decimal_text(k, 1),
        "b": decimal_text(k * 2 * (length + width), 1),
        "c": decimal_text(abs(k - 1) * dist, 1),
    }


def wrong_chain(q):
    length, width, area, dist = _facts(q)
    k = _k(length, width, area) + Fraction(7, 10)
    return {
        "a": decimal_text(k, 1),
        "b": decimal_text(k * 2 * (length + width), 1),
        "c": decimal_text(abs(k - 1) * dist, 1),
    }


def test_variety_and_new_shapes():
    check_variety(TOPIC, lambda s: make("enlargement_invariants", seed=s, difficulty="hard"))


def test_wired():
    check_wired(SKILL, TOPIC)


def test_oracle_agrees():
    check_oracle(TOPIC, oracle)


def test_answer_space():
    check_answer_space(TOPIC)


def test_key_accepted_distractors_rejected():
    check_key_accepted_distractors_rejected(SKILL, TOPIC)


def test_follow_through_from_wrong_scale_factor():
    check_follow_through(SKILL, TOPIC, wrong_chain)


def test_both_enlargements_and_reductions_appear():
    factors = [Fraction(make(TOPIC, seed=s, difficulty="exam").parts[0].answer_text) for s in range(200)]
    assert any(k < 1 for k in factors) and any(k > 1 for k in factors)


def test_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL, TOPIC, {"a": "9.9", "b": "901.1", "c": "902.2"})
