"""Exam-tier checks for `f10-congruence-similarity-criteria`: SAS similarity to find a side, the third side, the perimeter."""

import re
from fractions import Fraction

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
    text_of,
)

TOPIC = "exd_x_sc2_similar_sas"
SKILL = "f10-congruence-similarity-criteria"


def _facts(q):
    text = text_of(q)
    ab, bc, ac = (int(n) for n in re.search(r"AB = (\d+) \w+, BC = (\d+) \w+ and AC = (\d+)", text).groups())
    given = re.search(r"angle Q = \d+° and (PQ|QR) = (\d+)", text)
    return ab, bc, ac, given.group(1), int(given.group(2))


def _solve(q):
    ab, bc, ac, side, value = _facts(q)
    ratio = Fraction(value, ab if side == "PQ" else bc)
    pq, qr = (value, bc * ratio) if side == "PQ" else (ab * ratio, value)
    return ab, bc, ac, side, value, ratio, pq, qr, ac * ratio


def oracle(q):
    ab, bc, ac, side, value, ratio, pq, qr, pr = _solve(q)
    asked = qr if side == "PQ" else pq
    assert asked.denominator == pr.denominator == 1
    return {"a": str(asked), "b": str(pr), "c": str(pq + qr + pr)}


def wrong_chain(q):
    """A wrong missing side, then PR and the perimeter worked from it; None when those are not whole numbers."""
    ab, bc, ac, side, value, ratio, pq, qr, pr = _solve(q)
    asked = qr if side == "PQ" else pq
    other = bc if side == "PQ" else ab
    for delta in range(1, 40):
        wrong = asked + delta
        new_pr = Fraction(ac) * wrong / other
        if new_pr.denominator == 1:
            return {"a": str(wrong), "b": str(new_pr), "c": str(value + wrong + new_pr)}
    return None


def test_no_text_defects():
    check_no_text_defects(TOPIC)


def test_variety_and_new_shapes():
    check_variety(TOPIC, lambda s: make("f10c_congruence_criteria", seed=s, difficulty="hard"))


def test_wired():
    check_wired(SKILL, TOPIC)


def test_oracle_agrees():
    check_oracle(TOPIC, oracle)


def test_answer_space():
    check_answer_space(TOPIC)


def test_deterministic():
    check_deterministic(TOPIC)


def test_key_accepted_distractors_rejected():
    check_key_accepted_distractors_rejected(SKILL, TOPIC)


def test_follow_through_from_wrong_missing_side():
    check_follow_through(SKILL, TOPIC, wrong_chain)


def test_stated_angle_agrees_with_the_sides():
    import math

    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam")
        ab, bc, ac, _, _ = _facts(q)
        angle = int(re.search(r"angle B = (\d+)°", q.stem).group(1))
        cosine = (ab * ab + bc * bc - ac * ac) / (2 * ab * bc)
        assert abs(cosine - math.cos(math.radians(angle))) < 1e-9, q.stem


def test_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL, TOPIC, {"a": "901", "b": "902", "c": "903"})
