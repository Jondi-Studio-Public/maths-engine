"""Exam-tier checks for `f10-geometric-construction-algorithm`: refined PM, then MQ, then AQ."""

import math
import re

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

TOPIC = "exd_x_sc2_construction_chain"
SKILL = "f10-geometric-construction-algorithm"


def _facts(q):
    text = text_of(q)
    length = int(re.search(r"segment \w\w, (\d+) mm", text).group(1))
    pq = int(re.search(r"(?:so that \w\w = |, | point \w )(\d+) mm(?: from \w)?\.", text.split("Step 4:")[1]).group(1))
    refined = int(re.search(r"compasses are set to (\d+) mm\.", text).group(1))
    return length, pq, refined


def oracle(q):
    length, pq, refined = _facts(q)
    half = length // 2
    pm = math.isqrt(refined**2 - half**2)
    assert pm * pm == refined**2 - half**2
    mq = pm - pq
    aq_sq = half**2 + mq**2
    aq = math.isqrt(aq_sq)
    assert aq * aq == aq_sq
    return {"a": str(pm), "b": str(mq), "c": str(aq)}


def wrong_chain(q):
    """A wrong PM, then MQ and AQ worked from it; None when AQ would not be a whole number of millimetres."""
    length, pq, refined = _facts(q)
    half = length // 2
    right = math.isqrt(refined**2 - half**2)
    for delta in range(1, 80):
        pm = right + delta
        mq = pm - pq
        aq_sq = half**2 + mq**2
        if mq > 0 and math.isqrt(aq_sq) ** 2 == aq_sq:
            return {"a": str(pm), "b": str(mq), "c": str(math.isqrt(aq_sq))}
    return None


def test_variety_and_new_shapes():
    check_variety(TOPIC, lambda s: make("f10_geometric_construction_algorithm", seed=s, variant="propagation"))


def test_wired():
    check_wired(SKILL, TOPIC)


def test_oracle_agrees():
    check_oracle(TOPIC, oracle)


def test_answer_space():
    check_answer_space(TOPIC)


def test_key_accepted_distractors_rejected():
    check_key_accepted_distractors_rejected(SKILL, TOPIC)


def test_follow_through_from_wrong_pm():
    check_follow_through(SKILL, TOPIC, wrong_chain)


def test_refinement_changes_the_answer():
    """The original compass setting must give a different PM, so ignoring the refinement is a wrong answer."""
    for seed in range(150):
        q = make(TOPIC, seed=seed, difficulty="exam")
        length, pq, refined = _facts(q)
        original = int(re.search(r"Set the compasses to (\d+) mm", q.stem).group(1))
        assert original != refined
        assert original**2 - (length // 2) ** 2 != int(q.parts[0].answer_text) ** 2


def test_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL, TOPIC, {"a": "901", "b": "5", "c": "7"})
