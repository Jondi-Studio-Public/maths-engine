"""Tests for the multi-part exam source `exd_x_r2h_abelian_orders_chain` (skill `ucv1-structure-classification`).

The oracle reads the two cyclic factors and the order asked about out of the rendered text and finds every element
order of the product group by brute force.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.engine import grade
from tests.exd_x_r2h_support import exam, follow_through_check, skill_of, standard_checks
from math import gcd

TOPIC = "exd_x_r2h_abelian_orders_chain"
SKILL_ID = "ucv1-structure-classification"
HARD_TOPIC = "p15a_structure_classification"
LABELS = list("abcd")
MARKS = [1, 2, 1, 2]
FT_LABELS = ["c", "d"]


def parse(q):
    m, n = (int(v) for v in re.search(r"Z(\d+) x Z(\d+)", q.text).groups())
    d = int(re.search(r"order (?:exactly )?(\d+)", q.parts[1].prompt).group(1))
    return m, n, d


def element_orders(m, n):
    orders = []
    for a in range(m):
        for b in range(n):
            oa, ob = m // gcd(a, m), n // gcd(b, n)
            orders.append(oa * ob // gcd(oa, ob))
    return orders


def oracle(q):
    m, n, d = parse(q)
    orders = element_orders(m, n)
    count = orders.count(d)
    phi = sum(1 for k in range(1, d + 1) if gcd(k, d) == 1)
    assert count % phi == 0
    return {"a": str(max(orders)), "b": str(count), "c": str(count // phi), "d": str(count - phi)}


def _phi(q):
    return int(q.parts[1].answer_text) // int(q.parts[2].answer_text)


def test_follow_through_from_wrong_count():
    def wrong_chain(q):
        phi = _phi(q)
        # Avoid a carried half-integer, where the rounding direction is the grader's business.
        wrong = next(int(d) for d in q.parts[1].distractors if (int(d) % phi) * 2 != phi)
        replies = {
            "a": q.parts[0].answer_text,
            "b": str(wrong),
            "c": str(round(Fraction(wrong, phi))),
            "d": str(wrong - phi),
        }
        return "b", replies, 4

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_all_wrong_earns_nothing():
    skill = skill_of(SKILL_ID)
    for s in range(30):
        q = exam(TOPIC, s)
        assert grade(skill, q, {"a": "1009", "b": "211", "c": "307", "d": "401"}).marks == 0


CHECKS = standard_checks(TOPIC, SKILL_ID, oracle, LABELS, MARKS, FT_LABELS, HARD_TOPIC)


def test_wired():
    CHECKS["wired"]()


def test_shape():
    CHECKS["shape"]()


def test_prompt_shapes():
    CHECKS["shapes"]()


def test_oracle_agrees_over_300_seeds():
    CHECKS["oracle_agrees"]()


def test_composed_answer_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_tagged_and_rejected():
    CHECKS["distractors_rejected"]()


def test_answer_space():
    CHECKS["answer_space"]()


def test_deterministic():
    CHECKS["deterministic"]()
