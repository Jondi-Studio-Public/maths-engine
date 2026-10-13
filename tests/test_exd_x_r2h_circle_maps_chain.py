"""Tests for the multi-part exam source `exd_x_r2h_circle_maps_chain` (skill `ucv1-fundamental-group-map`).

The oracle reads the two powers, the winding count of the test loop and the target out of the rendered text, and
counts windings by summing the wrapped argument increments of the sampled loop.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen.engine import grade
from tests.exd_x_r2h_support import exam, follow_through_check, skill_of, standard_checks
import cmath
import math

TOPIC = "exd_x_r2h_circle_maps_chain"
SKILL_ID = "ucv1-fundamental-group-map"
HARD_TOPIC = "p15b_fundamental_group_map"
LABELS = list("abcd")
MARKS = [2, 1, 1, 2]
FT_LABELS = ["b", "c", "d"]


def _exponent(text, name):
    m = re.search(name + r"\(z\) = z(?:\^\(?(-?\d+)\)?)?", text)
    return int(m.group(1)) if m.group(1) else 1


def parse(q):
    m, n = _exponent(q.text, "f"), _exponent(q.text, "g")
    j = int(re.search(r"\[lambda\] = (\d+)", q.parts[1].prompt).group(1))
    t = re.search(r"winds (-?\d+) times|\] = (-?\d+) in Z|sends r to (-?\d+)", q.parts[3].prompt)
    return m, n, j, int(next(g for g in t.groups() if g is not None))


def winding(m, n, r):
    """Winding number of t -> g(f(exp(2 pi i r t))), counted from the argument increments of a fine sampling."""
    steps = 8 * (abs(r * m * n) + 2)
    total, previous = 0.0, 1 + 0j
    for i in range(1, steps + 1):
        z = cmath.exp(2j * math.pi * r * i / steps)
        value = (z**m) ** n
        total += cmath.phase(value / previous)
        previous = value
    return round(total / (2 * math.pi))


def oracle(q):
    m, n, j, target = parse(q)
    k = winding(m, n, 1)
    r = target // k
    assert k * r == target and winding(m, n, r) == target
    return {"a": str(k), "b": str(winding(m, n, j)), "c": str(k * k), "d": str(r)}


def test_follow_through_from_wrong_k():
    def wrong_chain(q):
        _, _, j, target = parse(q)
        # Keep away from a carried half-integer, where the rounding direction is the grader's business.
        wrong = next(
            (int(d) for d in q.parts[0].distractors if (2 * target) % int(d) != 0 or target % int(d) == 0),
            int(q.parts[0].distractors[0]),
        )
        replies = {"a": str(wrong), "b": str(wrong * j), "c": str(wrong * wrong), "d": str(round(target / wrong))}
        return "a", replies, 4

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_follow_through_from_wrong_winding():
    def wrong_chain(q):
        replies = {label: p.answer_text for label, p in zip(LABELS, q.parts)}
        replies["b"] = q.parts[1].distractors[0]
        return "b", replies, 5

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_all_wrong_earns_nothing():
    skill = skill_of(SKILL_ID)
    for s in range(30):
        q = exam(TOPIC, s)
        assert grade(skill, q, {"a": "1009", "b": "211", "c": "-307", "d": "401"}).marks == 0


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


def test_chain_is_not_copyable():
    """No part repeats an earlier answer or follows through as a mere restatement, and junk replies earn nothing."""
    from tests.exd_junk_probe import assert_chain_is_not_copyable

    assert_chain_is_not_copyable(TOPIC, SKILL_ID)
