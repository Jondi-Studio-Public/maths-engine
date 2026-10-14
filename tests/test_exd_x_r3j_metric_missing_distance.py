"""Exam source of `ucv1-metric-axiom-check`: the values of one unknown distance that make a metric."""

from __future__ import annotations

import itertools
import re

from exd_r3j_support import (
    canonical,
    check_answer_space,
    check_deterministic,
    check_distractors,
    check_full_marks,
    check_shapes,
    check_source_wired,
    skill,
)

from mathgen import make
from mathgen.engine import grade

TOPIC = "exd_x_r3j_metric_missing_distance"
SKILL_ID = "ucv1-metric-axiom-check"


def _is_metric(known: dict, t: int) -> bool:
    points = "ABCD"

    def d(x, y):
        if x == y:
            return 0
        pair = frozenset((x, y))
        return t if pair == frozenset("AD") else known[pair]

    positive = all(d(x, y) > 0 for x, y in itertools.permutations(points, 2))
    triangle = all(d(x, z) <= d(x, y) + d(y, z) for x, y, z in itertools.product(points, repeat=3))
    return positive and triangle


def _oracle(q):
    values = {a + b: int(v) for a, b, v in re.findall(r"d\(([A-D]), ([A-D])\) = (\d+)", q.stem)}
    known = {frozenset(k): v for k, v in values.items()}
    assert len(known) == 5
    good = [t for t in range(1, 300) if _is_metric(known, t)]
    assert good == list(range(good[0], good[-1] + 1))  # an interval, as the generator relies on
    return {"a": str(good[0]), "b": str(good[-1]), "c": str(len(good))}


def test_source_wired():
    check_source_wired(SKILL_ID, TOPIC)


def test_shape_and_marks():
    for s in range(60):
        q = make(TOPIC, seed=s)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 2, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompt_shapes_new_against_hard_tier():
    check_shapes(TOPIC, "p15b_metric_axiom_check", {"difficulty": "hard"})


def test_key_rederived_from_prompt():
    for s in range(300):
        q = make(TOPIC, seed=s)
        assert _oracle(q) == canonical(q), (s, q.text)


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_bounds():
    sk = skill(SKILL_ID)
    done = 0
    for s in range(100):
        q = make(TOPIC, seed=s)
        canon = canonical(q)
        lo, hi = int(canon["a"]) - 2, int(canon["b"]) + 3
        count = hi - lo + 1
        reply = {"a": str(lo), "b": str(hi), "c": str(count)}
        if lo <= 0 or len({lo, hi, count}) < 3:
            continue
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 0
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through, (s, reply)
        done += 1
    assert done >= 60


def test_all_wrong_earns_nothing():
    q = make(TOPIC, seed=3)
    assert grade(skill(SKILL_ID), q, {"a": "1", "b": "99", "c": "50"}).marks == 0


def test_deterministic():
    check_deterministic(TOPIC)
