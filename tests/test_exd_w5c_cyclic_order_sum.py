"""Cyclic-group-element-order hard tier: order of a sum."""

from __future__ import annotations

import re
from math import gcd

from mathgen.grading import GRADERS
from mathgen import make
from mathgen.text_quality import prompt_shape

TOPIC = "cyclic_group_element_order"
PATTERN = re.compile(r"Z/(\d+)Z, find the order of the element \[(\d+)\] \+ \[(\d+)\]\.")


def _order(x, n):
    return n // gcd(x % n, n)


def test_sum_kind_matches_recomputation_from_prompt():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard", kind="sum")
        n, a, b = (int(g) for g in PATTERN.search(q.text).groups())
        assert n >= 8 and a + b >= n and (a + b) % n != 0
        expected = _order(a + b, n)
        current, count = 0, 0
        while True:
            current = (current + a + b) % n
            count += 1
            if current == 0:
                break
        assert count == expected == int(q.answer_text)
        oa, ob = _order(a, n), _order(b, n)
        rivals = {oa * ob // gcd(oa, ob), oa + ob, oa * ob, max(oa, ob)}
        assert expected not in rivals


def test_default_path_unchanged_and_sum_differs():
    for seed in range(20):
        omitted = make(TOPIC, seed=seed, difficulty="hard")
        explicit = make(TOPIC, seed=seed, difficulty="hard", kind=None)
        assert omitted.text == explicit.text and omitted.answer_text == explicit.answer_text
        assert " + " not in omitted.text
        assert make(TOPIC, seed=seed, difficulty="hard", kind="sum").text != omitted.text


def test_distractors_clear_the_real_grader():
    grader = GRADERS["number"]
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard", kind="sum")
        assert grader(q, q.answer_text).correct
        for distractor in q.distractors:
            assert distractor != q.answer_text
            assert not grader(q, distractor).correct, (seed, distractor, q.answer_text)


def test_prompt_shape_differs_from_default():
    default = make(TOPIC, seed=3, difficulty="hard")
    summed = make(TOPIC, seed=3, difficulty="hard", kind="sum")
    assert prompt_shape(default) != prompt_shape(summed)
