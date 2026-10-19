"""Tests the exam chain for ``f10-stem-and-leaf-plots``: Q1, Q3, the interquartile range and the outlier boundary."""

from __future__ import annotations

import re
import statistics

import pytest

from mathgen import make
from tests.exd_x_r2c_support import (
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through,
    check_grading,
    check_shapes,
    check_wired,
    exam,
)

TOPIC = "exd_x_fc_stem_and_leaf"
SKILL_ID = "f10-stem-and-leaf-plots"


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(200)]


def values_of(q) -> list[int]:
    """The data, read from the plot rows in the stem only."""
    body = re.search(r"rows written as stem \| leaves: (.*?)\. (?:Leaves|The leaves)", q.stem).group(1)
    out = []
    for row in body.split("; "):
        stem, leaves = row.split(" | ")
        out += [10 * int(stem) + int(x) for x in leaves.split()]
    return sorted(out)


def oracle(q):
    """Q1, Q3, IQR and the boundary, with the median left out of both halves (the rule the stem states)."""
    v = values_of(q)
    n = len(v)
    assert n % 2 == 1
    q1, q3 = statistics.median(v[: n // 2]), statistics.median(v[n // 2 + 1 :])
    return q1, q3, q3 - q1, q3 + 1.5 * (q3 - q1)


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number"] * 4
        assert [p.marks for p in q.parts] == [2, 2, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert q.figure is None


def test_oracle_agrees_over_200_seeds(questions):
    bad = []
    for q in questions:
        keys = oracle(q)
        got = tuple(float(p.answer_text) for p in q.parts)
        if got != tuple(float(k) for k in keys):
            bad.append((q.stem, got, keys))
    assert bad == [], bad[:2]


def test_count_is_eleven_or_fifteen_and_iqr_even(questions):
    seen = set()
    for q in questions:
        seen.add(len(values_of(q)))
        assert int(q.parts[2].answer_text) % 2 == 0
    assert seen == {11, 15}


def test_both_quartiles_are_inside_the_data(questions):
    for q in questions:
        v = values_of(q)
        q1, q3 = int(q.parts[0].answer_text), int(q.parts[1].answer_text)
        assert v[0] < q1 < q3 < v[-1]


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_follow_through_from_wrong_lower_quartile():
    def carried(q):
        q3, iqr = int(q.parts[1].answer_text), int(q.parts[2].answer_text)
        # A carried IQR that repeats the wrong Q1 it came from is "the same number again", which the marker withholds.
        wrong_q1 = next(
            w for w in (int(q.parts[0].answer_text) + d for d in (-2, 2, -4, 4)) if q3 - w != w and q3 - w != q3
        )
        wrong_iqr = q3 - wrong_q1
        assert wrong_iqr != iqr
        return "a", str(wrong_q1), {"c": str(wrong_iqr), "d": str(q3 + 3 * wrong_iqr // 2)}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_follow_through_from_wrong_iqr():
    def carried(q):
        q3 = int(q.parts[1].answer_text)
        wrong_iqr = int(q.parts[2].answer_text) + 4
        return "c", str(wrong_iqr), {"d": str(q3 + 3 * wrong_iqr // 2)}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)
