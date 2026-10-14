"""Tests the exam chain for the Cartesian equation of a plane through three points: a, b, d and a missing coordinate."""

from __future__ import annotations

import re

import pytest
import sympy

from mathgen import make
from tests.exd_x_r2c_support import (
    canonical,
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through,
    check_grading,
    check_shapes,
    check_wired,
    exam,
)

TOPIC = "exd_x_r2c_vector_plane"
SKILL_ID = "senior-determine-vector-and-cartesian-equations-of-a-plane"
HARD_TOPIC = "determine_vector_and_cartesian_equations_of_a_plane"
a, b, d, k = sympy.symbols("a b d k")


def oracle(q):
    """a, b, d from the three printed points (a linear system), then k from the printed P."""
    points = [tuple(int(v) for v in match) for match in re.findall(r"[ABC]\((-?\d+), (-?\d+), (-?\d+)\)", q.stem)]
    assert len(points) == 3, q.stem
    solved = sympy.solve([a * x + b * y + z - d for x, y, z in points], [a, b, d], dict=True)[0]
    x1, y1 = (int(v) for v in re.search(r"P\((-?\d+), (-?\d+), k\)", q.parts[3].prompt).groups())
    k_val = solved[d] - solved[a] * x1 - solved[b] * y1
    return solved[a], solved[b], solved[d], k_val


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert all(p.answer_kind == "fraction" for p in q.parts)
        assert [p.marks for p in q.parts] == [2, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        expected = oracle(q)
        got = tuple(sympy.Rational(p.answer_text) for p in q.parts)
        if got != expected:
            bad.append((q.stem, canonical(q), expected))
    assert bad == [], bad[:2]


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_follow_through_from_wrong_a():
    """A wrong a carried into d and k earns both later parts."""

    def carried(q):
        points = [tuple(int(v) for v in m) for m in re.findall(r"[ABC]\((-?\d+), (-?\d+), (-?\d+)\)", q.stem)]
        x1, y1 = (int(v) for v in re.search(r"P\((-?\d+), (-?\d+), k\)", q.parts[3].prompt).groups())
        b_val = sympy.Rational(q.parts[1].answer_text)
        for shift in (1, 2, 3, 5):
            a_wrong = sympy.Rational(q.parts[0].answer_text) + shift
            d_wrong = a_wrong * points[0][0] + b_val * points[0][1] + points[0][2]
            k_wrong = d_wrong - a_wrong * x1 - b_val * y1
            if len({a_wrong, d_wrong, k_wrong, b_val}) == 4 and 0 not in (a_wrong, d_wrong, k_wrong):
                break
        return "a", str(a_wrong), {"c": str(d_wrong), "d": str(k_wrong)}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)
