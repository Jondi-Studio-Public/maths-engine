"""Tests the exam chain for reading a histogram: class frequency, total, median class, percentage of the data."""

from __future__ import annotations

import re
from bisect import bisect_right
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import round_half_up
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

TOPIC = "exd_x_w12a_histogram_chain"
SKILL_ID = "f7-read-a-histogram"


def oracle(q):
    """(a) to (d) from the drawn bars and the two ranges named in the prompts, by listing the data."""
    panel = q.figure["panels"][0]
    edges, freqs = panel["edges"], panel["frequencies"]
    (a_lo, _a_hi), (d_lo, d_hi) = re.findall(r"from (\d+) to less than (\d+)", q.text)
    data = sorted(edges[i] for i, f in enumerate(freqs) for _ in range(f))
    n = len(data)
    middle = data[(n - 1) // 2], data[n // 2]
    assert middle[0] == middle[1], "the two middle values must sit in one class"
    median_class_low = edges[bisect_right(edges, middle[0]) - 1]
    in_range = sum(1 for v in data if int(d_lo) <= v < int(d_hi))
    return freqs[edges.index(int(a_lo))], n, median_class_low, Fraction(100 * in_range, n)


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_source_wired():
    check_wired(SKILL_ID, TOPIC)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    check_shapes(TOPIC, "stat4_histogram", {"tier": "hard"})


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert q.figure["kind"] == "histogram"
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number"] * 4
        assert [p.marks for p in q.parts] == [1, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]
        assert all(len(p.distractors) == 3 and p.sensible_range == (0, None) for p in q.parts)


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        freq, total, median_low, pct = oracle(q)
        got = [Fraction(t) for t in canonical(q).values()]
        if got != [freq, total, median_low, pct]:
            bad.append((q.text, canonical(q)))
    assert bad == [], bad[:2]


def test_percentage_is_whole_and_total_is_one_of_four(questions):
    for q in questions[:300]:
        _, total, _, pct = oracle(q)
        assert total in (20, 25, 40, 50) and pct.denominator == 1 and 0 < pct < 100


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_total():
    """A wrong total carries into the percentage: the same count over the learner's own total."""

    def carried(q):
        _, total, _, pct = oracle(q)
        count = pct * total / 100
        wrong = total * 2
        return "b", str(wrong), {"d": str(round_half_up(Fraction(100 * count, wrong), 0))}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nonsense")
