"""Tests the exam chain for reading a box plot: IQR, values between the quartiles, upper fence, outlier verdict."""

from __future__ import annotations

import re
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

TOPIC = "exd_x_w12a_box_plot_read_chain"
SKILL_ID = "f10-read-a-box-plot"


def _further_value(q) -> int:
    """The one number in the sentence that ends "Is it an outlier?"."""
    last = q.text.split("(d) ")[1]
    return int(re.search(r"\d+", last).group())


def oracle(q):
    """(a) to (d) from the plotted quartiles and the two numbers in the prompts, with exact fractions."""
    plot = q.figure["plots"][0]
    q1, q3 = Fraction(plot["q1"]), Fraction(plot["q3"])
    iqr = q3 - q1
    count = int(re.search(r"(?:There were|Of the) (\d+)", q.text).group(1))
    value = Fraction(_further_value(q))
    fence = q3 + Fraction(3, 2) * iqr
    return iqr, Fraction(count, 2), fence, value > fence


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_source_wired():
    check_wired(SKILL_ID, TOPIC)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    check_shapes(TOPIC, "stat5_box_plot_read", {"tier": "hard"})


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert q.figure["kind"] == "box_plot" and len(q.figure["plots"]) == 1
        assert q.figure["plots"][0]["outliers"] == []
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "mcq"]
        assert [p.marks for p in q.parts] == [2, 1, 2, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, False]
        assert all(len(p.distractors) == 3 for p in q.parts)


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        iqr, between, fence, outlier = oracle(q)
        a, b, c, d = canonical(q).values()
        verdict = d.startswith("Yes")
        if (Fraction(a), Fraction(b), Fraction(c), verdict) != (iqr, between, fence, outlier):
            bad.append((q.text, canonical(q)))
    assert bad == [], bad[:2]


def test_further_value_is_above_the_whisker_and_both_verdicts_occur(questions):
    verdicts = set()
    for q in questions[:300]:
        plot = q.figure["plots"][0]
        value = _further_value(q)
        assert value > plot["max"]
        verdicts.add(oracle(q)[3])
    assert verdicts == {True, False}


def test_fence_is_whole_and_distractors_reach_the_wrong_verdict_or_reason(questions):
    """The key's verdict is the oracle's, and no distractor states the same verdict with the same reason."""
    for q in questions[:120]:
        iqr, _, fence, outlier = oracle(q)
        assert fence.denominator == 1 and iqr % 2 == 0
        (d_part,) = [p for p in q.parts if p.label == "d"]
        assert d_part.answer_text.startswith("Yes") == outlier
        assert d_part.answer_text not in d_part.distractors
        assert len(set(d_part.distractors)) == 3


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_iqr():
    """An IQR two too big moves the fence by three: the upper quartile plus 1.5 of the learner's own IQR."""

    def carried(q):
        iqr, _, fence, _ = oracle(q)
        wrong = iqr + 2
        later = {"c": str(round_half_up(fence + Fraction(3, 2) * 2, 0))}
        return "a", str(wrong), later

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, tier="nonsense")
