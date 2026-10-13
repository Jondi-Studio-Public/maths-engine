"""Tests for the exam source of senior-implement-the-statistical-investigation-process-to-answer-questions-acmgm093.

The rate of change, the later trend value and the seasonal forecast are re-derived from the stem.
"""

from __future__ import annotations

import re
from fractions import Fraction

import mathgen.generators.exd_x_r2k_trend_forecast  # noqa: F401  (registers the topic)
from tests import exd_x_r2k_kit as kit

SKILL = "senior-implement-the-statistical-investigation-process-to-answer-questions-acmgm093"
TOPIC = "exd_x_r2k_trend_forecast"


def parse(q):
    stem = q.stem
    per_year = int(re.search(r"and (\d+) (?:quarters|months) in each year", stem).group(1))
    (q1, y1), (q2, y2) = (
        (int(a), Fraction(b))
        for a, b in re.findall(r"trend value for (?:quarter|month) (\d+) of Year 2 is (\d+\.\d)", stem)
    )
    qt = int(re.search(r"seasonal index for (?:quarter|month) (\d+) is", stem).group(1))
    s = Fraction(re.search(r"index for (?:quarter|month) \d+ is (\d\.\d\d)", stem).group(1))
    year = int(re.search(r"value for (?:quarter|month) \d+ of Year (\d+), correct", q.parts[1].prompt).group(1))
    return per_year, q1, y1, q2, y2, qt, s, year


def oracle(q) -> dict[str, str]:
    per_year, q1, y1, q2, y2, qt, s, year = parse(q)
    m = (y2 - y1) / (q2 - q1)
    t1 = per_year + q1
    tt = per_year * (year - 1) + qt
    trend = y1 + m * (tt - t1)
    return {"a": kit.fixed(m, 1), "b": kit.fixed(trend, 1), "c": kit.fixed(trend * s, 1)}


def test_wiring_and_shapes():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_oracle_agrees_over_300_seeds():
    bad = [s for s in range(300) if oracle(kit.exam_q(SKILL, s)) != kit.canon(kit.exam_q(SKILL, s))]
    assert bad == []


def test_quarters_and_months_both_occur():
    stems = " ".join(kit.exam_q(SKILL, s).stem for s in range(60))
    assert "quarters in each year" in stems and "months in each year" in stems


def test_grading_and_tags():
    kit.check_grading(SKILL)
    kit.check_tags(SKILL)


def test_answer_space():
    kit.check_answer_space(SKILL)


def test_follow_through_from_a_wrong_rate():
    def chain(q):
        per_year, q1, y1, q2, y2, qt, s, year = parse(q)
        c = kit.canon(q)
        m_wrong = Fraction(c["a"]) + Fraction(1, 2)
        if m_wrong == 0:  # a zero rate earns no follow-through, by design
            m_wrong = Fraction(c["a"]) + Fraction(7, 10)
        trend = y1 + m_wrong * (per_year * (year - 1) + qt - (per_year + q1))
        return "a", {
            **c,
            "a": kit.fixed(m_wrong, 1),
            "b": kit.fixed(trend, 1),
            "c": kit.fixed(kit.half_up(trend, 1) * s, 1),
        }

    kit.check_follow_through(SKILL, chain)
