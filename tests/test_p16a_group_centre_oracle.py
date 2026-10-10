"""Independent oracle for the `p16a_group_centre` questions: it re-derives each answer from the rendered question
text.

Parses only ``question.text`` / ``question.answer_text`` and re-derives the
answer from scratch. Does NOT import
``mathgen.generators.p16a_group_centre_build``.
"""

from __future__ import annotations

import re
import statistics
from fractions import Fraction

import pytest

from mathgen import make

GC_TOPIC = "p16a_group_centre_difference"
TV_TOPIC = "p16a_trend_value"

_LIST_RE = re.compile(r"(?:-?\d+, )+-?\d+")
_TABLE_RE = re.compile(r"(-?\d+) \((\d+) times\)")


def _extract_two_lists(text: str) -> tuple[list[int], list[int]]:
    matches = _LIST_RE.findall(text)
    assert len(matches) == 2, text
    return (
        [int(x) for x in matches[0].split(", ")],
        [int(x) for x in matches[1].split(", ")],
    )


def _extract_two_tables(text: str) -> tuple[dict, dict]:
    # Two "Results for X: v (f times), ..." clauses, in order.
    parts = text.split("Results for")
    assert len(parts) == 3, text
    tables = []
    for part in parts[1:]:
        pairs = _TABLE_RE.findall(part)
        tables.append({int(v): int(f) for v, f in pairs})
    return tables[0], tables[1]


@pytest.mark.parametrize("seed", range(60))
def test_group_centre_easy_oracle(seed):
    q = make(GC_TOPIC, seed=seed, difficulty="easy")
    a, b = _extract_two_lists(q.text)
    expected = Fraction(sum(a), len(a)) - Fraction(sum(b), len(b))
    given = Fraction(q.answer["p"], q.answer["q"])
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


@pytest.mark.parametrize("seed", range(60))
def test_group_centre_medium_oracle(seed):
    q = make(GC_TOPIC, seed=seed, difficulty="medium")
    a, b = _extract_two_lists(q.text)
    expected = Fraction(statistics.median(a)).limit_denominator(1000) - Fraction(
        statistics.median(b)
    ).limit_denominator(1000)
    given = Fraction(q.answer["p"], q.answer["q"])
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


@pytest.mark.parametrize("seed", range(60))
def test_group_centre_hard_oracle(seed):
    q = make(GC_TOPIC, seed=seed, difficulty="hard")
    table_a, table_b = _extract_two_tables(q.text)
    flat_a = [v for v, f in table_a.items() for _ in range(f)]
    flat_b = [v for v, f in table_b.items() for _ in range(f)]
    expected = Fraction(statistics.mean(flat_a)).limit_denominator(10000) - Fraction(
        statistics.mean(flat_b)
    ).limit_denominator(10000)
    given = Fraction(q.answer["p"], q.answer["q"])
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


@pytest.mark.parametrize("seed", range(60))
def test_group_centre_exam_oracle(seed):
    # Redesigned: neither tier names its measure any more. The measure is implied by the REASON
    # given instead: an outlier ("...not affected...") means median, "none of the data..." means
    # mean.
    q = make(GC_TOPIC, seed=seed, difficulty="exam")
    a, b = _extract_two_lists(q.text)
    given = Fraction(q.answer["p"], q.answer["q"])
    if "outlier" in q.text and "not affected" in q.text:
        expected = Fraction(statistics.median(a)).limit_denominator(1000) - Fraction(
            statistics.median(b)
        ).limit_denominator(1000)
    else:
        assert "None of the data" in q.text, q.text
        expected = Fraction(sum(a), len(a)) - Fraction(sum(b), len(b))
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


# --------------------------------------------------------------------------
# p16a_trend_value
# --------------------------------------------------------------------------

_EQ_RE = re.compile(r"T\(t\) = (-?\d+) \+ \(?(-?\d+(?:/\d+)?)\)?×t")
_TSTAR_DIRECT_RE = re.compile(r"Find the trend value T\((\d+)\)")
_PERIOD_RE = re.compile(r"for (\w+) (\d+) of Year (\d+)")
_PER_GROUP_RE = re.compile(r"each Year has (\d+) \w+s")
_TWO_POINTS_RE = re.compile(r"trend value in \w+ (\d+) is (-?\d+\.\d+) and in \w+ (\d+) is (-?\d+\.\d+)")
_TSTAR_HARD_RE = re.compile(r"Find the trend value in \w+ (\d+), to 1")

# The seasonal_two_point (exam) format: two period-labelled trend values,
# "X units per Year", and a third period label as the target.
_SEASONAL_GIVEN_RE = re.compile(
    r"the trend value for (\w+) (\d+) of Year (\d+) is (-?\d+\.\d+) and the trend "
    r"value for \w+ (\d+) of Year (\d+) is (-?\d+\.\d+)"
)
_SEASONAL_PER_GROUP_RE = re.compile(r"(\d+) (\w+)s per Year")
_SEASONAL_TARGET_RE = re.compile(r"Find the trend value for \w+ (\d+) of Year (\d+), to 1 decimal place")


@pytest.mark.parametrize("seed", range(60))
def test_trend_value_easy_oracle(seed):
    q = make(TV_TOPIC, seed=seed, difficulty="easy")
    m = _EQ_RE.search(q.text)
    assert m, q.text
    a = Fraction(int(m.group(1)))
    b = Fraction(m.group(2))
    t_star = int(_TSTAR_DIRECT_RE.search(q.text).group(1))
    expected = a + b * t_star
    assert float(expected) == pytest.approx(float(q.answer_text), abs=0.05)


@pytest.mark.parametrize("seed", range(60))
def test_trend_value_medium_oracle(seed):
    q = make(TV_TOPIC, seed=seed, difficulty="medium")
    m = _EQ_RE.search(q.text)
    assert m, q.text
    a = Fraction(int(m.group(1)))
    b = Fraction(m.group(2))
    per_group = int(_PER_GROUP_RE.search(q.text).group(1))
    pm = _PERIOD_RE.search(q.text)
    sub, year = int(pm.group(2)), int(pm.group(3))
    t_star = (year - 1) * per_group + sub
    expected = a + b * t_star
    assert float(expected) == pytest.approx(float(q.answer_text), abs=0.05)


@pytest.mark.parametrize("seed", range(60))
def test_trend_value_hard_oracle(seed):
    q = make(TV_TOPIC, seed=seed, difficulty="hard")
    m = _TWO_POINTS_RE.search(q.text)
    assert m, q.text
    t1, y1, t2, y2 = int(m.group(1)), Fraction(m.group(2)), int(m.group(3)), Fraction(m.group(4))
    t_star = int(_TSTAR_HARD_RE.search(q.text).group(1))
    b = (y2 - y1) / (t2 - t1)
    a = y1 - b * t1
    expected = a + b * t_star
    assert float(expected) == pytest.approx(float(q.answer_text), abs=0.05)


@pytest.mark.parametrize("seed", range(60))
def test_trend_value_exam_oracle(seed):
    # seasonal_two_point: two NON-consecutive period-labelled trend values, requiring label->t
    # conversion for both (via the stated "N units per Year") plus a genuine two-point line fit,
    # evaluated at a third period label that must also be converted.
    q = make(TV_TOPIC, seed=seed, difficulty="exam")
    gm = _SEASONAL_GIVEN_RE.search(q.text)
    assert gm, q.text
    label_word1, sub1, year1, y1, sub2, year2 = (
        gm.group(1),
        int(gm.group(2)),
        int(gm.group(3)),
        Fraction(gm.group(4)),
        int(gm.group(5)),
        int(gm.group(6)),
    )
    y2 = Fraction(gm.group(7))
    pgm = _SEASONAL_PER_GROUP_RE.search(q.text)
    assert pgm, q.text
    per_group = int(pgm.group(1))
    tm = _SEASONAL_TARGET_RE.search(q.text)
    assert tm, q.text
    sub3, year3 = int(tm.group(1)), int(tm.group(2))

    t1 = (year1 - 1) * per_group + sub1
    t2 = (year2 - 1) * per_group + sub2
    t_star = (year3 - 1) * per_group + sub3
    assert t2 != t1
    b = (y2 - y1) / (t2 - t1)
    a = y1 - b * t1
    expected = a + b * t_star
    assert float(expected) == pytest.approx(float(q.answer_text), abs=0.05)
    for d in q.distractors:
        assert abs(float(d) - float(expected)) > 0.051
