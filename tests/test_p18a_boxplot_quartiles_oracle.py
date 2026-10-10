"""Seed count is `N_SEEDS` (default 250 per tier, >= 1000 questions over the
four tiers), overridable by `P18A_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re
import statistics
from fractions import Fraction

import pytest

from mathgen import make

TOPIC = "p18a_boxplot_quartiles"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("P18A_ORACLE_SEEDS", "250"))

_ANSWER_RE = re.compile(r"^Q1 = (-?\d+(?:\.\d+)?), Q3 = (-?\d+(?:\.\d+)?), IQR = (-?\d+(?:\.\d+)?)$")
_PLAIN_VALUE_LIST_RE = re.compile(r": ((?:-?\d+, )*-?\d+)\. Use the exclusive method")
_FREQ_PAIR_RE = re.compile(r"(-?\d+): (\d+)")
_GROUP_RE = re.compile(
    r"Group A: ([^.;]+)[.;] ?Group B: ([^.]+)\.|"
    r"group A: ([^;]+); group B: ([^.]+)\."
)


def _median(values: list[int]) -> Fraction:
    s = sorted(values)
    m = len(s)
    if m % 2 == 1:
        return Fraction(s[m // 2])
    return Fraction(s[m // 2 - 1] + s[m // 2], 2)


def _exclusive_quartiles(values: list[int]) -> tuple[Fraction, Fraction]:
    """A genuinely different route from the generator's own `_median_of` +
    index-slicing: builds the two halves by REMOVING one occurrence of the
    overall median value's position (using `statistics.median_low`/`median_high`
    style reasoning is avoided in favour of an explicit odd/even split, then
    hands each half to Python's own `statistics.median`)."""
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        lower, upper = s[: n // 2], s[n // 2 :]
    else:
        lower, upper = s[: n // 2], s[n // 2 + 1 :]
    return Fraction(statistics.median(lower)), Fraction(statistics.median(upper))


def _parse_plain_list(text: str) -> list[int]:
    m = _PLAIN_VALUE_LIST_RE.search(text)
    assert m is not None, f"oracle: no value list found in {text!r}"
    return [int(t) for t in m.group(1).split(", ")]


def _parse_frequency_table(text: str) -> list[int]:
    pairs = _FREQ_PAIR_RE.findall(text)
    assert pairs, f"oracle: no frequency table found in {text!r}"
    expanded = []
    for v, c in pairs:
        expanded.extend([int(v)] * int(c))
    return expanded


def _parse_groups(text: str) -> tuple[list[int], list[int]]:
    m = _GROUP_RE.search(text)
    assert m is not None, f"oracle: no two-group data found in {text!r}"
    groups = m.groups()
    a_text, b_text = (groups[0], groups[1]) if groups[0] is not None else (groups[2], groups[3])
    a = [int(t) for t in a_text.split(",")]
    b = [int(t) for t in b_text.split(",")]
    return a, b


def _oracle_triple(question, tier: str) -> tuple[Fraction, Fraction, Fraction]:
    if tier in ("easy", "medium"):
        values = _parse_plain_list(question.text)
    elif tier == "hard":
        values = _parse_frequency_table(question.text)
    else:
        a, b = _parse_groups(question.text)
        q1_a, q3_a = _exclusive_quartiles(a)
        q1_b, q3_b = _exclusive_quartiles(b)
        iqr_a, iqr_b = q3_a - q1_a, q3_b - q1_b
        assert iqr_a != iqr_b, f"oracle: equal group IQRs in {question.text!r}"
        if iqr_a > iqr_b:
            return q1_a, q3_a, iqr_a
        return q1_b, q3_b, iqr_b
    q1, q3 = _exclusive_quartiles(values)
    return q1, q3, q3 - q1


def _all(tier, seeds=None):
    seeds = seeds if seeds is not None else range(N_SEEDS)
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_answer_text(tier):
    mismatches = []
    for q in _all(tier):
        oracle = _oracle_triple(q, tier)
        m = _ANSWER_RE.match(q.answer_text)
        assert m is not None, q.answer_text
        actual = tuple(Fraction(v) for v in m.groups())
        if oracle != actual:
            mismatches.append((q.seed, q.text, tuple(str(x) for x in oracle), q.answer_text))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_answer_shape_is_recognised(tier):
    for q in _all(tier, seeds=range(20)):
        assert _ANSWER_RE.match(q.answer_text), q.answer_text


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_every_distractor_disagrees_with_the_oracle(tier):
    for q in _all(tier, seeds=range(30)):
        oracle = _oracle_triple(q, tier)
        for d in q.distractors:
            m = _ANSWER_RE.match(d)
            assert m is not None, d
            actual = tuple(Fraction(v) for v in m.groups())
            assert actual != oracle, f"{tier} seed {q.seed}: distractor {d!r} matches the oracle"


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_oracle_tukey_distractor_present_whenever_it_differs(tier):
    """Whenever the exclusive and Tukey-inclusive methods genuinely differ
    (odd n only), the Tukey-inclusive triple must be among the offered
    distractors -- re-derived here from the oracle's own parse, never the
    generator's internal state."""
    checked_a_differing_case = False
    for q in _all(tier, seeds=range(60)):
        if tier == "hard":
            values = _parse_frequency_table(q.text)
        else:
            values = _parse_plain_list(q.text)
        n = len(values)
        if n % 2 == 0:
            continue
        s = sorted(values)
        mid = n // 2
        incl_q1 = Fraction(statistics.median(s[: mid + 1]))
        incl_q3 = Fraction(statistics.median(s[mid:]))
        excl_q1, excl_q3 = _exclusive_quartiles(values)
        if (incl_q1, incl_q3) == (excl_q1, excl_q3):
            continue
        checked_a_differing_case = True
        incl_iqr = incl_q3 - incl_q1
        incl_text = f"Q1 = {incl_q1 if incl_q1.denominator == 1 else float(incl_q1)}"
        found = any(
            _ANSWER_RE.match(d)
            and Fraction(_ANSWER_RE.match(d).group(1)) == incl_q1
            and Fraction(_ANSWER_RE.match(d).group(2)) == incl_q3
            for d in q.distractors
        )
        assert found, f"{tier} seed {q.seed}: tukey triple ({incl_q1}, {incl_q3}, {incl_iqr}) missing"
    assert checked_a_differing_case
