"""Independent oracle for the `p16a_conditional_dist` questions: it re-derives each answer from the rendered
question text.

Parses only ``question.text`` / ``question.answer_text`` and re-derives the
answer from scratch. Does NOT import
``mathgen.generators.p16a_conditional_dist_build``.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

CI_TOPIC = "p16a_conditional_independence"
ED_TOPIC = "p16a_estimated_distribution"


_EASY_TABLE_RE = re.compile(
    r"Of (\d+) students surveyed, .+?: (\d+) satisfy both A and B; "
    r"(\d+) satisfy A but not B; (\d+) satisfy B but not A; and "
    r"(\d+) satisfy neither"
)


@pytest.mark.parametrize("seed", range(60))
def test_conditional_independence_easy_oracle(seed):
    q = make(CI_TOPIC, seed=seed, difficulty="easy")
    m = _EASY_TABLE_RE.search(q.text)
    assert m, q.text
    n_total, a, b, c, d = (int(m.group(i)) for i in range(1, 6))
    assert a + b + c + d == n_total, q.text
    expected = Fraction(a, a + c) - Fraction(a + b, n_total)
    given = Fraction(q.answer["p"], q.answer["q"])
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


@pytest.mark.parametrize("seed", range(60))
def test_conditional_independence_medium_oracle(seed):
    q = make(CI_TOPIC, seed=seed, difficulty="medium")
    m = re.search(
        r"Of (\d+) students surveyed, .+?: (\d+) satisfy A and (\d+) "
        r"satisfy B\. Of the students who satisfy A, (\d+) also satisfy B",
        q.text,
    )
    assert m, q.text
    n_total, row_a_total, col_b_total, a = (int(m.group(i)) for i in range(1, 5))
    expected = Fraction(a, col_b_total) - Fraction(row_a_total, n_total)
    given = Fraction(q.answer["p"], q.answer["q"])
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


@pytest.mark.parametrize("seed", range(60))
def test_conditional_independence_hard_oracle(seed):
    q = make(CI_TOPIC, seed=seed, difficulty="hard")
    n_total = int(re.search(r"survey of (\d+) students", q.text).group(1))
    pcts = [float(x) for x in re.findall(r"([\d.]+)%", q.text)]
    assert len(pcts) == 4, q.text
    counts = [round(p * n_total / 100) for p in pcts]
    a, b, c, d = counts
    expected = Fraction(a, a + c) - Fraction(a + b, n_total)
    given = Fraction(q.answer["p"], q.answer["q"])
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


@pytest.mark.parametrize("seed", range(60))
def test_conditional_independence_exam_oracle(seed):
    q = make(CI_TOPIC, seed=seed, difficulty="exam")
    rows = re.findall(r"[Oo]f the students who travel by ([\w ]+?), (\d+) satisfy A and (\d+) do not", q.text)
    assert len(rows) == 3, q.text
    labels = [r[0] for r in rows]
    counts_a = [int(r[1]) for r in rows]
    counts_not_a = [int(r[2]) for r in rows]
    target_label = re.search(r"student travels by ([\w ]+?)'", q.text).group(1)
    target = labels.index(target_label)
    p_a_given_b = Fraction(counts_a[target], counts_a[target] + counts_not_a[target])
    p_a = Fraction(sum(counts_a), sum(counts_a) + sum(counts_not_a))
    expected = p_a_given_b - p_a
    given = Fraction(q.answer["p"], q.answer["q"])
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


# --------------------------------------------------------------------------
# p16a_estimated_distribution
# --------------------------------------------------------------------------

_ROW_RE = re.compile(r"x=(-?\d+): (\d+|\?) \w+")


def _parse_rows(text: str):
    return [(int(x), (None if f == "?" else int(f))) for x, f in _ROW_RE.findall(text)]


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_estimated_distribution_direct_oracle(seed, tier):
    q = make(ED_TOPIC, seed=seed, difficulty=tier)
    rows = _parse_rows(q.text)
    n_total = int(re.search(r"Over (\d+)", q.text).group(1))
    values_order = [int(x) for x in re.search(r"x = ([\d, ]+)\)", q.text).group(1).split(", ")]
    freq_by_value = {v: f for v, f in rows}
    expected = [Fraction(freq_by_value[v], n_total) for v in values_order]
    given = [Fraction(x) for x in q.answer_text.split(", ")]
    assert expected == given, f"seed {seed}/{tier}: expected {expected}, got {given}. {q.text}"


@pytest.mark.parametrize("seed", range(60))
def test_estimated_distribution_hard_oracle(seed):
    q = make(ED_TOPIC, seed=seed, difficulty="hard")
    rows = _parse_rows(q.text)
    n_total = int(re.search(r"Over (\d+) \w+ in total", q.text).group(1))
    values_order = [int(x) for x in re.search(r"x = ([\d, ]+)\)", q.text).group(1).split(", ")]
    known_sum = sum(f for _, f in rows if f is not None)
    missing = n_total - known_sum
    freq_by_value = {v: (f if f is not None else missing) for v, f in rows}
    expected = [Fraction(freq_by_value[v], n_total) for v in values_order]
    given = [Fraction(x) for x in q.answer_text.split(", ")]
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


_RATIO_RE = re.compile(r"the frequency for x=(-?\d+) is (\d+) times the frequency for x=(-?\d+)")


@pytest.mark.parametrize("seed", range(60))
def test_estimated_distribution_exam_oracle(seed):
    q = make(ED_TOPIC, seed=seed, difficulty="exam")
    rows = _parse_rows(q.text)
    n_total = int(re.search(r"Over (\d+) \w+ in total", q.text).group(1))
    values_order = [int(x) for x in re.search(r"x = ([\d, ]+)\)", q.text).group(1).split(", ")]
    m = _RATIO_RE.search(q.text)
    assert m, q.text
    x_large, ratio, x_small = int(m.group(1)), int(m.group(2)), int(m.group(3))
    known_sum = sum(f for _, f in rows if f is not None)
    missing_sum = n_total - known_sum
    assert missing_sum % (ratio + 1) == 0, f"seed {seed}: ratio system has no integer solution"
    f_small = missing_sum // (ratio + 1)
    f_large = ratio * f_small
    freq_by_value = {v: f for v, f in rows if f is not None}
    freq_by_value[x_small] = f_small
    freq_by_value[x_large] = f_large
    expected = [Fraction(freq_by_value[v], n_total) for v in values_order]
    given = [Fraction(x) for x in q.answer_text.split(", ")]
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"
