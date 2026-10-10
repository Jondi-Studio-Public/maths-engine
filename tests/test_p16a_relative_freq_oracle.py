"""Independent oracle for the `p16a_relative_freq` questions: it re-derives each answer from the rendered
question text.

Parses only ``question.text`` / ``question.answer_text`` and re-derives the
answer from scratch. Does NOT import
``mathgen.generators.p16a_relative_freq_build``.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

RF_TOPIC = "p16a_relative_frequency_estimate"
HP_TOPIC = "p16a_histogram_probability"

_TABLE_RE = re.compile(r"([A-Za-z0-9 ]+?): (\d+)")


def _parse_table(text: str) -> dict:
    body = text.split("recorded: ")[1].split(". Based on")[0]
    entries = body.split(", ")
    table = {}
    for entry in entries:
        label, count = entry.rsplit(": ", 1)
        table[label.strip()] = int(count)
    return table


@pytest.mark.parametrize("seed", range(60))
def test_relative_frequency_single_oracle(seed):
    q = make(RF_TOPIC, seed=seed, difficulty="easy")
    table = _parse_table(q.text)
    n_total = sum(table.values())
    target = re.search(r"next trial gives (\w+)", q.text).group(1)
    expected = Fraction(table[target], n_total)
    given = Fraction(q.answer["p"], q.answer["q"])
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


@pytest.mark.parametrize("seed", range(60))
def test_relative_frequency_union_oracle(seed):
    q = make(RF_TOPIC, seed=seed, difficulty="medium")
    table = _parse_table(q.text)
    n_total = sum(table.values())
    m = re.search(r"next trial gives (\w+) or (\w+)", q.text)
    expected = Fraction(table[m.group(1)] + table[m.group(2)], n_total)
    given = Fraction(q.answer["p"], q.answer["q"])
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


@pytest.mark.parametrize("seed", range(60))
def test_relative_frequency_complement_oracle(seed):
    q = make(RF_TOPIC, seed=seed, difficulty="hard")
    table = _parse_table(q.text)
    n_total = sum(table.values())
    target = re.search(r"does NOT give (\w+)", q.text).group(1)
    expected = 1 - Fraction(table[target], n_total)
    given = Fraction(q.answer["p"], q.answer["q"])
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


@pytest.mark.parametrize("seed", range(60))
def test_relative_frequency_projection_oracle(seed):
    q = make(RF_TOPIC, seed=seed, difficulty="exam")
    table = _parse_table(q.text)
    n_total = sum(table.values())
    m = re.search(
        r"relative frequency of (\w+) in these trials, estimate how many of the "
        r"next (\d+) trials will give \1",
        q.text,
    )
    assert m, q.text
    target, future_n = m.group(1), int(m.group(2))
    prob = Fraction(table[target], n_total)
    raw = prob * future_n
    # round-half-up, matching mathgen.core.round_half_up
    import math

    expected = math.floor(raw + Fraction(1, 2))
    given = Fraction(q.answer["p"], q.answer["q"])
    assert given == expected, f"seed {seed}: expected {expected}, got {given}. {q.text}"


# --------------------------------------------------------------------------
# p16a_histogram_probability
# --------------------------------------------------------------------------

_CLASS_RE = re.compile(r"\[(\d+), (\d+)\): (\d+) \w+")


def _parse_classes(text: str):
    body = text.split("by class: ")[1].split(". Estimate")[0].split(". Assuming")[0]
    return [(int(lo), int(hi), int(f)) for lo, hi, f in _CLASS_RE.findall(body)]


@pytest.mark.parametrize("seed", range(60))
def test_histogram_single_class_oracle(seed):
    q = make(HP_TOPIC, seed=seed, difficulty="easy")
    classes = _parse_classes(q.text)
    n_total = sum(f for _, _, f in classes)
    m = re.search(r"value in \[(\d+), (\d+)\)", q.text)
    lo, hi = int(m.group(1)), int(m.group(2))
    freq = next(f for clo, chi, f in classes if clo == lo and chi == hi)
    expected = Fraction(freq, n_total)
    given = Fraction(q.answer["p"], q.answer["q"])
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


@pytest.mark.parametrize("seed", range(60))
def test_histogram_union_classes_oracle(seed):
    q = make(HP_TOPIC, seed=seed, difficulty="medium")
    classes = _parse_classes(q.text)
    n_total = sum(f for _, _, f in classes)
    m = re.search(r"value in \[(\d+), (\d+)\)", q.text)
    lo, hi = int(m.group(1)), int(m.group(2))
    count = sum(f for clo, chi, f in classes if clo >= lo and chi <= hi)
    expected = Fraction(count, n_total)
    given = Fraction(q.answer["p"], q.answer["q"])
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


@pytest.mark.parametrize("seed", range(60))
def test_histogram_tail_oracle(seed):
    q = make(HP_TOPIC, seed=seed, difficulty="hard")
    classes = _parse_classes(q.text)
    n_total = sum(f for _, _, f in classes)
    cutoff = int(re.search(r"value of at least (\d+)", q.text).group(1))
    count = sum(f for lo, hi, f in classes if lo >= cutoff)
    expected = Fraction(count, n_total)
    given = Fraction(q.answer["p"], q.answer["q"])
    assert expected == given, f"seed {seed}: expected {expected}, got {given}. {q.text}"


@pytest.mark.parametrize("seed", range(60))
def test_histogram_interpolate_oracle(seed):
    q = make(HP_TOPIC, seed=seed, difficulty="exam")
    classes = _parse_classes(q.text)
    n_total = sum(f for _, _, f in classes)
    cutoff = int(re.search(r"value of at least (\d+)", q.text).group(1))
    total = Fraction(0)
    for lo, hi, f in classes:
        if lo >= cutoff:
            total += Fraction(f, n_total)
        elif hi > cutoff > lo:
            fraction_above = Fraction(hi - cutoff, hi - lo)
            total += Fraction(f, n_total) * fraction_above
    given = Fraction(q.answer["p"], q.answer["q"])
    assert total == given, f"seed {seed}: expected {total}, got {given}. {q.text}"
