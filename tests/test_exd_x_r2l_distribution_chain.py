"""Tests for the exam source of senior-use-relative-frequencies-obtained-from-data-to-obtain.

The exam tier is a four-part chain on a frequency table with two unknown frequencies: their sum from the total, one
of them from the stated mean, the other, and then the estimated distribution as an ordered list. Every key is
re-derived from the rendered prompt by solving the two equations with SymPy.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2l_distribution_chain"
SKILL_ID = "senior-use-relative-frequencies-obtained-from-data-to-obtain"
SEEDS = range(320)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _show(v: Fraction) -> str:
    return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)


def _facts(q):
    n = int(re.search(r"(\d+) recorded values", q.text).group(1))
    mean = Fraction(re.search(r"is (\d+(?:\.\d+)?)\. \(a\)", q.text).group(1))
    table = re.search(r"frequency table (.+?)\. The frequencies", q.text).group(1)
    rows = [(int(x), f) for x, f in re.findall(r"x = (\d+): (\w+)", table)]
    return n, mean, rows


def oracle(q) -> dict[str, str]:
    n, mean, rows = _facts(q)
    m, k = sympy.symbols("m k")
    pos = {name: x for x, name in rows if name in ("m", "k")}
    known = [(x, int(f)) for x, f in rows if f not in ("m", "k")]
    total = sum(f for _, f in known)
    sol = sympy.solve(
        [
            sympy.Eq(m + k, n - total),
            sympy.Eq(
                pos["m"] * m + pos["k"] * k,
                n * sympy.Rational(mean.numerator, mean.denominator) - sum(x * f for x, f in known),
            ),
        ],
        [m, k],
        dict=True,
    )[0]
    freqs = {name: int(sol[sym]) for name, sym in (("m", m), ("k", k))}
    probs = [Fraction(freqs.get(f) if f in freqs else int(f), n) for _, f in rows]
    return {
        "a": str(n - total),
        "b": str(freqs["m"]),
        "c": str(freqs["k"]),
        "d": ", ".join(_show(p) for p in probs),
    }


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]  # (d) follows through too
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "ordered_list"]


def test_oracle_agrees_over_300_seeds(questions):
    bad = [(q.text, oracle(q), _canonical(q)) for q in questions if oracle(q) != _canonical(q)]
    assert bad == []


def test_composed_answer_full_marks(questions):
    for q in questions[:120]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6


def test_every_distractor_rejected_and_tagged(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) == 3 and len(set(p.distractors)) == 3
            tags = [p.option_notes.get(d) for d in p.distractors]
            assert all(t is None or t in CATALOGUE for t in tags), (p.label, tags)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_every_part_has_a_tagged_distractor(questions):
    for q in questions:
        for p in q.parts:
            assert any(p.option_notes.get(d) in CATALOGUE for d in p.distractors), (q.text, p.label)


def test_follow_through_from_wrong_sum(questions):
    checked = 0
    for q in questions[:150]:
        n, mean, rows = _facts(q)
        pos = {name: x for x, name in rows if name in ("m", "k")}
        gap = pos["k"] - pos["m"]
        # shifting the sum by the gap between the two x values moves m by exactly k's value, a whole number
        wrong_a = int(q.parts[0].answer_text) + gap
        known = [(x, int(f)) for x, f in rows if f not in ("m", "k")]
        r = n * mean - sum(x * f for x, f in known)
        wrong_b = (pos["k"] * wrong_a - r) / gap
        assert wrong_b.denominator == 1 and wrong_b > 0
        wrong_c = wrong_a - wrong_b
        if wrong_c <= 0 or wrong_c == wrong_b or wrong_b == int(q.parts[1].answer_text):
            continue
        result = grade(_SKILL, q, {"a": str(wrong_a), "b": _show(wrong_b), "c": _show(wrong_c)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True
        if wrong_c != int(q.parts[2].answer_text):
            assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True
        checked += 1
    assert checked >= 100


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "999", "b": "1000", "c": "3000", "d": "1/2, 1/2"})
        assert result.marks == 0


def test_list_order_matters(questions):
    for q in questions[:60]:
        reversed_reply = ", ".join(reversed(q.parts[3].answer_text.split(", ")))
        if reversed_reply != q.parts[3].answer_text:
            assert grade(_SKILL, q, {**_canonical(q), "d": reversed_reply}).parts["d"].marks == 0
