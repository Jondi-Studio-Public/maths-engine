"""Tests for the multi-part exam tier `exam_parts_normal_distribution`
(skill `senior-calculate-probabilities-and-quantiles-associated-with-a-given`).

The oracle (`_oracle`) re-derives every part's answer from the question text alone, with `math.erf` and
`statistics.NormalDist`; it never imports the generator module.
"""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal
from statistics import NormalDist

import pytest

from mathgen import make
from mathgen.engine import grade
from mathgen.skills import load_skills

TOPIC = "exam_parts_normal_distribution"
SKILL_ID = "senior-calculate-probabilities-and-quantiles-associated-with-a-given"
SEEDS = range(120)
STD = NormalDist(0, 1)


@pytest.fixture(scope="module")
def skill():
    return next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _fmt(value, places):
    quantum = Decimal(1).scaleb(-places)
    return str(Decimal(repr(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def _phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _oracle(text):
    """{label: answer_text} worked out from the question text alone."""
    mu = float(re.search(r"mean (\d+) ", text).group(1))
    sigma = float(re.search(r"standard deviation (\d+) ", text).group(1))
    x = float(re.search(r"\(a\) Find the z-score for a [a-z ]+ of (\d+\.\d\d) ", text).group(1))
    sym = re.search(r"\(b\) Hence find P\(X ([<>]) ", text).group(1)
    n = int(re.search(r"sample of (\d+) ", text).group(1))
    p = float(re.search(r"P\(X < k\) = (0\.\d+)", text).group(1))
    z = (x - mu) / sigma
    z2 = float(_fmt(z, 2))
    prob = _phi(z2) if sym == "<" else _phi(-z2)
    prob4 = float(_fmt(prob, 4))
    return {
        "a": _fmt(z, 2),
        "b": _fmt(prob, 4),
        "c": _fmt(n * prob4, 0),
        "d": _fmt(mu + sigma * STD.inv_cdf(p), 1),
    }


def _replies(q):
    return {part.label: part.answer_text for part in q.parts}


def test_skill_exam_source_uses_new_topic(skill):
    exam = [s for s in skill.sources if s.difficulty == "exam"]
    assert len(exam) == 1
    assert exam[0].topic == TOPIC
    assert exam[0].params == {}


def test_shape_marks_and_follow_through():
    q = _q(0)
    assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
    assert [p.marks for p in q.parts] == [1, 2, 1, 2]
    assert [p.answer_kind for p in q.parts] == ["number"] * 4
    assert q.parts[0].follow_through is None
    assert "erf" in q.parts[1].follow_through
    assert q.parts[2].follow_through.endswith("*part_b")
    assert q.parts[3].follow_through is None


@pytest.mark.parametrize("seed", SEEDS)
def test_composed_answer_earns_full_marks(skill, seed):
    q = _q(seed)
    result = grade(skill, q, q.answer_text)
    assert result.correct
    assert result.marks == result.max_marks == 6
    assert all(not r.follow_through for r in result.parts.values())


@pytest.mark.parametrize("seed", SEEDS)
def test_distractors_are_wrong_with_other_parts_canonical(skill, seed):
    q = _q(seed)
    for part in q.parts:
        assert len(part.distractors) >= 2
        for d in part.distractors:
            reply = _replies(q)
            reply[part.label] = d
            result = grade(skill, q, reply)
            assert result.parts[part.label].marks == 0, (part.label, d)
            assert not result.correct


@pytest.mark.parametrize("seed", SEEDS)
def test_independent_oracle_agrees(seed):
    q = _q(seed)
    expected = _oracle(q.text)
    for part in q.parts:
        assert part.answer_text == expected[part.label], (part.label, q.text)


def test_stated_precision_matches_prompts():
    for seed in SEEDS:
        q = _q(seed)
        places = [len(p.answer_text.split(".")[1]) if "." in p.answer_text else 0 for p in q.parts]
        assert places == [2, 4, 0, 1]
        assert "2 decimal places" in q.parts[0].prompt
        assert "4 decimal places" in q.parts[1].prompt
        assert "nearest whole number" in q.parts[2].prompt
        assert "1 decimal place" in q.parts[3].prompt


def test_variety_and_determinism():
    texts = {_q(s).text for s in range(60)}
    assert len(texts) >= 40
    assert _q(7).text == _q(7).text
    assert _q(7).answer_text == _q(7).answer_text
    assert {"<", ">"} <= {re.search(r"P\(X ([<>]) ", _q(s).text).group(1) for s in range(40)}


def test_follow_through_from_wrong_z(skill):
    checked = 0
    for seed in SEEDS:
        q = _q(seed)
        mu, sigma, x, sym, n, _ = _parse(q.text)
        z = float(q.parts[0].answer_text)
        wrong_z = round(z + 0.30, 2)
        prob = _phi(wrong_z) if sym == "<" else _phi(-wrong_z)
        scaled = prob * 1e4
        if abs(scaled - math.floor(scaled) - 0.5) < 1e-6:
            continue
        b_from_wrong = _fmt(prob, 4)
        reply = _replies(q)
        reply["a"] = _fmt(wrong_z, 2)
        reply["b"] = b_from_wrong
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        if b_from_wrong != q.parts[1].answer_text:
            assert result.parts["b"].marks == 2
            assert result.parts["b"].follow_through is True
            checked += 1
        assert result.marks < result.max_marks
        assert not result.correct
    assert checked > 60


def test_follow_through_from_wrong_probability(skill):
    checked = 0
    for seed in SEEDS:
        q = _q(seed)
        _, _, _, _, n, _ = _parse(q.text)
        wrong_b = float(q.parts[1].answer_text) + 0.0500
        if wrong_b >= 1:
            wrong_b = float(q.parts[1].answer_text) - 0.0500
        raw = n * float(_fmt(wrong_b, 4))
        if abs(raw - math.floor(raw) - 0.5) < 1e-6:
            continue
        reply = _replies(q)
        reply["b"] = _fmt(wrong_b, 4)
        reply["c"] = _fmt(raw, 0)
        result = grade(skill, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 1
        assert result.parts["c"].follow_through is True
        assert result.marks == 4
        checked += 1
    assert checked > 100


def test_part_d_is_independent_of_earlier_parts(skill):
    q = _q(3)
    reply = _replies(q)
    reply.update(a="9.99", b="0.1234", c="1")
    result = grade(skill, q, reply)
    assert result.parts["d"].marks == 2
    assert result.marks == 2
    reply = _replies(q)
    reply["d"] = q.parts[3].distractors[0]
    assert grade(skill, q, reply).parts["d"].marks == 0


@pytest.mark.parametrize("seed", range(20))
def test_wrong_everywhere_earns_nothing(skill, seed):
    q = _q(seed)
    reply = {"a": "7.77", "b": "0.1111", "c": "9999", "d": "-5.5"}
    result = grade(skill, q, reply)
    assert result.marks == 0
    assert not result.correct


def _parse(text):
    mu = float(re.search(r"mean (\d+) ", text).group(1))
    sigma = float(re.search(r"standard deviation (\d+) ", text).group(1))
    x = float(re.search(r"of (\d+\.\d\d) ", text).group(1))
    sym = re.search(r"P\(X ([<>]) ", text).group(1)
    n = int(re.search(r"sample of (\d+) ", text).group(1))
    p = float(re.search(r"P\(X < k\) = (0\.\d+)", text).group(1))
    return mu, sigma, x, sym, n, p
