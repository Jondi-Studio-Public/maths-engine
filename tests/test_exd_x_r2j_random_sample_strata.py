"""Exam chain for `senior-understand-the-concept-of-a-random-sample`: a proportional stratified sample, part by part."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade
from tests import exd_x_r2j_kit as kit

SKILL = "senior-understand-the-concept-of-a-random-sample"
TOPIC = "exd_x_r2j_random_sample_strata"
SEEDS = range(300)


@pytest.fixture(scope="module")
def questions():
    return [kit.exam_q(SKILL, s) for s in SEEDS]


def oracle(q) -> dict[str, str]:
    """Re-derive every part from the rendered stem alone."""
    stem = q.stem
    sizes = [int(v.replace(",", "")) for v in re.findall(r"(\d[\d,]*) (?:in|at) the", stem)]
    total = int(re.search(r"(?:has|employs) ([\d,]+) ", stem).group(1).replace(",", ""))
    n = int(re.search(r"sample of (\d+)", stem).group(1))
    assert len(sizes) == 3
    assert sum(sizes) == total
    frac = Fraction(n, sum(sizes))
    take = [frac * s for s in sizes]
    assert all(t.denominator == 1 for t in take)
    out = {"a": f"{frac.numerator}/{frac.denominator}", "b": str(int(take[0])), "c": str(int(take[1]))}
    out["d"] = str(n - int(take[0]) - int(take[1]))
    assert out["d"] == str(int(take[2]))
    return out


def test_wiring_shapes_defects_determinism():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["fraction", "number", "number", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == kit.canon(q), q.text


def test_answer_space():
    kit.check_answer_space(SKILL)


def test_grading():
    kit.check_grading(SKILL)


def test_follow_through_from_wrong_sampling_fraction(questions):
    for q in questions[:100]:
        sizes = [int(v.replace(",", "")) for v in re.findall(r"(\d[\d,]*) (?:in|at) the", q.stem)]
        n = int(re.search(r"sample of (\d+)", q.stem).group(1))
        wrong = Fraction(n, sum(sizes) + sizes[0])  # a plausible but wrong fraction
        take = [wrong * s for s in sizes]
        if any(t.denominator != 1 for t in take[:2]):
            wrong = Fraction(1, 2)
            take = [wrong * s for s in sizes]
        b, c = take[0], take[1]
        if b.denominator != 1 or c.denominator != 1:
            continue
        reply = {
            "a": f"{wrong.numerator}/{wrong.denominator}",
            "b": str(int(b)),
            "c": str(int(c)),
            "d": str(n - int(b) - int(c)),
        }
        result = grade(kit.skill(SKILL), q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label)


def test_follow_through_in_last_part_only(questions):
    for q in questions[:100]:
        canon = kit.canon(q)
        b_wrong = str(int(canon["b"]) + 3)
        n = int(re.search(r"sample of (\d+)", q.stem).group(1))
        d_from_wrong = str(n - int(b_wrong) - int(canon["c"]))
        if d_from_wrong in (b_wrong, canon["c"]):  # repeating an earlier reply is not follow-through
            continue
        result = grade(kit.skill(SKILL), q, {**canon, "b": b_wrong, "d": d_from_wrong})
        assert result.parts["b"].marks == 0
        assert result.parts["d"].marks == q.parts[3].marks and result.parts["d"].follow_through
        assert result.parts["a"].marks and result.parts["c"].marks


def test_topic_registered():
    assert make(TOPIC, seed=3, difficulty="exam").parts
