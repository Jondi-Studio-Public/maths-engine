"""Exam chain for `f10-shape-sorting-algorithm`: count, total and mean perimeter of the shapes sent to one bin."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen.engine import grade
from tests import exd_x_r2j_kit as kit

SKILL = "f10-shape-sorting-algorithm"
TOPIC = "exd_x_r2j_shape_sorting_chain"
SEEDS = range(300)


@pytest.fixture(scope="module")
def questions():
    return [kit.exam_q(SKILL, s) for s in SEEDS]


def _predicate(question: str):
    """The test a numbered question asks, as a function of (sides, right angle, parallel pairs)."""
    q = question.lower()
    if "exactly 3" in q:
        return lambda sides, right, pairs: len(sides) == 3
    if "equal in length" in q or "same length" in q:
        return lambda sides, right, pairs: len(set(sides)) == 1
    if "right angle" in q:
        return lambda sides, right, pairs: right
    if "at least one pair" in q:
        return lambda sides, right, pairs: pairs >= 1
    if "both pairs" in q or "second pair" in q:
        return lambda sides, right, pairs: pairs == 2
    raise AssertionError(question)


def _procedure(stem: str):
    tests = {}
    for num, body in re.findall(r"Test (\d+): (.+?)(?= Test \d+:| Six shapes)", stem):
        question, rest = body.split("? ", 1)
        yes = re.search(r"If yes, (go to Test (\d+)|assign (Bin \w))", rest)
        no = re.search(r"If no, (go to Test (\d+)|assign (Bin \w))", rest)
        tests[int(num)] = (
            _predicate(question),
            yes.group(2) and int(yes.group(2)) or yes.group(3),
            no.group(2) and int(no.group(2)) or no.group(3),
        )
    return tests


def _shapes(stem: str):
    found = {}
    for label, kind, sides, tail in re.findall(
        r"Shape (\w) is a (triangle|quadrilateral) with sides ([\d, ]+) units\.((?: (?!Shape|A seventh|It is)[A-Z][^.]*\.){1,2})",
        stem,
    ):
        found[label] = _facts(sides, tail)
    return found


def _facts(sides: str, tail: str):
    sides = [int(v) for v in sides.split(",")]
    right = "One of its angles is a right angle" in tail
    pairs = 2 if "Both pairs" in tail else 1 if "Exactly one pair" in tail else 0
    return sides, right, pairs


def _bin(tests, facts):
    node = 1
    while isinstance(node, int):
        predicate, yes, no = tests[node]
        node = yes if predicate(*facts) else no
    return node


def one_dp(x: Fraction) -> str:
    return f"{(x * 10 + Fraction(1, 2)) // 1 / 10:.1f}"


def oracle(q) -> dict[str, str]:
    stem = q.stem
    tests = _procedure(stem)
    target = re.search(r"placed in (Bin \w)\?", q.parts[0].prompt).group(1)
    shapes = _shapes(stem.split("Six shapes are sorted by this procedure.")[1])
    assert len(shapes) == 6
    members = [facts for facts in shapes.values() if _bin(tests, facts) == target]
    late = re.search(
        r"Shape V, is a (?:triangle|quadrilateral) with sides ([\d, ]+) units\.((?: (?!Shape|A seventh|It is)[A-Z][^.]*\.){1,2}) It is",
        q.parts[3].prompt,
    )
    late_facts = _facts(late.group(1), late.group(2))
    assert _bin(tests, late_facts) == target
    count, total = len(members), sum(sum(f[0]) for f in members)
    mean = Fraction(total, count)
    new_mean = Fraction(total + sum(late_facts[0]), count + 1)

    return {"a": str(count), "b": str(total), "c": one_dp(mean), "d": one_dp(new_mean)}


def test_wiring_shapes_defects_determinism():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == kit.canon(q), q.text


def test_grading():
    kit.check_grading(SKILL)


def test_follow_through_from_wrong_count(questions):
    checked = 0
    for q in questions[:100]:
        c = kit.canon(q)
        count_wrong = int(c["a"]) + 1
        total = int(c["b"])
        mean_wrong = Fraction(total, count_wrong)
        late_perim = sum(
            int(v)
            for v in re.search(r"Shape V, is a \w+ with sides ([\d, ]+) units", q.parts[3].prompt).group(1).split(",")
        )
        new_wrong = Fraction(total + late_perim, count_wrong + 1)
        reply = {"a": str(count_wrong), "b": c["b"], "c": one_dp(mean_wrong), "d": one_dp(new_wrong)}
        if reply["c"] == c["c"] or reply["d"] == c["d"]:
            continue
        result = grade(kit.skill(SKILL), q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks and result.parts["d"].marks == q.parts[3].marks, (
            q.text,
            reply,
        )
        checked += 1
    assert checked >= 70


def test_count_in_the_bin_is_spread():
    """Part (a), how many shapes reach the bin, is 2, 3 or 4 in a spread (it was 2 or 3, and 3 on over half)."""
    from tests.exd_junk_probe import assert_answer_spread

    assert_answer_spread("exd_x_r2j_shape_sorting_chain", "a", max_share=0.45, at_least=3)
