"""Exam source of `senior-check-results-of-calculations-for-accuracy`: locate, repair and carry a slip forward."""

from __future__ import annotations

import re

from mathgen.engine import grade
from tests import exd_x_sc1_support as sup

SKILL_ID = "senior-check-results-of-calculations-for-accuracy"
TOPIC = "exd_x_sc1_check_calculation"
_STEP = re.compile(r"Step (\d): (\d+) ([+\-×]) (\d+) = (\d+)")
_APPLY = {"+": lambda p, q: p + q, "-": lambda p, q: p - q, "×": lambda p, q: p * q}


def steps(q):
    found = [(int(n), int(left), op, int(right), int(res)) for n, left, op, right, res in _STEP.findall(q.stem)]
    assert [s[0] for s in found] == [1, 2, 3, 4]
    return found


def oracle(q):
    shown = steps(q)
    bad = [n for n, left, op, right, res in shown if _APPLY[op](left, right) != res]
    assert len(bad) == 1 and bad[0] < 4
    n = bad[0]
    _, left, op, right, _res = shown[n - 1]
    correct = _APPLY[op](left, right)
    value = correct
    for _n, _l, o, r, _res in shown[n:]:
        value = _APPLY[o](value, r)
    return {"a": n, "b": correct, "c": value}


def test_wiring_shapes_and_text():
    sup.check_wiring(SKILL_ID, TOPIC)
    sup.check_shapes(SKILL_ID)
    sup.check_no_text_defects(SKILL_ID)


def test_shape_of_the_question():
    for seed in range(60):
        q = sup.exam(SKILL_ID, seed)
        assert [p.marks for p in q.parts] == [1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]


def test_key_rederived_from_prompt():
    seen_steps = set()
    for seed in range(300):
        q = sup.exam(SKILL_ID, seed)
        expected = oracle(q)
        seen_steps.add(expected["a"])
        assert {p.label: int(p.answer_text) for p in q.parts} == expected, (seed, q.text)
    assert seen_steps == {1, 2, 3}


def test_grading_and_tags():
    sup.check_grading(SKILL_ID)
    sup.check_tags(SKILL_ID)


def test_follow_through_from_a_wrong_repair():
    """A wrong corrected value, carried through the later steps correctly, still earns the final result."""
    for seed in range(100):
        q = sup.exam(SKILL_ID, seed)
        expected = oracle(q)
        shown = steps(q)
        wrong = expected["b"] + 7
        value = wrong
        for _n, _l, o, r, _res in shown[expected["a"] :]:
            value = _APPLY[o](value, r)
        result = grade(sup.skill(SKILL_ID), q, {"a": str(expected["a"]), "b": str(wrong), "c": str(value)})
        assert result.parts["a"].marks == 1 and result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (seed, wrong, value)


def test_the_stated_final_result_is_not_the_key():
    for seed in range(300):
        q = sup.exam(SKILL_ID, seed)
        assert steps(q)[-1][4] != int(q.parts[2].answer_text)


def test_answer_space_and_determinism():
    sup.check_answer_space(SKILL_ID)
    sup.check_deterministic(SKILL_ID)


def test_chain_is_not_copyable():
    """The later steps never cancel out ((c) is not (b) again) and junk replies earn nothing."""
    from tests.exd_junk_probe import assert_chain_is_not_copyable

    assert_chain_is_not_copyable("exd_x_sc1_check_calculation", "senior-check-results-of-calculations-for-accuracy")
