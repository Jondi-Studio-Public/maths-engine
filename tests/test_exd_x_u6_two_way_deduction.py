"""Exam source of `f10-two-way-table-relationship`: deduce blank cells from totals, then a conditional fraction."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u6_two_way_deduction"
SKILL_ID = "f10-two-way-table-relationship"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _numbers(stem):
    nums = [int(v) for v in re.findall(r"\d+", re.sub(r"Year 9|under 30|30 or older", "", stem))]
    if stem.startswith("A survey"):  # N, rowA, colX, J
        n, row, col, joint = nums
    elif stem.startswith("Records"):  # N, colX, rowA, J
        n, col, row, joint = nums
    else:  # N, rowA, colX, J
        n, row, col, joint = nums
    return n, row, col, joint


def _oracle(q):
    n, row, col, joint = _numbers(q.stem)
    a, b = row - joint, col - joint
    c = n - joint - a - b
    return {"a": str(a), "b": str(b), "c": str(c), "d": str(Fraction(b, n - row))}


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        assert _oracle(q) == {p.label: p.answer_text for p in q.parts}, (seed, q.text)


def test_chain_shape_and_ranges():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "fraction"]
        assert q.parts[3].sensible_range == (0, 1)
        for d in q.parts[3].distractors:
            assert 0 <= Fraction(d) <= 1


def test_grader_accepts_key_and_rejects_distractors():
    shown = 0
    for seed in range(100):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 6
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            for d in p.distractors:
                shown += 1
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d)
    assert shown >= 300


def test_follow_through_from_wrong_earlier_cells():
    for seed in range(100):
        q = _exam(seed)
        n, row, col, joint = _numbers(q.stem)
        a_wrong = int(q.parts[0].distractors[0])
        b_wrong = int(q.parts[1].distractors[0])
        c_follow = n - joint - a_wrong - b_wrong
        if c_follow <= 0 or c_follow in (a_wrong, b_wrong):
            continue
        reply = {"a": str(a_wrong), "b": str(b_wrong), "c": str(c_follow), "d": str(Fraction(b_wrong, n - row))}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks, (seed, reply)
        if 0 < Fraction(b_wrong, n - row) < 1:
            assert result.parts["d"].marks == result.parts["d"].max_marks, (seed, reply)
