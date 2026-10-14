"""Exam tier of `simplify-surds`: the area and diagonal of a rectangle whose sides are surds."""

from __future__ import annotations

import re

import sympy

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_n4_surd_rectangle"
SKILL = next(s for s in load_skills()[1] if s.id == "simplify-surds")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _sides(q):
    found = re.findall(r"(\d*)√(\d+) m", q.stem)
    (a, m), (c, n) = [(int(a or 1), int(m)) for a, m in found]
    return a * sympy.sqrt(m), c * sympy.sqrt(n)


def _same(text, expected):
    return sympy.simplify(sympy.sympify(text) - expected) == 0


def test_prompt_shapes_are_new():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("t1_simplify_surds", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        x, y = _sides(q)
        a, b, c = (p.answer_text for p in q.parts)
        assert _same(a, x * y), (seed, q.text)
        assert sympy.Integer(b) == sympy.expand(x**2 + y**2), (seed, q.text)
        assert _same(c, sympy.sqrt(x**2 + y**2)), (seed, q.text)
        # the surd answers are in the form k*sqrt(s) with s square-free and k, s whole numbers
        for text in (a, c):
            match = re.fullmatch(r"(\d+)\*sqrt\((\d+)\)", text)
            assert match and set(sympy.factorint(int(match.group(2))).values()) <= {1}, (seed, text)


def test_shape():
    q = _exam(0)
    assert [p.label for p in q.parts] == ["a", "b", "c"]
    assert [p.answer_kind for p in q.parts] == ["expression", "number", "expression"]
    assert [bool(p.follow_through) for p in q.parts] == [False, False, True]
    assert all(len(p.distractors) >= 2 for p in q.parts)


def test_key_accepted_and_distractors_rejected():
    shown = 0
    for seed in range(120):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        full = grade(SKILL, q, canon)
        assert full.correct and full.marks == full.max_marks == 5, (seed, q.text)
        for part in q.parts:
            for d in part.distractors:
                shown += 1
                result = grade(SKILL, q, {**canon, part.label: d})
                assert result.parts[part.label].marks == 0, (seed, part.label, d)
                assert part.option_notes.get(d)
    assert shown >= 600


def test_unsimplified_surd_is_not_the_key():
    for seed in range(40):
        q = _exam(seed)
        x, y = _sides(q)
        k, s = [int(v) for v in re.fullmatch(r"(\d+)\*sqrt\((\d+)\)", q.parts[2].answer_text).groups()]
        canon = {p.label: p.answer_text for p in q.parts}
        unsimplified = f"sqrt({k * k * s})"
        assert not grade(SKILL, q, {**canon, "c": unsimplified}).parts["c"].marks


def test_follow_through_from_wrong_squared_diagonal():
    carried = 0
    for seed in range(100):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        wrong_b = int(q.parts[1].distractors[0])
        wrong_c = str(sympy.sqrt(wrong_b))
        result = grade(SKILL, q, {**canon, "b": str(wrong_b), "c": wrong_c})
        assert result.parts["b"].marks == 0
        carried += result.parts["c"].marks == q.parts[2].marks
        if result.parts["c"].marks:
            assert result.parts["c"].follow_through
    assert carried >= 90


def test_all_wrong_earns_nothing():
    result = grade(SKILL, _exam(3), {"a": "7", "b": "7", "c": "7"})
    assert result.marks == 0 and not result.correct
