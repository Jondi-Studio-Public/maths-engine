"""Exam source of `f10-angle-relationships`: the fourth angle at a point, then x, then one angle."""

from __future__ import annotations

import re

import pytest
import sympy as sp

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_pr2_angle_point_unknown"
SKILL = next(s for s in load_skills()[1] if s.id == "f10-angle-relationships")
_EXPR = re.compile(r"\((\d*)x \+ (\d+)\)°")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _parse(q):
    given = [int(v) for v in re.findall(r"(\d+)°", re.sub(_EXPR, "", q.stem))]
    (p1, k1), (p2, k2) = [(int(p or 1), int(k)) for p, k in _EXPR.findall(q.stem)]
    return given, (p1, k1), (p2, k2)


def oracle(q) -> dict[str, str]:
    """Solve the whole problem from the rendered stem with sympy."""
    given, (p1, k1), (p2, k2) = _parse(q)
    x = sp.symbols("x")
    fourth = 360 - sum(given)
    (solution,) = sp.solve(sp.Eq(p1 * x + k1 + p2 * x + k2, fourth), x)
    return {"a": str(fourth), "b": str(solution), "c": str(p1 * solution + k1)}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages, s.params) for s in exam] == [(TOPIC, ["exam"], {"tier": "exam"})]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("p10e_angle_relationships", seed=s)) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_shape_and_no_text_defects():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert [p.marks for p in q.parts] == [2, 2, 1]
        assert find_text_defects(q) == []


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        assert oracle(q) == _canonical(q), (seed, q.text)


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) / len(answers) <= 0.05


def test_grader_accepts_key_and_rejects_distractors():
    for seed in range(120):
        q = _exam(seed)
        full = grade(SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 5
        canon = _canonical(q)
        for part in q.parts:
            assert len(part.distractors) >= 2
            for d in part.distractors:
                result = grade(SKILL, q, {**canon, part.label: str(d)})
                assert result.parts[part.label].marks == 0, (seed, part.label, str(d))


def test_follow_through_from_wrong_fourth_angle():
    checked = 0
    for seed in range(200):
        q = _exam(seed)
        given, (p1, k1), (p2, k2) = _parse(q)
        for d in q.parts[0].distractors:
            wrong = int(str(d))
            x_wrong = sp.Rational(wrong - k1 - k2, p1 + p2)
            if x_wrong.q != 1 or x_wrong <= 0 or wrong == int(q.parts[0].answer_text):
                continue
            c_wrong = p1 * x_wrong + k1
            reply = {"a": str(wrong), "b": str(x_wrong), "c": str(c_wrong)}
            if len({reply["a"], reply["b"], reply["c"]}) < 3:
                continue
            result = grade(SKILL, q, reply)
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 2 and result.parts["b"].follow_through
            assert result.parts["c"].marks == 1 and result.parts["c"].follow_through
            assert result.marks == 3 and not result.correct
            checked += 1
            break
    assert checked >= 40


def test_all_wrong_earns_zero():
    for seed in range(40):
        result = grade(SKILL, _exam(seed), {"a": "999", "b": "1", "c": "2"})
        assert result.marks == 0 and not result.correct


def test_deterministic():
    assert _exam(7).to_dict() == _exam(7).to_dict()


@pytest.mark.parametrize("seed", range(5))
def test_parts_do_not_print_keys(seed):
    q = _exam(seed)
    numbers = set(re.findall(r"\d+", " ".join([q.stem, *(p.prompt for p in q.parts)])))
    assert all(p.answer_text not in numbers for p in q.parts)
