"""Exam chain of `senior-the-opposite-angles-of-a-cyclic-quadrilateral-are`: x, y, a named angle, an exterior angle."""

from __future__ import annotations

import re

import pytest
import sympy

from tests import exd_x_r3a_support as S

TOPIC = "exd_x_r3a_cyclic_chain"
SKILL = S.load_skill("senior-the-opposite-angles-of-a-cyclic-quadrilateral-are")
X, Y = sympy.symbols("x y")
NEXT = {"A": "B", "B": "C", "C": "D", "D": "A"}


def _angles(q):
    out = {}
    for name, body in re.findall(r"angle ([A-D]) = \(?([^()]*?)\)? degrees", q.stem):
        out[name] = sympy.sympify(re.sub(r"(\d)([xy])", r"\1*\2", body))
    assert set(out) == set("ABCD"), q.stem
    return out


def oracle(q):
    ang = _angles(q)
    (sol,) = sympy.solve([ang["A"] + ang["C"] - 180, ang["B"] + ang["D"] - 180], [X, Y], dict=True)
    value = {n: ang[n].subs(sol) for n in "ABCD"}
    target = re.search(r"find the size of angle ([A-D]), in degrees", q.parts[2].prompt).group(1)
    ext = re.search(r"side ([A-D])([A-D]) is extended beyond ([A-D]) to a point E", q.parts[3].prompt)
    vertex = ext.group(3)
    return {"a": str(sol[X]), "b": str(sol[Y]), "c": str(value[target]), "d": str(180 - value[vertex])}


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_x(questions):
    def chain(q):
        ang = _angles(q)
        wrong_x = int(q.parts[0].distractors[0])
        eq = ang["A"] + ang["C"] - 180
        if eq.coeff(X) and eq.coeff(Y):
            y_val = sympy.solve(eq.subs(X, wrong_x), Y)[0]
        else:
            y_val = sympy.solve((ang["B"] + ang["D"] - 180).subs(X, wrong_x), Y)[0]
        if not y_val.is_Integer or y_val <= 0:
            return None
        target = re.search(r"find the size of angle ([A-D]), in degrees", q.parts[2].prompt).group(1)
        vertex = re.search(r"beyond ([A-D]) to a point E", q.parts[3].prompt).group(1)
        sub = {X: wrong_x, Y: y_val}
        c_val, d_val = ang[target].subs(sub), 180 - ang[vertex].subs(sub)
        if not (0 < c_val < 360 and 0 < d_val < 180) or len({wrong_x, y_val, c_val, d_val}) < 4:
            return None
        return {"a": str(wrong_x), "b": str(y_val), "c": str(c_val), "d": str(d_val)}, {"b", "c", "d"}

    S.check_follow_through(SKILL, questions, chain, n=300, min_seen=10)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "999", "b": "888", "c": "777", "d": "666"})


def test_deterministic():
    S.check_deterministic(TOPIC)
