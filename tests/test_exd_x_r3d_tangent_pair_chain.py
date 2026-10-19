"""Exam chain of `senior-the-alternate-segment-theorem`: x, angle AQB, angle between tangents, angle OAB."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from tests import exd_x_r3d_support as S

TOPIC = "exd_x_r3d_tangent_pair_chain"
SKILL = S.load_skill("senior-the-alternate-segment-theorem")
_X = sympy.Symbol("x")
_TR = standard_transformations + (implicit_multiplication_application,)


def _expressions(q):
    found = re.findall(r"\(([^()]*x[^()]*)\) degrees", q.stem)
    assert len(found) == 2, q.stem
    return [parse_expr(t, transformations=_TR, local_dict={"x": _X}) for t in found]


def oracle(q):
    e1, e2 = _expressions(q)
    (x,) = sympy.solve(e1 - e2, _X)
    theta = e1.subs(_X, x)
    # Triangle PAB is isosceles (equal tangents); radius OA is perpendicular to the tangent PA.
    return {"a": str(x), "b": str(theta), "c": str(180 - 2 * theta), "d": str(90 - theta)}


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


def test_angles_are_a_valid_configuration(questions):
    for q in questions:
        theta, apb, oab = (int(q.parts[i].answer_text) for i in (1, 2, 3))
        assert 0 < theta < 90 and apb + 2 * theta == 180 and oab + theta == 90


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_x(questions):
    def chain(q):
        e1, e2 = _expressions(q)
        x = next((int(d) for d in q.parts[0].distractors if int(d) > 0), None)
        if x is None:
            return None
        theta = int(e2.subs(_X, x))
        if not 0 < theta < 90 or len({x, theta, 180 - 2 * theta, 90 - theta}) < 4:
            return None
        return {"a": str(x), "b": str(theta), "c": str(180 - 2 * theta), "d": str(90 - theta)}, {"b", "c", "d"}

    S.check_follow_through(SKILL, questions, chain, n=300, minimum=25)


def test_follow_through_from_wrong_angle_only(questions):
    seen = 0
    for q in questions[:100]:
        x = int(q.parts[0].answer_text)
        theta = next((int(d) for d in q.parts[1].distractors if 0 < int(d) < 90), 0)
        if not 0 < theta < 90 or len({x, theta, 180 - 2 * theta, 90 - theta}) < 4:
            continue
        result = S.grade(SKILL, q, {"a": str(x), "b": str(theta), "c": str(180 - 2 * theta), "d": str(90 - theta)})
        assert result.parts["a"].marks == q.parts[0].marks and result.parts["b"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks and result.parts["d"].marks == q.parts[3].marks
        seen += 1
    assert seen >= 20


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "999", "b": "998", "c": "997", "d": "996"})


def test_deterministic():
    S.check_deterministic(TOPIC)
