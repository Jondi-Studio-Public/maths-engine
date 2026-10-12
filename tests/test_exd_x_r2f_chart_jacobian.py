"""Exam source of `ucv1-manifold-chart-jacobian`: a four-part chain from the Jacobian determinant to a linear estimate."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2f_chart_jacobian"
SKILL_ID = "ucv1-manifold-chart-jacobian"
SEEDS = range(300)
X, Y = sympy.symbols("x y")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_STEM = re.compile(r"u = (.*?) and v = (.*?)\. Let J .*point \((-?\d+), (-?\d+)\)\.")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _expr(text):
    return parse_expr(text.replace("²", "**2"), transformations=_TRANSFORMS, local_dict={"x": X, "y": Y})


def _frac(value) -> str:
    value = sympy.Rational(value)
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def oracle(q):
    """Re-derive every part from the rendered stem and prompts, by differentiating the displayed chart."""
    u_text, v_text, x0, y0 = _STEM.search(q.stem).groups()
    u, v = _expr(u_text), _expr(v_text)
    x0, y0 = int(x0), int(y0)
    step = sympy.Rational(re.search(r"u₀ \+ ([\d.]+)", q.parts[3].prompt).group(1))
    jac = sympy.Matrix([[sympy.diff(f, var) for var in (X, Y)] for f in (u, v)])
    det = sympy.expand(jac.det())
    at_p = jac.subs({X: x0, Y: y0})
    inv = at_p.inv()
    return det, det.subs({X: x0, Y: y0}), inv[0, 0], x0 + step * inv[0, 0]


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert [p.answer_kind for p in q.parts] == ["expression", "fraction", "fraction", "fraction"]
        assert all(len(p.distractors) >= 2 and p.option_notes for p in q.parts)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("p15d_chart_jacobian", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects(questions):
    for q in questions[:60]:
        assert find_text_defects(q) == []


def test_oracle_agrees(questions):
    for q in questions:
        det, det_p, dxdu, estimate = oracle(q)
        assert sympy.expand(_expr(q.parts[0].answer_text) - det) == 0, q.text
        assert q.parts[1].answer_text == _frac(det_p), q.text
        assert q.parts[2].answer_text == _frac(dxdu), q.text
        assert q.parts[3].answer_text == _frac(estimate), q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 7


def test_equivalent_forms_of_part_a(questions):
    for q in questions[:40]:
        det = _expr(q.parts[0].answer_text)
        spaced = str(sympy.expand(det)).replace("**", "^").replace("*", "")
        canon = _canonical(q)
        assert grade(_SKILL, q, {**canon, "a": spaced}).parts["a"].marks == 2, (q.text, spaced)


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_determinant(questions):
    """A slip in det J at P, carried through (c) and (d), loses only (b)."""
    checked = 0
    for q in questions[:100]:
        u_text, v_text, x0, y0 = _STEM.search(q.stem).groups()
        x0, y0 = int(x0), int(y0)
        step = sympy.Rational(re.search(r"u₀ \+ ([\d.]+)", q.parts[3].prompt).group(1))
        b_wrong = sympy.Rational(q.parts[1].distractors[0])
        v_y = sympy.diff(_expr(v_text), Y).subs({X: x0, Y: y0})
        c_wrong = v_y / b_wrong
        d_wrong = x0 + step * c_wrong
        canon = _canonical(q)
        reply = {**canon, "b": _frac(b_wrong), "c": _frac(c_wrong), "d": _frac(d_wrong)}
        result = grade(_SKILL, q, reply)
        assert result.parts["b"].marks == 0
        # The engine withholds follow-through from a reply that merely repeats an earlier one.
        if b_wrong != 0 and c_wrong != 0 and c_wrong != b_wrong and d_wrong not in (b_wrong, c_wrong):
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (q.text, reply)
            assert result.parts["d"].marks == 2, (q.text, reply)
            assert result.marks == 6 and not result.correct
            checked += 1
    assert checked >= 70


def test_follow_through_from_wrong_expression(questions):
    """A wrong det J expression evaluated correctly at P still earns (b)."""
    for q in questions[:60]:
        u_text, v_text, x0, y0 = _STEM.search(q.stem).groups()
        wrong = _expr(q.parts[0].distractors[0])
        value = wrong.subs({X: int(x0), Y: int(y0)})
        if value == 0:
            continue
        result = grade(_SKILL, q, {**_canonical(q), "a": q.parts[0].distractors[0], "b": _frac(value)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1, (q.text, value)


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "x", "b": "9999", "c": "7/3", "d": "5/11"})
        assert result.marks == 0 and not result.correct


def test_answer_space(questions):
    tuples = Counter(q.answer_text for q in questions)
    assert len(tuples) >= 100
    assert max(tuples.values()) / len(questions) <= 0.05


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
