"""Exam chain of `senior-integrate-expressions-of-the-form-frac-pm1-sqrt`: complete the square, exact area, mean."""

from __future__ import annotations

import math
import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from exd_x_r3f_support import (
    assert_all_wrong_earns_nothing,
    assert_answer_space,
    assert_clean_and_deterministic,
    assert_follow_through,
    assert_full_marks_and_distractors,
    assert_shapes,
    assert_wired,
    canonical,
    exam_question,
)

TOPIC = "exd_x_r3f_inverse_trig_chain"
SKILL_ID = "senior-integrate-expressions-of-the-form-frac-pm1-sqrt"
X = sympy.Symbol("x")
_T = standard_transformations + (implicit_multiplication_application,)


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_T, local_dict={"x": X})


def _simpson(f, lo, hi, steps=4000):
    h = (hi - lo) / steps
    total = f(lo) + f(hi) + sum((4 if i % 2 else 2) * f(lo + i * h) for i in range(1, steps))
    return total * h / 3


def read_question(q):
    """Everything the oracle needs, read back from the rendered stem and prompts."""
    match = re.match(r"The function f is given by f\(x\) = (\d+)/(sqrt\()?(.*?)\)?\.$", q.stem)
    assert match, q.stem
    k, root, quad_text = int(match.group(1)), bool(match.group(2)), match.group(3)
    if not root:
        quad_text = quad_text[1:] if quad_text.startswith("(") else quad_text
        quad_text = quad_text.rstrip(")")
    else:
        quad_text = quad_text.rstrip(")")
    quad = sympy.Poly(_parse(quad_text), X)
    limits = re.search(r"the lines x = (-?\d+) and x = (-?\d+)", q.parts[1].prompt)
    return k, root, quad, int(limits.group(1)), int(limits.group(2))


def oracle(q):
    k, root, quad, lo, hi = read_question(q)
    a2, b, c = quad.all_coeffs()
    if root:
        assert a2 == -1
        constant = c + sympy.Rational(b, 2) ** 2  # q in  q - (x + p)^2
        f = lambda v: k / math.sqrt(float(a2 * v * v + b * v + c))  # noqa: E731
    else:
        assert a2 == 1
        constant = c - sympy.Rational(b, 2) ** 2  # q in  (x + p)^2 + q
        f = lambda v: k / float(a2 * v * v + b * v + c)  # noqa: E731
    area = _simpson(f, lo, hi)
    return int(constant), area, area / (hi - lo), (k, root, b, lo, hi)


@pytest.fixture(scope="module")
def questions():
    return [exam_question(TOPIC, s) for s in range(300)]


def test_source_wired():
    assert_wired(SKILL_ID, TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 3, 2]
        assert [p.answer_kind for p in q.parts] == ["number", "expression", "expression"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_prompts_and_determinism():
    assert_shapes(SKILL_ID, TOPIC)
    assert_clean_and_deterministic(TOPIC)


def test_key_rederived_from_rendered_prompt(questions):
    families = set()
    for q in questions:
        m, area, mean, (_k, root, *_rest) = oracle(q)
        families.add(root)
        assert int(q.parts[0].answer_text) == m, q.text
        assert abs(float(sympy.N(_parse(q.parts[1].answer_text.replace("pi", "pi")))) - area) < 1e-7, q.text
        assert abs(float(sympy.N(_parse(q.parts[2].answer_text))) - mean) < 1e-7, q.text
    assert families == {True, False}


def test_grader_accepts_key_and_rejects_distractors():
    assert_full_marks_and_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    assert_answer_space(TOPIC)


def test_every_part_has_tagged_distractors(questions):
    for q in questions[:60]:
        for p in q.parts:
            assert len(p.distractors) >= 2
            assert all(p.option_notes[d] for d in p.distractors)


def _wrong_constant_replies(q):
    """A wrong q in (a) (the constant left unchanged), then (b) and (c) worked out correctly from it."""
    _m, _area, _mean, (k, root, b, lo, hi) = oracle(q)
    wrong = int(q.parts[0].distractors[0])
    if wrong <= 0:
        return None
    s = sympy.sqrt(wrong)
    h = sympy.Rational(-b, 2) if root else sympy.Rational(b, 2)
    inverse = sympy.asin if root else sympy.atan
    scale = 1 if root else sympy.Rational(1, 1) / s
    area = k * scale * (inverse((hi + h) / s) - inverse((lo + h) / s))
    if not area.is_real:
        return None
    area = sympy.simplify(area)
    return {"a": str(wrong), "b": str(area), "c": str(sympy.simplify(area / (hi - lo)))}


def test_follow_through_from_a_wrong_constant():
    assert_follow_through(SKILL_ID, TOPIC, _wrong_constant_replies, ["a"], ["b", "c"])


def test_follow_through_into_the_mean_value_only(questions):
    """A wrong area in (b) followed through correctly into (c) earns (c)."""
    from mathgen.engine import grade
    from exd_x_r3f_support import skill

    for q in questions[:60]:
        wrong_area = q.parts[1].distractors[0]
        mean = sympy.simplify(_parse(wrong_area) / (oracle(q)[3][4] - oracle(q)[3][3]))
        result = grade(skill(SKILL_ID), q, {"a": q.parts[0].answer_text, "b": wrong_area, "c": str(mean)})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks and result.parts["c"].follow_through


def test_all_wrong_earns_nothing():
    assert_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "999", "b": "pi/97", "c": "pi/89"})


def test_canonical_text_labels():
    q = exam_question(TOPIC, 3)
    assert set(canonical(q)) == {"a", "b", "c"}
