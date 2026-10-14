"""Exam source of `centered-difference-polynomial`: centred differences at h and h/2 and their extrapolation."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from mathgen.generators._u4_kit import survives_nsimplify
from tests.exd_x_u4_support import (
    check_answer_space,
    check_distractors_rejected,
    check_full_marks,
    check_no_text_defects,
    check_tagged,
    exam,
    frac,
    shapes,
    skill,
)

TOPIC = "exd_x_u4_richardson_difference"
SKILL_ID = "centered-difference-polynomial"
_T = sympy.Symbol("t")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _oracle(text: str) -> dict[str, Fraction]:
    body = re.search(r"s\(t\) = (.+?)\. The velocity", text).group(1)
    s = parse_expr(body.replace("³", "**3").replace("²", "**2"), transformations=_TRANSFORMS, local_dict={"t": _T})
    t0 = sympy.Integer(re.search(r"velocity at t = (\d+)", text).group(1))
    h = sympy.Rational(1, int(re.search(r"Take h = 1/(\d+)", text).group(1)))

    def centred(step):
        return (s.subs(_T, t0 + step) - s.subs(_T, t0 - step)) / (2 * step)

    d1, d2 = centred(h), centred(h / 2)
    improved = (4 * d2 - d1) / 3
    assert improved == sympy.diff(s, _T).subs(_T, t0)  # exact for a cubic

    def to_fraction(v):
        return Fraction(int(v.p), int(v.q))

    return {"a": to_fraction(d1), "b": to_fraction(d2), "c": to_fraction(improved)}


def test_prompt_variety_and_new_shape():
    exam_shapes = shapes(TOPIC, 60, difficulty="exam")
    hard = shapes("centered_difference_polynomial", 60, difficulty="hard")
    assert len(exam_shapes) >= 3
    assert exam_shapes - hard


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam(TOPIC, seed)
        assert {p.label: frac(p.answer_text) for p in q.parts} == _oracle(q.text), (seed, q.text)


def test_structure():
    for seed in range(60):
        q = exam(TOPIC, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors_rejected(SKILL_ID, TOPIC)
    check_tagged(TOPIC)


def test_follow_through_from_wrong_estimates():
    sk = skill(SKILL_ID)
    tested = 0
    for seed in range(100):
        q = exam(TOPIC, seed)
        a = frac(q.parts[0].answer_text) + Fraction(1, 2)
        b = frac(q.parts[1].answer_text) - Fraction(1, 4)
        c = (4 * b - a) / 3
        if len({a, b, c}) < 3 or not survives_nsimplify(a, b, c):
            continue
        tested += 1
        result = grade(sk, q, {"a": str(a), "b": str(b), "c": str(c)})
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks
        assert result.parts["c"].follow_through is True
        assert not result.correct
    assert tested >= 40


def test_all_wrong_earns_zero():
    sk = skill(SKILL_ID)
    for seed in range(40):
        assert grade(sk, exam(TOPIC, seed), {"a": "9999", "b": "9998", "c": "9997"}).marks == 0


def test_deterministic():
    assert exam(TOPIC, 7).to_json() == exam(TOPIC, 7).to_json()
