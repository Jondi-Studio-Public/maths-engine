"""Exam source of `condition-number-diagonal`: a condition number used to bound and budget measurement error."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

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

TOPIC = "exd_x_u4_condition_error_bound"
SKILL_ID = "condition-number-diagonal"


def _oracle(text: str) -> dict[str, Fraction]:
    entries = [sympy.Rational(v.replace(" ", "")) for v in re.search(r"diag\(([^)]*)\)", text).group(1).split(",")]
    matrix = sympy.diag(*entries)
    if "A⁻¹ =" in text:
        matrix = matrix.inv()  # the matrix shown is the inverse
    kappa = sympy.nsimplify(sympy.simplify(matrix.norm(2) * matrix.inv().norm(2)))
    eps = sympy.Rational(re.search(r"at most ([\d.]+)%", text).group(1))
    tau = sympy.Rational(re.search(r"must not exceed (\d+)%", text).group(1))

    def to_fraction(v):
        return Fraction(int(v.p), int(v.q))

    return {"a": to_fraction(kappa), "b": to_fraction(kappa * eps), "c": to_fraction(tau / kappa)}


def test_prompt_variety_and_new_shape():
    exam_shapes = shapes(TOPIC, 60, difficulty="exam")
    hard = shapes("condition_number_diagonal", 60, difficulty="hard", size=3)
    assert len(exam_shapes) >= 3
    assert exam_shapes - hard


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam(TOPIC, seed)
        assert {p.label: frac(p.answer_text) for p in q.parts} == _oracle(q.text), (seed, q.text)


def test_both_forms_and_signs_appear():
    texts = [exam(TOPIC, s).text for s in range(60)]
    assert any("A⁻¹ =" in t for t in texts) and any(
        "A = diag" in t or "A = {" in t or "scaled by A" in t for t in texts
    )
    assert any("diag(-" in t or ", -" in t.split("diag(")[1].split(")")[0] for t in texts if "diag(" in t)


def test_structure():
    for seed in range(60):
        q = exam(TOPIC, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors_rejected(SKILL_ID, TOPIC)
    check_tagged(TOPIC)


def test_follow_through_from_wrong_condition_number():
    sk = skill(SKILL_ID)
    tested = 0
    for seed in range(100):
        q = exam(TOPIC, seed)
        text = q.text
        eps = Fraction(re.search(r"at most ([\d.]+)%", text).group(1))
        tau = Fraction(re.search(r"must not exceed (\d+)%", text).group(1))
        a = frac(q.parts[0].answer_text) + Fraction(1, 2)  # a wrong, sensible kappa
        b, c = a * eps, tau / a
        if len({a, b, c}) < 3 or not survives_nsimplify(a, b, c):
            continue
        tested += 1
        result = grade(sk, q, {"a": str(a), "b": str(b), "c": str(c)})
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks
            assert result.parts[label].follow_through is True
        assert not result.correct
    assert tested >= 80


def test_all_wrong_earns_zero():
    sk = skill(SKILL_ID)
    for seed in range(40):
        assert grade(sk, exam(TOPIC, seed), {"a": "999", "b": "998", "c": "997"}).marks == 0


def test_deterministic():
    assert exam(TOPIC, 7).to_json() == exam(TOPIC, 7).to_json()
