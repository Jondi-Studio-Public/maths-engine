"""Tests the multi-part exam tier of `senior-use-factorial-notation`: simplify, evaluate, solve."""

from __future__ import annotations

import math
import re

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade
from tests import exd_x_r3b_support as S

TOPIC = "exd_x_r3b_factorial"
SKILL_ID = "senior-use-factorial-notation"
_SKILL = S.skill(SKILL_ID)
n = sympy.Symbol("n")
_FACT = r"(?:n|\(n ([+-]) (\d+)\))!"
_EXPRESSION = re.compile(rf"S\(n\) = ({_FACT})/({_FACT}) ([+-]) ({_FACT})/({_FACT})")


@pytest.fixture(scope="module")
def questions():
    return S.exam_questions(TOPIC, 300)


def _offset(sign: str | None, amount: str | None) -> int:
    return 0 if sign is None else (1 if sign == "+" else -1) * int(amount)


def _value_at(q, at: int) -> int:
    """S(at), computed from the stem alone with math.factorial."""
    g = _EXPRESSION.search(q.stem).groups()
    a1, b1, op, a2, b2 = (_offset(g[1], g[2]), _offset(g[4], g[5]), g[6], _offset(g[8], g[9]), _offset(g[11], g[12]))
    first = math.factorial(at + a1) // math.factorial(at + b1)
    second = math.factorial(at + a2) // math.factorial(at + b2)
    return first + second if op == "+" else first - second


def test_source_wired():
    S.check_wiring(TOPIC, SKILL_ID)


def test_prompt_variety_and_new_shape_over_hard_tier():
    S.check_shapes(TOPIC, SKILL_ID)


def test_structure():
    S.check_structure(TOPIC, "abc", [2, 1, 2])


def test_keys_match_direct_factorial_evaluation_over_300_seeds(questions):
    for q in questions:
        key = sympy.sympify(q.parts[0].answer_text.replace("^", "**"))
        for at in range(2, 12):
            assert key.subs(n, at) == _value_at(q, at), (q.stem, at)
        n0 = int(re.search(r"S\((\d+)\)", q.parts[1].prompt).group(1))
        assert int(q.parts[1].answer_text) == _value_at(q, n0), q.stem
        target = int(re.search(r"S\(n\) = (\d+)", q.parts[2].prompt).group(1))
        solutions = [m for m in range(2, 200) if _value_at(q, m) == target]
        assert solutions == [int(q.parts[2].answer_text)], q.stem


def test_grading_accepts_key_and_rejects_every_distractor():
    S.check_grading(TOPIC, SKILL_ID, 120)


def test_distractors_are_tagged_and_formatted():
    S.check_tags(TOPIC)


def test_follow_through_from_a_wrong_simplification(questions):
    checked = 0
    for q in questions[:100]:
        wrong = q.parts[0].distractors[0]
        poly = sympy.sympify(wrong.replace("^", "**"))
        n0 = int(re.search(r"S\((\d+)\)", q.parts[1].prompt).group(1))
        target = int(re.search(r"S\(n\) = (\d+)", q.parts[2].prompt).group(1))
        b_wrong = poly.subs(n, n0)
        sols = [m for m in sympy.solve(poly - target, n) if m.is_real and m > 0]
        if (
            sympy.Poly(poly, n).degree() != 2
            or sympy.Poly(poly, n).LC() != sympy.Poly(sympy.sympify(q.parts[0].answer_text.replace("^", "**")), n).LC()
        ):
            continue  # the follow-through uses the stated leading coefficient, so only same-lead wrong forms apply
        if not sols or str(b_wrong) == q.parts[1].answer_text:
            continue
        c_wrong = max(sols)
        c_text = str(int(round(float(c_wrong))))
        result = grade(_SKILL, q, {"a": wrong, "b": str(b_wrong), "c": c_text})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (q.stem, wrong)
        assert result.parts["c"].marks == 2, (q.stem, wrong, c_text)
        checked += 1
    assert checked >= 5


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "n^2", "b": "7", "c": "3"})
        assert result.marks == 0 and not result.correct


def test_answer_space_over_300_seeds():
    S.check_answer_space(TOPIC)


def test_clean_and_deterministic():
    S.check_clean_and_deterministic(TOPIC)
    assert make(TOPIC, seed=7, difficulty="exam").to_dict() == make(TOPIC, seed=7, difficulty="exam").to_dict()
