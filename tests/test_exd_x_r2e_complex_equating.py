"""Tests for the multi-part exam source of `senior-review-cartesian-form` (`exd_x_r2e_complex_equating`): solving
z^2 = w by equating real and imaginary parts, then the modulus or argument of z."""

from __future__ import annotations

import collections
import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2e_complex_equating"
SKILL_ID = "senior-review-cartesian-form"
HARD = ("p11c_cartesian_form_equating", {"mode": "hard"})
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _c(text: str):
    """A complex number written a + bi, as a sympy number."""
    t = re.sub(r"(\d)i", r"\1*I", text)
    t = re.sub(r"(?<![\w*])i", "1*I", t)
    return sympy.sympify(t)


def _two_dp(x) -> str:
    return str(Decimal(str(sympy.N(x, 30))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _facts(q):
    m = re.search(r"z\^2 = (.+?) and Re\(z\) ([<>]) 0\.", q.stem)
    return _c(m.group(1)), (1 if m.group(2) == ">" else -1)


def _solve(q):
    """x^2, x and y from the quadratic in x^2, in exact arithmetic."""
    w, side = _facts(q)
    a_, b_ = sympy.re(w), sympy.im(w)
    x2 = (a_ + sympy.sqrt(a_**2 + b_**2)) / 2
    x = side * sympy.sqrt(x2)
    return x2, x, b_ / (2 * x)


def _d(q, x, y) -> str:
    z = x + y * sympy.I
    return _two_dp(sympy.Abs(z)) if "modulus" in q.parts[3].prompt else _two_dp(sympy.arg(z))


def oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the stem alone."""
    x2, x, y = _solve(q)
    return {"a": str(x2), "b": str(x), "c": str(y), "d": _d(q, x, y)}


def test_wired_to_the_skill():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.params, s.stages) for s in exam] == [(TOPIC, {"tier": "exam"}, ["exam"])]


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [3, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompt_shapes_add_to_the_hard_tier():
    shapes = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[0], seed=s, difficulty="hard", **HARD[1])) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert shapes - hard


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 8


def test_distractors_rejected_and_tagged(questions):
    for q in questions[:100]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (p.label, d)
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_both_kinds_of_final_part_occur(questions):
    kinds = collections.Counter("modulus" in q.parts[3].prompt for q in questions)
    assert kinds[True] >= 80 and kinds[False] >= 80


def test_follow_through_from_a_wrong_square(questions):
    """A wrong x^2 (the square of another factor of xy) carried through x, y and the modulus or argument."""
    checked = 0
    for q in questions[:150]:
        x2, x, y = _solve(q)
        _w, side = _facts(q)
        product = abs(int(x * y))
        roots = [k for k in range(1, product + 1) if product % k == 0 and k != abs(int(x))]
        if not roots:
            continue
        k = roots[0]
        a_wrong, b_wrong = k * k, side * k
        c_wrong = sympy.im(_facts(q)[0]) / (2 * b_wrong)
        reply = {"a": str(a_wrong), "b": str(b_wrong), "c": str(c_wrong), "d": _d(q, b_wrong, c_wrong)}
        if len({reply["a"], reply["b"], reply["c"], reply["d"]}) < 4:
            continue
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through, (q.text, reply)
        assert result.parts["c"].marks == 2 and result.parts["d"].marks == 2, (q.text, reply)
        checked += 1
    assert checked >= 60


def test_follow_through_from_a_wrong_y(questions):
    for q in questions[:80]:
        canon = _canonical(q)
        x, y = sympy.Integer(canon["b"]), sympy.Integer(canon["c"])
        y += 4 if y == -3 else 3  # never 0: an argument or modulus of a purely real number is trivial
        result = grade(_SKILL, q, {**canon, "c": str(y), "d": _d(q, x, y)})
        assert result.parts["c"].marks == 0
        assert result.parts["d"].marks == 2, q.text


def test_all_wrong_earns_zero(questions):
    for q in questions[:50]:
        result = grade(_SKILL, q, {"a": "-4", "b": "0", "c": "-7777", "d": "-99"})
        assert result.marks == 0 and not result.correct
