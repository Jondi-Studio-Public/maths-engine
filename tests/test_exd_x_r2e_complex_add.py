"""Tests for the multi-part exam source of `add-complex-numbers` (`exd_x_r2e_complex_add`): k times a conjugate,
p from an equation, then the modulus and principal argument of p."""

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

TOPIC = "exd_x_r2e_complex_add"
SKILL_ID = "add-complex-numbers"
HARD = ("complex_number", {"kind": "add"})
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


def _fmt(z) -> str:
    re_, im_ = sympy.re(z), sympy.im(z)
    mag = "" if abs(im_) == 1 else str(abs(im_))
    if im_ == 0:
        return str(re_)
    if re_ == 0:
        return f"{'-' if im_ < 0 else ''}{mag}i"
    return f"{re_} {'+' if im_ > 0 else '-'} {mag}i"


def _two_dp(x) -> str:
    return str(Decimal(str(sympy.N(x, 30))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _facts(q):
    m = re.search(
        r"Let q = (.+?)\. The complex number p satisfies p ([+-]) (\d+) times the conjugate of q = (.+)\.$", q.stem
    )
    return _c(m.group(1)), (1 if m.group(2) == "+" else -1), int(m.group(3)), _c(m.group(4))


def _p(q):
    qq, sign, k, s_ = _facts(q)
    return sympy.expand(s_ - sign * k * sympy.conjugate(qq))


def oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the stem alone."""
    qq, _sign, k, _s = _facts(q)
    p = _p(q)
    return {
        "a": _fmt(sympy.expand(k * sympy.conjugate(qq))),
        "b": _fmt(p),
        "c": _two_dp(sympy.Abs(p)),
        "d": _two_dp(sympy.arg(p)),
    }


def test_wired_to_the_skill():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam" and s.effective_key == "exam"]
    assert [(s.topic, s.params, s.stages) for s in exam] == [(TOPIC, {"tier": "exam"}, ["exam"])]


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert [p.answer_kind for p in q.parts] == ["expression", "expression", "number", "number"]


def test_prompt_shapes_add_to_the_hard_tier():
    shapes = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[0], seed=s, difficulty="hard", **HARD[1])) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert shapes - hard


def test_no_text_defects(questions):
    for q in questions[:60]:
        assert find_text_defects(q) == []


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_answer_space(questions):
    tuples = collections.Counter(tuple(p.answer_text for p in q.parts) for q in questions)
    assert len(tuples) >= 100
    assert max(tuples.values()) / len(questions) <= 0.05


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 7


def test_distractors_rejected_and_tagged(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (p.label, d)
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_oracle_agrees_as_numbers(questions):
    """The complex answers are compared as numbers, not as strings."""
    for q in questions:
        for label in "ab":
            assert _c(_canonical(q)[label]) == _c(oracle(q)[label])


def test_follow_through_from_a_wrong_conjugate(questions):
    """A wrong (a) used correctly earns (b), then (c) and (d) from that (b)."""
    for q in questions[:120]:
        _qq, sign, _k, s_ = _facts(q)
        a_wrong = _c(_canonical(q)["a"]) + 1  # the real part one too high
        p_wrong = sympy.expand(s_ - sign * a_wrong)
        reply = {
            "a": _fmt(a_wrong),
            "b": _fmt(p_wrong),
            "c": _two_dp(sympy.Abs(p_wrong)),
            "d": _two_dp(sympy.arg(p_wrong)),
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through, (q.text, reply)
        assert result.parts["c"].marks == 2 and result.parts["d"].marks == 2, (q.text, reply)
        assert result.marks == 6 and not result.correct


def test_follow_through_from_a_wrong_p(questions):
    for q in questions[:80]:
        canon = _canonical(q)
        p_wrong = _c(canon["b"]) + 2 * sympy.I
        if sympy.im(p_wrong) == 0:
            continue  # an argument of 0 is a trivial reply, so no follow-through is given for it
        reply = {**canon, "b": _fmt(p_wrong), "c": _two_dp(sympy.Abs(p_wrong)), "d": _two_dp(sympy.arg(p_wrong))}
        result = grade(_SKILL, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["d"].marks == 2, (q.text, reply)


def test_all_wrong_earns_zero(questions):
    for q in questions[:50]:
        result = grade(_SKILL, q, {"a": "0", "b": "0", "c": "-9", "d": "-99"})
        assert result.marks == 0 and not result.correct


def test_deterministic():
    assert make(TOPIC, seed=11).to_dict() == make(TOPIC, seed=11).to_dict()
