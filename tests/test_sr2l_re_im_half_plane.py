"""locus-re-im-half-plane: a condition on Re or Im of a transformation of z, turned into a condition on x or y."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from tests.sr2l_support import (
    check_answer_space,
    check_clean,
    check_graded_exam,
    check_graded_single,
    check_structure,
    check_tiers_differ,
    check_wired,
    expr,
    line_form,
    normal_line,
    number_close,
    questions,
    stem_of,
    x,
    y,
)

SKILL = "locus-re-im-half-plane"
TOPICS = {"easy": "sr2l_re_im", "medium": "sr2l_re_im", "hard": "sr2l_re_im", "exam": "sr2l_re_im_exam"}
_ZB = x - sympy.I * y
_Z = x + sympy.I * y
_CONDITION = re.compile(r"((?:-?\d+ [<≤] )?(?:Re|Im)\([^)]*\)(?: [<≤>≥] -?\d+)?)")
_GRID = [Fraction(i, 6) for i in range(-72, 73)]
_ROWS = [Fraction(-3), Fraction(0), Fraction(5, 7)]


def _w(text: str):
    """The expression inside Re( ) or Im( ), with z = x + iy and the conjugate z̄ = x - iy."""
    t = text.replace("z̄", "W").replace("z", "V")
    t = re.sub(r"(\d)i", r"\1*I", t)
    t = re.sub(r"(?<![A-Za-z])i(?=[VW])", "I*", t)
    t = re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "I", t)
    t = re.sub(r"I(?=[VW])", "I*", t)
    return sympy.expand(expr(t, V=_Z, W=_ZB))


def _linear(part: str, text: str):
    """(a, b, c) with part(w) = a x + b y + c, for the inside text of Re( ) or Im( )."""
    v = sympy.re(_w(text)) if part == "Re" else sympy.im(_w(text))
    v = sympy.expand(v)
    return sympy.Rational(v.coeff(x, 1)), sympy.Rational(v.coeff(y, 1)), sympy.Rational(v.subs({x: 0, y: 0}))


_CMP = {"<": lambda a, b: a < b, "≤": lambda a, b: a <= b, ">": lambda a, b: a > b, "≥": lambda a, b: a >= b}


def _stem_region(text: str):
    """A predicate on (x, y) read from the printed condition alone."""
    cond = _CONDITION.search(text).group(1)
    double = re.fullmatch(r"(-?\d+) ([<≤]) (Re|Im)\(([^)]*)\) ([<≤]) (-?\d+)", cond)
    if double:
        lo, o1, part, inner, o2, hi = double.groups()
        a, b, c = _linear(part, inner)
        return lambda px, py: (
            _CMP[o1](Fraction(lo), Fraction(a * px + b * py + c))
            and _CMP[o2](Fraction(a * px + b * py + c), Fraction(hi))
        )
    part, inner, op, k = re.fullmatch(r"(Re|Im)\(([^)]*)\) ([<≤>≥]) (-?\d+)", cond).groups()
    a, b, c = _linear(part, inner)
    return lambda px, py: _CMP[op](Fraction(a * px + b * py + c), Fraction(k))


def _answer_region(answer: str):
    double = re.fullmatch(r"(-?[\d/]+) ([<≤]) ([xy]) ([<≤]) (-?[\d/]+)", answer)
    if double:
        lo, o1, var, o2, hi = double.groups()
        return lambda px, py: (
            _CMP[o1](Fraction(lo), px if var == "x" else py) and _CMP[o2](px if var == "x" else py, Fraction(hi))
        )
    var, op, bound = re.fullmatch(r"([xy]) ([<≤>≥]) (-?[\d/]+)", answer).groups()
    return lambda px, py: _CMP[op](px if var == "x" else py, Fraction(bound))


def _agree(stem_pred, answer: str):
    ans = _answer_region(answer)
    for px in _GRID:
        for py in _ROWS:
            assert stem_pred(px, py) == ans(px, py), (answer, px, py)


def test_wired():
    check_wired(SKILL, TOPICS)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_agrees_with_the_printed_condition(tier):
    for q in questions("sr2l_re_im", tier, 80):
        _agree(_stem_region(stem_of(q)), q.answer_text)


def test_each_tier_has_its_features():
    medium = " ".join(stem_of(q) for q in questions("sr2l_re_im", "medium", 80))
    hard = " ".join(stem_of(q) for q in questions("sr2l_re_im", "hard", 120))
    assert re.search(r"\(-\d", medium), "no negative coefficient"
    assert "z̄" in hard and re.search(r"\d?iz", hard), "no conjugate or multiple of i"
    assert re.search(r"\d [<≤] (Re|Im)", hard), "no strip"
    answers = {q.answer_text for q in questions("sr2l_re_im", "hard", 120)}
    assert any(a.count("<") + a.count("≤") == 2 for a in answers)


def test_reversal_happens_with_a_negative_coefficient():
    reversed_seen = 0
    for q in questions("sr2l_re_im", "medium", 80):
        cond = _CONDITION.search(stem_of(q)).group(1)
        m = re.fullmatch(r"(Re|Im)\((-?\d*)z[^)]*\) ([<≤>≥]) -?\d+", cond)
        if m and m.group(2).startswith("-"):
            sign_in, sign_out = m.group(3), q.answer_text.split()[1]
            assert sign_in.replace("≤", "<").replace("≥", ">") != sign_out.replace("≤", "<").replace("≥", ">")
            reversed_seen += 1
    assert reversed_seen >= 5


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_graded(tier):
    check_graded_single(SKILL, "sr2l_re_im", tier)


def test_tiers_differ():
    check_tiers_differ(SKILL, TOPICS)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    check_answer_space("sr2l_re_im", tier, minimum=10)


def _exam_region(q):
    m = re.search(r"((?:Re|Im)\([^)]*\)) ([≥≤]) ((?:Re|Im)\([^)]*\))", stem_of(q))
    left, op, right = m.groups()

    def side(t):
        part, inner = re.fullmatch(r"(Re|Im)\((.*)\)", t).groups()
        return _linear(part, inner)

    (a1, b1, c1), (a2, b2, c2) = side(left), side(right)
    return op, (a1 - a2, b1 - b2, c1 - c2)


def test_exam_matches_an_independent_reading():
    for q in questions("sr2l_re_im_exam", "exam", 60):
        op, (a, b, c) = _exam_region(q)
        pa, pb, pc, pd = q.parts
        assert b != 0 and a != 0
        assert line_form(pa.answer_text) == normal_line(a * x + b * y + c), stem_of(q)
        assert number_close(pb.answer_text, -c / a, 0.0001), stem_of(q)
        on_axis = _CMP[op](Fraction(c), Fraction(0))  # the origin, to confirm the stem's claim that it is not in S
        assert not on_axis, stem_of(q)
        _agree(lambda px, py, a=a, c=c, op=op: _CMP[op](Fraction(a * px + c), Fraction(0)), pc.answer_text)
        assert number_close(pd.answer_text, abs(c) / sympy.sqrt(a**2 + b**2), 0.0051), stem_of(q)


def test_exam_structure_and_grading():
    check_structure(
        "sr2l_re_im_exam", ["pair", "number", "inequality", "number"], [1, 1, 2, 2], [False, False, False, True]
    )
    check_graded_exam(SKILL, "sr2l_re_im_exam")


def test_exam_least_modulus_follows_through_from_a_wrong_intercept():
    from mathgen.engine import grade

    from tests.sr2l_support import skill

    sk, used = skill(SKILL), 0
    for q in questions("sr2l_re_im_exam", "exam", 40):
        pa, pb, pc, pd = q.parts
        wrong = pb.distractors[0]
        value = abs(float(sympy.Rational(wrong))) / 2**0.5
        result = grade(sk, q, {"a": pa.answer_text, "b": wrong, "c": pc.answer_text, "d": f"{value + 1e-9:.2f}"})
        assert result.parts["b"].marks == 0
        assert result.parts["d"].marks == result.parts["d"].max_marks, (q.text, wrong, value)
        used += 1
    assert used


def test_clean_and_deterministic():
    check_clean("sr2l_re_im", ["easy", "medium", "hard"])
    check_clean("sr2l_re_im_exam", ["exam"])
