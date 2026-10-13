"""Tests `ucv1-trig-substitution-integrals` (topic `ucv1_trig_substitution`) and its exam chain.

The oracle reads the integrand and the limits back out of the question TEXT and evaluates the integral numerically
with mpmath, which shares nothing with the closed forms in the generator.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import mpmath as mp
import pytest
import sympy

import mathgen.generators.fy1cd_trig_substitution_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "ucv1_trig_substitution"
EXAM = "exd_x_fy_trig_substitution"
SKILL_ID = "ucv1-trig-substitution-integrals"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_X = sympy.Symbol("x")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _expr(text):
    """A plain-text expression such as '3√3/2', '1/(x√(x² - 36))' or '3 + 2√3' as a SymPy expression."""
    t = text.replace("x²", "x**2").replace("^(3/2)", "**(3/2)")
    t = re.sub(r"√(\d+)", r"sqrt(\1)", t)
    t = t.replace("√(", "sqrt(")
    t = re.sub(r"(\d|x|\))\s*sqrt", r"\1*sqrt", t)
    t = re.sub(r"(\d)x", r"\1*x", t)
    return sympy.sympify(t, locals={"x": _X})


def _integral_parts(text):
    m = re.search(r"∫ (.*?) dx from x = (.*?) to x = (.*?)(?:,|\.)(?: |$)", text)
    return _expr(m.group(1)), _expr(m.group(2)), _expr(m.group(3))


def _value(f, lo, hi):
    fn = sympy.lambdify(_X, f, "mpmath")
    with mp.workdps(25):
        return float(mp.quad(fn, [float(lo), float(hi)]))


# -- wiring -------------------------------------------------------------------------------------------------------


def test_skill_wiring():
    assert _SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert _SKILL.sources[-1].stages == ["exam"]
    assert all(m.id in CATALOGUE for m in _SKILL.misconceptions)


# -- single-answer tiers ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_key_is_the_integral_to_three_places(tier):
    for seed in range(80):
        q = _q(tier, seed)
        f, lo, hi = _integral_parts(q.text)
        assert abs(_value(f, lo, hi) - float(q.answer)) <= 0.0005 + 1e-9, (seed, q.text, q.answer_text)
        assert len(q.answer_text.split(".")[1]) == 3


def test_tiers_use_the_expected_integrands():
    forms = {t: Counter() for t in TIERS}
    for tier in TIERS:
        for seed in range(120):
            text = _q(tier, seed).text
            if "x² - " in text and "√(x²" in text:
                forms[tier]["secant"] += 1
            elif "x²/√" in text:
                forms[tier]["x-squared"] += 1
            elif "^(3/2)" in text:
                forms[tier]["three-halves"] += 1
            elif "√(" in text:
                forms[tier]["root"] += 1
            elif re.search(r"x² - \d+x \+", text):
                forms[tier]["completed-square"] += 1
            else:
                forms[tier]["standard"] += 1
    assert set(forms["easy"]) == {"root", "standard"}
    assert set(forms["medium"]) == {"root", "three-halves"}
    assert set(forms["hard"]) == {"secant", "x-squared", "completed-square"}
    assert min(forms["hard"].values()) >= 20


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(100):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, seed
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, text, q.answer_text)
            assert len(text.split(".")[1]) == 3  # same number of decimal places as the answer
            assert q.option_notes[text] in CATALOGUE
            assert text in q.distractor_values


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, find_text_defects(q, include_working=True))
        assert q.to_json() == _q(tier, seed).to_json()


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    texts = [_q(tier, s).text for s in range(100)]
    answers = [_q(tier, s).answer_text for s in range(100)]
    assert len(set(texts)) >= 20
    assert len(set(answers)) >= 6


# -- exam chain ---------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["fraction", "number", "number", "number"]
        assert [p.marks for p in q.parts] == [2, 2, 2, 2]
        assert [bool(p.follow_through) for p in q.parts] == [False, False, True, True]
        assert find_text_defects(q, include_working=True) == []
        assert q.to_json() == _exam(seed).to_json()


def test_exam_keys_match_independent_derivation():
    for seed in range(100):
        q = _exam(seed)
        f, lo, hi = _integral_parts(q.text.replace("Let I = ", "").replace(". (a)", ", (a)").split(", (a)")[0] + ".")
        a = int(re.search(r"x = (\d+) sin", q.parts[0].prompt).group(1))
        theta1 = sympy.asin(hi / a)
        assert Fraction(q.parts[0].answer_text) == Fraction(int((theta1 / sympy.pi).p), int((theta1 / sympy.pi).q))
        assert int(q.parts[1].answer) == a * a
        inner = _value(sympy.sin(_X) ** 2, 0, theta1)
        assert abs(float(q.parts[2].answer) - inner) <= 0.00005 + 1e-9
        assert abs(float(q.parts[3].answer) - _value(f, lo, hi)) <= 0.0005 + 1e-9
        assert len(q.parts[2].answer_text.split(".")[1]) == 4 and len(q.parts[3].answer_text.split(".")[1]) == 3


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(80):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 8
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)
                assert p.option_notes[text] in CATALOGUE


def test_exam_follow_through_from_a_wrong_limit_and_a_wrong_constant():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        # (c) from a wrong (a)
        for wrong in q.parts[0].distractors:
            p = Fraction(wrong)
            own_c = float(
                sympy.pi * sympy.Rational(p.numerator, p.denominator) / 2
                - sympy.sin(2 * sympy.pi * sympy.Rational(p.numerator, p.denominator)) / 4
            )
            text = str(Decimal(repr(own_c)).quantize(Decimal("0.0001"), ROUND_HALF_UP))
            if text == q.parts[2].answer_text:
                continue
            result = grade(_SKILL, q, {**canon, "a": wrong, "c": text})
            if result.parts["c"].marks == 2:
                assert result.parts["c"].follow_through is True
                checked += 1
        # (d) from a wrong (b)
        for wrong in q.parts[1].distractors:
            text = str((Decimal(wrong) * Decimal(q.parts[2].answer_text)).quantize(Decimal("0.001"), ROUND_HALF_UP))
            if text == q.parts[3].answer_text:
                continue
            result = grade(_SKILL, q, {**canon, "b": wrong, "d": text})
            assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (seed, wrong, text)
            checked += 1
    assert checked >= 60


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(200)]
    assert len(set(answers)) >= 20
    assert len({_exam(s).text for s in range(200)}) >= 20
