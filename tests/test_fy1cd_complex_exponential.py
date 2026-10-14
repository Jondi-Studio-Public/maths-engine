"""Tests `ucv1-complex-exponential-form` (topic `ucv1_complex_exponential`) and its exam chain.

The oracle reads the numbers back out of the question TEXT and re-derives the key with SymPy (`arg`, `Abs`, powers).
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy

import mathgen.generators.fy1cd_complex_exponential_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "ucv1_complex_exponential"
EXAM = "exd_x_fy_complex_exponential"
SKILL_ID = "ucv1-complex-exponential-form"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _cartesian(text):
    """The number after 'z = ' in a stem such as 'z = -3 + 3√3 i'."""
    body = re.search(r"z = ([^.]*?) in|z = ([^.]*)\.", text)
    body = (body.group(1) or body.group(2)).replace(" - ", " + -")
    total = 0
    for term in body.split(" + "):
        sign = -1 if term.startswith("-") else 1
        term = term.lstrip("-")
        imag = term.endswith("i")
        if imag:
            term = term[:-1].strip()
        root = "√3" in term
        coef = term.replace("√3", "").strip()
        value = sign * (int(coef) if coef else 1) * (sympy.sqrt(3) if root else 1)
        total += value * (sympy.I if imag else 1)
    return total


def _angle(s):
    """'-5π/6', 'π/4', '2π/3', '0' -> a Fraction of pi."""
    s = s.replace("π", "")
    if s in ("", "-"):
        s += "1"
    return Fraction(s) if "/" in s or s.lstrip("-").isdigit() else Fraction(1)


def _expo(text, name):
    m = re.search(rf"{name} = (\d*)e\^\((-?)i(\d*)π(?:/(\d+))?\)", text)
    r = int(m.group(1) or 1)
    num = int(m.group(3) or 1)
    p = Fraction(num, int(m.group(4) or 1)) * (-1 if m.group(2) else 1)
    return r, p


def _arg_over_pi(expr):
    value = sympy.simplify(sympy.arg(sympy.expand(expr)) / sympy.pi)
    return Fraction(int(value.p), int(value.q))


def _unit(p):
    return sympy.exp(sympy.I * sympy.pi * sympy.Rational(p.numerator, p.denominator))


# -- wiring -------------------------------------------------------------------------------------------------------


def test_skill_wiring():
    assert _SKILL.answer == "fraction"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert _SKILL.sources[-1].stages == ["exam"]
    ids = {m.id for m in _SKILL.misconceptions}
    assert all(m in CATALOGUE for m in ids)


# -- single-answer tiers ------------------------------------------------------------------------------------------


def test_easy_key_is_the_argument_of_the_stated_number():
    seen = Counter()
    for seed in range(150):
        q = _q("easy", seed)
        z = _cartesian(q.text)
        assert Fraction(q.answer_text) == _arg_over_pi(z), (seed, q.text)
        seen[q.answer_text] += 1
    assert len(seen) >= 12, seen


def test_medium_key_is_the_argument_of_the_combination():
    forms = Counter()
    for seed in range(150):
        q = _q("medium", seed)
        (_, a), (_, b) = _expo(q.text, "z₁"), _expo(q.text, "z₂")
        z1, z2 = _unit(a), _unit(b)
        plain = re.search(r"Write (.*?) in the form", q.text).group(1)
        combo = {
            "z₁z₂": z1 * z2,
            "z₁²z₂": z1**2 * z2,
            "z₁z₂²": z1 * z2**2,
            "z₁/z₂": z1 / z2,
            "z₁²/z₂": z1**2 / z2,
            "z₁/z₂²": z1 / z2**2,
            "z₁ times the conjugate of z₂": z1 * sympy.conjugate(z2),
        }[plain]
        assert Fraction(q.answer_text) == _arg_over_pi(combo), (seed, q.text)
        forms[plain] += 1
    assert len(forms) == 7, forms


def test_hard_key_is_the_smallest_positive_argument_among_the_roots():
    outside = 0
    for seed in range(150):
        q = _q("hard", seed)
        n = "²³⁴⁵⁶".index(re.search(r"z([²³⁴⁵⁶])", q.text).group(1)) + 2
        rhs = re.search(r"= (.*?), the one", q.text).group(1)
        if "e^" in rhs:
            r, phi = _expo("z = " + rhs, "z")
            w = r * _unit(phi)
        else:
            w = sympy.sympify(rhs.replace("i", "*I")) if rhs.endswith("i") else sympy.sympify(rhs)
        roots = [sympy.expand(x) for x in sympy.Poly(sympy.Symbol("t") ** n - w, sympy.Symbol("t")).nroots(n=20)]
        args = [float(sympy.arg(x) / sympy.pi) for x in roots]
        positive = min(a for a in args if a > 1e-9)
        assert abs(float(Fraction(q.answer_text)) - positive) < 1e-7, (seed, q.text, args)
        outside += Fraction(q.answer_text) != Fraction(args[0]).limit_denominator(1000)
    assert outside > 20


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(100):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, seed
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
            assert text in q.distractor_values


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, find_text_defects(q, include_working=True))
        assert q.to_json() == _q(tier, seed).to_json()


def test_tiers_differ_in_shape():
    easy = _q("easy", 3).text
    medium = _q("medium", 3).text
    hard = _q("hard", 3).text
    assert "e^(" not in easy and "e^(" in medium and "e^(" in hard or "i is" in hard or "z" in hard
    assert "z₁" in medium and "z₁" not in easy and "z₁" not in hard
    assert re.search(r"z[²³⁴⁵⁶] =", hard) and "smallest positive" in hard


# -- exam chain ---------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number", "fraction", "fraction", "number"]
        assert [p.marks for p in q.parts] == [2, 2, 2, 2]
        assert [bool(p.follow_through) for p in q.parts] == [False, False, True, True]
        assert q.parts[0].sensible_range == (0, None)
        assert find_text_defects(q, include_working=True) == []
        assert q.to_json() == _exam(seed).to_json()


def test_exam_keys_match_independent_derivation():
    for seed in range(200):
        q = _exam(seed)
        z = _cartesian(q.text)
        n = "²³⁴⁵⁶".index(re.search(r"z([²³⁴⁵⁶])", q.parts[3].prompt).group(1)) + 2
        assert sympy.simplify(sympy.Abs(z) - int(q.parts[0].answer)) == 0
        assert Fraction(q.parts[1].answer_text) == _arg_over_pi(z)
        assert Fraction(q.parts[2].answer_text) == _arg_over_pi(z**n)
        assert sympy.expand(z**n) == int(q.parts[3].answer), (seed, q.text)


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


def test_exam_follow_through_from_a_wrong_argument():
    checked = 0
    for seed in range(80):
        q = _exam(seed)
        n = "²³⁴⁵⁶".index(re.search(r"z([²³⁴⁵⁶])", q.parts[3].prompt).group(1)) + 2
        canon = {p.label: p.answer_text for p in q.parts}
        for wrong in (w for w in q.parts[1].distractors if Fraction(w) != 0):  # a zero earlier answer carries nothing
            # the learner's own (b) is wrong and (c) is right for it, so (c) earns its marks by follow-through
            own_c = _reduce(n * Fraction(wrong))
            result = grade(_SKILL, q, {**canon, "b": wrong, "c": str(own_c)})
            assert result.parts["b"].marks == 0
            if own_c != Fraction(q.parts[2].answer_text):
                assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, wrong)
                checked += 1
    assert checked >= 100


def _reduce(p):
    p = Fraction(p)
    while p > 1:
        p -= 2
    while p <= -1:
        p += 2
    return p


def test_exam_answer_space():
    texts = [_exam(s).text for s in range(300)]
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(texts)) >= 25
    assert len(set(answers)) >= 25
    assert max(Counter(answers).values()) <= 30
