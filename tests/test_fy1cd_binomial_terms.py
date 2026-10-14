"""Tests `ucv1-binomial-theorem-terms` (topic `ucv1_binomial_terms`) and its exam chain `exd_x_fy_binomial_terms`.

The oracle reads the expression and the asked-for power of x back out of the question TEXT, expands it with SymPy and
reads the coefficient off.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy

import mathgen.generators.fy1cd_binomial_terms_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "ucv1_binomial_terms"
EXAM = "exd_x_fy_binomial_terms"
SKILL_ID = "ucv1-binomial-theorem-terms"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_X = sympy.Symbol("x")
_SUP = str.maketrans("²³⁴⁵⁶", "23456")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _expression(text):
    """The product of bracketed powers in the text, as a SymPy expression."""
    body = re.search(r"expansion of (.*?)\.(?: |$)", text).group(1)
    body = re.sub(r"x([²³⁴⁵⁶]+)", lambda m: "x**" + m.group(1).translate(_SUP), body)
    body = re.sub(r"\^\((-?\d+)\)", r"**(\1)", body)
    body = body.replace("^", "**").replace(")(", ")*(").replace(")  (", ")*(")
    body = re.sub(r"(\d)x", r"\1*x", body)
    body = re.sub(r"(\d|\))\s+\(", r"\1*(", body)
    return sympy.sympify(body, locals={"x": _X})


def _power(text):
    m = re.search(r"coefficient of x(?:\^\((-?\d+)\)|\^(\d+)|([²³⁴⁵⁶])|\b)", text)
    if m.group(1):
        return int(m.group(1))
    if m.group(2):
        return int(m.group(2))
    if m.group(3):
        return int(m.group(3).translate(_SUP))
    return 1


def _coefficient(expr, t):
    shifted = sympy.expand(expr * _X**80)
    return int(sympy.Poly(shifted, _X).coeff_monomial(_X ** (80 + t)))


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


def test_easy_key_is_the_coefficient_in_a_linear_binomial():
    for seed in range(150):
        q = _q("easy", seed)
        assert _coefficient(_expression(q.text), _power(q.text)) == int(q.answer), (seed, q.text)


def test_medium_key_is_the_coefficient_of_the_stated_power():
    powers = set()
    for seed in range(150):
        q = _q("medium", seed)
        t = _power(q.text)
        assert t != 0
        assert _coefficient(_expression(q.text), t) == int(q.answer), (seed, q.text)
        powers.add(t)
    assert len(powers) >= 8


def test_hard_key_is_the_independent_term_or_a_product_coefficient():
    seen = Counter()
    for seed in range(150):
        q = _q("hard", seed)
        if "independent" in q.text:
            assert _coefficient(_expression(q.text), 0) == int(q.answer), (seed, q.text)
            seen["independent"] += 1
        else:
            assert _coefficient(_expression(q.text), _power(q.text)) == int(q.answer), (seed, q.text)
            seen["product"] += 1
    assert seen["independent"] >= 60 and seen["product"] >= 30, seen


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
    easy = [_q("easy", s).text for s in range(40)]
    medium = [_q("medium", s).text for s in range(40)]
    hard = [_q("hard", s).text for s in range(40)]
    assert not any("/x" in t for t in easy)  # a plain (a + bx)^n
    assert all("/x" in t for t in medium)  # a power of x in the denominator
    assert any("independent" in t for t in hard) and any(re.search(r"\)\^\d+ \(", t) for t in hard)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    texts = [_q(tier, s).text for s in range(80)]
    answers = [_q(tier, s).answer_text for s in range(80)]
    assert len(set(texts)) >= 40, len(set(texts))
    assert len(set(answers)) >= 30


# -- exam chain ---------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number"] * 4
        assert [p.marks for p in q.parts] == [2, 1, 2, 3]
        assert [bool(p.follow_through) for p in q.parts] == [False, True, True, True]
        assert find_text_defects(q, include_working=True) == []
        assert q.to_json() == _exam(seed).to_json()


def test_exam_keys_match_independent_derivation():
    for seed in range(150):
        q = _exam(seed)
        text = q.text.replace("Consider the expansion of", "expansion of").replace(
            ", written in descending powers of x", ""
        )
        expr = _expression(text + ".")
        indep = _coefficient(expr, 0)
        assert int(q.parts[2].answer) == indep, (seed, q.text)
        # r is the index at which the power of x vanishes, and C(n, r) is the coefficient with the numbers removed
        n = int(re.search(r"\)\^(\d+)", q.text).group(1))
        r = int(q.parts[0].answer)
        assert int(q.parts[1].answer) == sympy.binomial(n, r)
        expo = int(re.search(r"power (-?\d+)", q.parts[3].prompt).group(1))
        assert _coefficient(expr, expo) == int(q.parts[3].answer), (seed, q.text)


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


def test_exam_follow_through_from_a_wrong_r():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        n = int(re.search(r"\)\^(\d+)", q.text).group(1))
        canon = {p.label: p.answer_text for p in q.parts}
        for wrong in q.parts[0].distractors:
            own_b = int(sympy.binomial(n, int(wrong))) if 0 <= int(wrong) <= n else None
            if own_b is None or own_b == int(q.parts[1].answer):
                continue
            result = grade(_SKILL, q, {**canon, "a": wrong, "b": str(own_b)})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (seed, wrong)
            checked += 1
    assert checked >= 60


def test_exam_answer_space():
    texts = [_exam(s).text for s in range(300)]
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(texts)) >= 60
    assert len(set(answers)) >= 60
    assert max(Counter(answers).values()) <= 20
