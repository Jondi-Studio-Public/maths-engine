"""Tests `ucv1-complex-roots-real-polynomials` (topic `ucv1_complex_poly_roots`) and its exam chain.

The oracle reads the polynomial and the stated root back out of the question TEXT and re-derives each key with SymPy
(`Poly.real_roots`, `Poly.nroots`, and a solve for the unknown coefficient).
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy

import mathgen.generators.fy1cd_complex_poly_roots_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "ucv1_complex_poly_roots"
EXAM = "exd_x_fy_complex_poly_roots"
SKILL_ID = "ucv1-complex-roots-real-polynomials"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_X, _K = sympy.symbols("x k")
_SUP = str.maketrans("²³⁴", "234")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _poly(text):
    """The polynomial between 'The cubic/quartic ' and ' =' (or ' has'), as a SymPy expression in x (and k)."""
    body = re.search(r"The (?:cubic|quartic) (.*?)(?: = 0)? has", text).group(1)
    body = body.translate(_SUP)
    body = re.sub(r"x(\d)", r"x**\1", body)
    body = re.sub(r"(\d)(x|k)", r"\1*\2", body)
    body = body.replace("kx", "k*x")
    return sympy.sympify(body, locals={"x": _X, "k": _K})


def _stated_root(text):
    candidates = re.findall(r"(?<![\w²³⁴])(-?\d+ [+-] \d*i|-?\d*i)(?![\w])", text)
    raw = candidates[-1].replace(" ", "")
    raw = re.sub(r"(\d)i", r"\1*I", raw)
    raw = raw.replace("-i", "-I").replace("+i", "+I")
    if raw == "i":
        raw = "I"
    return sympy.sympify(raw)


def _real_roots(poly):
    return sorted(int(r) for r in sympy.Poly(poly, _X).real_roots())


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


def test_easy_key_is_the_real_root_of_the_stated_cubic():
    for seed in range(150):
        q = _q("easy", seed)
        poly = _poly(q.text)
        root = _stated_root(q.text)
        assert sympy.expand(poly.subs(_X, root)) == 0, (seed, q.text)
        assert _real_roots(poly) == [int(q.answer)], (seed, q.text)


def test_medium_key_is_the_product_or_the_sum_of_squares_of_the_real_roots():
    asked = Counter()
    for seed in range(150):
        q = _q("medium", seed)
        poly = _poly(q.text)
        assert sympy.expand(poly.subs(_X, _stated_root(q.text))) == 0
        r1, r2 = _real_roots(poly)
        if "product" in q.text:
            assert int(q.answer) == r1 * r2, (seed, q.text)
            asked["product"] += 1
        else:
            assert "sum of the squares" in q.text
            assert int(q.answer) == r1 * r1 + r2 * r2, (seed, q.text)
            asked["squares"] += 1
    assert min(asked.values()) >= 30, asked


def test_hard_key_is_the_unknown_coefficient_or_the_largest_real_root():
    asked = Counter()
    for seed in range(150):
        q = _q("hard", seed)
        poly = _poly(q.text)
        root = _stated_root(q.text)
        if "Find k" in q.text:
            solved = sympy.solve(sympy.expand(poly.subs(_X, root)), _K)
            assert solved == [int(q.answer)], (seed, q.text)
            full = poly.subs(_K, int(q.answer))
            assert sympy.expand(full.subs(_X, sympy.conjugate(root))) == 0
            asked["k"] += 1
        else:
            assert sympy.expand(poly.subs(_X, root)) == 0
            assert max(_real_roots(poly)) == int(q.answer), (seed, q.text)
            asked["largest"] += 1
    assert min(asked.values()) >= 30, asked


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
    degrees = {t: Counter("cubic" if "cubic" in _q(t, s).text else "quartic" for s in range(60)) for t in TIERS}
    assert set(degrees["easy"]) == {"cubic"}
    assert set(degrees["medium"]) == {"quartic"}
    assert set(degrees["hard"]) == {"cubic", "quartic"}
    assert all("Find its real root" in _q("easy", s).text for s in range(20))


# -- exam chain ---------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["expression", "number", "number", "number"]
        assert [p.marks for p in q.parts] == [1, 2, 2, 3]
        assert [bool(p.follow_through) for p in q.parts] == [False, True, True, True]
        assert find_text_defects(q, include_working=True) == []
        assert q.to_json() == _exam(seed).to_json()


def test_exam_keys_match_independent_derivation():
    for seed in range(150):
        q = _exam(seed)
        poly = _exam_poly(q.text)
        root = _stated_root(q.text)
        assert sympy.expand(poly.subs(_X, root)) == 0
        other = sympy.sympify(q.parts[0].answer_text.replace(" ", "").replace("i", "*I").replace("-*I", "-I"))
        assert sympy.expand(other - sympy.conjugate(root)) == 0, (seed, q.text)
        m = int(q.parts[1].answer)
        quadratic = sympy.expand((_X - root) * (_X - sympy.conjugate(root)))
        assert quadratic == _X**2 + (quadratic.coeff(_X, 1)) * _X + m
        quotient, remainder = sympy.div(sympy.Poly(poly, _X), sympy.Poly(quadratic, _X))
        assert remainder.is_zero
        assert int(q.parts[2].answer) == quotient.all_coeffs()[2]
        assert int(q.parts[3].answer) == max(_real_roots(poly))


def _exam_poly(text):
    body = re.search(r"The quartic (.*?) has real", text).group(1).translate(_SUP)
    body = re.sub(r"x(\d)", r"x**\1", body)
    body = re.sub(r"(\d)x", r"\1*x", body)
    return sympy.sympify(body, locals={"x": _X})


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


def test_exam_follow_through_from_a_wrong_conjugate_and_a_wrong_constant():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        # (b): the learner's own (a) is wrong, and (b) is right for it
        wrong_a = q.parts[0].distractors[0]
        w = sympy.sympify(wrong_a.replace(" ", "").replace("i", "*I").replace("-*I", "-I").replace("+*I", "+I"))
        own_m = int(sympy.expand(sympy.re(w) ** 2 + sympy.im(w) ** 2))
        result = grade(_SKILL, q, {**canon, "a": wrong_a, "b": str(own_m)})
        if own_m != int(q.parts[1].answer):
            assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, wrong_a)
            checked += 1
        # (c) from a wrong (b)
        e = int(_exam_poly(q.text).subs(_X, 0))
        wrong_b = int(q.parts[1].distractors[0])
        if wrong_b and e % wrong_b == 0:
            own_v = e // wrong_b
            result = grade(_SKILL, q, {**canon, "b": str(wrong_b), "c": str(own_v)})
            if own_v != int(q.parts[2].answer) and own_v != 0:
                assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, wrong_b)
                checked += 1
    assert checked >= 25


def test_exam_answer_space():
    texts = [_exam(s).text for s in range(300)]
    assert len(set(texts)) >= 200
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(Counter(answers).values()) <= 20
