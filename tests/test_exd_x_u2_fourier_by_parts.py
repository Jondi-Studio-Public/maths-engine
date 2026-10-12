"""Exam tier for fourier-coefficient-simple: a sine integral by parts, then the coefficient it gives.

(a) several prompt shapes, some absent from the practice tiers; (b) each part's key re-derived from the RENDERED
PROMPT with SymPy's integrator over 300 seeds; (c) the skill's grader accepts the key and rejects every
distractor; (d) a wrong integral used correctly in part (b) earns that part's marks.
"""

from __future__ import annotations

import re

import mpmath
import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u2_fourier_by_parts"
SKILL_ID = "fourier-coefficient-simple"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_X = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"pi": sympy.pi})


def _oracle(q):
    """The integral and the coefficient, by numerical quadrature of the printed function and the printed n."""
    poly = re.search(r"f\(x\) = ([-+\dx^ ]+?)(?= is defined|\. )", q.stem).group(1)
    n = int(re.search(r"sin\((\d+)x\) dx", q.text).group(1))
    f = sympy.lambdify(_X, _parse(poly.replace(" ", "")), "mpmath")
    integral = mpmath.quad(lambda t: f(t) * mpmath.sin(n * t), [0, mpmath.pi])
    return integral, 2 * integral / mpmath.pi, n


@pytest.fixture(scope="module")
def questions():
    return [_q(s) for s in SEEDS]


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_q(seed)) == []


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages) for s in exam] == [(TOPIC, ["exam"])]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("fourier_coefficient_simple", seed=s, kind="step")) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b"]
        assert [p.marks for p in q.parts] == [3, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, True]
        assert all(p.answer_kind == "expression" for p in q.parts)


def test_both_parities_occur(questions):
    parities = {_oracle(q)[2] % 2 for q in questions[:60]}
    assert parities == {0, 1}


def test_deterministic():
    assert _q(3).to_dict() == _q(3).to_dict()


def test_key_matches_independent_derivation_over_300_seeds(questions):
    bad = []
    for q in questions:
        integral, coefficient, _ = _oracle(q)
        a, b = (_parse(p.answer_text) for p in q.parts)
        if not (
            abs(float(sympy.N(a, 30)) - float(integral)) < 1e-9
            and abs(float(sympy.N(b, 30)) - float(coefficient)) < 1e-9
        ):
            bad.append((q.text, q.answer_text))
    assert bad == []


def test_answer_space(questions):
    texts = [q.answer_text for q in questions]
    assert len(set(texts)) >= 100
    assert max(texts.count(t) for t in set(texts)) <= 15


def test_key_accepted_and_every_distractor_rejected(questions):
    bad = []
    for q in questions[:40]:
        full = grade(_SKILL, q, q.answer_text)
        if not (full.correct and full.marks == full.max_marks == 4):
            bad.append(("key rejected", q.text, q.answer_text))
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            assert all(note in CATALOGUE for note in p.option_notes.values())
            for d in p.distractors:
                if grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks != 0:
                    bad.append(("distractor accepted", q.text, p.label, d))
    assert bad == []


def test_follow_through_from_a_wrong_integral(questions):
    for q in questions[:40]:
        wrong = q.parts[0].distractors[0]
        b_wrong = str(sympy.expand(2 * _parse(wrong) / sympy.pi)).replace("**", "^")
        result = grade(_SKILL, q, {"a": wrong, "b": b_wrong})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through, (q.text, wrong, b_wrong)
        assert result.marks == 1 < result.max_marks and not result.correct


def test_all_wrong_earns_zero(questions):
    for q in questions[:30]:
        assert grade(_SKILL, q, {"a": "1", "b": "2"}).marks == 0
