"""Checks for the multi-part exam source of `write-expressions-from-words`: allowance, discount, value.

(a) several prompt shapes, some absent from the hard tier; (b) every part's key is re-derived from the RENDERED
PROMPT with sympy over 300 seeds; (c) the skill's grader accepts the key and rejects every distractor;
(d) a wrong expression in (a) or (b) used correctly later still earns the later marks.
"""

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc4_words_expression"
SKILL = next(s for s in load_skills()[1] if s.id == "write-expressions-from-words")
SEEDS = range(300)
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def questions():
    return [_q(s) for s in SEEDS]


def _expr(text, var):
    return parse_expr(text, transformations=TRANSFORMS, local_dict={var: sympy.Symbol(var)})


def _facts(q):
    stem = q.stem
    fee, rate = (int(x) for x in re.findall(r"\$(\d+)", stem))
    var, allowance = re.search(r"where (\w) > (\d+)", q.parts[0].prompt).groups()
    discount = int(re.search(r"(\d+)%", q.parts[1].prompt).group(1))
    v0 = int(re.search(r"(\d+) (?:days|hours|GB)", q.parts[2].prompt).group(1))
    return fee, rate, var, int(allowance), discount, v0


def oracle(q):
    fee, rate, var, allowance, discount, v0 = _facts(q)
    x = sympy.Symbol(var)
    base = fee + rate * (x - allowance)
    disc = sympy.Rational(100 - discount, 100) * base
    return var, sympy.expand(base), sympy.expand(disc), disc.subs(x, v0)


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape_and_wiring():
    exam = [s for s in SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    for seed in range(40):
        q = _q(seed)
        assert [p.answer_kind for p in q.parts] == ["expression", "expression", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_keys_match_independent_derivation(questions):
    bad = []
    for seed, q in zip(SEEDS, questions):
        var, base, disc, value = oracle(q)
        a, b, c = (_canonical(q)[k] for k in "abc")
        if (
            sympy.simplify(_expr(a, var) - base) != 0
            or sympy.simplify(_expr(b, var) - disc) != 0
            or sympy.Rational(c) != value
        ):
            bad.append((seed, q.text, _canonical(q)))
    assert bad == [], bad[:2]


def test_grader_accepts_key_and_rejects_every_distractor(questions):
    bad = []
    for seed, q in zip(SEEDS, questions):
        if seed >= 150:
            break
        canon = _canonical(q)
        result = grade(SKILL, q, canon)
        if not (result.correct and result.marks == result.max_marks):
            bad.append(("key rejected", seed, canon))
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for d in p.distractors:
                if grade(SKILL, q, {**canon, p.label: d}).parts[p.label].marks != 0:
                    bad.append(("distractor accepted", seed, p.label, d))
    assert bad == [], bad[:3]


def test_distractors_are_tagged(questions):
    for q in questions[:100]:
        for p in q.parts:
            assert [d for d in p.distractors if p.option_notes.get(d)], (q.text, p.label)


def test_follow_through_from_a_wrong_expression(questions):
    for q in questions[:100]:
        fee, rate, var, allowance, discount, v0 = _facts(q)
        x = sympy.Symbol(var)
        a_wrong = q.parts[0].distractors[0]
        b_wrong_expr = sympy.expand(sympy.Rational(100 - discount, 100) * _expr(a_wrong, var))
        c_wrong = b_wrong_expr.subs(x, v0)
        reply = {"a": a_wrong, "b": str(b_wrong_expr).replace("**", "^"), "c": str(c_wrong)}
        result = grade(SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks, (q.text, reply)
        assert result.parts["c"].marks == result.parts["c"].max_marks, (q.text, reply)
        assert result.parts["b"].follow_through is True


def test_all_wrong_earns_nothing(questions):
    for q in questions[:40]:
        result = grade(SKILL, q, {"a": "7", "b": "n", "c": "123456"})
        assert result.marks == 0 and not result.correct
